import torch
from typing import Literal
from einops import rearrange


def _chunk_range(T: int, chunk: int):
    """Yield (start, end) pairs that partition [0, T) into slices of size `chunk`."""
    for t0 in range(0, T, chunk):
        yield t0, min(t0 + chunk, T)


def _effective_chunk(T: int, *chunk_sizes: int) -> int:
    """Return the smallest positive chunk_size, or T (no-op) if none are positive."""
    active = [c for c in chunk_sizes if c > 0]
    return min(active) if active else T


def _logprobs_at(logits_slice: torch.Tensor, idx: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Gather log-probs at idx from a [B, C, V] logits slice, given its logsumexp [B, C]."""
    return logits_slice.gather(-1, idx) - rearrange(lse, "b c -> b c 1")


# ---------------------------------------------------------------------------
# Per-rank helpers used by the distributed training loop
# ---------------------------------------------------------------------------

def student_topk_indices(
    student_logits: torch.Tensor,   # [B, T, V]
    top_k: int,
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Selects top-K vocab indices from student logits (no grad).

    Args:
        student_logits: Student logits, `[B, T, V]`.
        top_k: Number of vocab indices to keep per position.
        chunk_size: Chunk size along T (-1 = no chunking).

    Returns:
        Top-K vocab indices, `[B, T, K]`.
    """
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []

    for t0, t1 in _chunk_range(T, chunk):
        curr_student_logits = student_logits[:,t0:t1]
        with torch.no_grad():
            parts.append(curr_student_logits.topk(top_k, dim=-1).indices)
    return torch.cat(parts, dim=1)


def teacher_logprobs_at_indices(
    teacher_logits: torch.Tensor,   # [B, T, V]
    topk_idx: torch.Tensor,         # [B, T, K]
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Computes teacher log-probs at pre-selected top-K indices (no grad).

    Args:
        teacher_logits: Teacher logits, `[B, T, V]`.
        topk_idx: Vocab indices to gather log-probs at, `[B, T, K]`.
        chunk_size: Chunk size along T (-1 = no chunking).

    Returns:
        Teacher log-probs at `topk_idx`, `[B, T, K]`.
    """
    T = teacher_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = teacher_logits[:, t0:t1]
            lse = torch.logsumexp(sl, dim=-1)
            parts.append(_logprobs_at(sl, topk_idx[:, t0:t1], lse))
    del teacher_logits
    return torch.cat(parts, dim=1)


def teacher_topk_logprobs(
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int,
    chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:  # ([B, T, K], [B, T, K])
    """Teacher selects top-K indices and computes its own log-probs (no grad).

    Args:
        teacher_logits: Teacher logits, `[B, T, V]`.
        top_k: Number of vocab indices to keep per position.
        chunk_size: Chunk size along T (-1 = no chunking).

    Returns:
        `(topk_idx, topk_logprobs)`, each `[B, T, K]`.
    """
    T = teacher_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    idx_parts, lp_parts = [], []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = teacher_logits[:, t0:t1]
            lse = torch.logsumexp(sl, dim=-1)
            _, idx = sl.topk(top_k, dim=-1)
            idx_parts.append(idx)
            lp_parts.append(_logprobs_at(sl, idx, lse))
    del teacher_logits
    return torch.cat(idx_parts, dim=1), torch.cat(lp_parts, dim=1)


def student_logprob_at_sampled_tokens(
    student_logits: torch.Tensor,   # [B, T, V]
    token_ids: torch.Tensor,        # [B, T]
) -> torch.Tensor:                  # [B, T]
    """Returns the log-prob of the specific sampled token at each position (no grad).

    Used to compute per-token TIS weights:
        w_t = exp(log π_train(y_t) − log π_vllm(y_t))
    which correct for the numerical gap between vLLM inference log-probs and
    the training-time forward pass.

    Args:
        student_logits: Student logits, `[B, T, V]`.
        token_ids: The sampled token id at each position, `[B, T]`.

    Returns:
        Log-prob of `token_ids` under `student_logits`, `[B, T]`.
    """
    with torch.no_grad():
        lse = torch.logsumexp(student_logits, dim=-1)          # [B, T]
        gathered = student_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
        return gathered - lse


def student_logprobs_at_indices(
    student_logits: torch.Tensor,   # [B, T, V]  — retains grad
    topk_idx: torch.Tensor,         # [B, T, K]
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Computes student log-probs at given indices, preserving the gradient graph.

    Args:
        student_logits: Student logits, `[B, T, V]`, with grad enabled.
        topk_idx: Vocab indices to gather log-probs at, `[B, T, K]`.
        chunk_size: Chunk size along T (-1 = no chunking).

    Returns:
        Student log-probs at `topk_idx`, `[B, T, K]`, differentiable.
    """
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    for t0, t1 in _chunk_range(T, chunk):
        sl = student_logits[:, t0:t1]
        lse = torch.logsumexp(sl, dim=-1)
        parts.append(_logprobs_at(sl, topk_idx[:, t0:t1], lse))
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Combined helper (non-distributed / single-process use)
# ---------------------------------------------------------------------------

def _topk_logprobs_slice(
    student_logits_slice: torch.Tensor,   # [B, C, V]
    teacher_logits_slice: torch.Tensor,   # [B, C, V]
    top_k: int,
    select_topk_by: Literal["student", "teacher"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core top-K log-prob computation for one [B, C, V] slice.

    Args:
        student_logits_slice: Student logits for this T-chunk, `[B, C, V]`.
        teacher_logits_slice: Teacher logits for this T-chunk, `[B, C, V]`.
        top_k: Number of vocab indices to keep per position.
        select_topk_by: Whether the student or the teacher picks the top-K
          indices used by both.

    Returns:
        `(student_logprobs, teacher_logprobs, topk_idx)`, each `[B, C, K]`
        (`student_logprobs` is the only one with a gradient graph).
    """
    s_lse = torch.logsumexp(student_logits_slice, dim=-1)              # [B, C]
    with torch.no_grad():
        t_lse = torch.logsumexp(teacher_logits_slice, dim=-1)          # [B, C]

    if select_topk_by == "student":
        _, topk_idx = student_logits_slice.topk(top_k, dim=-1)         # [B, C, K]
    else:
        with torch.no_grad():
            _, topk_idx = teacher_logits_slice.topk(top_k, dim=-1)

    s_lp = _logprobs_at(student_logits_slice, topk_idx, s_lse)
    with torch.no_grad():
        t_lp = _logprobs_at(teacher_logits_slice, topk_idx, t_lse)

    return s_lp, t_lp, topk_idx


def compute_topk_logprobs_for_distillation(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int = 100,
    select_topk_by: Literal["student", "teacher"] = "student",
    student_chunk_size: int = -1,
    teacher_chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes top-K log-probabilities for distillation (non-distributed).

    Args:
        student_logits: Student logits, `[B, T, V]`.
        teacher_logits: Teacher logits, `[B, T, V]`.
        top_k: Number of vocab indices to keep per position.
        select_topk_by: Whether the student or the teacher picks the top-K
          indices used by both.
        student_chunk_size: Chunk size along T for the student (-1 = no
          chunking).
        teacher_chunk_size: Chunk size along T for the teacher (-1 = no
          chunking).

    Returns:
        A tuple `(student_topk_logprobs, teacher_topk_logprobs, topk_idx)`,
        each `[B, T, K]`.
    """
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, student_chunk_size, teacher_chunk_size)
    s_parts, t_parts, idx_parts = [], [], []
    for t0, t1 in _chunk_range(T, chunk):
        s_lp, t_lp, idx = _topk_logprobs_slice(
            student_logits[:, t0:t1],
            teacher_logits[:, t0:t1],
            top_k, select_topk_by,
        )
        s_parts.append(s_lp)
        t_parts.append(t_lp)
        idx_parts.append(idx)
    return torch.cat(s_parts, dim=1), torch.cat(t_parts, dim=1), torch.cat(idx_parts, dim=1)
