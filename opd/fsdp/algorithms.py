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
    input_ids: torch.Tensor,  #[B,T]
    K: int,
    chunk_size: int = -1,
) -> tuple[torch.Tensor,torch.Tensor]:                  # [B, T, K]
    """Select top-K vocab indices from student logits (no grad)."""
    T = student_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    parts = []
    gathered_logits = []
   
    for t0, t1 in _chunk_range(T, chunk):
        curr_student_logits = student_logits[:,t0:t1]
        curr_input_ids = input_ids[:,t0:t1]
        with torch.no_grad():
            parts.append(curr_student_logits.topk(K, dim=-1).indices)
        lp = torch.log_softmax(curr_student_logits.float(), dim=-1)
        gathered_logits.append(torch.gather(lp,dim=-1,index=curr_input_ids.unsqueeze(-1)).squeeze(-1))
    return torch.cat(parts, dim=1), torch.cat(gathered_logits,dim=1)


def teacher_logprobs_at_indices(
    teacher_logits: torch.Tensor,   # [B, T, V]
    topk_idx: torch.Tensor,         # [B, T, K]
    chunk_size: int = -1,
) -> torch.Tensor:                  # [B, T, K]
    """Compute teacher log-probs at pre-selected top-K indices (no grad)."""
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
    K: int,
    chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:  # ([B, T, K], [B, T, K])
    """Teacher selects top-K indices and computes its own log-probs (no grad)."""
    T = teacher_logits.shape[1]
    chunk = _effective_chunk(T, chunk_size)
    idx_parts, lp_parts = [], []
    with torch.no_grad():
        for t0, t1 in _chunk_range(T, chunk):
            sl = teacher_logits[:, t0:t1]
            lse = torch.logsumexp(sl, dim=-1)
            _, idx = sl.topk(K, dim=-1)
            idx_parts.append(idx)
            lp_parts.append(_logprobs_at(sl, idx, lse))
    del teacher_logits
    return torch.cat(idx_parts, dim=1), torch.cat(lp_parts, dim=1)


def student_logprob_at_sampled_tokens(
    student_logits: torch.Tensor,   # [B, T, V]
    token_ids: torch.Tensor,        # [B, T]
) -> torch.Tensor:                  # [B, T]
    """Log-prob of the specific sampled token at each position (no grad).

    Used to compute per-token TIS weights:
        w_t = exp(log π_train(y_t) − log π_vllm(y_t))
    which correct for the numerical gap between vLLM inference log-probs and
    the training-time forward pass (SDPO paper, Appendix A.4).
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
    """Compute student log-probs at given indices, preserving the gradient graph."""
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
    s: torch.Tensor,   # [B, C, V]
    t: torch.Tensor,   # [B, C, V]
    top_k: int,
    select_topk_by: Literal["student", "teacher"],
):
    """Core top-K log-prob computation for one [B, C, V] slice."""
    s_lse = torch.logsumexp(s, dim=-1)              # [B, C]
    with torch.no_grad():
        t_lse = torch.logsumexp(t, dim=-1)          # [B, C]

    if select_topk_by == "student":
        _, topk_idx = s.topk(top_k, dim=-1)         # [B, C, K]
    else:
        with torch.no_grad():
            _, topk_idx = t.topk(top_k, dim=-1)

    s_lp = _logprobs_at(s, topk_idx, s_lse)
    with torch.no_grad():
        t_lp = _logprobs_at(t, topk_idx, t_lse)

    return s_lp, t_lp, topk_idx


def compute_topk_logprobs_for_distillation(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int = 100,
    select_topk_by: Literal["student", "teacher"] = "student",
    student_chunk_size: int = -1,
    teacher_chunk_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-K log-probabilities for distillation (non-distributed).

    Returns:
        s_topk_logprobs: [B, T, K]
        t_topk_logprobs: [B, T, K]
        topk_idx:        [B, T, K]
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
