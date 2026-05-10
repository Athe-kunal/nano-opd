import torch
from typing import Literal

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

    s_lp = s.gather(-1, topk_idx) - s_lse.unsqueeze(-1)
    with torch.no_grad():
        t_lp = t.gather(-1, topk_idx) - t_lse.unsqueeze(-1)

    return s_lp, t_lp, topk_idx


def compute_topk_logprobs_for_distillation(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int = 100,
    select_topk_by: Literal["student", "teacher"] = "student",
    student_chunk_size: int = -1,
    teacher_chunk_size: int = -1,
):
    """
    Compute top-K log-probabilities for distillation, optionally processing
    the T dimension in chunks to bound peak memory.

    ``student_chunk_size`` / ``teacher_chunk_size`` set the maximum number of
    sequence positions processed at once for the student / teacher tensors
    respectively.  Both tensors are iterated over the same T-slices, so the
    effective chunk size is ``min(active flags)``.  Set either flag to ``-1``
    (default) to disable chunking.

    Returns:
        s_topk_logprobs: [B, T, K]
        t_topk_logprobs: [B, T, K]
        topk_idx:        [B, T, K]
    """
    active = [c for c in (student_chunk_size, teacher_chunk_size) if c > 0]
    if not active:
        return _topk_logprobs_slice(student_logits, teacher_logits, top_k, select_topk_by)

    chunk = min(active)
    T = student_logits.shape[1]
    s_parts, t_parts, idx_parts = [], [], []
    for start in range(0, T, chunk):
        s_lp, t_lp, idx = _topk_logprobs_slice(
            student_logits[:, start : start + chunk],
            teacher_logits[:, start : start + chunk],
            top_k, select_topk_by,
        )
        s_parts.append(s_lp)
        t_parts.append(t_lp)
        idx_parts.append(idx)

    return torch.cat(s_parts, dim=1), torch.cat(t_parts, dim=1), torch.cat(idx_parts, dim=1)


