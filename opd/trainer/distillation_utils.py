from typing import Literal

import torch
import torch.distributed as dist

from opd.fsdp.algorithms import (
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
)


def broadcast_minibatch(
    is_student: bool,
    mb_ids: torch.Tensor | None,
    mb_attn: torch.Tensor | None,
    device: torch.device,
    all_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast input_ids and attention_mask from student rank 0 to all ranks."""
    if is_student:
        assert mb_ids is not None and mb_attn is not None
        shape_t = torch.tensor([mb_ids.shape[0], mb_ids.shape[1]], dtype=torch.long, device=device)
    else:
        shape_t = torch.zeros(2, dtype=torch.long, device=device)

    dist.broadcast(shape_t, src=0, group=all_group)
    B, T = int(shape_t[0].item()), int(shape_t[1].item())

    if not is_student:
        mb_ids  = torch.zeros(B, T, dtype=torch.long, device=device)
        mb_attn = torch.zeros(B, T, dtype=torch.long, device=device)

    assert mb_ids is not None and mb_attn is not None
    dist.broadcast(mb_ids,  src=0, group=all_group)
    dist.broadcast(mb_attn, src=0, group=all_group)
    return mb_ids, mb_attn


def broadcast_teacher_inputs(is_student, t_mb_ids, t_mb_attn, t_mb_mask, device, all_group):
    """Broadcast teacher-specific tensors from student rank 0 to all ranks.

    Used when teacher and student have different inputs (e.g. SDFT, where the
    teacher prompt includes a worked demonstration).
    """
    if is_student:
        shape_t = torch.tensor([t_mb_ids.shape[0], t_mb_ids.shape[1]], dtype=torch.long, device=device)
    else:
        shape_t = torch.zeros(2, dtype=torch.long, device=device)
    dist.broadcast(shape_t, src=0, group=all_group)
    B, T_t = int(shape_t[0].item()), int(shape_t[1].item())
    if not is_student:
        t_mb_ids  = torch.zeros(B, T_t, dtype=torch.long,  device=device)
        t_mb_attn = torch.zeros(B, T_t, dtype=torch.long,  device=device)
        t_mb_mask = torch.zeros(B, T_t, dtype=torch.float, device=device)
    dist.broadcast(t_mb_ids,  src=0, group=all_group)
    dist.broadcast(t_mb_attn, src=0, group=all_group)
    dist.broadcast(t_mb_mask, src=0, group=all_group)
    return t_mb_ids, t_mb_attn, t_mb_mask


def exchange_topk(
    *,
    select_topk_by: Literal["student", "teacher"],
    is_student: bool,
    is_teacher: bool,
    student_logits: torch.Tensor | None,          # [B, T, V], student rank only
    teacher_logits: torch.Tensor | None,          # [B, T, V], teacher rank only
    t_compact_mask: torch.Tensor | None = None,   # [B, T], teacher rank only; None → broadcast all-ones
    B: int = 0,
    T: int = 0,
    K: int = 0,
    s_chunk: int = -1,
    t_chunk: int = -1,
    teacher_global_rank: int = 0,
    all_group=None,
    device: torch.device | None = None,
) -> dict:
    """Top-K log-prob exchange between student and teacher ranks.

    Works for both full-sequence logits [B, T-1, V] (OPD) and packed
    response-aligned logits [B, R_max, V] (SDFT). The caller is responsible
    for computing teacher_logits on the teacher rank (and optionally packing
    it and providing t_compact_mask when student/teacher sequences differ).

    Returns a dict with keys: topk_idx, t_logprobs, t_compact_mask,
    student_topk_idx, teacher_topk_idx, t_logprobs_at_student, teacher_own_logprobs.
    """
    if is_teacher and t_compact_mask is None:
        t_compact_mask = torch.ones(B, T, dtype=torch.float32, device=device)

    if select_topk_by == "student":
        if is_student:
            topk_idx = student_topk_indices(student_logits, K, s_chunk)
        else:
            topk_idx = torch.empty(B, T, K, dtype=torch.long, device=device)
        dist.broadcast(topk_idx, src=0, group=all_group)

        if is_teacher:
            t_logprobs           = teacher_logprobs_at_indices(teacher_logits, topk_idx, t_chunk)
            teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(teacher_logits, K, t_chunk)
        else:
            t_logprobs           = torch.empty(B, T, K, dtype=torch.bfloat16, device=device)
            teacher_topk_idx     = torch.empty(B, T, K, dtype=torch.long,     device=device)
            teacher_own_logprobs = torch.empty(B, T, K, dtype=torch.bfloat16, device=device)
            t_compact_mask       = torch.empty(B, T,    dtype=torch.float32,  device=device)

        dist.broadcast(t_logprobs,           src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_topk_idx,     src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_own_logprobs, src=teacher_global_rank, group=all_group)
        dist.broadcast(t_compact_mask,       src=teacher_global_rank, group=all_group)

        student_topk_idx      = topk_idx
        t_logprobs_at_student = t_logprobs

    else:  # forward_kl: teacher picks top-K
        if is_student:
            student_topk_idx = student_topk_indices(student_logits, K, s_chunk)
        else:
            student_topk_idx = torch.empty(B, T, K, dtype=torch.long, device=device)
        dist.broadcast(student_topk_idx, src=0, group=all_group)

        if is_teacher:
            teacher_topk_idx, t_logprobs = teacher_topk_logprobs(teacher_logits, K, t_chunk)
            teacher_own_logprobs  = t_logprobs
            t_logprobs_at_student = teacher_logprobs_at_indices(teacher_logits, student_topk_idx, t_chunk)
        else:
            teacher_topk_idx      = torch.empty(B, T, K, dtype=torch.long,     device=device)
            t_logprobs            = torch.empty(B, T, K, dtype=torch.bfloat16, device=device)
            teacher_own_logprobs  = torch.empty(B, T, K, dtype=torch.bfloat16, device=device)
            t_logprobs_at_student = torch.empty(B, T, K, dtype=torch.bfloat16, device=device)
            t_compact_mask        = torch.empty(B, T,    dtype=torch.float32,  device=device)

        dist.broadcast(teacher_topk_idx,      src=teacher_global_rank, group=all_group)
        dist.broadcast(t_logprobs,            src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_own_logprobs,  src=teacher_global_rank, group=all_group)
        dist.broadcast(t_logprobs_at_student, src=teacher_global_rank, group=all_group)
        dist.broadcast(t_compact_mask,        src=teacher_global_rank, group=all_group)

        topk_idx = teacher_topk_idx

    return {
        "topk_idx":              topk_idx,
        "t_logprobs":            t_logprobs,
        "t_compact_mask":        t_compact_mask,
        "student_topk_idx":      student_topk_idx,
        "teacher_topk_idx":      teacher_topk_idx,
        "t_logprobs_at_student": t_logprobs_at_student,
        "teacher_own_logprobs":  teacher_own_logprobs,
    }
