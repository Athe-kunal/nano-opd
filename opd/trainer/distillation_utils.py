from typing import Literal

import torch
import torch.distributed as dist

from opd.fsdp.algorithms import (
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
)


def bcast_or_alloc_async(*, tensor, is_owner, shape, dtype, src, device, group):
    """If not the owning rank, allocate a placeholder; issue the broadcast without waiting for it.

    Returns (tensor, handle). Call handle.wait() before reading the tensor.
    Lets several independent broadcasts overlap on the wire instead of
    completing one at a time.
    """
    if not is_owner:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    handle = dist.broadcast(tensor, src=src, group=group, async_op=True)
    return tensor, handle


def broadcast_minibatch(
    is_student: bool,
    mb_ids: torch.Tensor | None,
    mb_attn: torch.Tensor | None,
    device: torch.device,
    all_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast input_ids and attention_mask from student rank 0 to all ranks."""
    shape_t = torch.tensor([mb_ids.shape[0], mb_ids.shape[1]], dtype=torch.long, device=device) if is_student else None
    shape_t, h = bcast_or_alloc_async(tensor=shape_t, is_owner=is_student, shape=(2,), dtype=torch.long, src=0, device=device, group=all_group)
    h.wait()
    B, T = int(shape_t[0].item()), int(shape_t[1].item())

    mb_ids,  h1 = bcast_or_alloc_async(tensor=mb_ids,  is_owner=is_student, shape=(B, T), dtype=torch.long, src=0, device=device, group=all_group)
    mb_attn, h2 = bcast_or_alloc_async(tensor=mb_attn, is_owner=is_student, shape=(B, T), dtype=torch.long, src=0, device=device, group=all_group)
    h1.wait()
    h2.wait()
    return mb_ids, mb_attn


def broadcast_teacher_inputs(is_student, t_mb_ids, t_mb_attn, t_mb_mask, device, all_group):
    """Broadcast teacher-specific tensors from student rank 0 to all ranks.

    Used when teacher and student have different inputs (e.g. SDFT, where the
    teacher prompt includes a worked demonstration).
    """
    shape_t = torch.tensor([t_mb_ids.shape[0], t_mb_ids.shape[1]], dtype=torch.long, device=device) if is_student else None
    shape_t, h = bcast_or_alloc_async(tensor=shape_t, is_owner=is_student, shape=(2,), dtype=torch.long, src=0, device=device, group=all_group)
    h.wait()
    B, T_t = int(shape_t[0].item()), int(shape_t[1].item())

    t_mb_ids,  h1 = bcast_or_alloc_async(tensor=t_mb_ids,  is_owner=is_student, shape=(B, T_t), dtype=torch.long,  src=0, device=device, group=all_group)
    t_mb_attn, h2 = bcast_or_alloc_async(tensor=t_mb_attn, is_owner=is_student, shape=(B, T_t), dtype=torch.long,  src=0, device=device, group=all_group)
    t_mb_mask, h3 = bcast_or_alloc_async(tensor=t_mb_mask, is_owner=is_student, shape=(B, T_t), dtype=torch.float, src=0, device=device, group=all_group)
    for h in (h1, h2, h3):
        h.wait()
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

    if select_topk_by == "student": #reverse KL
        topk_idx = student_topk_indices(student_logits, K, s_chunk) if is_student else None
        topk_idx, h = bcast_or_alloc_async(tensor=topk_idx, is_owner=is_student, shape=(B, T, K), dtype=torch.long, src=0, device=device, group=all_group)
        h.wait()

        if is_teacher:
            t_logprobs                             = teacher_logprobs_at_indices(teacher_logits, topk_idx, t_chunk)
            teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(teacher_logits, K, t_chunk)
        else:
            t_logprobs = teacher_topk_idx = teacher_own_logprobs = None

        t_logprobs,           h1 = bcast_or_alloc_async(tensor=t_logprobs,           is_owner=is_teacher, shape=(B, T, K), dtype=torch.bfloat16, src=teacher_global_rank, device=device, group=all_group)
        teacher_topk_idx,     h2 = bcast_or_alloc_async(tensor=teacher_topk_idx,     is_owner=is_teacher, shape=(B, T, K), dtype=torch.long,     src=teacher_global_rank, device=device, group=all_group)
        teacher_own_logprobs, h3 = bcast_or_alloc_async(tensor=teacher_own_logprobs, is_owner=is_teacher, shape=(B, T, K), dtype=torch.bfloat16, src=teacher_global_rank, device=device, group=all_group)
        t_compact_mask,       h4 = bcast_or_alloc_async(tensor=t_compact_mask,       is_owner=is_teacher, shape=(B, T),    dtype=torch.float32,  src=teacher_global_rank, device=device, group=all_group)
        for h in (h1, h2, h3, h4):
            h.wait()

        student_topk_idx      = topk_idx
        t_logprobs_at_student = t_logprobs

    else:  # forward_kl: teacher picks top-K
        student_topk_idx = student_topk_indices(student_logits, K, s_chunk) if is_student else None
        student_topk_idx, h = bcast_or_alloc_async(tensor=student_topk_idx, is_owner=is_student, shape=(B, T, K), dtype=torch.long, src=0, device=device, group=all_group)
        h.wait()

        if is_teacher:
            teacher_topk_idx, t_logprobs = teacher_topk_logprobs(teacher_logits, K, t_chunk)
            teacher_own_logprobs  = t_logprobs
            t_logprobs_at_student = teacher_logprobs_at_indices(teacher_logits, student_topk_idx, t_chunk)
        else:
            teacher_topk_idx = t_logprobs = teacher_own_logprobs = t_logprobs_at_student = None

        teacher_topk_idx,      h1 = bcast_or_alloc_async(tensor=teacher_topk_idx,      is_owner=is_teacher, shape=(B, T, K), dtype=torch.long,     src=teacher_global_rank, device=device, group=all_group)
        t_logprobs,            h2 = bcast_or_alloc_async(tensor=t_logprobs,            is_owner=is_teacher, shape=(B, T, K), dtype=torch.bfloat16, src=teacher_global_rank, device=device, group=all_group)
        teacher_own_logprobs,  h3 = bcast_or_alloc_async(tensor=teacher_own_logprobs,  is_owner=is_teacher, shape=(B, T, K), dtype=torch.bfloat16, src=teacher_global_rank, device=device, group=all_group)
        t_logprobs_at_student, h4 = bcast_or_alloc_async(tensor=t_logprobs_at_student, is_owner=is_teacher, shape=(B, T, K), dtype=torch.bfloat16, src=teacher_global_rank, device=device, group=all_group)
        t_compact_mask,        h5 = bcast_or_alloc_async(tensor=t_compact_mask,        is_owner=is_teacher, shape=(B, T),    dtype=torch.float32,  src=teacher_global_rank, device=device, group=all_group)
        for h in (h1, h2, h3, h4, h5):
            h.wait()

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
