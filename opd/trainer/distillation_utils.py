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
    """Broadcast input_ids and attention_mask from student rank 0 to all ranks.

    Student rank 0 owns the real tensors; teacher rank allocates zeros of the
    right shape. Returns (mb_ids, mb_attn) filled on every rank.
    """
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


def exchange_topk(
    *,
    select_topk_by: Literal["student", "teacher"],
    is_student: bool,
    is_teacher: bool,
    student_logits: torch.Tensor | None,  # [B, T-1, V], only valid on student
    teacher,                               # TeacherModel, only valid on teacher rank
    mb_ids: torch.Tensor,
    mb_attn: torch.Tensor,
    K: int,
    s_chunk: int,
    t_chunk: int,
    teacher_global_rank: int,
    all_group,
    device: torch.device,
) -> dict:
    """Coordinate the top-K log-prob exchange between student and teacher ranks.

    Returns a dict with keys:
      topk_idx            - indices used for the distillation loss
      t_logprobs          - teacher log-probs at topk_idx          [B, T-1, K]
      student_topk_idx    - student's own top-K indices            [B, T-1, K]
      teacher_topk_idx    - teacher's own top-K indices            [B, T-1, K]
      t_logprobs_at_student - teacher log-probs at student top-K   [B, T-1, K]
      teacher_own_logprobs  - teacher log-probs at teacher top-K   [B, T-1, K]
    """
    B, T_shifted = mb_ids.shape[0], mb_ids.shape[1] - 1

    if select_topk_by == "student":
        # Student picks top-K; teacher evaluates at those indices.
        if is_student:
            assert student_logits is not None
            topk_idx = student_topk_indices(student_logits, K, s_chunk)
        else:
            topk_idx = torch.empty(B, T_shifted, K, dtype=torch.long, device=device)
        dist.broadcast(topk_idx, src=0, group=all_group)

        if is_teacher:
            t_logits = teacher.get_logits(mb_ids, mb_attn)[:, :-1]
            t_logprobs = teacher_logprobs_at_indices(t_logits, topk_idx, t_chunk)
            teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(t_logits, K, t_chunk)
        else:
            t_logprobs           = torch.empty(B, T_shifted, K, dtype=torch.bfloat16, device=device)
            teacher_topk_idx     = torch.empty(B, T_shifted, K, dtype=torch.long,     device=device)
            teacher_own_logprobs = torch.empty(B, T_shifted, K, dtype=torch.bfloat16, device=device)

        dist.broadcast(t_logprobs,           src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_topk_idx,     src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_own_logprobs, src=teacher_global_rank, group=all_group)

        student_topk_idx      = topk_idx
        t_logprobs_at_student = t_logprobs  # teacher was already evaluated at student indices

    else:  # forward_kl: teacher picks top-K
        if is_student:
            assert student_logits is not None
            student_topk_idx = student_topk_indices(student_logits, K, s_chunk)
        else:
            student_topk_idx = torch.empty(B, T_shifted, K, dtype=torch.long, device=device)
        dist.broadcast(student_topk_idx, src=0, group=all_group)

        if is_teacher:
            t_logits = teacher.get_logits(mb_ids, mb_attn)[:, :-1]
            teacher_topk_idx, t_logprobs = teacher_topk_logprobs(t_logits, K, t_chunk)
            teacher_own_logprobs  = t_logprobs
            t_logprobs_at_student = teacher_logprobs_at_indices(t_logits, student_topk_idx, t_chunk)
        else:
            teacher_topk_idx      = torch.empty(B, T_shifted, K, dtype=torch.long,     device=device)
            t_logprobs            = torch.empty(B, T_shifted, K, dtype=torch.bfloat16, device=device)
            teacher_own_logprobs  = torch.empty(B, T_shifted, K, dtype=torch.bfloat16, device=device)
            t_logprobs_at_student = torch.empty(B, T_shifted, K, dtype=torch.bfloat16, device=device)

        dist.broadcast(teacher_topk_idx,      src=teacher_global_rank, group=all_group)
        dist.broadcast(t_logprobs,            src=teacher_global_rank, group=all_group)
        dist.broadcast(teacher_own_logprobs,  src=teacher_global_rank, group=all_group)
        dist.broadcast(t_logprobs_at_student, src=teacher_global_rank, group=all_group)

        topk_idx = teacher_topk_idx

    return {
        "topk_idx":             topk_idx,
        "t_logprobs":           t_logprobs,
        "student_topk_idx":     student_topk_idx,
        "teacher_topk_idx":     teacher_topk_idx,
        "t_logprobs_at_student": t_logprobs_at_student,
        "teacher_own_logprobs": teacher_own_logprobs,
    }

