"""Shared minibatch step for the three self-teacher training scripts (SDPO/OPSD/SDFT).

All three build a separate, packed teacher batch (the teacher's prompt is
richer than the student's — feedback, a reference solution, or a
demonstration — so the two sequences differ in length) and share the same
exchange -> loss -> backward -> health-metric shape. What differs per script
is "privileged information" each passes in explicitly: the loss divisor
(an adaptive grad-accumulation window vs. a plain epoch-size divisor), and any
extra masking (SDPO's self-distillation mask, SDFT's token-skip mask, OPSD's
per-token KL clip). OPD is not part of this — it has no separate teacher
batch and calls `exchange_topk`/`exchange_sampled_teacher_logprob` directly.
"""

from typing import Any, Literal

import torch
import torch.nn.functional as F

from opd.fsdp.algorithms import student_logprob_at_sampled_tokens
from opd.fsdp.model import StudentModel, TeacherModel
from opd.metrics import (
    compute_entropy_gap,
    compute_overlap_ratio,
    compute_overlap_token_advantage,
)
from opd.trainer.distillation_utils import (
    minibatch_exchange,
    mopd_pg_loss_and_backward,
    pack_response_logits,
)
from opd.trainer.setup_utils import DistributedContext, print0
from opd.trainer.trainer_utils import MinibatchTensors, StepAccumulator


def self_distill_minibatch(
    mb: MinibatchTensors,
    acc: StepAccumulator,
    *,
    ctx: DistributedContext,
    student: StudentModel | None,
    teacher: TeacherModel | None,
    select_topk_by: Literal["student", "teacher"],
    top_k: int,
    student_chunk_size: int,
    teacher_chunk_size: int,
    loss_fn: Any,
    is_pg: bool,
    tis_clip: float,
    divisor: float,
    extra_mask: torch.Tensor | None = None,
    mask_fn: Any = None,
    kl_clip: float | None = None,
) -> None:
    """Exchange + loss + backward + health metrics, shared by SDPO/OPSD/SDFT.

    Calls `minibatch_exchange` (a collective — invoked on every rank), then
    does all further computation only on student ranks. Callers supply the
    parts that vary per algorithm: `divisor` (adaptive grad-accumulation
    window vs. epoch size), and one of `extra_mask`/`mask_fn`/`kl_clip` for
    any extra masking their loss needs.

    Args:
        mb: This minibatch's broadcast tensors.
        acc: Accumulator to record this minibatch's loss/health metrics into.
        ctx: Distributed rank/topology info.
        student: Student model wrapper (student ranks only; `None` on the
          teacher rank).
        teacher: Teacher model (teacher rank only; `None` on student ranks).
        select_topk_by: Whether the student or the teacher selects top-K
          indices (ignored when `is_pg` is True).
        top_k: Top-K vocab size for the distillation exchange.
        student_chunk_size: Chunk size along T for student top-K computation.
        teacher_chunk_size: Chunk size along T for teacher top-K computation.
        loss_fn: The distillation loss function (from `opd.loss.ALGORITHMS`).
        is_pg: Whether `args.algorithm == "mopd_pg_loss"` — selects the
          lighter sampled-token-only exchange instead of a full top-K one.
        tis_clip: TIS importance-weight clip C (0 disables).
        divisor: Value the loss is divided by before `.backward()` (an
          adaptive grad-accumulation window size, or a plain step-size
          divisor like `mb.n_mb`).
        extra_mask: Extra multiplicative mask applied to the effective mask
          and forwarded to the PG loss (e.g. SDPO's self-distillation mask,
          `[B, 1]`). `None` disables.
        mask_fn: Extra mask-transform applied to the effective mask and
          forwarded to the PG loss (e.g. SDFT's token-skip mask). `None`
          disables.
        kl_clip: Per-token pointwise KL clip, forwarded to `loss_fn` in the
          non-PG path only (OPSD; `mopd_pg_loss_and_backward` has no
          equivalent). `None` disables.
    """
    result = minibatch_exchange(
        ctx, mb.mb_ids, mb.mb_attn, mb.mb_mask,
        mb.t_mb_ids, mb.t_mb_attn, mb.t_mb_mask,
        student, teacher,
        select_topk_by, top_k, student_chunk_size, teacher_chunk_size,
        is_pg=is_pg,
    )

    if not ctx.is_student:
        return

    if result.is_pg:
        loss = mopd_pg_loss_and_backward(
            student=student, pg=result.pg, loss_fn=loss_fn,
            student_logits=result.student_logits, sampled_ids=mb.mb_ids[:, 1:], s_shift_mask=result.s_shift_mask,
            inf_lp_shifted=mb.mb_inf_lp[:, 1:] if tis_clip > 0.0 else None,
            tis_clip=tis_clip, divisor=divisor,
            extra_mask=extra_mask, mask_fn=mask_fn,
        )
        acc.add_loss(loss)
        return

    tk = result.topk
    student_logits = result.student_logits
    s_log_resp = F.log_softmax(result.s_resp.float(), dim=-1)
    s_logprobs = s_log_resp.gather(-1, tk.topk_idx)
    s_lp_at_student = s_log_resp.gather(-1, tk.student_topk_idx)

    # Exclude positions where the teacher sequence was truncated (its longer,
    # richer-context prompt may hit max_seq_len). Those padded positions have
    # log_softmax(0) = -log(V) — a spurious uniform distribution that would
    # corrupt the loss signal.
    effective_mask = result.s_compact_mask * tk.t_compact_mask   # [B, R_max]
    if extra_mask is not None:
        effective_mask = effective_mask * extra_mask
    if mask_fn is not None:
        effective_mask = mask_fn(effective_mask)

    if effective_mask.sum() == 0:
        msg = (
            f"[warn mb] effective_mask is all-zero: "
            f"s_mask={result.s_compact_mask.sum().item():.0f} "
            f"t_mask={tk.t_compact_mask.sum().item():.0f}"
        )
        if extra_mask is not None:
            msg += f" extra_mask={extra_mask.sum().item():.0f}"
        print0(msg, flush=True)

    if tis_clip > 0.0:
        sampled_ids = mb.mb_ids[:, 1:]
        s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
        inf_lp_shifted = mb.mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)
        tis_full = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=tis_clip)
        tis_resp, _ = pack_response_logits(
            tis_full.unsqueeze(-1).expand_as(student_logits), result.s_shift_mask
        )
        tis_weights = tis_resp[..., 0]   # [B, R_max]
    else:
        tis_weights = None

    loss = loss_fn(
        s_logprobs, tk.t_logprobs, effective_mask,
        tis_weights=tis_weights, kl_clip=kl_clip,
    ) / divisor
    student._scale_loss(loss).backward()
    acc.add_loss(loss)

    with torch.no_grad():
        ratio = compute_overlap_ratio(tk.student_topk_idx, tk.teacher_topk_idx).item()
        advantage = compute_overlap_token_advantage(
            tk.student_topk_idx, tk.teacher_topk_idx, s_lp_at_student, tk.t_logprobs_at_student
        ).item()
        entropy_gap = compute_entropy_gap(s_lp_at_student, tk.teacher_own_logprobs).item()
        acc.add_health_metrics(ratio, advantage, entropy_gap)
