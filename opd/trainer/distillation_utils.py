"""Shared minibatch-exchange helpers used by OPD/SDPO/OPSD/SDFT training loops.

Student and teacher live on different ranks (see `setup_utils.init_distributed`)
and communicate only through `torch.distributed` collectives. This module holds
every piece of that cross-rank plumbing: broadcasting raw minibatch tensors,
exchanging top-K log-prob distributions for the KL losses, the lighter
sampled-token-only exchange for the MOPD policy-gradient loss, and syncing the
student's weights into a self-teacher (SDPO/SDFT).

Most functions here take a `ctx: DistributedContext` (the same object
`init_distributed` returns once at startup) instead of separately re-passing
`is_student`/`is_teacher`/`teacher_global_rank`/`all_group`/`device` — that
cluster of rank-topology facts is fixed for the whole run, so it travels as
one named object rather than five independent arguments at every call site.
The low-level broadcast primitives go one step further and wrap `ctx` in a
`RankBroadcaster` instance, since those calls repeat `device`/`group` at
nearly every line.

Sections:
  1. Low-level broadcast primitives (RankBroadcaster; includes minibatch
     input broadcast, student -> all ranks)
  2. Batch preparation (packing, padding, alignment)
  3. Teacher weight sync (SDPO / SDFT self-teacher)
  4. MOPD policy-gradient exchange
  5. Top-K KL exchange
  6. Minibatch orchestration (entry point used by the training loops)
"""

from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from opd.fsdp.algorithms import (
    student_logprobs_at_indices,
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
)
from opd.fsdp.model import StudentModel, TeacherModel
from opd.loss import compute_tis_weights
from opd.trainer.models import DistributedContext, MinibatchExchangeResult, MopdPGExchange, TopKExchange
from opd.trainer.setup_utils import print0
from opd.trainer.sync_teacher import TeacherSyncer

# ---------------------------------------------------------------------------
# 1. Low-level broadcast primitives
# ---------------------------------------------------------------------------


class RankBroadcaster:
    """Broadcasts tensors between ranks within one run's process groups.

    Every broadcast in this module needs the same rank-topology facts
    (`device`, `all_group`) — wrapping `ctx` here once removes that repeated
    `device=ctx.device, group=ctx.all_group` boilerplate from every call site
    below. `ctx` is fixed for the whole run, so one instance can be reused
    across every exchange in a minibatch.
    """

    def __init__(self, ctx: DistributedContext) -> None:
        self.ctx = ctx

    def bcast_or_alloc_async(
        self,
        *,
        tensor: torch.Tensor | None,
        is_owner: bool,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        src: int,
    ) -> tuple[torch.Tensor, dist.Work]:
        """If not the owning rank, allocate a placeholder; issue the broadcast without waiting for it.

        Args:
            tensor: The tensor to broadcast, if this rank owns it; otherwise ignored.
            is_owner: Whether this rank is `src` and holds the real data.
            shape: Shape to allocate the placeholder with, on non-owning ranks.
            dtype: Dtype to allocate the placeholder with, on non-owning ranks.
            src: Source rank of the broadcast.

        Returns:
            `(tensor, handle)`. Call `handle.wait()` before reading `tensor`. Lets
            several independent broadcasts overlap on the wire instead of
            completing one at a time.
        """
        if not is_owner:
            tensor = torch.empty(shape, dtype=dtype, device=self.ctx.device)
        handle = dist.broadcast(tensor, src=src, group=self.ctx.all_group, async_op=True)
        return tensor, handle

    def bcast_or_alloc_many(
        self,
        *,
        specs: list[tuple[torch.Tensor | None, tuple[int, ...], torch.dtype]],
        is_owner: bool,
        src: int,
    ) -> list[torch.Tensor]:
        """Issues `bcast_or_alloc_async` for each `(tensor, shape, dtype)` in `specs`, then waits on all.

        Broadcasting several independent tensors is a repeated pattern in this
        module (a minibatch's ids + attention mask, a top-K exchange's four
        result tensors, ...). Doing it through one call instead of a manual
        `h1, h2, h3, ... = ...; h1.wait(); h2.wait(); ...` sequence removes that
        per-call-site bookkeeping while keeping the same overlap-on-the-wire
        behavior (all broadcasts are issued before any wait happens).

        Args:
            specs: `(tensor, shape, dtype)` triples, one per tensor to broadcast.
              `tensor` is only read on the owning rank; non-owning ranks use
              `shape`/`dtype` to allocate a placeholder.
            is_owner: Whether this rank is `src` and holds the real data for
              every tensor in `specs`.
            src: Source rank of the broadcast.

        Returns:
            The broadcast tensors, in the same order as `specs`.
        """
        pending = [
            self.bcast_or_alloc_async(tensor=tensor, is_owner=is_owner, shape=shape, dtype=dtype, src=src)
            for tensor, shape, dtype in specs
        ]
        tensors, handles = zip(*pending)
        for handle in handles:
            handle.wait()
        return list(tensors)

    def broadcast_shape(self, local_shape: tuple[int, int] | None) -> tuple[int, int]:
        """Broadcasts a 2D tensor's `(dim0, dim1)` shape from student rank 0 to all ranks.

        Used by `broadcast_minibatch` and `broadcast_teacher_inputs`, which both
        need to tell non-owning ranks how large a placeholder to allocate before
        the actual payload broadcast.

        Args:
            local_shape: This rank's local `(dim0, dim1)` shape (student ranks only).

        Returns:
            The broadcast `(dim0, dim1)` shape.
        """
        shape_t = torch.tensor(local_shape, dtype=torch.long, device=self.ctx.device) if self.ctx.is_student else None
        shape_t, handle = self.bcast_or_alloc_async(
            tensor=shape_t, is_owner=self.ctx.is_student, shape=(2,), dtype=torch.long, src=0,
        )
        handle.wait()
        return int(shape_t[0].item()), int(shape_t[1].item())

    def broadcast_minibatch(
        self, mb_ids: torch.Tensor | None, mb_attn: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcasts input_ids and attention_mask from student rank 0 to all ranks."""
        B, T = self.broadcast_shape((mb_ids.shape[0], mb_ids.shape[1]) if self.ctx.is_student else None)
        mb_ids, mb_attn = self.bcast_or_alloc_many(
            specs=[(mb_ids, (B, T), torch.long), (mb_attn, (B, T), torch.long)],
            is_owner=self.ctx.is_student, src=0,
        )
        return mb_ids, mb_attn

    def broadcast_teacher_inputs(
        self,
        t_mb_ids: torch.Tensor | None,
        t_mb_attn: torch.Tensor | None,
        t_mb_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Broadcasts teacher-specific tensors from student rank 0 to all ranks.

        Used when teacher and student have different inputs (e.g. SDFT, where the
        teacher prompt includes a worked demonstration).
        """
        B, T_t = self.broadcast_shape((t_mb_ids.shape[0], t_mb_ids.shape[1]) if self.ctx.is_student else None)
        t_mb_ids, t_mb_attn, t_mb_mask = self.bcast_or_alloc_many(
            specs=[
                (t_mb_ids, (B, T_t), torch.long),
                (t_mb_attn, (B, T_t), torch.long),
                (t_mb_mask, (B, T_t), torch.float),
            ],
            is_owner=self.ctx.is_student, src=0,
        )
        return t_mb_ids, t_mb_attn, t_mb_mask


# ---------------------------------------------------------------------------
# 2. Batch preparation
# ---------------------------------------------------------------------------


def pack_response_logits(
    logits: torch.Tensor, shift_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts response-position logits into a compact [B, R_max, V] tensor.

    Student and teacher process sequences of different lengths (teacher prompt
    includes feedback, so it is longer). To compute the KL loss we must align
    both distributions at the same response token positions. This removes
    prompt positions and packs the result into a dense tensor.

    Args:
        logits: [B, T-1, V] — model output logits.
        shift_mask: [B, T-1] — float mask, 1 at response positions.

    Returns:
        `(resp_logits, compact_mask)`: `resp_logits` is `[B, R_max, V]`;
        `compact_mask` is `[B, R_max]`, 1 where a response token exists.
    """
    B, _, V = logits.shape
    resp_counts = torch.einsum("bt->b", shift_mask.long())     # [B]
    R_max = int(resp_counts.max().item())
    if R_max == 0:
        return logits.new_zeros(B, 0, V), shift_mask.new_zeros(B, 0)

    out          = logits.new_zeros(B, R_max, V)
    compact_mask = torch.zeros(B, R_max, dtype=torch.float, device=logits.device)
    for b in range(B):
        r_b = int(resp_counts[b].item())
        if r_b > 0:
            out[b, :r_b]          = logits[b][shift_mask[b].bool()]
            compact_mask[b, :r_b] = 1.0
    return out, compact_mask


def align_to_rmax(
    t_resp: torch.Tensor, t_compact_mask: torch.Tensor, R_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads or trims teacher packed logits/mask to match the student's R_max.

    Padded positions would receive log_softmax(0) = -log(V) — a spurious
    uniform distribution — so `t_compact_mask` tracks which positions are
    real vs. padded, letting the loss exclude them.

    Args:
        t_resp: Teacher packed logits (or logprobs), `[B, T_t, ...]`.
        t_compact_mask: Teacher packed response mask, `[B, T_t]`.
        R_max: Target length to align to (the student's packed length).

    Returns:
        `(t_resp, t_compact_mask)`, both with their second dimension equal to `R_max`.
    """
    if t_resp.shape[1] < R_max:
        pad = R_max - t_resp.shape[1]
        t_resp = torch.cat([t_resp, t_resp.new_zeros(t_resp.shape[0], pad, t_resp.shape[-1])], dim=1)
        t_compact_mask = torch.cat([t_compact_mask, t_compact_mask.new_zeros(t_compact_mask.shape[0], pad)], dim=1)
    elif t_resp.shape[1] > R_max:
        t_resp, t_compact_mask = t_resp[:, :R_max], t_compact_mask[:, :R_max]
    return t_resp, t_compact_mask


def prepare_teacher_batch(
    rollouts: list[dict[str, Any]], tokenizer: Any, device: torch.device
) -> dict[str, torch.Tensor]:
    """Builds padded teacher sequences from augmented prompts.

    Each rollout must carry a `teacher_prompt` string (chat template applied
    to the teacher's richer-context messages — feedback, a reference solution,
    or a demonstration, depending on the caller). The same response token IDs
    as the student are appended after the teacher prompt, so the teacher
    re-evaluates the student's exact on-policy response from its own context.

    No max_seq_len cap is applied here: the teacher never generates tokens, so
    memory is bounded by sequence length alone (no KV cache growth during
    decoding). The student batch enforces max_seq_len separately.

    Args:
        rollouts: Rollout dicts, each with `teacher_prompt` and `response_ids`.
        tokenizer: Tokenizer used to encode `teacher_prompt`.
        device: Device to place the returned tensors on.

    Returns:
        A dict with `input_ids`, `attention_mask`, and `response_mask` tensors,
        each `[B, T_t]`.
    """
    input_ids_list, response_mask_list = [], []

    for r in rollouts:
        t_prompt_ids = tokenizer.encode(r["teacher_prompt"], add_special_tokens=False)
        response_ids = list(r["response_ids"])

        full_ids = t_prompt_ids + response_ids
        r_len = len(response_ids)
        input_ids_list.append(full_ids)
        response_mask_list.append([0] * len(t_prompt_ids) + [1] * r_len)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids   = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m   + [0]      * (max_len - len(m))   for m   in response_mask_list]
    attn_masks   = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

    return {
        "input_ids":      torch.tensor(padded_ids,   dtype=torch.long,  device=device),
        "attention_mask": torch.tensor(attn_masks,   dtype=torch.long,  device=device),
        "response_mask":  torch.tensor(padded_masks, dtype=torch.float, device=device),
    }


# ---------------------------------------------------------------------------
# 3. Teacher weight sync (SDPO / SDFT self-teacher)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sync_student_to_teacher(
    student_fsdp_model: torch.nn.Module | None,
    teacher: TeacherModel | None,
    syncer: TeacherSyncer,
    global_step: int,
    ctx: DistributedContext,
) -> None:
    """Broadcasts full student parameters to the teacher rank and applies the syncer.

    FSDP shards student parameters across student ranks; summon_full_params
    temporarily gathers them so rank 0 can broadcast the full tensors. The
    teacher rank receives each parameter and updates its weights via the chosen
    sync method (EMA, trust-region, hard-sync, or on-policy).

    All ranks must call this function together because dist.broadcast is a
    collective — the loops on both sides must execute the same number of times
    in the same order (guaranteed because student and teacher share the same
    architecture).

    Args:
        student_fsdp_model: The FSDP-wrapped student model (student ranks only).
        teacher: The teacher model (teacher rank only).
        syncer: The sync strategy to apply on the teacher rank.
        global_step: Current training step, passed through to `syncer.step`.
        ctx: Distributed rank/topology info for this run.
    """
    if ctx.is_student:
        with FSDP.summon_full_params(student_fsdp_model, writeback=False, recurse=True):
            for s_param in student_fsdp_model.parameters():
                # Rank 0 sends; other student ranks and teacher rank receive.
                # writeback=False ensures the receive on non-zero student ranks
                # does not corrupt the FSDP shards.
                dist.broadcast(s_param.data, src=0, group=ctx.all_group)

    if ctx.is_teacher:
        received = []
        for t_param in teacher.model.parameters():
            buf = torch.empty_like(t_param.data)
            dist.broadcast(buf, src=0, group=ctx.all_group)
            received.append(buf)
        syncer.step(received, teacher.model.parameters(), global_step)
        print(f"[sync step={global_step}] teacher updated via {syncer.__class__.__name__}", flush=True)


# ---------------------------------------------------------------------------
# 4. MOPD policy-gradient exchange
# ---------------------------------------------------------------------------


def fetch_teacher_sampled_logprob(
    *,
    ctx: DistributedContext,
    teacher_logits: torch.Tensor | None,   # [B, T, V], teacher rank only
    token_ids: torch.Tensor,               # [B, T], sampled tokens (already on every rank via broadcast_minibatch)
    B: int,
    T: int,
    teacher_chunk_size: int = -1,
) -> torch.Tensor:
    """Broadcasts log π_φd(y_t): teacher log-prob at the student's sampled token.

    Used for the MOPD policy-gradient advantage. Only a [B, T] tensor
    crosses the wire here, versus [B, T, K] for fetch_teacher_topk — the PG form
    needs no top-K selection at all, just this one scalar per position.
    """
    if ctx.is_teacher:
        t_logprob = teacher_logprobs_at_indices(teacher_logits, token_ids.unsqueeze(-1), teacher_chunk_size).squeeze(-1)
    else:
        t_logprob = None
    t_logprob, h = RankBroadcaster(ctx).bcast_or_alloc_async(
        tensor=t_logprob, is_owner=ctx.is_teacher, shape=(B, T), dtype=torch.bfloat16,
        src=ctx.teacher_global_rank,
    )
    h.wait()
    return t_logprob


def exchange_mopd_pg_packed(
    *,
    ctx: DistributedContext,
    student_logits: torch.Tensor | None,   # [B, T_s-1, V], student rank only, grad-carrying
    teacher_logits: torch.Tensor | None,   # [B, T_t-1, V], teacher rank only, no-grad
    student_ids: torch.Tensor | None,      # [B, T_s], student rank only (full, unshifted)
    teacher_ids: torch.Tensor | None,      # [B, T_t], teacher rank only (full, unshifted)
    s_shift_mask: torch.Tensor | None,     # [B, T_s-1], student rank only
    t_shift_mask: torch.Tensor | None,     # [B, T_t-1], teacher rank only
    R_max: int,
    B: int,
    teacher_chunk_size: int = -1,
) -> MopdPGExchange:
    """MOPD policy-gradient exchange for packed-response setups (SDPO / OPSD /
    SDFT self-teacher training, where teacher prompts include extra feedback
    so student and teacher sequences differ in length).

    Same principle as fetch_teacher_sampled_logprob: gather each policy's
    log-prob at its own sampled token in the cheap, unpacked [B, T-1] space —
    no top-K, no full-vocab packing — then pack to response-only positions
    [B, R_max] and broadcast only the teacher's small result.
    """
    if ctx.is_student:
        s_logprob_full = student_logprobs_at_indices(
            student_logits, student_ids[:, 1:].unsqueeze(-1), -1
        ).squeeze(-1)                                                  # [B, T_s-1], grad
        s_logprob_resp, s_compact_mask = pack_response_logits(
            s_logprob_full.unsqueeze(-1), s_shift_mask
        )
        s_logprob_resp = s_logprob_resp[..., 0]
    else:
        s_logprob_resp = s_compact_mask = None

    if ctx.is_teacher:
        t_logprob_full = teacher_logprobs_at_indices(
            teacher_logits, teacher_ids[:, 1:].unsqueeze(-1), teacher_chunk_size
        ).squeeze(-1)                                                  # [B, T_t-1], no grad
        t_logprob_resp, t_compact_mask = pack_response_logits(
            t_logprob_full.unsqueeze(-1), t_shift_mask
        )
        t_logprob_resp = t_logprob_resp[..., 0]
        t_logprob_resp, t_compact_mask = align_to_rmax(
            t_logprob_resp.unsqueeze(-1), t_compact_mask, R_max
        )
        t_logprob_resp = t_logprob_resp[..., 0]
    else:
        t_logprob_resp = t_compact_mask = None

    t_logprob_resp, t_compact_mask = RankBroadcaster(ctx).bcast_or_alloc_many(
        specs=[
            (t_logprob_resp, (B, R_max), torch.bfloat16),
            (t_compact_mask, (B, R_max), torch.float32),
        ],
        is_owner=ctx.is_teacher, src=ctx.teacher_global_rank,
    )

    return MopdPGExchange(
        s_logprob=s_logprob_resp,
        t_logprob=t_logprob_resp,
        s_compact_mask=s_compact_mask,
        t_compact_mask=t_compact_mask,
    )


def _effective_mask(
    s_mask: torch.Tensor,
    t_mask: torch.Tensor,
    extra_mask: torch.Tensor | None,
    mask_fn: Any,
    warn: bool,
) -> torch.Tensor:
    """Combines student/teacher masks (+ optional extra mask / mask_fn), warning if empty.

    Shared by `mopd_pg_loss_and_backward` and `topk_kl_loss_and_backward` — both
    build their loss mask from the same student-mask * teacher-mask * extras
    recipe before calling into `loss_fn`.
    """
    effective_mask = s_mask * t_mask
    if extra_mask is not None:
        effective_mask = effective_mask * extra_mask
    if mask_fn is not None:
        effective_mask = mask_fn(effective_mask)

    if warn and effective_mask.sum() == 0:
        msg = f"[warn mb] effective_mask is all-zero: s_mask={s_mask.sum().item():.0f} t_mask={t_mask.sum().item():.0f}"
        if extra_mask is not None:
            msg += f" extra_mask={extra_mask.sum().item():.0f}"
        print0(msg, flush=True)

    return effective_mask


def mopd_pg_loss_and_backward(
    *,
    student: StudentModel,
    pg: MopdPGExchange,
    loss_fn: Any,                            # compute_mopd_pg_loss
    tis_weights: torch.Tensor | None,        # [B, R_max], already aligned with pg.s_logprob; None disables TIS
    divisor: float,
    extra_mask: torch.Tensor | None = None,  # e.g. SDPO's mb_sd_mask.unsqueeze(1)
    mask_fn: Any = None,                     # e.g. SDFT's apply_token_skip_mask
    warn: bool = True,
) -> torch.Tensor:
    """Shared MOPD-PG loss step, used by OPD directly and, via `exchange_mopd_pg_packed`,
    by the three packed-response self-teacher scripts (SDPO / OPSD / SDFT) —
    computing the effective mask, calling the loss, and backpropagating was
    identical across all call sites.

    `tis_weights` must already be computed (via `compute_tis_weights`) and, if
    the caller packs its sequences (self-teacher scripts), already packed to
    the same shape as `pg.s_logprob` — this function has no visibility into
    whether packing happened, since OPD's sequences never need it.

    Returns the (already backward()-ed) loss tensor for accounting.
    """
    effective_mask = _effective_mask(pg.s_compact_mask, pg.t_compact_mask, extra_mask, mask_fn, warn)

    loss = loss_fn(pg.s_logprob, pg.t_logprob, effective_mask, tis_weights=tis_weights) / divisor
    student._scale_loss(loss).backward()
    return loss


# ---------------------------------------------------------------------------
# 5. Top-K KL exchange
# ---------------------------------------------------------------------------


def fetch_teacher_topk(
    *,
    ctx: DistributedContext,
    select_topk_by: Literal["student", "teacher"],
    student_logits: torch.Tensor | None,          # [B, T, V], student rank only
    teacher_logits: torch.Tensor | None,          # [B, T, V], teacher rank only
    t_compact_mask: torch.Tensor | None = None,   # [B, T], teacher rank only; None → broadcast all-ones
    B: int = 0,
    T: int = 0,
    top_k: int = 0,
    student_chunk_size: int = -1,
    teacher_chunk_size: int = -1,
) -> TopKExchange:
    """Top-K log-prob exchange between student and teacher ranks.

    Works for both full-sequence logits [B, T-1, V] (OPD) and packed
    response-aligned logits [B, R_max, V] (SDFT). The caller is responsible
    for computing teacher_logits on the teacher rank (and optionally packing
    it and providing t_compact_mask when student/teacher sequences differ).

    Regardless of `select_topk_by`, the teacher always computes exactly the
    same two things: its own top-K + log-probs there (needed either as the
    loss's shared support, when the teacher selects, or as a diagnostic
    baseline for the entropy-gap metric, when the student selects), and its
    log-probs at the student's top-K (needed either as the loss's shared
    support, when the student selects, or as a diagnostic for the
    overlap/advantage metrics, when the teacher selects). Only the final
    selection of which pair becomes "the" loss support depends on
    `select_topk_by` — so the two cases share one code path instead of being
    near-duplicate branches.
    """
    if ctx.is_teacher and t_compact_mask is None:
        t_compact_mask = torch.ones(B, T, dtype=torch.float32, device=ctx.device)

    rb = RankBroadcaster(ctx)
    student_topk_idx = student_topk_indices(student_logits, top_k, student_chunk_size) if ctx.is_student else None
    student_topk_idx, h = rb.bcast_or_alloc_async(
        tensor=student_topk_idx, is_owner=ctx.is_student, shape=(B, T, top_k), dtype=torch.long, src=0,
    )
    h.wait()

    if ctx.is_teacher:
        teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(teacher_logits, top_k, teacher_chunk_size)
        t_logprobs_at_student = teacher_logprobs_at_indices(teacher_logits, student_topk_idx, teacher_chunk_size)
    else:
        teacher_topk_idx = teacher_own_logprobs = t_logprobs_at_student = None

    teacher_topk_idx, teacher_own_logprobs, t_logprobs_at_student, t_compact_mask = rb.bcast_or_alloc_many(
        specs=[
            (teacher_topk_idx, (B, T, top_k), torch.long),
            (teacher_own_logprobs, (B, T, top_k), torch.bfloat16),
            (t_logprobs_at_student, (B, T, top_k), torch.bfloat16),
            (t_compact_mask, (B, T), torch.float32),
        ],
        is_owner=ctx.is_teacher, src=ctx.teacher_global_rank,
    )

    if select_topk_by == "student":
        topk_idx, t_logprobs = student_topk_idx, t_logprobs_at_student
    else:  # "teacher": forward_kl / mopd_loss weight by the teacher distribution
        topk_idx, t_logprobs = teacher_topk_idx, teacher_own_logprobs

    return TopKExchange(
        topk_idx=topk_idx,
        t_logprobs=t_logprobs,
        t_compact_mask=t_compact_mask,
        student_topk_idx=student_topk_idx,
        teacher_topk_idx=teacher_topk_idx,
        t_logprobs_at_student=t_logprobs_at_student,
        teacher_own_logprobs=teacher_own_logprobs,
    )


def topk_kl_loss_and_backward(
    *,
    student: StudentModel,
    student_logits: torch.Tensor,            # [B, T, V], aligned with topk.topk_idx
    topk: TopKExchange,
    loss_fn: Any,                             # reverse_kl / forward_kl / jsd / mopd_loss
    response_mask: torch.Tensor,              # [B, T], student-side response mask, same T as student_logits
    tis_weights: torch.Tensor | None,         # [B, T], already aligned with response_mask; None disables TIS
    divisor: float,
    student_chunk_size: int = -1,
    extra_mask: torch.Tensor | None = None,   # e.g. SDPO's mb_sd_mask.unsqueeze(1)
    mask_fn: Any = None,                      # e.g. SDFT's apply_token_skip_mask
    kl_clip: float | None = None,             # OPSD's per-token KL clip
    warn: bool = True,
) -> torch.Tensor:
    """Shared step for the top-K KL family (reverse_kl / forward_kl / jsd /
    mopd_loss), used directly by OPD and, via a packed `student_logits`/`topk`/
    `response_mask`, by the three self-teacher scripts (SDPO / OPSD / SDFT).

    All four share the same shape: gather student log-probs at the exchanged
    top-K indices, assemble the effective mask, run `loss_fn` + backward.
    `tis_weights` must already be computed (via `compute_tis_weights`) and, if
    the caller packs its sequences, already packed to the same shape as
    `response_mask` — this function has no visibility into whether packing
    happened, since OPD's sequences never need it.

    Returns the (already backward()-ed) loss tensor for accounting.
    """
    s_logprobs = student_logprobs_at_indices(student_logits, topk.topk_idx, student_chunk_size)

    effective_mask = _effective_mask(response_mask, topk.t_compact_mask, extra_mask, mask_fn, warn)

    loss = loss_fn(
        s_logprobs, topk.t_logprobs, effective_mask,
        tis_weights=tis_weights, kl_clip=kl_clip,
    ) / divisor
    student._scale_loss(loss).backward()
    return loss


# ---------------------------------------------------------------------------
# 6. Minibatch orchestration (entry point used by the training loops)
# ---------------------------------------------------------------------------


def minibatch_exchange(
    ctx: DistributedContext,
    mb_ids: torch.Tensor | None,
    mb_attn: torch.Tensor | None,
    mb_mask: torch.Tensor | None,
    t_mb_ids: torch.Tensor | None,
    t_mb_attn: torch.Tensor | None,
    t_mb_mask: torch.Tensor | None,
    student: StudentModel | None,
    teacher: TeacherModel | None,
    select_topk_by: Literal["student", "teacher"],
    top_k: int,
    student_chunk_size: int,
    teacher_chunk_size: int,
    is_pg: bool = False,
) -> MinibatchExchangeResult:
    """Student + teacher forward, then top-K exchange (or, for MOPD-PG, the
    lighter sampled-token-only exchange). Shared by SDPO / OPSD / SDFT — the
    three self-teacher scripts where the teacher's prompt carries extra
    context (feedback, a reference solution, or a demonstration) and so
    produces a different-length sequence than the student's.

    Args:
        ctx: Distributed rank/topology info for this run.
        mb_ids: Student input ids, `[B, T_s]` (student ranks only).
        mb_attn: Student attention mask, `[B, T_s]` (student ranks only).
        mb_mask: Student response mask, `[B, T_s]` (student ranks only).
        t_mb_ids: Teacher input ids, `[B, T_t]` (teacher rank only).
        t_mb_attn: Teacher attention mask, `[B, T_t]` (teacher rank only).
        t_mb_mask: Teacher response mask, `[B, T_t]` (teacher rank only).
        student: The student model wrapper (student ranks only).
        teacher: The teacher model (teacher rank only).
        select_topk_by: Whether the student or the teacher selects top-K
          indices (ignored when `is_pg` is True).
        top_k: Top-K vocab size for the distillation exchange.
        student_chunk_size: Chunk size along T for student top-K computation.
        teacher_chunk_size: Chunk size along T for teacher top-K computation.
        is_pg: If True, use the lighter MOPD-PG sampled-token exchange
          instead of a full top-K exchange.

    Returns:
        A `MinibatchExchangeResult` with `pg` set (and `topk` `None`) when
        `is_pg` is True, or `topk` set (and `pg` `None`) otherwise.
    """
    if ctx.is_student:
        student_logits = student.get_logits(mb_ids, mb_attn)[:, :-1]
        s_shift_mask = mb_mask[:, 1:]
        if is_pg:
            # PG form only needs the sampled token's log-prob, not the full
            # top-K distribution — skip packing [B, T_s-1, V] into
            # [B, R_max, V], which would materialise a same-sized copy of the
            # logits for nothing.
            s_resp = s_compact_mask = None
            R_max_local = int(torch.einsum("bt->b", s_shift_mask.long()).max().item())
        else:
            s_resp, s_compact_mask = pack_response_logits(student_logits, s_shift_mask)
            R_max_local = s_resp.shape[1]
    else:
        student_logits = s_shift_mask = s_resp = s_compact_mask = None
        R_max_local = 0

    # Broadcast R_max so the teacher can allocate matching tensors.
    rb = RankBroadcaster(ctx)
    R_max_t, h = rb.bcast_or_alloc_async(
        tensor=torch.tensor([R_max_local], dtype=torch.long, device=ctx.device) if ctx.is_student else None,
        is_owner=ctx.is_student, shape=(1,), dtype=torch.long, src=0,
    )
    h.wait()
    R_max = int(R_max_t.item())

    if ctx.is_teacher:
        with torch.no_grad():
            teacher_logits = teacher.get_logits(t_mb_ids, t_mb_attn)[:, :-1]
        t_shift_mask = t_mb_mask[:, 1:]
        if not is_pg:
            t_resp, t_compact_mask = pack_response_logits(teacher_logits, t_shift_mask)
            t_resp, t_compact_mask = align_to_rmax(t_resp, t_compact_mask, R_max)
    else:
        teacher_logits = t_shift_mask = t_resp = t_compact_mask = None

    if is_pg:
        pg = exchange_mopd_pg_packed(
            ctx=ctx,
            student_logits=student_logits if ctx.is_student else None,
            teacher_logits=teacher_logits if ctx.is_teacher else None,
            student_ids=mb_ids if ctx.is_student else None,
            teacher_ids=t_mb_ids if ctx.is_teacher else None,
            s_shift_mask=s_shift_mask if ctx.is_student else None,
            t_shift_mask=t_shift_mask if ctx.is_teacher else None,
            R_max=R_max, B=mb_ids.shape[0],
            teacher_chunk_size=teacher_chunk_size,
        )
        return MinibatchExchangeResult(
            is_pg=True, topk=None, pg=pg,
            s_resp=s_resp, s_compact_mask=s_compact_mask,
            s_shift_mask=s_shift_mask, student_logits=student_logits,
        )

    topk = fetch_teacher_topk(
        ctx=ctx, select_topk_by=select_topk_by,
        student_logits=s_resp, teacher_logits=t_resp, t_compact_mask=t_compact_mask,
        B=mb_ids.shape[0], T=R_max, top_k=top_k,
        student_chunk_size=student_chunk_size, teacher_chunk_size=teacher_chunk_size,
    )
    return MinibatchExchangeResult(
        is_pg=False, topk=topk, pg=None,
        s_resp=s_resp, s_compact_mask=s_compact_mask,
        s_shift_mask=s_shift_mask, student_logits=student_logits,
    )
