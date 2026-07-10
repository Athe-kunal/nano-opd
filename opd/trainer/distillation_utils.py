"""Shared minibatch-exchange helpers used by OPD/SDPO/OPSD/SDFT training loops.

Student and teacher live on different ranks (see `setup_utils.init_distributed`)
and communicate only through `torch.distributed` collectives. This module holds
every piece of that cross-rank plumbing: broadcasting raw minibatch tensors,
exchanging top-K log-prob distributions for the KL losses, the lighter
sampled-token-only exchange for the MOPD policy-gradient loss, and syncing the
student's weights into a self-teacher (SDPO/SDFT).

Sections:
  1. Low-level broadcast primitives
  2. Batch preparation (packing, padding, alignment)
  3. Teacher weight sync (SDPO / SDFT self-teacher)
  4. Minibatch input broadcast (student -> all ranks)
  5. MOPD policy-gradient exchange
  6. Top-K KL exchange
  7. Minibatch orchestration (entry point used by the training loops)
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from opd.fsdp.algorithms import (
    student_logprob_at_sampled_tokens,
    student_logprobs_at_indices,
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
)
from opd.fsdp.model import StudentModel, TeacherModel
from opd.trainer.setup_utils import print0
from opd.trainer.sync_teacher import TeacherSyncer

# ---------------------------------------------------------------------------
# 1. Low-level broadcast primitives
# ---------------------------------------------------------------------------


def bcast_or_alloc_async(
    *,
    tensor: torch.Tensor | None,
    is_owner: bool,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    device: torch.device,
    group: Any,
) -> tuple[torch.Tensor, dist.Work]:
    """If not the owning rank, allocate a placeholder; issue the broadcast without waiting for it.

    Args:
        tensor: The tensor to broadcast, if this rank owns it; otherwise ignored.
        is_owner: Whether this rank is `src` and holds the real data.
        shape: Shape to allocate the placeholder with, on non-owning ranks.
        dtype: Dtype to allocate the placeholder with, on non-owning ranks.
        src: Source rank of the broadcast.
        device: Device to allocate the placeholder on.
        group: Process group to broadcast within.

    Returns:
        `(tensor, handle)`. Call `handle.wait()` before reading `tensor`. Lets
        several independent broadcasts overlap on the wire instead of
        completing one at a time.
    """
    if not is_owner:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    handle = dist.broadcast(tensor, src=src, group=group, async_op=True)
    return tensor, handle


def bcast_or_alloc_many(
    *,
    specs: list[tuple[torch.Tensor | None, tuple[int, ...], torch.dtype]],
    is_owner: bool,
    src: int,
    device: torch.device,
    group: Any,
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
        device: Device to allocate placeholders on.
        group: Process group to broadcast within.

    Returns:
        The broadcast tensors, in the same order as `specs`.
    """
    pending = [
        bcast_or_alloc_async(tensor=tensor, is_owner=is_owner, shape=shape, dtype=dtype, src=src, device=device, group=group)
        for tensor, shape, dtype in specs
    ]
    tensors, handles = zip(*pending)
    for handle in handles:
        handle.wait()
    return list(tensors)


def _broadcast_shape(
    is_student: bool,
    local_shape: tuple[int, int] | None,
    device: torch.device,
    all_group: Any,
) -> tuple[int, int]:
    """Broadcasts a 2D tensor's `(dim0, dim1)` shape from student rank 0 to all ranks.

    Shared by `broadcast_minibatch` and `broadcast_teacher_inputs`, which both
    need to tell non-owning ranks how large a placeholder to allocate before
    the actual payload broadcast.

    Args:
        is_student: Whether this rank owns `local_shape`.
        local_shape: This rank's local `(dim0, dim1)` shape (student ranks only).
        device: Device to allocate the broadcast buffer on.
        all_group: Process group spanning student and teacher ranks.

    Returns:
        The broadcast `(dim0, dim1)` shape.
    """
    shape_t = torch.tensor(local_shape, dtype=torch.long, device=device) if is_student else None
    shape_t, handle = bcast_or_alloc_async(
        tensor=shape_t, is_owner=is_student, shape=(2,), dtype=torch.long,
        src=0, device=device, group=all_group,
    )
    handle.wait()
    return int(shape_t[0].item()), int(shape_t[1].item())


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
    is_student: bool,
    is_teacher: bool,
    all_group: Any,
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
        is_student: Whether this rank is a student (FSDP) rank.
        is_teacher: Whether this rank is the teacher rank.
        all_group: Process group spanning student and teacher ranks.
    """
    if is_student:
        with FSDP.summon_full_params(student_fsdp_model, writeback=False, recurse=True):
            for s_param in student_fsdp_model.parameters():
                # Rank 0 sends; other student ranks and teacher rank receive.
                # writeback=False ensures the receive on non-zero student ranks
                # does not corrupt the FSDP shards.
                dist.broadcast(s_param.data, src=0, group=all_group)

    if is_teacher:
        received = []
        for t_param in teacher.model.parameters():
            buf = torch.empty_like(t_param.data)
            dist.broadcast(buf, src=0, group=all_group)
            received.append(buf)
        syncer.step(received, teacher.model.parameters(), global_step)
        # print directly — this runs only on the teacher rank, not rank 0, so
        # print0's rank-0-only gating would silently drop it.
        print(f"[sync step={global_step}] teacher updated via {syncer.__class__.__name__}", flush=True)


# ---------------------------------------------------------------------------
# 4. Minibatch input broadcast (student -> all ranks)
# ---------------------------------------------------------------------------


def broadcast_minibatch(
    is_student: bool,
    mb_ids: torch.Tensor | None,
    mb_attn: torch.Tensor | None,
    device: torch.device,
    all_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcasts input_ids and attention_mask from student rank 0 to all ranks."""
    B, T = _broadcast_shape(
        is_student, (mb_ids.shape[0], mb_ids.shape[1]) if is_student else None, device, all_group,
    )

    mb_ids, mb_attn = bcast_or_alloc_many(
        specs=[(mb_ids, (B, T), torch.long), (mb_attn, (B, T), torch.long)],
        is_owner=is_student, src=0, device=device, group=all_group,
    )
    return mb_ids, mb_attn


def broadcast_teacher_inputs(
    is_student: bool,
    t_mb_ids: torch.Tensor | None,
    t_mb_attn: torch.Tensor | None,
    t_mb_mask: torch.Tensor | None,
    device: torch.device,
    all_group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcasts teacher-specific tensors from student rank 0 to all ranks.

    Used when teacher and student have different inputs (e.g. SDFT, where the
    teacher prompt includes a worked demonstration).
    """
    B, T_t = _broadcast_shape(
        is_student, (t_mb_ids.shape[0], t_mb_ids.shape[1]) if is_student else None, device, all_group,
    )

    t_mb_ids, t_mb_attn, t_mb_mask = bcast_or_alloc_many(
        specs=[
            (t_mb_ids, (B, T_t), torch.long),
            (t_mb_attn, (B, T_t), torch.long),
            (t_mb_mask, (B, T_t), torch.float),
        ],
        is_owner=is_student, src=0, device=device, group=all_group,
    )
    return t_mb_ids, t_mb_attn, t_mb_mask


# ---------------------------------------------------------------------------
# 5. MOPD policy-gradient exchange
# ---------------------------------------------------------------------------


def exchange_sampled_teacher_logprob(
    *,
    is_teacher: bool,
    teacher_logits: torch.Tensor | None,   # [B, T, V], teacher rank only
    token_ids: torch.Tensor,               # [B, T], sampled tokens (already on every rank via broadcast_minibatch)
    B: int,
    T: int,
    t_chunk: int = -1,
    teacher_global_rank: int = 0,
    all_group: Any = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Broadcasts log π_φd(y_t): teacher log-prob at the student's sampled token.

    Used for the MOPD policy-gradient advantage (Eq. 3). Only a [B, T] tensor
    crosses the wire here, versus [B, T, K] for exchange_topk — the PG form
    needs no top-K selection at all, just this one scalar per position.
    """
    if is_teacher:
        t_logprob = teacher_logprobs_at_indices(teacher_logits, token_ids.unsqueeze(-1), t_chunk).squeeze(-1)
    else:
        t_logprob = None
    t_logprob, h = bcast_or_alloc_async(
        tensor=t_logprob, is_owner=is_teacher, shape=(B, T), dtype=torch.bfloat16,
        src=teacher_global_rank, device=device, group=all_group,
    )
    h.wait()
    return t_logprob


@dataclass
class MopdPGExchange:
    """Result of exchange_mopd_pg_packed: sampled-token log-probs for the MOPD-PG loss."""
    s_logprob: torch.Tensor | None       # [B, R_max], grad-carrying (student ranks only)
    t_logprob: torch.Tensor | None       # [B, R_max], no-grad
    s_compact_mask: torch.Tensor | None  # [B, R_max] (student ranks only)
    t_compact_mask: torch.Tensor | None  # [B, R_max]


def exchange_mopd_pg_packed(
    *,
    is_student: bool,
    is_teacher: bool,
    student_logits: torch.Tensor | None,   # [B, T_s-1, V], student rank only, grad-carrying
    teacher_logits: torch.Tensor | None,   # [B, T_t-1, V], teacher rank only, no-grad
    student_ids: torch.Tensor | None,      # [B, T_s], student rank only (full, unshifted)
    teacher_ids: torch.Tensor | None,      # [B, T_t], teacher rank only (full, unshifted)
    s_shift_mask: torch.Tensor | None,     # [B, T_s-1], student rank only
    t_shift_mask: torch.Tensor | None,     # [B, T_t-1], teacher rank only
    R_max: int,
    B: int,
    t_chunk: int = -1,
    teacher_global_rank: int = 0,
    all_group: Any = None,
    device: torch.device | None = None,
) -> MopdPGExchange:
    """MOPD policy-gradient exchange for packed-response setups (SDPO / OPSD /
    SDFT self-teacher training, where teacher prompts include extra feedback
    so student and teacher sequences differ in length).

    Same principle as exchange_sampled_teacher_logprob: gather each policy's
    log-prob at its own sampled token in the cheap, unpacked [B, T-1] space —
    no top-K, no full-vocab packing — then pack to response-only positions
    [B, R_max] and broadcast only the teacher's small result.
    """
    if is_student:
        s_logprob_full = student_logprobs_at_indices(
            student_logits, student_ids[:, 1:].unsqueeze(-1), -1
        ).squeeze(-1)                                                  # [B, T_s-1], grad
        s_logprob_resp, s_compact_mask = pack_response_logits(
            s_logprob_full.unsqueeze(-1), s_shift_mask
        )
        s_logprob_resp = s_logprob_resp[..., 0]
    else:
        s_logprob_resp = s_compact_mask = None

    if is_teacher:
        t_logprob_full = teacher_logprobs_at_indices(
            teacher_logits, teacher_ids[:, 1:].unsqueeze(-1), t_chunk
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

    t_logprob_resp, t_compact_mask = bcast_or_alloc_many(
        specs=[
            (t_logprob_resp, (B, R_max), torch.bfloat16),
            (t_compact_mask, (B, R_max), torch.float32),
        ],
        is_owner=is_teacher, src=teacher_global_rank, device=device, group=all_group,
    )

    return MopdPGExchange(
        s_logprob=s_logprob_resp,
        t_logprob=t_logprob_resp,
        s_compact_mask=s_compact_mask,
        t_compact_mask=t_compact_mask,
    )


def mopd_pg_loss_and_backward(
    *,
    student: StudentModel,
    pg: MopdPGExchange,
    loss_fn: Any,                            # compute_mopd_pg_loss
    student_logits: torch.Tensor,            # [B, T_s-1, V], for TIS gather
    sampled_ids: torch.Tensor,               # [B, T_s-1]
    s_shift_mask: torch.Tensor,              # [B, T_s-1]
    inf_lp_shifted: torch.Tensor | None,     # [B, T_s-1], vLLM inference log-probs; None disables TIS
    tis_clip: float,
    divisor: float,
    extra_mask: torch.Tensor | None = None,  # e.g. SDPO's mb_sd_mask.unsqueeze(1)
    mask_fn: Any = None,                     # e.g. SDFT's apply_token_skip_mask
    warn: bool = True,
) -> torch.Tensor:
    """Shared MOPD-PG loss step for the three packed-response training scripts
    (SDPO / OPSD / SDFT) — computing the effective mask, TIS weights, calling
    the loss, and backpropagating was identical across all three call sites.

    Returns the (already backward()-ed) loss tensor for accounting.
    """
    effective_mask = pg.s_compact_mask * pg.t_compact_mask
    if extra_mask is not None:
        effective_mask = effective_mask * extra_mask
    if mask_fn is not None:
        effective_mask = mask_fn(effective_mask)

    if warn and effective_mask.sum() == 0:
        print0(
            f"[warn mb] effective_mask is all-zero: "
            f"s_mask={pg.s_compact_mask.sum().item():.0f} t_mask={pg.t_compact_mask.sum().item():.0f}",
        )

    if tis_clip > 0.0:
        s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
        tis_full = (s_lp_sampled - inf_lp_shifted.to(s_lp_sampled.dtype)).exp().clamp(max=tis_clip)
        tis_resp, _ = pack_response_logits(tis_full.unsqueeze(-1), s_shift_mask)
        tis_weights = tis_resp[..., 0]
    else:
        tis_weights = None

    loss = loss_fn(pg.s_logprob, pg.t_logprob, effective_mask, tis_weights=tis_weights) / divisor
    student._scale_loss(loss).backward()
    return loss


# ---------------------------------------------------------------------------
# 6. Top-K KL exchange
# ---------------------------------------------------------------------------


@dataclass
class TopKExchange:
    """Result of exchange_topk: the top-K distributions used to compute the KL loss."""
    topk_idx: torch.Tensor
    t_logprobs: torch.Tensor
    t_compact_mask: torch.Tensor
    student_topk_idx: torch.Tensor
    teacher_topk_idx: torch.Tensor
    t_logprobs_at_student: torch.Tensor
    teacher_own_logprobs: torch.Tensor


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
    all_group: Any = None,
    device: torch.device | None = None,
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
    if is_teacher and t_compact_mask is None:
        t_compact_mask = torch.ones(B, T, dtype=torch.float32, device=device)

    # The student always selects and broadcasts its own top-K indices.
    student_topk_idx = student_topk_indices(student_logits, K, s_chunk) if is_student else None
    student_topk_idx, h = bcast_or_alloc_async(
        tensor=student_topk_idx, is_owner=is_student, shape=(B, T, K), dtype=torch.long,
        src=0, device=device, group=all_group,
    )
    h.wait()

    # The teacher computes its own top-K (+ log-probs there) and its
    # log-probs at the student's top-K — both needed regardless of
    # select_topk_by, either for the loss or for diagnostics.
    if is_teacher:
        teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(teacher_logits, K, t_chunk)
        t_logprobs_at_student = teacher_logprobs_at_indices(teacher_logits, student_topk_idx, t_chunk)
    else:
        teacher_topk_idx = teacher_own_logprobs = t_logprobs_at_student = None

    teacher_topk_idx, teacher_own_logprobs, t_logprobs_at_student, t_compact_mask = bcast_or_alloc_many(
        specs=[
            (teacher_topk_idx, (B, T, K), torch.long),
            (teacher_own_logprobs, (B, T, K), torch.bfloat16),
            (t_logprobs_at_student, (B, T, K), torch.bfloat16),
            (t_compact_mask, (B, T), torch.float32),
        ],
        is_owner=is_teacher, src=teacher_global_rank, device=device, group=all_group,
    )

    # The loss's shared support is whichever side select_topk_by names — the
    # other side's quantities above remain available for diagnostics only.
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


# ---------------------------------------------------------------------------
# 7. Minibatch orchestration (entry point used by the training loops)
# ---------------------------------------------------------------------------


@dataclass
class MinibatchExchangeResult:
    """Result of minibatch_exchange: exactly one of `topk`/`pg` is set, per `is_pg`.

    Packed student tensors (`s_resp`, `s_compact_mask`) are only produced in
    the top-K path (`is_pg=False`) — the PG path skips packing the full
    logits and leaves them `None`.
    """
    is_pg: bool
    topk: TopKExchange | None
    pg: MopdPGExchange | None
    s_resp: torch.Tensor | None
    s_compact_mask: torch.Tensor | None
    s_shift_mask: torch.Tensor | None
    student_logits: torch.Tensor | None


def minibatch_exchange(
    is_student: bool,
    is_teacher: bool,
    mb_ids: torch.Tensor | None,
    mb_attn: torch.Tensor | None,
    mb_mask: torch.Tensor | None,
    t_mb_ids: torch.Tensor | None,
    t_mb_attn: torch.Tensor | None,
    t_mb_mask: torch.Tensor | None,
    student_model: torch.nn.Module | None,
    teacher: TeacherModel | None,
    select_topk_by: Literal["student", "teacher"],
    K: int,
    s_chunk: int,
    t_chunk: int,
    teacher_global_rank: int,
    all_group: Any,
    device: torch.device,
    is_pg: bool = False,
) -> MinibatchExchangeResult:
    """Student + teacher forward, then top-K exchange (or, for MOPD-PG, the
    lighter sampled-token-only exchange). Shared by SDPO / OPSD / SDFT — the
    three self-teacher scripts where the teacher's prompt carries extra
    context (feedback, a reference solution, or a demonstration) and so
    produces a different-length sequence than the student's.

    Args:
        is_student: Whether this rank is a student (FSDP) rank.
        is_teacher: Whether this rank is the teacher rank.
        mb_ids: Student input ids, `[B, T_s]` (student ranks only).
        mb_attn: Student attention mask, `[B, T_s]` (student ranks only).
        mb_mask: Student response mask, `[B, T_s]` (student ranks only).
        t_mb_ids: Teacher input ids, `[B, T_t]` (teacher rank only).
        t_mb_attn: Teacher attention mask, `[B, T_t]` (teacher rank only).
        t_mb_mask: Teacher response mask, `[B, T_t]` (teacher rank only).
        student_model: The FSDP-wrapped student model (student ranks only).
        teacher: The teacher model (teacher rank only).
        select_topk_by: Whether the student or the teacher selects top-K
          indices (ignored when `is_pg` is True).
        K: Top-K vocab size for the distillation exchange.
        s_chunk: Chunk size along T for student top-K computation.
        t_chunk: Chunk size along T for teacher top-K computation.
        teacher_global_rank: Global rank of the teacher process.
        all_group: Process group spanning student and teacher ranks.
        device: Device to allocate broadcast tensors on.
        is_pg: If True, use the lighter MOPD-PG sampled-token exchange
          instead of a full top-K exchange.

    Returns:
        A `MinibatchExchangeResult` with `pg` set (and `topk` `None`) when
        `is_pg` is True, or `topk` set (and `pg` `None`) otherwise.
    """
    if is_student:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student_model(input_ids=mb_ids, attention_mask=mb_attn).logits[:, :-1]
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
    R_max_t = torch.tensor([R_max_local if is_student else 0], dtype=torch.long, device=device)
    dist.broadcast(R_max_t, src=0, group=all_group)
    R_max = int(R_max_t.item())

    if is_teacher:
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
            is_student=is_student, is_teacher=is_teacher,
            student_logits=student_logits if is_student else None,
            teacher_logits=teacher_logits if is_teacher else None,
            student_ids=mb_ids if is_student else None,
            teacher_ids=t_mb_ids if is_teacher else None,
            s_shift_mask=s_shift_mask if is_student else None,
            t_shift_mask=t_shift_mask if is_teacher else None,
            R_max=R_max, B=mb_ids.shape[0],
            t_chunk=t_chunk,
            teacher_global_rank=teacher_global_rank, all_group=all_group, device=device,
        )
        return MinibatchExchangeResult(
            is_pg=True, topk=None, pg=pg,
            s_resp=s_resp, s_compact_mask=s_compact_mask,
            s_shift_mask=s_shift_mask, student_logits=student_logits,
        )

    topk = exchange_topk(
        select_topk_by=select_topk_by, is_student=is_student, is_teacher=is_teacher,
        student_logits=s_resp, teacher_logits=t_resp, t_compact_mask=t_compact_mask,
        B=mb_ids.shape[0], T=R_max, K=K,
        s_chunk=s_chunk, t_chunk=t_chunk,
        teacher_global_rank=teacher_global_rank, all_group=all_group, device=device,
    )
    return MinibatchExchangeResult(
        is_pg=False, topk=topk, pg=None,
        s_resp=s_resp, s_compact_mask=s_compact_mask,
        s_shift_mask=s_shift_mask, student_logits=student_logits,
    )
