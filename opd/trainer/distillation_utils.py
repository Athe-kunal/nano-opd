from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from opd.trainer.setup_utils import print0
from opd.fsdp.algorithms import (
    student_logprob_at_sampled_tokens,
    student_logprobs_at_indices,
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


def pack_response_logits(logits, shift_mask):
    """Extract response-position logits into a compact [B, R_max, V] tensor.

    Student and teacher process sequences of different lengths (teacher prompt
    includes feedback, so it is longer). To compute the KL loss we must align
    both distributions at the same response token positions. This removes
    prompt positions and packs the result into a dense tensor.

    Args:
        logits:     [B, T-1, V] — model output logits.
        shift_mask: [B, T-1]    — float mask, 1 at response positions.

    Returns:
        resp_logits:  [B, R_max, V]
        compact_mask: [B, R_max] — 1 where response tokens exist.
    """
    B, _, V = logits.shape
    resp_counts = shift_mask.long().sum(dim=1)     # [B]
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


def prepare_teacher_batch(rollouts, tokenizer, device):
    """Build padded teacher sequences from augmented prompts.

    Each rollout must carry a ``teacher_prompt`` string (chat template applied
    to the teacher's richer-context messages — feedback, a reference solution,
    or a demonstration, depending on the caller). The same response token IDs
    as the student are appended after the teacher prompt, so the teacher
    re-evaluates the student's exact on-policy response from its own context.

    No max_seq_len cap is applied here: the teacher never generates tokens, so
    memory is bounded by sequence length alone (no KV cache growth during
    decoding). The student batch enforces max_seq_len separately.
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


@torch.no_grad()
def sync_student_to_teacher(
    student_fsdp_model, teacher, syncer, global_step,
    is_student, is_teacher, all_group,
):
    """Broadcast full student parameters to the teacher rank and apply the syncer.

    FSDP shards student parameters across student ranks; summon_full_params
    temporarily gathers them so rank 0 can broadcast the full tensors. The
    teacher rank receives each parameter and updates its weights via the chosen
    sync method (EMA, trust-region, hard-sync, or on-policy).

    All ranks must call this function together because dist.broadcast is a
    collective — the loops on both sides must execute the same number of times
    in the same order (guaranteed because student and teacher share the same
    architecture).
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
        student_proxy = (torch.nn.Parameter(r, requires_grad=False) for r in received)
        syncer.step(student_proxy, teacher.model.parameters(), global_step)
        # print directly — this runs only on the teacher rank, not rank 0
        print(f"[sync step={global_step}] teacher updated via {syncer.__class__.__name__}", flush=True)


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


def exchange_sampled_teacher_logprob(
    *,
    is_teacher: bool,
    teacher_logits: torch.Tensor | None,   # [B, T, V], teacher rank only
    token_ids: torch.Tensor,               # [B, T], sampled tokens (already on every rank via broadcast_minibatch)
    B: int,
    T: int,
    t_chunk: int = -1,
    teacher_global_rank: int = 0,
    all_group=None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Broadcast log π_φd(y_t): teacher log-prob at the student's sampled token.

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
    all_group=None,
    device: torch.device | None = None,
) -> dict:
    """MOPD policy-gradient exchange for packed-response setups (SDPO / OPSD /
    SDFT self-teacher training, where teacher prompts include extra feedback
    so student and teacher sequences differ in length).

    Same principle as exchange_sampled_teacher_logprob: gather each policy's
    log-prob at its own sampled token in the cheap, unpacked [B, T-1] space —
    no top-K, no full-vocab packing — then pack to response-only positions
    [B, R_max] and broadcast only the teacher's small result.

    Returns: {"s_logprob": [B,R_max] grad, "t_logprob": [B,R_max] no-grad,
              "s_compact_mask": [B,R_max], "t_compact_mask": [B,R_max]}
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
        if t_logprob_resp.shape[1] < R_max:
            pad = R_max - t_logprob_resp.shape[1]
            t_logprob_resp = torch.cat([t_logprob_resp, t_logprob_resp.new_zeros(B, pad)], dim=1)
            t_compact_mask = torch.cat([t_compact_mask, t_compact_mask.new_zeros(B, pad)], dim=1)
        elif t_logprob_resp.shape[1] > R_max:
            t_logprob_resp = t_logprob_resp[:, :R_max]
            t_compact_mask = t_compact_mask[:, :R_max]
    else:
        t_logprob_resp = t_compact_mask = None

    t_logprob_resp, h1 = bcast_or_alloc_async(
        tensor=t_logprob_resp, is_owner=is_teacher, shape=(B, R_max), dtype=torch.bfloat16,
        src=teacher_global_rank, device=device, group=all_group,
    )
    t_compact_mask, h2 = bcast_or_alloc_async(
        tensor=t_compact_mask, is_owner=is_teacher, shape=(B, R_max), dtype=torch.float32,
        src=teacher_global_rank, device=device, group=all_group,
    )
    h1.wait()
    h2.wait()

    return {
        "s_logprob":      s_logprob_resp,
        "t_logprob":      t_logprob_resp,
        "s_compact_mask": s_compact_mask,
        "t_compact_mask": t_compact_mask,
    }


def mopd_pg_loss_and_backward(
    *,
    student,                                # has ._scale_loss(...).backward()
    pg: dict,                               # from exchange_mopd_pg_packed
    loss_fn,                                # compute_mopd_pg_loss
    student_logits: torch.Tensor,           # [B, T_s-1, V], for TIS gather
    sampled_ids: torch.Tensor,              # [B, T_s-1]
    s_shift_mask: torch.Tensor,             # [B, T_s-1]
    inf_lp_shifted: torch.Tensor | None,    # [B, T_s-1], vLLM inference log-probs; None disables TIS
    tis_clip: float,
    divisor: float,
    extra_mask: torch.Tensor | None = None,  # e.g. SDPO's mb_sd_mask.unsqueeze(1)
    mask_fn=None,                            # e.g. SDFT's apply_token_skip_mask
    warn: bool = True,
) -> torch.Tensor:
    """Shared MOPD-PG loss step for the three packed-response training scripts
    (SDPO / OPSD / SDFT) — computing the effective mask, TIS weights, calling
    the loss, and backpropagating was identical across all three call sites.

    Returns the (already backward()-ed) loss tensor for accounting.
    """
    effective_mask = pg["s_compact_mask"] * pg["t_compact_mask"]
    if extra_mask is not None:
        effective_mask = effective_mask * extra_mask
    if mask_fn is not None:
        effective_mask = mask_fn(effective_mask)

    if warn and effective_mask.sum() == 0:
        print0(
            f"[warn mb] effective_mask is all-zero: "
            f"s_mask={pg['s_compact_mask'].sum().item():.0f} t_mask={pg['t_compact_mask'].sum().item():.0f}",
        )

    if tis_clip > 0.0:
        s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
        tis_full = (s_lp_sampled - inf_lp_shifted.to(s_lp_sampled.dtype)).exp().clamp(max=tis_clip)
        tis_resp, _ = pack_response_logits(tis_full.unsqueeze(-1), s_shift_mask)
        tis_weights = tis_resp[..., 0]
    else:
        tis_weights = None

    loss = loss_fn(pg["s_logprob"], pg["t_logprob"], effective_mask, tis_weights=tis_weights) / divisor
    student._scale_loss(loss).backward()
    return loss


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
    all_group=None,
    device: torch.device | None = None,
) -> TopKExchange:
    """Top-K log-prob exchange between student and teacher ranks.

    Works for both full-sequence logits [B, T-1, V] (OPD) and packed
    response-aligned logits [B, R_max, V] (SDFT). The caller is responsible
    for computing teacher_logits on the teacher rank (and optionally packing
    it and providing t_compact_mask when student/teacher sequences differ).
    """
    if is_teacher and t_compact_mask is None:
        t_compact_mask = torch.ones(B, T, dtype=torch.float32, device=device)

    if select_topk_by == "student": #reverse KL
        topk_idx = (
            student_topk_indices(student_logits, K, s_chunk) if is_student else None
        )
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
        student_topk_idx = (
            student_topk_indices(student_logits, K, s_chunk) if is_student else None
        )
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

    return TopKExchange(
        topk_idx=topk_idx,
        t_logprobs=t_logprobs,
        t_compact_mask=t_compact_mask,
        student_topk_idx=student_topk_idx,
        teacher_topk_idx=teacher_topk_idx,
        t_logprobs_at_student=t_logprobs_at_student,
        teacher_own_logprobs=teacher_own_logprobs,
    )


def align_to_rmax(t_resp, t_compact_mask, R_max):
    """Pad or trim teacher packed logits/mask to match student R_max.

    Padded positions would receive log_softmax(0) = -log(V) — a spurious
    uniform distribution — so t_compact_mask tracks which positions are real
    vs. padded, letting the loss exclude them.
    """
    if t_resp.shape[1] < R_max:
        pad = R_max - t_resp.shape[1]
        t_resp = torch.cat([t_resp, t_resp.new_zeros(t_resp.shape[0], pad, t_resp.shape[-1])], dim=1)
        t_compact_mask = torch.cat([t_compact_mask, t_compact_mask.new_zeros(t_compact_mask.shape[0], pad)], dim=1)
    elif t_resp.shape[1] > R_max:
        t_resp, t_compact_mask = t_resp[:, :R_max], t_compact_mask[:, :R_max]
    return t_resp, t_compact_mask


def minibatch_exchange(
    is_student, is_teacher, mb_ids, mb_attn, mb_mask,
    t_mb_ids, t_mb_attn, t_mb_mask, student_model, teacher,
    select_topk_by, K, s_chunk, t_chunk, teacher_global_rank, all_group, device,
    is_pg=False,
):
    """Student + teacher forward, then top-K exchange (or, for MOPD-PG, the
    lighter sampled-token-only exchange). Shared by SDPO / OPSD / SDFT — the
    three self-teacher scripts where the teacher's prompt carries extra
    context (feedback, a reference solution, or a demonstration) and so
    produces a different-length sequence than the student's.

    Returns (tk, s_resp, s_compact_mask, s_shift_mask, student_logits).

    When is_pg=True: s_resp/s_compact_mask are None (never packed — the PG
    form doesn't need the full top-K distribution) and tk is the dict
    {"s_logprob", "t_logprob", "s_compact_mask", "t_compact_mask"} from
    exchange_mopd_pg_packed, instead of a TopKExchange.
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
            R_max_local = int(s_shift_mask.long().sum(dim=1).max().item())
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
        tk = exchange_mopd_pg_packed(
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
        return tk, s_resp, s_compact_mask, s_shift_mask, student_logits

    tk = exchange_topk(
        select_topk_by=select_topk_by, is_student=is_student, is_teacher=is_teacher,
        student_logits=s_resp, teacher_logits=t_resp, t_compact_mask=t_compact_mask,
        B=mb_ids.shape[0], T=R_max, K=K,
        s_chunk=s_chunk, t_chunk=t_chunk,
        teacher_global_rank=teacher_global_rank, all_group=all_group, device=device,
    )
    return tk, s_resp, s_compact_mask, s_shift_mask, student_logits
