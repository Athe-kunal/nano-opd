
import os
import math
import time
import argparse
from typing import Literal

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from opd.common import compute_cleanup, print0
from opd.loss import ALGORITHMS
from opd.fsdp.algorithms import (
    student_logprob_at_sampled_tokens,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
)
from opd.trainer.distillation_utils import broadcast_minibatch
from opd.trainer.setup_utils import init_distributed, build_student, build_teacher, init_vllm_transfer
from opd.envs.opsd_dataset import OPSDMathEnv
from opd.envs.dataset import distributed_opd_loader
from opd.metrics import (
    compute_overlap_ratio,
    compute_overlap_token_advantage,
    compute_entropy_gap,
)
from opd.generator.rollout import (
    generate_rollouts_remote,
    sync_weights_to_vllm_inplace,
    prepare_batch,
)


# ---------------------------------------------------------------------------
# Teacher prompt construction (Figure 2 of the OPSD paper)
# ---------------------------------------------------------------------------

# The teacher sees the problem AND the ground-truth reference solution y*.
# This follows Figure 2 of the paper exactly: after reading the reference
# solution the teacher is asked to solve the problem in its own way — this
# rationalization is done implicitly through a single forward pass (no
# generation), so the teacher never actually produces new tokens here.
_TEACHER_TEMPLATE = (
    "{problem}\n"
    "Here is a reference solution:\n"
    "{solution}\n"
    "After understanding the reference solution, please try to solve this "
    "problem using your own approach below:"
)


def _build_teacher_messages(
    student_messages: list[dict],
    solution: str,
) -> list[dict]:
    """Construct the reference-conditioned teacher prompt (OPSD paper, Figure 2).

    Student sees: system (optional) + user(problem)
    Teacher sees: system (optional) + user(problem + reference solution template)

    Splicing into the last user turn preserves the chat template structure
    regardless of whether a system message is present.
    """
    problem_content = student_messages[-1]["content"]
    teacher_user = _TEACHER_TEMPLATE.format(
        problem=problem_content,
        solution=solution,
    )
    teacher_messages = list(student_messages[:-1])   # preserve system message if any
    teacher_messages.append({"role": "user", "content": teacher_user})
    return teacher_messages


# ---------------------------------------------------------------------------
# Batch preparation helpers
# Teacher sequences are longer than student sequences because the teacher
# prompt includes the reference solution, so they require separate padding.
# ---------------------------------------------------------------------------

def prepare_teacher_batch(rollouts, tokenizer, device):
    """Build padded teacher sequences from reference-conditioned prompts.

    Each rollout carries a ``teacher_prompt`` string (chat template applied to
    the reference-augmented messages). The same response token IDs as the
    student are appended after the teacher prompt so the teacher evaluates
    the student's exact on-policy response from its richer context.

    No max_seq_len cap: the teacher never generates tokens, so memory is
    bounded by sequence length alone (no KV cache growth during decoding).
    """
    input_ids_list, response_mask_list = [], []
    for r in rollouts:
        t_prompt_ids = tokenizer.encode(r["teacher_prompt"], add_special_tokens=False)
        response_ids = list(r["response_ids"])
        full_ids     = t_prompt_ids + response_ids
        r_len        = len(response_ids)
        input_ids_list.append(full_ids)
        response_mask_list.append([0] * len(t_prompt_ids) + [1] * r_len)

    max_len      = max(len(ids) for ids in input_ids_list)
    pad_id       = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids   = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m   + [0]      * (max_len - len(m))   for m   in response_mask_list]
    attn_masks   = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

    return {
        "input_ids":      torch.tensor(padded_ids,   dtype=torch.long,  device=device),
        "attention_mask": torch.tensor(attn_masks,   dtype=torch.long,  device=device),
        "response_mask":  torch.tensor(padded_masks, dtype=torch.float, device=device),
    }


def pack_response_logits(logits, shift_mask):
    """Extract response-position logits into a compact [B, R_max, V] tensor.

    Student and teacher process sequences of different lengths (teacher prompt
    is longer due to the injected reference solution). To align both
    distributions at response token positions, this removes prompt positions
    and packs the result into a dense tensor.

    Args:
        logits:     [B, T-1, V] — model output logits (already shifted).
        shift_mask: [B, T-1]    — float mask, 1 at response positions.

    Returns:
        resp_logits:  [B, R_max, V]
        compact_mask: [B, R_max] — 1 where response tokens exist.
    """
    B, _, V = logits.shape
    resp_counts = shift_mask.long().sum(dim=1)
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


def broadcast_teacher_inputs(is_student, t_mb_ids, t_mb_attn, t_mb_mask, device, all_group):
    """Broadcast teacher-specific tensors from student rank 0 to all ranks."""
    if is_student:
        shape_t = torch.tensor(
            [t_mb_ids.shape[0], t_mb_ids.shape[1]], dtype=torch.long, device=device
        )
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(
        description="On-Policy Self-Distillation (OPSD) — a single LLM acts as "
                    "both student (sees problem only) and teacher (sees problem + "
                    "reference solution). The teacher is the frozen initial policy."
    )
    # Model — student and teacher start from the same checkpoint; teacher is frozen
    parser.add_argument("--student-model", type=str, required=True,
                        help="HuggingFace model ID or path. Used for both student "
                             "(updated) and teacher (frozen initial policy).")
    parser.add_argument("--train-world-size", type=int, required=True,
                        help="Number of student (FSDP) ranks. The teacher occupies "
                             "rank train_world_size in the torchrun world.")
    # Dataset — siyanzhao/Openthoughts_math_30k_opsd (hardcoded)
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="HuggingFace split to load from siyanzhao/Openthoughts_math_30k_opsd.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="forward_kl",
                        choices=list(ALGORITHMS.keys()),
                        help="Distillation loss. OPSD paper (Table 3) finds forward KL "
                             "KL(p_T || p_S) consistently outperforms reverse KL and JSD.")
    parser.add_argument("--distill-top-k", type=int, default=100,
                        help="Top-K vocab for KL distillation. Larger K is more faithful "
                             "but uses more memory and bandwidth.")
    parser.add_argument("--student-chunk-size", type=int, default=-1)
    parser.add_argument("--teacher-chunk-size", type=int, default=-1)
    parser.add_argument("--tis-clip", type=float, default=0.0,
                        help="TIS importance-weight clip C (0 disables). Corrects for "
                             "log-prob gap between vLLM inference and training forward pass.")
    parser.add_argument("--kl-clip", type=float, default=0.0,
                        help="Per-token pointwise KL clip τ (0 disables). Clips each "
                             "token's divergence contribution to prevent stylistic tokens "
                             "from dominating the gradient signal (OPSD paper Section 3.2 "
                             "and Figure 4). Strongly recommended — the paper shows this "
                             "prevents performance collapse on Qwen3-1.7B.")
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100,
                        help="OPSD paper converges within 100 gradient update steps.")
    parser.add_argument("--prompts-per-step", type=int, default=8,
                        help="Number of distinct (problem, solution) pairs per step. "
                             "Each pair produces exactly one on-policy rollout (num_samples=1).")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Optimizer steps per rollout batch before collecting new rollouts.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="opsd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = os.environ.get("USE_WANDB", "1").strip().lower() not in ("0", "false", "no")
    if use_wandb:
        import wandb

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD/SDFT:
    #   ranks 0..train_world_size-1  →  student (FSDP, updated by optimizer)
    #   rank  train_world_size       →  teacher (plain nn.Module, frozen initial policy)
    ctx = init_distributed(args.device_type, args.train_world_size)
    ddp_rank            = ctx.ddp_rank
    ddp_world_size      = ctx.ddp_world_size
    device              = ctx.device
    train_world_size    = ctx.train_world_size
    teacher_global_rank = ctx.teacher_global_rank
    is_student          = ctx.is_student
    is_teacher          = ctx.is_teacher
    master_process      = ctx.master_process
    student_group       = ctx.student_group
    all_group           = ctx.all_group

    print0(f"Model: {args.student_model}  (teacher = frozen initial policy)")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}")
    if args.kl_clip > 0.0:
        print0(f"Per-token KL clip: {args.kl_clip}")

    if master_process and use_wandb:
        wandb.init(
            project="nano-opd",
            name=args.run_name,
            config={
                "student_model": args.student_model,
                "algorithm": args.algorithm,
                "distill_top_k": args.distill_top_k,
                "kl_clip": args.kl_clip,
                "lr": args.lr,
                "num_steps": args.num_steps,
                "prompts_per_step": args.prompts_per_step,
                "train_batch_size": args.train_batch_size,
                "epochs": args.epochs,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        )

    assert args.prompts_per_step % train_world_size == 0, (
        f"prompts_per_step ({args.prompts_per_step}) must be divisible by "
        f"train_world_size ({train_world_size})"
    )

    # -------------------------------------------------------------------------
    # Model setup
    if is_student:
        student = build_student(
            args.student_model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            sharding_strategy=args.sharding_strategy,
            train_world_size=train_world_size,
            student_group=student_group,
            total_steps=args.num_steps * args.epochs,
        )

    if is_teacher:
        # Frozen initial policy — weights are never updated after this load.
        # The paper (Section 4.1) finds that fixing the teacher to the initial
        # policy stabilises training and acts as an implicit regulariser that
        # prevents excessive deviation from the pretrained distribution.
        teacher = build_teacher(args.student_model)
        print(f"[teacher] Loaded initial policy from {args.student_model} (frozen)", flush=True)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    # OPSD paper (Table 3) recommends forward KL: KL(p_T || p_S).
    # For forward KL the teacher selects the top-K indices (the teacher-weighted
    # sum means we need tokens where the teacher has non-negligible probability).
    loss_fn = ALGORITHMS[args.algorithm]
    select_topk_by: Literal["student", "teacher"] = (
        "teacher" if args.algorithm == "forward_kl" else "student"
    )
    K = args.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only)
    model_update_group = init_vllm_transfer(
        args.rollout_worker_url,
        rollout_worker_world_size=args.rollout_worker_world_size,
        train_world_size=train_world_size,
        master_process=master_process,
        all_group=all_group,
    )

    # -------------------------------------------------------------------------
    # Dataset (student ranks only)
    if is_student:
        dataset     = OPSDMathEnv.load(split=args.dataset_split)
        loader      = distributed_opd_loader(
            dataset, args.prompts_per_step, train_world_size, ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Training loop — all ranks iterate together
    for step in range(args.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        if is_student:
            examples, _ = next(loader_iter)   # list[OPSDMathEnv], state_dict

            # Student prompt: problem only — p_S(· | x)
            prompts = [
                student.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex.problem}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in examples
            ]

            rollouts = generate_rollouts_remote(
                args.rollout_worker_url,
                prompts=prompts,
                num_samples=1,        # OPSD: single on-policy trajectory per prompt
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Attach reference-conditioned teacher prompt to each rollout.
            # The teacher sees: problem + ground-truth solution y* → richer
            # context than the student (problem only), following Figure 2.
            for i, ex in enumerate(examples):
                r            = rollouts[i]    # one rollout per prompt (num_samples=1)
                student_msgs = [{"role": "user", "content": ex.problem}]
                teacher_msgs = _build_teacher_messages(student_msgs, ex.solution)
                r["teacher_prompt"] = student.tokenizer.apply_chat_template(
                    teacher_msgs, tokenize=False, add_generation_prompt=True
                )

            if step == 0:
                print0(
                    f"[debug step=0] teacher prompt snippet:\n"
                    f"{rollouts[0]['teacher_prompt'][:400]}",
                    flush=True,
                )

            batch = prepare_batch(
                rollouts, tokenizer=student.tokenizer,
                max_seq_len=args.max_seq_len, device=device,
            )
            teacher_batch = prepare_teacher_batch(
                rollouts, tokenizer=student.tokenizer, device=device,
            )

            input_ids          = batch["input_ids"]           # [N, T_s]
            attention_mask     = batch["attention_mask"]
            response_mask      = batch["response_mask"]
            inference_logprobs = batch["inference_logprobs"]
            teacher_input_ids  = teacher_batch["input_ids"]   # [N, T_t]
            teacher_attn_mask  = teacher_batch["attention_mask"]
            teacher_resp_mask  = teacher_batch["response_mask"]
            student.model.train()

        total_loss        = 0.0
        n_batches         = 0
        overlap_ratio     = 0.0
        overlap_advantage = 0.0
        entropy_gap_val   = 0.0

        # -- Distillation epochs --
        for _epoch in range(args.epochs):

            if is_student:
                n_mb = math.ceil(input_ids.shape[0] / args.train_batch_size)
                perm = torch.randperm(input_ids.shape[0], device=device)
                n_mb_t = torch.tensor([n_mb], dtype=torch.long, device=device)
            else:
                n_mb_t = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(n_mb_t, src=0, group=all_group)
            n_mb = int(n_mb_t.item())

            for mb_idx in range(n_mb):

                # -- Slice minibatch (student ranks) --
                if is_student:
                    start     = mb_idx * args.train_batch_size
                    idx       = perm[start : start + args.train_batch_size]
                    mb_ids    = input_ids[idx]
                    mb_attn   = attention_mask[idx]
                    mb_mask   = response_mask[idx]
                    mb_inf_lp = inference_logprobs[idx]
                    t_mb_ids  = teacher_input_ids[idx]
                    t_mb_attn = teacher_attn_mask[idx]
                    t_mb_mask = teacher_resp_mask[idx]
                else:
                    mb_ids = mb_attn = t_mb_ids = t_mb_attn = t_mb_mask = None

                mb_ids, mb_attn = broadcast_minibatch(
                    is_student, mb_ids, mb_attn, device, all_group
                )
                t_mb_ids, t_mb_attn, t_mb_mask = broadcast_teacher_inputs(
                    is_student, t_mb_ids, t_mb_attn, t_mb_mask, device, all_group
                )

                # -- Student forward (with grad) --
                if is_student:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        student_logits = student.model(
                            input_ids=mb_ids, attention_mask=mb_attn
                        ).logits[:, :-1]                    # [B, T_s-1, V]

                    s_shift_mask = mb_mask[:, 1:]           # [B, T_s-1]
                    s_resp, s_compact_mask = pack_response_logits(student_logits, s_shift_mask)

                # -- Broadcast R_max so teacher can allocate matching tensors --
                R_max_t = torch.tensor(
                    [s_resp.shape[1] if is_student else 0], dtype=torch.long, device=device
                )
                dist.broadcast(R_max_t, src=0, group=all_group)
                R_max = int(R_max_t.item())

                # -- Teacher forward (frozen initial policy, no grad) --
                # The teacher p_T(· | x, y*) conditions on both the problem and
                # the reference solution. Gradients must NOT flow through the
                # teacher — it acts as a fixed target distribution (OPSD Eq. 1).
                # The teacher is the frozen initial policy and never updated.
                if is_teacher:
                    t_shift_mask = t_mb_mask[:, 1:]
                    with torch.no_grad():
                        teacher_logits = teacher.get_logits(t_mb_ids, t_mb_attn)[:, :-1]
                    t_resp, t_compact_mask = pack_response_logits(teacher_logits, t_shift_mask)
                    if t_resp.shape[1] < R_max:
                        pad_len = R_max - t_resp.shape[1]
                        t_resp = torch.cat(
                            [t_resp, t_resp.new_zeros(t_resp.shape[0], pad_len, t_resp.shape[-1])], dim=1
                        )
                        t_compact_mask = torch.cat(
                            [t_compact_mask, t_compact_mask.new_zeros(t_compact_mask.shape[0], pad_len)], dim=1
                        )
                    elif t_resp.shape[1] > R_max:
                        t_resp         = t_resp[:, :R_max]
                        t_compact_mask = t_compact_mask[:, :R_max]

                # -- Top-K selection and log-prob computation --
                B = mb_ids.shape[0]

                if select_topk_by == "student":
                    if is_student:
                        _, topk_idx = s_resp.topk(K, dim=-1)   # [B, R, K]
                    else:
                        topk_idx = torch.empty(B, R_max, K, dtype=torch.long, device=device)
                    dist.broadcast(topk_idx, src=0, group=all_group)

                    if is_teacher:
                        t_logprobs = teacher_logprobs_at_indices(
                            t_resp, topk_idx, chunk_size=args.teacher_chunk_size
                        ).float()
                        t_topk_idx, t_own_logprobs = teacher_topk_logprobs(
                            t_resp, K, chunk_size=args.teacher_chunk_size
                        )
                        t_own_logprobs = t_own_logprobs.float()
                        del t_resp
                    else:
                        t_logprobs     = torch.empty(B, R_max, K, dtype=torch.float32, device=device)
                        t_topk_idx     = torch.empty(B, R_max, K, dtype=torch.long,    device=device)
                        t_own_logprobs = torch.empty(B, R_max, K, dtype=torch.float32, device=device)
                        t_compact_mask = torch.zeros(B, R_max,    dtype=torch.float32, device=device)

                    dist.broadcast(t_logprobs,     src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_topk_idx,     src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_own_logprobs, src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_compact_mask, src=teacher_global_rank, group=all_group)

                    student_topk_idx = topk_idx
                    teacher_topk_idx = t_topk_idx
                    t_lp_at_student  = t_logprobs

                else:  # forward_kl: teacher selects top-K
                    if is_student:
                        _, student_topk_idx = s_resp.topk(K, dim=-1)
                    else:
                        student_topk_idx = torch.empty(B, R_max, K, dtype=torch.long, device=device)
                    dist.broadcast(student_topk_idx, src=0, group=all_group)

                    if is_teacher:
                        teacher_topk_idx, t_logprobs = teacher_topk_logprobs(
                            t_resp, K, chunk_size=args.teacher_chunk_size
                        )
                        t_logprobs      = t_logprobs.float()
                        t_own_logprobs  = t_logprobs
                        t_lp_at_student = teacher_logprobs_at_indices(
                            t_resp, student_topk_idx, chunk_size=args.teacher_chunk_size
                        ).float()
                        del t_resp
                    else:
                        teacher_topk_idx = torch.empty(B, R_max, K, dtype=torch.long,    device=device)
                        t_logprobs       = torch.empty(B, R_max, K, dtype=torch.float32, device=device)
                        t_own_logprobs   = torch.empty(B, R_max, K, dtype=torch.float32, device=device)
                        t_lp_at_student  = torch.empty(B, R_max, K, dtype=torch.float32, device=device)
                        t_compact_mask   = torch.zeros(B, R_max,    dtype=torch.float32, device=device)

                    dist.broadcast(teacher_topk_idx, src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_logprobs,       src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_own_logprobs,   src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_lp_at_student,  src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_compact_mask,   src=teacher_global_rank, group=all_group)

                    topk_idx = teacher_topk_idx

                # -- Loss and backward (student ranks only) --
                if is_student:
                    s_log_resp = F.log_softmax(s_resp.float(), dim=-1)
                    s_logprobs = s_log_resp.gather(-1, topk_idx)   # [B, R, K]

                    # Exclude positions where the teacher sequence was truncated
                    # (reference solution may push teacher prompt past max context).
                    effective_mask = s_compact_mask * t_compact_mask   # [B, R_max]
                    if effective_mask.sum() == 0:
                        print0(
                            f"[warn mb] effective_mask is all-zero: "
                            f"s_mask={s_compact_mask.sum().item():.0f} "
                            f"t_mask={t_compact_mask.sum().item():.0f}",
                            flush=True,
                        )

                    # Per-token pointwise KL clipping (OPSD paper Section 3.2).
                    # Stylistic tokens can exhibit much higher KL than math tokens,
                    # dominating the gradient signal. Clipping each token's divergence
                    # contribution to τ stabilises training and prevents performance
                    # collapse, especially for smaller models (Figure 4).
                    tis_weights = None
                    if args.tis_clip > 0.0:
                        sampled_ids    = mb_ids[:, 1:]
                        s_lp_sampled   = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
                        inf_lp_shifted = mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)
                        tis_full       = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=args.tis_clip)
                        tis_resp, _    = pack_response_logits(
                            tis_full.unsqueeze(-1).expand_as(student_logits), s_shift_mask
                        )
                        tis_weights = tis_resp[..., 0]   # [B, R_max]

                    loss = loss_fn(
                        s_logprobs, t_logprobs, effective_mask,
                        tis_weights=tis_weights,
                        kl_clip=args.kl_clip if args.kl_clip > 0.0 else None,
                    ) / n_mb
                    student._scale_loss(loss).backward()
                    total_loss += loss.item()
                    n_batches  += 1

                    with torch.no_grad():
                        s_lp_metrics = s_log_resp.gather(-1, student_topk_idx)
                        overlap_ratio     += compute_overlap_ratio(student_topk_idx, teacher_topk_idx).item()
                        overlap_advantage += compute_overlap_token_advantage(
                            student_topk_idx, teacher_topk_idx, s_lp_metrics, t_lp_at_student
                        ).item()
                        entropy_gap_val   += compute_entropy_gap(s_lp_metrics, t_own_logprobs).item()

            if is_student:
                student._optimizer_step()

            # No teacher sync: the teacher is the frozen initial policy and is
            # never updated. This is the key design choice in OPSD (Section 4.1):
            # fixing the teacher to the initial policy stabilises the distillation
            # target and acts as an implicit regulariser anchoring the student to
            # the pretrained distribution.

        # -- Push updated student weights into vLLM for next step's rollouts --
        if is_student:
            sync_weights_to_vllm_inplace(
                student.model, args.rollout_worker_url, model_update_group, fsdp=True,
            )

            dt = time.time() - t0
            avg_loss   = total_loss / max(n_batches, 1)
            current_lr = student.scheduler.get_last_lr()[0] if student.scheduler is not None else args.lr
            tokens     = input_ids.numel()
            print0(
                f"step {step + 1:4d}/{args.num_steps} | loss {avg_loss:.4f} "
                f"| lr {current_lr:.2e} | tokens {tokens} | dt {dt:.1f}s "
                f"| overlap {overlap_ratio / max(n_batches, 1):.3f} "
                f"| adv {overlap_advantage / max(n_batches, 1):.4f} "
                f"| ent_gap {entropy_gap_val / max(n_batches, 1):.4f}"
            )

            if master_process and use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/step_time_s": dt,
                        "train/tokens_per_step": tokens,
                        "metrics/overlap_ratio": overlap_ratio / max(n_batches, 1),
                        "metrics/overlap_token_advantage": overlap_advantage / max(n_batches, 1),
                        "metrics/entropy_gap": entropy_gap_val / max(n_batches, 1),
                    },
                    step=step + 1,
                )

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                save_path = f"{args.save_dir}/step_{step + 1}"
                student.save_model(save_path)
                print0(f"Saved checkpoint to {save_path}")

        dist.barrier(group=all_group)

    compute_cleanup()
    if master_process and use_wandb:
        wandb.finish()
