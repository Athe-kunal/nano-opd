
import os
import math
import time
import argparse
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F

from opd.common import compute_cleanup, print0
from opd.loss import ALGORITHMS
from opd.fsdp.algorithms import student_logprob_at_sampled_tokens
from opd.trainer.distillation_utils import (
    broadcast_minibatch,
    broadcast_teacher_inputs,
    minibatch_exchange,
    mopd_pg_loss_and_backward,
    pack_response_logits,
    prepare_teacher_batch,
    sync_student_to_teacher,
)
from opd.trainer.setup_utils import init_distributed, build_student, build_teacher, init_vllm_transfer
from opd.trainer.sync_teacher import SYNC_METHODS, build_syncer
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
from opd.envs.dataset import distributed_opd_loader, build_opd_dataset
from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

_ENV_CLS = {
    "dapo_math": DapoMathEnv,
    "livecodebench": LiveCodeBenchEnv,
    "sciknoweval": SciKnowEvalEnv,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_teacher_messages(init_messages, env_output, successful_rollout):
    """Construct the self-teacher prompt following Table 2 of the SDPO paper.

    The teacher sees the original question augmented with:
      - a successful rollout from the group (if any) as a correct reference
      - environment feedback from the current failed attempt (if it failed)

    Conditioning on this richer context lets the same model evaluate its own
    original response from a hindsight perspective, assigning dense logit-level
    credit without an external teacher.
    """
    user_content = init_messages[-1]["content"]
    parts = [user_content]
    has_extra = False
    if successful_rollout is not None:
        parts.append(f"\nCorrect solution:\n{successful_rollout}")
        has_extra = True
    if env_output:
        parts.append(
            f"\nThe following is feedback from your unsuccessful earlier attempt:\n{env_output}"
        )
        has_extra = True
    if has_extra:
        parts.append("\nCorrectly solve the original question.")

    teacher_messages = list(init_messages[:-1])  # preserve system message if any
    teacher_messages.append({"role": "user", "content": "\n".join(parts)})
    return teacher_messages, has_extra


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="Self-Distillation Policy Optimization (SDPO)")
    # Model — student and teacher share the same checkpoint; teacher is synced
    # to the student after every optimizer step via the chosen sync method.
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--train-world-size", type=int, required=True,
                        help="Number of student (FSDP) ranks. The teacher occupies "
                             "rank train_world_size in the torchrun world.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="jsd", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--distill-top-k", type=int, default=100,
                        help="Top-K vocab for KL distillation")
    parser.add_argument("--student-chunk-size", type=int, default=-1)
    parser.add_argument("--teacher-chunk-size", type=int, default=-1)
    parser.add_argument("--tis-clip", type=float, default=0.0,
                        help="TIS importance-weight clip C (0 disables)")
    # Generation
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Completions per prompt (group size G in Algorithm 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Optimizer step every N minibatches. "
                             "Effective batch size = train_batch_size * grad_accum_steps.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=512,
                        help="Hard cap on prompt tokens. Raises if exceeded.")
    parser.add_argument("--max-response-len", type=int, default=1536,
                        help="Cap on response tokens. Truncates silently if exceeded.")
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    # Teacher sync — controls how the self-teacher tracks the student
    parser.add_argument("--sync-method", type=str, default="ema",
                        choices=list(SYNC_METHODS.keys()),
                        help="How the self-teacher's weights follow the student after each step.")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="[ema] teacher ← α·student + (1−α)·teacher. "
                             "Small α → stable but lagging teacher.")
    parser.add_argument("--trust-region-beta", type=float, default=0.05,
                        help="[trust_region] teacher ← β·student + (1−β)·initial_weights. "
                             "Anchors the teacher to the pre-trained distribution.")
    parser.add_argument("--hard-sync-every-n", type=int, default=100,
                        help="[hard_sync] Full copy every N optimizer steps.")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="opd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-k", type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=4096)
    parser.add_argument("--sciknoweval-test-size", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, required=True, choices=list(_ENV_CLS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = os.environ.get("USE_WANDB", "1").strip().lower() not in ("0", "false", "no")
    if use_wandb:
        import wandb

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD:
    #   ranks 0..train_world_size-1  →  student (FSDP)
    #   rank  train_world_size       →  teacher (plain nn.Module, same model)
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

    print0(f"Model: {args.student_model}  (student = teacher, synced via {args.sync_method})")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}")

    if master_process and use_wandb:
        wandb.init(
            project="nano-opd",
            name=args.run_name,
            config={
                "student_model": args.student_model,
                "algorithm": args.algorithm,
                "distill_top_k": args.distill_top_k,
                "sync_method": args.sync_method,
                "ema_alpha": args.ema_alpha,
                "trust_region_beta": args.trust_region_beta,
                "hard_sync_every_n": args.hard_sync_every_n,
                "lr": args.lr,
                "num_steps": args.num_steps,
                "prompts_per_step": args.prompts_per_step,
                "train_batch_size": args.train_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "epochs": args.epochs,
                "num_samples": args.num_samples,
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
            scheduler_name=args.scheduler,
            warmup_ratio=args.warmup_ratio,
        )

    if is_teacher:
        # Same checkpoint as the student; weights will be synced after each step.
        teacher = build_teacher(args.student_model)

    # -------------------------------------------------------------------------
    # Teacher syncer — instantiated on all ranks so hyperparameters are visible,
    # but step() is only called on the teacher rank inside sync_student_to_teacher.
    syncer_kwargs: dict = {}
    if args.sync_method == "ema":
        syncer_kwargs["alpha"] = args.ema_alpha
    elif args.sync_method == "trust_region":
        if is_teacher:
            # Snapshot the initial weights as the regularization anchor.
            syncer_kwargs["initial_params"] = [
                p.data.clone() for p in teacher.model.parameters()
            ]
        else:
            syncer_kwargs["initial_params"] = []   # unused on student ranks
        syncer_kwargs["beta"] = args.trust_region_beta
    elif args.sync_method == "hard_sync":
        syncer_kwargs["sync_every_n_steps"] = args.hard_sync_every_n
    # "on_policy" takes no kwargs

    syncer = build_syncer(args.sync_method, **syncer_kwargs)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    loss_fn = ALGORITHMS[args.algorithm]
    select_topk_by: Literal["student", "teacher"] = (
        "teacher" if args.algorithm in ("forward_kl", "mopd_loss") else "student"
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
        dataset = build_opd_dataset(args.dataset, eval_test_size=args.sciknoweval_test_size, seed=args.seed)
        loader = distributed_opd_loader(
            dataset, args.prompts_per_step, train_world_size, ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Training loop — all ranks iterate together
    for step in range(args.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        if is_student:
            examples, _ = next(loader_iter)
            prompts = [
                student.tokenizer.apply_chat_template(
                    env.init([])[0], tokenize=False, add_generation_prompt=True
                )
                for env in examples
            ]
            rollouts = generate_rollouts_remote(
                args.rollout_worker_url,
                prompts=prompts,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Build feedback-augmented teacher prompts for each rollout.
            # For each question, if any rollout succeeded, pass it as a correct
            # reference for the failed ones (Table 2 of the SDPO paper).
            for i, env in enumerate(examples):
                group = rollouts[i * args.num_samples : (i + 1) * args.num_samples]
                rewards = [env.compute_reward(r["response"])[0] for r in group]
                successful_text = next(
                    (group[j]["response"] for j, rw in enumerate(rewards) if rw > 0), None
                )
                init_msgs, _ = env.init([])
                if i == 0:
                    print0(f"[debug step={step}] rewards={rewards} has_success={successful_text is not None}", flush=True)
                for j, r in enumerate(group):
                    env_output   = env.get_feedback(r["response"]) if rewards[j] == 0 else ""
                    # Paper Table 2: successful attempts pass their own response as the correct
                    # solution; failed attempts pass a different successful rollout (if any).
                    success_hint = r["response"] if rewards[j] > 0 else successful_text
                    teacher_msgs, has_distillation = _build_teacher_messages(init_msgs, env_output, success_hint)
                    r["has_distillation"] = has_distillation
                    r["teacher_prompt"] = student.tokenizer.apply_chat_template(
                        teacher_msgs, tokenize=False, add_generation_prompt=True
                    )

            batch = prepare_batch(
                rollouts, tokenizer=student.tokenizer,
                max_prompt_len=args.max_prompt_len,
                max_response_len=args.max_response_len,
                device=device,
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
            # 1 for rollouts where the teacher received augmented context (solution or feedback),
            # 0 for rollouts where teacher == student context (distillation signal is meaningless).
            sd_mask = torch.tensor(
                [r["has_distillation"] for r in rollouts], dtype=torch.float, device=device
            )  # [N]
            student.model.train()

        total_loss        = 0.0
        n_batches         = 0
        overlap_ratio     = 0.0
        overlap_advantage = 0.0
        entropy_gap_val   = 0.0

        # -- Distillation epochs --
        for _epoch in range(args.epochs):

            # Broadcast n_mb so the teacher rank knows how many iterations to do.
            if is_student:
                n_mb = math.ceil(input_ids.shape[0] / args.train_batch_size)
                perm = torch.randperm(input_ids.shape[0], device=device)
                n_mb_t = torch.tensor([n_mb], dtype=torch.long, device=device)
            else:
                n_mb_t = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(n_mb_t, src=0, group=all_group)
            n_mb = int(n_mb_t.item())

            G = args.grad_accum_steps
            for mb_idx in range(n_mb):
                # How many minibatches are in this accumulation window?
                # The last window may be smaller than G if n_mb % G != 0.
                window_start = (mb_idx // G) * G
                window_size  = min(window_start + G, n_mb) - window_start

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
                    mb_sd_mask = sd_mask[idx]              # [B]
                else:
                    mb_ids = mb_attn = t_mb_ids = t_mb_attn = t_mb_mask = mb_sd_mask = None

                # -- Broadcast student inputs to teacher (teacher needs B/T info) --
                mb_ids, mb_attn = broadcast_minibatch(
                    is_student, mb_ids, mb_attn, device, all_group
                )

                # -- Broadcast teacher-specific inputs (feedback-augmented sequences) --
                t_mb_ids, t_mb_attn, t_mb_mask = broadcast_teacher_inputs(
                    is_student, t_mb_ids, t_mb_attn, t_mb_mask, device, all_group
                )

                # -- Broadcast self-distillation mask --
                if not is_student:
                    mb_sd_mask = torch.zeros(mb_ids.shape[0], dtype=torch.float, device=device)
                dist.broadcast(mb_sd_mask, src=0, group=all_group)

                # -- Student + teacher forward, then top-K (or PG) exchange --
                # The self-teacher uses the same weights as the student but sees a
                # richer context: the question + feedback + original response.
                # stopgrad (no_grad, inside minibatch_exchange) prevents gradients
                # from flowing through the teacher back into the student's
                # computation graph (SDPO Eq. 1).
                is_pg = args.algorithm == "mopd_pg_loss"
                tk, s_resp, s_compact_mask, s_shift_mask, student_logits = minibatch_exchange(
                    is_student, is_teacher, mb_ids, mb_attn, mb_mask,
                    t_mb_ids, t_mb_attn, t_mb_mask,
                    student.model if is_student else None, teacher if is_teacher else None,
                    select_topk_by, K, args.student_chunk_size, args.teacher_chunk_size,
                    teacher_global_rank, all_group, device,
                    is_pg=is_pg,
                )

                # -- PG form: no top-K exchange at all, just the sampled-token
                # log-prob under each policy, packed to response positions. --
                if is_pg:
                    if is_student:
                        loss = mopd_pg_loss_and_backward(
                            student=student, pg=tk, loss_fn=loss_fn,
                            student_logits=student_logits, sampled_ids=mb_ids[:, 1:], s_shift_mask=s_shift_mask,
                            inf_lp_shifted=mb_inf_lp[:, 1:] if args.tis_clip > 0.0 else None,
                            tis_clip=args.tis_clip, divisor=window_size,
                            extra_mask=mb_sd_mask.unsqueeze(1),
                        )
                        total_loss += loss.item()
                        n_batches  += 1

                # -- Loss and backward (student ranks only; PG form already did its own above) --
                elif is_student:
                    s_log_resp      = F.log_softmax(s_resp.float(), dim=-1)
                    s_logprobs      = s_log_resp.gather(-1, tk.topk_idx)
                    s_lp_at_student = s_log_resp.gather(-1, tk.student_topk_idx)

                    # Exclude positions where the teacher sequence was truncated (due to
                    # its longer feedback-augmented prompt hitting max_seq_len). Those
                    # padded positions have log_softmax(0) = -log(V) — a spurious uniform
                    # distribution that would corrupt the loss signal.
                    # Also exclude samples where the teacher had no augmented context —
                    # those have teacher == student prompt so the KL signal is meaningless.
                    effective_mask = s_compact_mask * tk.t_compact_mask * mb_sd_mask.unsqueeze(1)  # [B, R_max]
                    if effective_mask.sum() == 0:
                        print0(f"[warn mb] effective_mask is all-zero: s_mask={s_compact_mask.sum().item():.0f} t_mask={tk.t_compact_mask.sum().item():.0f} sd_mask={mb_sd_mask.sum().item():.0f}", flush=True)

                    if args.tis_clip > 0.0:
                        sampled_ids    = mb_ids[:, 1:]
                        s_lp_sampled   = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
                        inf_lp_shifted = mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)
                        tis_full       = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=args.tis_clip)
                        tis_resp, _    = pack_response_logits(
                            tis_full.unsqueeze(-1).expand_as(student_logits), s_shift_mask
                        )
                        tis_weights = tis_resp[..., 0]   # [B, R_max]
                    else:
                        tis_weights = None

                    loss = loss_fn(s_logprobs, tk.t_logprobs, effective_mask, tis_weights=tis_weights) / window_size
                    student._scale_loss(loss).backward()
                    total_loss += loss.item()
                    n_batches  += 1

                    with torch.no_grad():
                        overlap_ratio     += compute_overlap_ratio(tk.student_topk_idx, tk.teacher_topk_idx).item()
                        overlap_advantage += compute_overlap_token_advantage(
                            tk.student_topk_idx, tk.teacher_topk_idx, s_lp_at_student, tk.t_logprobs_at_student
                        ).item()
                        entropy_gap_val   += compute_entropy_gap(s_lp_at_student, tk.teacher_own_logprobs).item()

                # Step after every G minibatches, and always on the final one.
                if is_student and ((mb_idx + 1) % G == 0 or mb_idx == n_mb - 1):
                    student._optimizer_step()

        # -- Sync updated student weights to the teacher rank (once per step) --
        # The teacher should track the fully-updated student after all epochs are
        # done, not after each intermediate epoch — syncing inside the epoch loop
        # would make the EMA teacher chase intermediate weights too aggressively.
        sync_student_to_teacher(
            student_fsdp_model=student.model if is_student else None,
            teacher=teacher if is_teacher else None,
            syncer=syncer,
            global_step=step,
            is_student=is_student,
            is_teacher=is_teacher,
            all_group=all_group,
        )

        # -- Sync updated student weights into vLLM (student ranks only) --
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
                f"| lr {current_lr:.2e} | tokens {tokens} | dt {dt:.1f}s"
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

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if master_process:
                _ENV_CLS[args.dataset].evaluate(
                    rollout_worker_url=args.rollout_worker_url,
                    step=step + 1,
                    tokenizer=student.tokenizer,
                    eval_k=args.eval_k,
                    eval_max_tokens=args.eval_max_tokens,
                    test_size=args.sciknoweval_test_size,
                )
            # All ranks wait so non-master ranks don't race ahead into the next
            # step's collectives while rank 0 is still running eval.
            dist.barrier(group=all_group)

    compute_cleanup()
    if master_process and use_wandb:
        wandb.finish()
