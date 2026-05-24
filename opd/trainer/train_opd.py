
import os
import math
import time
import argparse
from typing import Literal

import torch
import torch.distributed as dist

from opd.common import compute_cleanup, print0
from opd.loss import ALGORITHMS
from opd.fsdp.algorithms import (
    student_logprobs_at_indices,
    student_logprob_at_sampled_tokens,
)
from opd.trainer.distillation_utils import broadcast_minibatch, exchange_topk
from opd.trainer.setup_utils import init_distributed, build_student, build_teacher, init_vllm_transfer
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

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="On-policy distillation training")
    # Model
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--train-world-size", type=int, required=True,
                        help="Number of student (FSDP) ranks. Teacher ranks occupy the "
                             "remaining ranks in the torchrun world.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="reverse_kl", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--distill-top-k", type=int, default=100, help="Top-K vocab for KL distillation")
    parser.add_argument("--student-chunk-size", type=int, default=-1,
                        help="Chunk size along T for student logits in top-K computation (-1 = no chunking)")
    parser.add_argument("--teacher-chunk-size", type=int, default=-1,
                        help="Chunk size along T for teacher logits in top-K computation (-1 = no chunking)")
    parser.add_argument("--tis-clip", type=float, default=0.0, help="TIS importance-weight clip C (0 disables)")
    # Generation
    parser.add_argument("--num-samples", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50, help="vLLM sampling top-k")
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=4,
                        help="Sequences per gradient accumulation step. Optimizer updates after all "
                             "prompts_per_step * num_samples sequences are processed.")
    parser.add_argument("--epochs", type=int, default=1, help="Optimizer steps per rollout batch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="opd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0, help="Eval on AIME every N steps (0=disabled)")
    parser.add_argument("--eval-k", type=int, default=4, help="Number of samples per problem for pass@k eval")
    parser.add_argument("--eval-max-tokens", type=int, default=4096, help="Max tokens for eval generation")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["livecodebench", "sciknoweval", "dapo_math"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = os.environ.get("USE_WANDB", "1").strip().lower() not in ("0", "false", "no")
    if use_wandb:
        import wandb

    # -----------------------------------------------------------------------------
    # Distributed init
    ctx = init_distributed(args.device_type, args.train_world_size)
    ddp_rank          = ctx.ddp_rank
    ddp_world_size    = ctx.ddp_world_size
    device            = ctx.device
    train_world_size  = ctx.train_world_size
    teacher_global_rank = ctx.teacher_global_rank
    is_student        = ctx.is_student
    is_teacher        = ctx.is_teacher
    master_process    = ctx.master_process
    student_group     = ctx.student_group
    all_group         = ctx.all_group

    print0(f"Student: {args.student_model}")
    print0(f"Teacher: {args.teacher_model}")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}")

    if master_process and use_wandb:
        wandb.init(
            project="nano-opd",
            name=args.run_name,
            config={
                "student_model": args.student_model,
                "teacher_model": args.teacher_model,
                "algorithm": args.algorithm,
                "distill_top_k": args.distill_top_k,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "num_steps": args.num_steps,
                "prompts_per_step": args.prompts_per_step,
                "train_batch_size": args.train_batch_size,
                "epochs": args.epochs,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_len": args.max_seq_len,
                "temperature": args.temperature,
                "sharding_strategy": args.sharding_strategy,
            },
        )

    assert args.prompts_per_step % train_world_size == 0, (
        f"prompts_per_step ({args.prompts_per_step}) must be divisible by "
        f"train_world_size ({train_world_size})"
    )

    # -----------------------------------------------------------------------------
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
        teacher = build_teacher(args.teacher_model)

    # -----------------------------------------------------------------------------
    # Loss function and top-K selection
    loss_fn = ALGORITHMS[args.algorithm]
    # reverse_kl and jsd weight by the student distribution → student selects top-K
    # forward_kl weights by the teacher distribution → teacher selects top-K
    select_topk_by: Literal["student", "teacher"] = (
        "teacher" if args.algorithm == "forward_kl" else "student"
    )

    # -----------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only; teacher is not involved)
    model_update_group = init_vllm_transfer(
        args.rollout_worker_url,
        rollout_worker_world_size=args.rollout_worker_world_size,
        train_world_size=train_world_size,
        master_process=master_process,
        all_group=all_group,
    )

    # -----------------------------------------------------------------------------
    # Dataset (student ranks only)
    if is_student:
        dataset = build_opd_dataset(args.dataset)
        loader = distributed_opd_loader(
            dataset, args.prompts_per_step, train_world_size, ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -----------------------------------------------------------------------------
    # Training loop — all ranks iterate together; students and teacher take
    # different code paths but participate in the same NCCL collectives.
    for step in range(args.num_steps):
        t0 = time.time()

        # ---- Rollout generation (student ranks only) ----
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
            batch = prepare_batch(
                rollouts,
                tokenizer=student.tokenizer,
                max_seq_len=args.max_seq_len,
                device=device,
            )
            input_ids         = batch["input_ids"]          # [N, T]
            attention_mask    = batch["attention_mask"]
            response_mask     = batch["response_mask"]
            inference_logprobs = batch["inference_logprobs"] # [N, T]
            student.model.train()

        total_loss       = 0.0
        n_batches        = 0
        overlap_ratio    = 0.0
        overlap_advantage = 0.0
        entropy_gap_val  = 0.0

        # ---- Distillation epochs ----
        for _epoch in range(args.epochs):

            # Broadcast the number of minibatches this epoch so the teacher rank
            # knows how many iterations to participate in.
            if is_student:
                n_mb = math.ceil(input_ids.shape[0] / args.train_batch_size)
                perm = torch.randperm(input_ids.shape[0], device=device)
                n_mb_t = torch.tensor([n_mb], dtype=torch.long, device=device)
            else:
                n_mb_t = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(n_mb_t, src=0, group=all_group)
            n_mb = int(n_mb_t.item())

            for mb_idx in range(n_mb):

                # -- Broadcast minibatch shape then data to teacher rank --
                if is_student:
                    start     = mb_idx * args.train_batch_size
                    idx       = perm[start : start + args.train_batch_size]
                    mb_ids    = input_ids[idx]
                    mb_attn   = attention_mask[idx]
                    mb_mask   = response_mask[idx]
                    mb_inf_lp = inference_logprobs[idx]
                else:
                    mb_ids = mb_attn = None

                mb_ids, mb_attn = broadcast_minibatch(
                    is_student, mb_ids, mb_attn, device, all_group
                )

                # -- Student forward (with grad) --
                if is_student:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        student_logits = student.model(
                            input_ids=mb_ids, attention_mask=mb_attn
                        ).logits[:, :-1]               # [B, T-1, V]

                # -- Teacher: compute top-K log-probs and broadcast --
                # Broadcasts only [B, T-1, K] instead of the full [B, T-1, V] logit
                # tensor, reducing per-minibatch communication by ~vocab/K (>1000×).
                topk = exchange_topk(
                    select_topk_by=select_topk_by,
                    is_student=is_student,
                    is_teacher=is_teacher,
                    student_logits=student_logits if is_student else None,
                    teacher=teacher if is_teacher else None,
                    mb_ids=mb_ids,
                    mb_attn=mb_attn,
                    K=args.distill_top_k,
                    s_chunk=args.student_chunk_size,
                    t_chunk=args.teacher_chunk_size,
                    teacher_global_rank=teacher_global_rank,
                    all_group=all_group,
                    device=device,
                )
                topk_idx             = topk["topk_idx"]
                t_logprobs           = topk["t_logprobs"]
                student_topk_idx     = topk["student_topk_idx"]
                teacher_topk_idx     = topk["teacher_topk_idx"]
                t_logprobs_at_student = topk["t_logprobs_at_student"]
                teacher_own_logprobs  = topk["teacher_own_logprobs"]

                # -- Student: compute TIS weights then loss and backward --
                if is_student:
                    s_logprobs = student_logprobs_at_indices(student_logits, topk_idx, args.student_chunk_size)
                    shift_mask = mb_mask[:, 1:]                             # [B, T-1]

                    # TIS weight: corrects for numerical gap between vLLM inference
                    # log-probs and training-time log-probs (SDPO paper, Eq. 12 / A.4).
                    # w_t = exp(log π_train(y_t) − log π_vllm(y_t)), clipped to C.
                    if args.tis_clip > 0.0:
                        sampled_ids = mb_ids[:, 1:]                         # [B, T-1]
                        s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
                        inf_lp_shifted = mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)  # [B, T-1]
                        tis_weights = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=args.tis_clip)
                    else:
                        tis_weights = None

                    loss = loss_fn(s_logprobs, t_logprobs, shift_mask, tis_weights=tis_weights) / n_mb
                    student._scale_loss(loss).backward()
                    total_loss += loss.item()
                    n_batches  += 1

                    # Compute distillation health metrics (no grad)
                    with torch.no_grad():
                        s_lp_for_metrics = student_logprobs_at_indices(student_logits, student_topk_idx, args.student_chunk_size)
                        overlap_ratio     += compute_overlap_ratio(student_topk_idx, teacher_topk_idx).item()
                        overlap_advantage += compute_overlap_token_advantage(
                            student_topk_idx, teacher_topk_idx, s_lp_for_metrics, t_logprobs_at_student
                        ).item()
                        entropy_gap_val   += compute_entropy_gap(s_lp_for_metrics, teacher_own_logprobs).item()

            if is_student:
                student._optimizer_step()

        # ---- Sync updated student weights into vLLM (student ranks only) ----
        if is_student:
            sync_weights_to_vllm_inplace(
                student.model,
                args.rollout_worker_url,
                model_update_group,
                fsdp=True,
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
                student.save_model(save_path)   # barriers within student_group only
                print0(f"Saved checkpoint to {save_path}")

        # All ranks sync before eval so FSDP/NCCL state is settled.
        # Eval only runs on rank 0 and can take minutes (vLLM generation);
        # the barrier must fire first so other ranks don't time out waiting.
        dist.barrier(group=all_group)

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if master_process:
                _ENV_CLS[args.dataset].evaluate(
                    rollout_worker_url=args.rollout_worker_url,
                    step=step + 1,
                    tokenizer=student.tokenizer,
                    eval_k=args.eval_k,
                    eval_max_tokens=args.eval_max_tokens,
                )

    compute_cleanup()
    if master_process and use_wandb:
        wandb.finish()
