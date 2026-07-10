"""On-Policy Distillation (OPD) training loop — student + separate frozen teacher."""

import argparse
import time
from typing import Literal

import torch

from opd.loss import ALGORITHMS
from opd.fsdp.algorithms import (
    student_logprobs_at_indices,
    student_logprob_at_sampled_tokens,
)
from opd.trainer.distillation_utils import exchange_topk, exchange_sampled_teacher_logprob
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
from opd.trainer.setup_utils import (
    assert_prompts_divisible,
    build_student,
    build_teacher,
    compute_cleanup,
    init_distributed,
    init_vllm_transfer,
    print0,
    topk_selector_for,
)
from opd.trainer.trainer_utils import MinibatchTensors, StepAccumulator, Trainer
from opd.metrics import compute_topk_health_metrics
from opd.generator.rollout import generate_rollouts_remote
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
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-response-len", type=int, default=1536)
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

    use_wandb = should_use_wandb()

    # -----------------------------------------------------------------------------
    # Distributed init
    ctx = init_distributed(args.device_type, args.train_world_size)
    ddp_rank          = ctx.ddp_rank
    ddp_world_size    = ctx.ddp_world_size
    device            = ctx.device
    train_world_size  = ctx.train_world_size
    is_student        = ctx.is_student
    is_teacher        = ctx.is_teacher
    master_process    = ctx.master_process
    student_group     = ctx.student_group
    all_group         = ctx.all_group

    print0(f"Student: {args.student_model}")
    print0(f"Teacher: {args.teacher_model}")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}")

    init_wandb(
        args.run_name, master_process, use_wandb,
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
            "max_prompt_len": args.max_prompt_len,
            "max_response_len": args.max_response_len,
            "temperature": args.temperature,
            "sharding_strategy": args.sharding_strategy,
        },
    )

    assert_prompts_divisible(args.prompts_per_step, train_world_size)

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
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(args.algorithm)

    # -----------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only; teacher is not involved)
    model_update_group = init_vllm_transfer(
        args.rollout_worker_url,
        rollout_worker_world_size=args.rollout_worker_world_size,
        train_world_size=train_world_size,
        master_process=master_process,
        all_group=all_group,
    )

    trainer = Trainer(
        args, ctx,
        student if is_student else None, teacher if is_teacher else None,
        model_update_group, use_wandb,
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
    # Per-minibatch exchange + loss + backward. OPD has no separate teacher
    # batch — the teacher forward runs on the same broadcast mb_ids/mb_attn as
    # the student (same-length sequences, no packing needed).
    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        B, T = mb.mb_ids.shape[0], mb.mb_ids.shape[1] - 1

        # -- Student forward (with grad) --
        if is_student:
            student_logits = student.get_logits(mb.mb_ids, mb.mb_attn)[:, :-1]  # [B, T-1, V]

        # -- Teacher: compute top-K log-probs and broadcast --
        # Broadcasts only [B, T-1, K] instead of the full [B, T-1, V] logit
        # tensor, reducing per-minibatch communication by ~vocab/K (>1000×).
        if is_teacher:
            teacher_logits = teacher.get_logits(mb.mb_ids, mb.mb_attn)[:, :-1]
        else:
            teacher_logits = None

        if args.algorithm == "mopd_pg_loss":
            # -- PG form: no top-K exchange needed for the loss
            # itself, just the sampled token's log-prob under each policy.
            # mb_ids is already on every rank (broadcast by the Trainer), so
            # sampled_ids needs no extra communication — only the
            # teacher's [B, T-1] logprob at those ids crosses the wire.
            sampled_ids = mb.mb_ids[:, 1:]                             # [B, T-1]
            t_logprob = exchange_sampled_teacher_logprob(
                ctx=ctx,
                teacher_logits=teacher_logits,
                token_ids=sampled_ids,
                B=B, T=T,
                teacher_chunk_size=args.teacher_chunk_size,
            )
            topk = exchange_topk(
                ctx=ctx,
                select_topk_by=select_topk_by,
                student_logits=student_logits if is_student else None,
                teacher_logits=teacher_logits,
                B=B, T=T,
                top_k=args.distill_top_k,
                student_chunk_size=args.student_chunk_size,
                teacher_chunk_size=args.teacher_chunk_size,
            )

            if is_student:
                s_logprob = student_logprobs_at_indices(
                    student_logits, sampled_ids.unsqueeze(-1), args.student_chunk_size
                ).squeeze(-1)                                      # [B, T-1], grad-carrying
                shift_mask = mb.mb_mask[:, 1:]                     # [B, T-1]

                if args.tis_clip > 0.0:
                    s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
                    inf_lp_shifted = mb.mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)
                    tis_weights = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=args.tis_clip)
                else:
                    tis_weights = None

                loss = loss_fn(s_logprob, t_logprob, shift_mask, tis_weights=tis_weights) / mb.n_mb
                student._scale_loss(loss).backward()
                acc.add_loss(loss)

                with torch.no_grad():
                    ratio, adv, ent_gap = compute_topk_health_metrics(
                        student_logits, topk, args.student_chunk_size
                    )
                    acc.add_health_metrics(ratio, adv, ent_gap)
            return

        topk = exchange_topk(
            ctx=ctx,
            select_topk_by=select_topk_by,
            student_logits=student_logits if is_student else None,
            teacher_logits=teacher_logits,
            B=B, T=T,
            top_k=args.distill_top_k,
            student_chunk_size=args.student_chunk_size,
            teacher_chunk_size=args.teacher_chunk_size,
        )
        # -- Student: compute TIS weights then loss and backward --
        if is_student:
            s_logprobs = student_logprobs_at_indices(student_logits, topk.topk_idx, args.student_chunk_size)
            shift_mask = mb.mb_mask[:, 1:]                             # [B, T-1]

            # TIS weight: corrects for numerical gap between vLLM inference
            # log-probs and training-time log-probs (SDPO paper, Eq. 12 / A.4).
            # w_t = exp(log π_train(y_t) − log π_vllm(y_t)), clipped to C.
            if args.tis_clip > 0.0:
                sampled_ids = mb.mb_ids[:, 1:]                         # [B, T-1]
                s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
                inf_lp_shifted = mb.mb_inf_lp[:, 1:].to(s_lp_sampled.dtype)  # [B, T-1]
                tis_weights = (s_lp_sampled - inf_lp_shifted).exp().clamp(max=args.tis_clip)
            else:
                tis_weights = None

            loss = loss_fn(s_logprobs, topk.t_logprobs, shift_mask, tis_weights=tis_weights) / mb.n_mb
            student._scale_loss(loss).backward()
            acc.add_loss(loss)

            # Compute distillation health metrics (no grad)
            with torch.no_grad():
                ratio, adv, ent_gap = compute_topk_health_metrics(
                    student_logits, topk, args.student_chunk_size
                )
                acc.add_health_metrics(ratio, adv, ent_gap)

    # -----------------------------------------------------------------------------
    # Training loop — all ranks iterate together; students and teacher take
    # different code paths but participate in the same NCCL collectives.
    for step in range(args.num_steps):
        t0 = time.time()

        # ---- Rollout generation (student ranks only) ----
        rollouts = None
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

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=False)

        trainer.step(step, t0, batch, teacher_batch, do_minibatch, has_teacher_batch=False)

        # All ranks sync before eval so FSDP/NCCL state is settled.
        # Eval only runs on rank 0 and can take minutes (vLLM generation);
        # the barrier must fire first so other ranks don't time out waiting.
        trainer.barrier()

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
    finish_wandb(master_process, use_wandb)
