
import time
import argparse
import socket as _socket
from typing import Literal

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from nanoopd.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanoopd.loss import compute_reverse_kl_loss, compute_forward_kl_loss, ALGORITHMS
from nanoopd.fsdp.algorithms import compute_topk_logprobs_for_distillation
from nanoopd.fsdp.model import TeacherModel, StudentModel
from nanoopd.rollout import (
    generate_rollouts_remote,
    remote_vllm_init_weight_transfer,
    sync_weights_to_vllm_inplace,
    prepare_batch,
    wait_for_rollout_worker,
)
from nanoopd.data.dataset import distributed_opd_loader, build_opd_dataset
from nanoopd.eval_aime import run_eval

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="On-policy distillation training")
    # Model
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-gpu-id", type=int, default=0,
                        help="Rank ID (within training process) that loads the teacher.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="reverse_kl", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--distill-top-k", type=int, default=100, help="Top-K vocab for KL distillation")
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
    parser.add_argument("--train-batch-size", type=int, default=4)
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Distributed init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    print0(f"Student: {args.student_model}")
    print0(f"Teacher: {args.teacher_model}")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  World size: {ddp_world_size}")

    if master_process:
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

    assert args.prompts_per_step % ddp_world_size == 0, (
        f"prompts_per_step ({args.prompts_per_step}) must be divisible by "
        f"world_size ({ddp_world_size})"
    )

    # -----------------------------------------------------------------------------
    # Student model (FSDP)
    student_config = OmegaConf.create({
        "model_name": args.student_model,
        "dtype": "bfloat16",
        "enable_gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "attn_implementation": "flash_attention_2",
        "sharding_strategy": args.sharding_strategy,
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_ratio": 0.05,
        },
    })
    student = StudentModel(student_config, data_parallel_size=ddp_world_size)
    student.prepare_scheduler(total_steps=args.num_steps * args.epochs)

    # Teacher model (single rank, no TP)
    teacher = TeacherModel(
        model_name=args.teacher_model,
        gpu_ids=[args.teacher_gpu_id],
        dtype="bfloat16",
    )

    # -----------------------------------------------------------------------------
    # Loss function and top-K selection
    if args.algorithm == "reverse_kl":
        loss_fn = compute_reverse_kl_loss
        select_topk_by: Literal["student", "teacher"] = "student"
    else:  # forward_kl
        loss_fn = compute_forward_kl_loss
        select_topk_by = "teacher"

    # -----------------------------------------------------------------------------
    # vLLM weight-transfer setup
    wait_for_rollout_worker(args.rollout_worker_url)

    master_addr = _socket.gethostbyname(_socket.gethostname())
    nccl_port = 29600
    if master_process:
        remote_vllm_init_weight_transfer(
            args.rollout_worker_url,
            master_address=master_addr,
            master_port=nccl_port,
            rank_offset=ddp_world_size,
            world_size=ddp_world_size + args.rollout_worker_world_size,
        )
    if ddp:
        dist.barrier()

    model_update_group = dist.new_group(list(range(ddp_world_size))) if ddp else None

    # -----------------------------------------------------------------------------
    # Dataset
    dataset = build_opd_dataset()
    loader = distributed_opd_loader(
        dataset, args.prompts_per_step, ddp_world_size, ddp_rank, seed=args.seed
    )

    # -----------------------------------------------------------------------------
    # Training loop
    for step, (examples, _) in enumerate(loader):
        if step >= args.num_steps:
            break

        t0 = time.time()

        # ---- Rollout generation (each rank generates for its own prompts) ----
        prompts = [ex.prompt for ex in examples]
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
            rewards=[0.0] * len(rollouts),
            tokenizer=student.tokenizer,
            max_seq_len=args.max_seq_len,
            device=device,
        )
        input_ids      = batch["input_ids"]        # [N, T]
        attention_mask = batch["attention_mask"]
        response_mask  = batch["response_mask"]

        # ---- Distillation epochs ----
        student.model.train()
        total_loss = 0.0
        n_batches  = 0

        for _epoch in range(args.epochs):
            perm = torch.randperm(input_ids.shape[0], device=device)
            for start in range(0, input_ids.shape[0], args.train_batch_size):
                idx     = perm[start : start + args.train_batch_size]
                mb_ids  = input_ids[idx]
                mb_attn = attention_mask[idx]
                mb_mask = response_mask[idx]

                # Student forward (with grad)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = student.model(
                        input_ids=mb_ids, attention_mask=mb_attn
                    ).logits[:, :-1]                    # [B, T-1, V]

                # Teacher forward on the single teacher rank; broadcast to all ranks.
                teacher_logits = teacher.get_logits(mb_ids, mb_attn)
                if teacher_logits is None:
                    B, T = mb_ids.shape
                    V = student_logits.shape[-1]
                    teacher_logits = torch.empty(B, T, V, dtype=torch.bfloat16, device=device)
                dist.broadcast(teacher_logits, src=args.teacher_gpu_id, group=model_update_group)
                teacher_logits = teacher_logits[:, :-1]  # [B, T-1, V]

                s_logprobs, t_logprobs, _ = compute_topk_logprobs_for_distillation(
                    student_logits, teacher_logits,
                    top_k=args.distill_top_k,
                    select_topk_by=select_topk_by,
                )

                shift_mask = mb_mask[:, 1:]              # [B, T-1]
                loss = loss_fn(s_logprobs, t_logprobs, shift_mask)
                student._scale_loss(loss).backward()

                total_loss += loss.item()
                n_batches  += 1

            student._optimizer_step()

        # ---- Sync updated student weights into vLLM ----
        sync_weights_to_vllm_inplace(
            student.model,
            args.rollout_worker_url,
            model_update_group,
            fsdp=True,
        )

        dt = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = student.scheduler.get_last_lr()[0] if student.scheduler is not None else args.lr
        print0(f"step {step + 1:4d}/{args.num_steps} | loss {avg_loss:.4f} | lr {current_lr:.2e} | dt {dt:.1f}s")

        if master_process:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/step_time_s": dt,
                    "train/tokens_per_step": input_ids.numel(),
                },
                step=step + 1,
            )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_path = f"{args.save_dir}/step_{step + 1}"
            student.save_model(save_path)
            print0(f"Saved checkpoint to {save_path}")

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if master_process:
                run_eval(
                    rollout_worker_url=args.rollout_worker_url,
                    tokenizer=student.tokenizer,
                    eval_k=args.eval_k,
                    eval_max_tokens=args.eval_max_tokens,
                    step=step + 1,
                )

    compute_cleanup()
    if master_process:
        wandb.finish()
