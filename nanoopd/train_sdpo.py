
import os
import math
import time
import argparse
from functools import partial
from typing import Literal, Optional, Dict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from nanoopd.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanoopd.loss import compute_reverse_kl_loss, compute_forward_kl_loss, compute_jsd_loss, ALGORITHMS
from nanoopd.fsdp.model import StudentModel
from nanoopd.fsdp.algorithms import (
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
    student_logprobs_at_indices,
)
from nanoopd.rollout import (
    generate_rollouts_remote,
    remote_vllm_init_weight_transfer,
    sync_weights_to_vllm_inplace,
    prepare_batch,
    wait_for_rollout_worker,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from nanoopd.data.dataset import distributed_opd_loader, build_opd_dataset
from nanoopd.eval_aime import run_eval


def update_teacher_ema(
    student_fsdp_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    ema_alpha: float,
    sync_method: str,
    initial_teacher_params: Optional[Dict[str, torch.Tensor]],
) -> None:
    """Update EMA teacher weights from student after each optimizer step.

    EMA:           ϕ ← α·θ + (1-α)·ϕ
    Trust-region:  ϕ ← α·θ + (1-α)·ϕ₀

    Runs locally on each student rank — no cross-rank communication needed.
    All ranks see identical student params (FSDP shards params, not data) and
    the same teacher state, so the EMA update is consistent without any barrier.
    """
    with FSDP.summon_full_params(student_fsdp_model, rank0_only=False, writeback=False):
        s_params = dict(student_fsdp_model.named_parameters())
        for name, t_param in teacher_model.named_parameters():
            s_data = s_params[name].data.to(dtype=t_param.dtype)
            if sync_method == "trust_region":
                assert initial_teacher_params is not None
                t_param.data.copy_(ema_alpha * s_data + (1.0 - ema_alpha) * initial_teacher_params[name])
            else:  # ema
                t_param.data.mul_(1.0 - ema_alpha).add_(ema_alpha * s_data)


if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="Self-policy distillation training (SDPO)")
    # Model
    parser.add_argument("--student-model", type=str, required=True)
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="jsd", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--distill-top-k", type=int, default=100, help="Top-K vocab for KL/JSD distillation")
    parser.add_argument("--student-chunk-size", type=int, default=-1,
                        help="Chunk size along T for student logits in top-K computation (-1 = no chunking)")
    parser.add_argument("--teacher-chunk-size", type=int, default=-1,
                        help="Chunk size along T for teacher logits in top-K computation (-1 = no chunking)")
    parser.add_argument("--jsd-alpha", type=float, default=0.5,
                        help="JSD mixture weight: 0.5=symmetric JSD, 0.0=forward KL, 1.0=reverse KL")
    # EMA / teacher-sync
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="EMA update rate α: teacher ← α·student + (1-α)·teacher")
    parser.add_argument("--ema-sync-method", type=str, default="ema",
                        choices=["ema", "trust_region"],
                        help="ema: exponential moving average; "
                             "trust_region: blend with initial weights ϕ₀")
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

    use_wandb = os.environ.get("USE_WANDB", "1").strip().lower() not in ("0", "false", "no")
    if use_wandb:
        import wandb

    # -----------------------------------------------------------------------------
    # Distributed init — all ranks are student (FSDP) ranks; no separate teacher rank.
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    print0(f"Student/Teacher model: {args.student_model}")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}"
           + (f"  jsd-alpha: {args.jsd_alpha}" if args.algorithm == "jsd" else ""))
    print0(f"EMA α={args.ema_alpha}  method={args.ema_sync_method}")
    print0(f"Device: {device}  World size: {ddp_world_size}")

    if master_process and use_wandb:
        wandb.init(
            project="nano-opd",
            name=args.run_name,
            config={
                "student_model": args.student_model,
                "algorithm": args.algorithm,
                "distill_top_k": args.distill_top_k,
                "jsd_alpha": args.jsd_alpha,
                "ema_alpha": args.ema_alpha,
                "ema_sync_method": args.ema_sync_method,
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
    student = StudentModel(
        student_config,
        data_parallel_size=ddp_world_size,
    )
    student.prepare_scheduler(total_steps=args.num_steps * args.epochs)

    # EMA teacher — same architecture and initial weights as student, kept as a plain
    # nn.Module on every rank. Never updated by the optimizer; never wrapped in FSDP.
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # Trust-region mixing anchors to the initial weights throughout training.
    initial_teacher_params: Optional[Dict[str, torch.Tensor]] = None
    if args.ema_sync_method == "trust_region":
        initial_teacher_params = {
            name: param.data.clone()
            for name, param in teacher_model.named_parameters()
        }

    # -----------------------------------------------------------------------------
    # Loss function and top-K selection
    if args.algorithm == "reverse_kl":
        loss_fn = compute_reverse_kl_loss
        select_topk_by: Literal["student", "teacher"] = "student"
    elif args.algorithm == "forward_kl":
        loss_fn = compute_forward_kl_loss
        select_topk_by = "teacher"
    else:  # jsd — student selects top-K (SDPO paper default)
        loss_fn = partial(compute_jsd_loss, jsd_alpha=args.jsd_alpha)
        select_topk_by = "student"

    # -----------------------------------------------------------------------------
    # vLLM weight-transfer setup
    wait_for_rollout_worker(args.rollout_worker_url)

    master_addr = os.environ.get("NCCL_MASTER_ADDR", "127.0.0.1")
    nccl_port = 29600
    transfer_world_size = ddp_world_size + args.rollout_worker_world_size
    if master_process:
        remote_vllm_init_weight_transfer(
            args.rollout_worker_url,
            master_address=master_addr,
            master_port=nccl_port,
            rank_offset=ddp_world_size,
            world_size=transfer_world_size,
        )
        model_update_group = NCCLWeightTransferEngine.trainer_init({
            "master_address": master_addr,
            "master_port": nccl_port,
            "world_size": transfer_world_size,
        })
    else:
        model_update_group = None

    # -----------------------------------------------------------------------------
    # Dataset
    dataset = build_opd_dataset()
    loader = distributed_opd_loader(
        dataset, args.prompts_per_step, ddp_world_size, ddp_rank, seed=args.seed
    )
    loader_iter = iter(loader)

    # -----------------------------------------------------------------------------
    # Training loop
    for step in range(args.num_steps):
        t0 = time.time()

        # ---- Rollout generation ----
        examples, _ = next(loader_iter)
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
        input_ids      = batch["input_ids"]       # [N, T]
        attention_mask = batch["attention_mask"]
        response_mask  = batch["response_mask"]
        student.model.train()

        total_loss = 0.0
        n_batches  = 0

        # ---- Distillation epochs ----
        for _epoch in range(args.epochs):
            n_mb = math.ceil(input_ids.shape[0] / args.train_batch_size)
            perm = torch.randperm(input_ids.shape[0], device=device)

            for mb_idx in range(n_mb):
                start  = mb_idx * args.train_batch_size
                idx    = perm[start : start + args.train_batch_size]
                mb_ids  = input_ids[idx]        # [B, T]
                mb_attn = attention_mask[idx]
                mb_mask = response_mask[idx]

                # -- Student forward (with grad) --
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = student.model(
                        input_ids=mb_ids, attention_mask=mb_attn
                    ).logits[:, :-1]               # [B, T-1, V]

                # -- EMA teacher forward (no grad, local — no cross-rank broadcast) --
                K = args.distill_top_k
                T_shifted = mb_ids.shape[1] - 1
                s_chunk = args.student_chunk_size
                t_chunk = args.teacher_chunk_size

                with torch.no_grad():
                    t_logits = teacher_model(
                        input_ids=mb_ids, attention_mask=mb_attn
                    ).logits[:, :-1]               # [B, T-1, V]

                if select_topk_by == "student":
                    topk_idx   = student_topk_indices(student_logits, K, s_chunk)
                    t_logprobs = teacher_logprobs_at_indices(t_logits, topk_idx, t_chunk)
                else:  # teacher selects (forward_kl)
                    topk_idx, t_logprobs = teacher_topk_logprobs(t_logits, K, t_chunk)

                # -- Loss and backward --
                s_logprobs = student_logprobs_at_indices(student_logits, topk_idx, s_chunk)
                shift_mask = mb_mask[:, 1:]                             # [B, T-1]
                loss = loss_fn(s_logprobs, t_logprobs, shift_mask)
                student._scale_loss(loss).backward()
                total_loss += loss.item()
                n_batches  += 1

            student._optimizer_step()

            # ---- EMA sync: teacher ← α·student + (1-α)·teacher ----
            # Applied once per epoch (after each optimizer step).
            # Local on each rank — no communication needed.
            update_teacher_ema(
                student.model, teacher_model,
                args.ema_alpha, args.ema_sync_method, initial_teacher_params,
            )

        # ---- Sync updated student weights into vLLM ----
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
        )

        if master_process and use_wandb:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/step_time_s": dt,
                    "train/tokens_per_step": tokens,
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
    if master_process and use_wandb:
        wandb.finish()
