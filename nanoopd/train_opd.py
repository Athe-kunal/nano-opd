
import os
import math
import time
import argparse
from typing import Literal

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nanoopd.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanoopd.loss import ALGORITHMS
from nanoopd.fsdp.model import TeacherModel, StudentModel
from nanoopd.fsdp.algorithms import (
    student_topk_indices,
    teacher_logprobs_at_indices,
    teacher_topk_logprobs,
    student_logprobs_at_indices,
    student_logprob_at_sampled_tokens,
)
from nanoopd.metrics import (
    compute_overlap_ratio,
    compute_overlap_token_advantage,
    compute_entropy_gap,
)
from nanoopd.rollout import (
    generate_rollouts_remote,
    remote_vllm_init_weight_transfer,
    sync_weights_to_vllm_inplace,
    prepare_batch,
    wait_for_rollout_worker,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from nanoopd.data.dataset import distributed_opd_loader, build_opd_dataset, DatasetType
import nanoopd.eval_aime as _eval_aime
import nanoopd.eval_livecodebench as _eval_lcb
import nanoopd.eval_sciknoweval as _eval_sciknow

_EVAL_FN = {
    "dapo_math": _eval_aime.run_eval,
    "livecodebench": _eval_lcb.run_eval,
    "sciknoweval": _eval_sciknow.run_eval,
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
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Partition ranks: 0..train_world_size-1 are student (FSDP) ranks;
    # train_world_size..ddp_world_size-1 are teacher ranks.
    train_world_size = args.train_world_size
    teacher_global_rank = train_world_size   # first (and typically only) teacher rank
    is_student = ddp_rank < train_world_size
    is_teacher = not is_student
    master_process = ddp_rank == 0           # student rank 0 drives logging/eval/save

    # Process groups
    student_group = dist.new_group(list(range(train_world_size)))
    all_group     = dist.new_group(list(range(ddp_world_size)))

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
            data_parallel_size=train_world_size,
            process_group=student_group,
        )
        student.prepare_scheduler(total_steps=args.num_steps * args.epochs)

    if is_teacher:
        teacher = TeacherModel(
            model_name=args.teacher_model,
            dtype="bfloat16",
        )

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
    wait_for_rollout_worker(args.rollout_worker_url)

    master_addr = os.environ.get("NCCL_MASTER_ADDR", "127.0.0.1")
    nccl_port = 29600
    transfer_world_size = train_world_size + args.rollout_worker_world_size
    if master_process:
        remote_vllm_init_weight_transfer(
            args.rollout_worker_url,
            master_address=master_addr,
            master_port=nccl_port,
            rank_offset=train_world_size,   # vLLM joins after student ranks
            world_size=transfer_world_size,
        )
        # Open the TCPStore server at nccl_port (rank 0) so the vLLM background
        # thread (rank train_world_size) can complete the NCCL rendezvous.
        model_update_group = NCCLWeightTransferEngine.trainer_init({
            "master_address": master_addr,
            "master_port": nccl_port,
            "world_size": transfer_world_size,
        })
    else:
        model_update_group = None
    dist.barrier(group=all_group)

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
                    start  = mb_idx * args.train_batch_size
                    idx    = perm[start : start + args.train_batch_size]
                    mb_ids   = input_ids[idx]          # [B, T]
                    mb_attn  = attention_mask[idx]
                    mb_mask  = response_mask[idx]
                    mb_inf_lp = inference_logprobs[idx] # [B, T]
                    shape_t = torch.tensor(
                        [mb_ids.shape[0], mb_ids.shape[1]], dtype=torch.long, device=device
                    )
                else:
                    shape_t = torch.zeros(2, dtype=torch.long, device=device)

                dist.broadcast(shape_t, src=0, group=all_group)
                B_mb, T_mb = int(shape_t[0].item()), int(shape_t[1].item())

                if is_teacher:
                    mb_ids  = torch.zeros(B_mb, T_mb, dtype=torch.long,  device=device)
                    mb_attn = torch.zeros(B_mb, T_mb, dtype=torch.long,  device=device)
                dist.broadcast(mb_ids,  src=0, group=all_group)
                dist.broadcast(mb_attn, src=0, group=all_group)

                # -- Student forward (with grad) --
                if is_student:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        student_logits = student.model(
                            input_ids=mb_ids, attention_mask=mb_attn
                        ).logits[:, :-1]               # [B, T-1, V]

                # -- Teacher: compute top-K log-probs and broadcast --
                # Broadcasts only [B, T-1, K] instead of the full [B, T-1, V] logit
                # tensor, reducing per-minibatch communication by ~vocab/K (>1000×).
                K = args.distill_top_k
                T_shifted = T_mb - 1
                s_chunk = args.student_chunk_size
                t_chunk = args.teacher_chunk_size

                if select_topk_by == "student":
                    # Student selects top-K indices; teacher gathers at those indices.
                    if is_student:
                        topk_idx = student_topk_indices(student_logits, K, s_chunk)
                    else:
                        topk_idx = torch.empty(B_mb, T_shifted, K, dtype=torch.long, device=device)
                    dist.broadcast(topk_idx, src=0, group=all_group)

                    if is_teacher:
                        t_logits = teacher.get_logits(mb_ids, mb_attn)[:, :-1]
                        t_logprobs = teacher_logprobs_at_indices(t_logits, topk_idx, t_chunk)
                        # Also compute teacher's own top-K for metrics
                        teacher_topk_idx, teacher_own_logprobs = teacher_topk_logprobs(t_logits, K, t_chunk)
                    else:
                        t_logprobs = torch.empty(B_mb, T_shifted, K, dtype=torch.bfloat16, device=device)
                        teacher_topk_idx = torch.empty(B_mb, T_shifted, K, dtype=torch.long, device=device)
                        teacher_own_logprobs = torch.empty(B_mb, T_shifted, K, dtype=torch.bfloat16, device=device)
                    dist.broadcast(t_logprobs, src=teacher_global_rank, group=all_group)
                    dist.broadcast(teacher_topk_idx, src=teacher_global_rank, group=all_group)
                    dist.broadcast(teacher_own_logprobs, src=teacher_global_rank, group=all_group)
                    student_topk_idx = topk_idx  # student selected; teacher_topk_idx already set
                    t_logprobs_at_student = t_logprobs  # teacher already evaluated at student top-K

                else:  # forward_kl: teacher selects top-K
                    if is_student:
                        student_topk_idx = student_topk_indices(student_logits, K, s_chunk)
                    else:
                        student_topk_idx = torch.empty(B_mb, T_shifted, K, dtype=torch.long, device=device)
                    dist.broadcast(student_topk_idx, src=0, group=all_group)

                    if is_teacher:
                        t_logits = teacher.get_logits(mb_ids, mb_attn)[:, :-1]
                        teacher_topk_idx, t_logprobs = teacher_topk_logprobs(t_logits, K, t_chunk)
                        teacher_own_logprobs = t_logprobs
                        # Teacher log-probs at student top-K for overlap-token advantage
                        t_logprobs_at_student = teacher_logprobs_at_indices(t_logits, student_topk_idx, t_chunk)
                    else:
                        teacher_topk_idx = torch.empty(B_mb, T_shifted, K, dtype=torch.long, device=device)
                        t_logprobs = torch.empty(B_mb, T_shifted, K, dtype=torch.bfloat16, device=device)
                        teacher_own_logprobs = torch.empty(B_mb, T_shifted, K, dtype=torch.bfloat16, device=device)
                        t_logprobs_at_student = torch.empty(B_mb, T_shifted, K, dtype=torch.bfloat16, device=device)
                    dist.broadcast(teacher_topk_idx, src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_logprobs, src=teacher_global_rank, group=all_group)
                    dist.broadcast(teacher_own_logprobs, src=teacher_global_rank, group=all_group)
                    dist.broadcast(t_logprobs_at_student, src=teacher_global_rank, group=all_group)
                    topk_idx = teacher_topk_idx

                # -- Student: compute TIS weights then loss and backward --
                if is_student:
                    s_logprobs = student_logprobs_at_indices(student_logits, topk_idx, s_chunk)
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
                        s_lp_for_metrics = student_logprobs_at_indices(student_logits, student_topk_idx, s_chunk)
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

            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                if master_process:
                    _EVAL_FN[args.dataset](
                        rollout_worker_url=args.rollout_worker_url,
                        tokenizer=student.tokenizer,
                        eval_k=args.eval_k,
                        eval_max_tokens=args.eval_max_tokens,
                        step=step + 1,
                    )

        # All ranks sync at the end of each step before the next n_mb broadcast.
        dist.barrier(group=all_group)

    compute_cleanup()
    if master_process and use_wandb:
        wandb.finish()
