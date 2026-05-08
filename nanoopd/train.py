
import os
import math
import time
import argparse
import socket as _socket
from statistics import mean
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from nanoopd.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanoopd.loss import compute_reverse_kl_loss, compute_forward_kl_loss, ALGORITHMS
from nanoopd.rollout import (
    get_logprobs,
    generate_rollouts_remote,
    remote_vllm_init_weight_transfer,
    sync_weights_to_vllm_inplace,
    prepare_batch,
    wait_for_rollout_worker,
)

from nanoopd.data.dataset import distributed_opd_loader, build_opd_dataset

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="RL training for HF models")
    # Model
    parser.add_argument("--student-model", type=str, required=True, help="HF model path student model(e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--teacher-model", type=str, required=True, help="HF model path to teacher model(e.g. Qwen/Qwen3-0.6B)")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="grpo", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--kl-coeff", type=float, default=0.0, help="KL penalty coefficient")
    # Generation
    parser.add_argument("--num-samples", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of OPD steps")
    parser.add_argument("--prompts-per-step", type=int, default=8, help="Prompts per OPD step")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training micro-batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Optimizer steps per rollout batch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=20)
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy", help="wandb run name")
    parser.add_argument("--save-dir", type=str, default="opd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0, help="Save a checkpoint every N steps (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    print0(f"Loading model: {args.model}")
    print0(f"Algorithm: {args.algorithm}")
    print0(f"Device: {device}, World size: {ddp_world_size}")

    global_rollout_samples = args.prompts_per_step * args.num_samples

    assert global_rollout_samples % ddp_world_size == 0, (
        f"prompts_per_step * num_samples ({global_rollout_samples}) must be divisible by "
        f"world_size ({ddp_world_size})"
    )

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    student_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

