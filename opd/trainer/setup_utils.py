import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf

from opd.fsdp.model import StudentModel, TeacherModel
from opd.generator.rollout import remote_vllm_init_weight_transfer, wait_for_rollout_worker
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine


def print0(s="", **kwargs):
    """Log only from rank 0, so multi-process training doesn't spam N copies of every line."""
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        logger.info(s, **kwargs)


def compute_cleanup():
    """Companion to init_distributed: destroy the process group before exit, if one was created."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class DistributedContext:
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: torch.device
    train_world_size: int
    teacher_global_rank: int
    is_student: bool
    is_teacher: bool
    master_process: bool
    student_group: Any   # dist.ProcessGroup
    all_group: Any       # dist.ProcessGroup


def init_distributed(device_type_arg: str, train_world_size: int) -> DistributedContext:
    """Init torch.distributed and partition ranks into student and teacher sets.

    Ranks 0..train_world_size-1 are student (FSDP) ranks.
    Rank train_world_size is the teacher rank.
    """
    if device_type_arg == "":
        # Prefer CUDA if available, otherwise MPS, otherwise fall back to CPU.
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        print0(f"Autodetected device type: {device_type}")
    else:
        device_type = device_type_arg
    assert device_type in ("cuda", "mps", "cpu"), "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility. Note most of the code uses explicit rng objects; the
    # only place a global rng might matter is nn.Module weight init.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")  # tf32 instead of fp32 for matmuls

    is_ddp = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if is_ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1

    if is_ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    print0(f"Distributed world size: {ddp_world_size}")

    teacher_global_rank = train_world_size
    is_student = ddp_rank < train_world_size
    is_teacher = not is_student
    master_process = ddp_rank == 0

    student_group = dist.new_group(list(range(train_world_size)))
    all_group     = dist.new_group(list(range(ddp_world_size)))

    return DistributedContext(
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        train_world_size=train_world_size,
        teacher_global_rank=teacher_global_rank,
        is_student=is_student,
        is_teacher=is_teacher,
        master_process=master_process,
        student_group=student_group,
        all_group=all_group,
    )


def build_student(
    model_name: str,
    *,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    sharding_strategy: str,
    train_world_size: int,
    student_group: Any,
    total_steps: int,
    scheduler_name: str = "cosine",
    warmup_ratio: float = 0.05,
) -> StudentModel:
    config = OmegaConf.create({
        "model_name": model_name,
        "dtype": "bfloat16",
        "enable_gradient_checkpointing": gradient_checkpointing,
        "max_grad_norm": max_grad_norm,
        "attn_implementation": "flash_attention_2",
        "sharding_strategy": sharding_strategy,
        "optimizer": {"lr": lr, "weight_decay": weight_decay},
        "scheduler": {"name": scheduler_name, "warmup_ratio": warmup_ratio},
    })
    student = StudentModel(config, data_parallel_size=train_world_size, process_group=student_group)
    student.prepare_scheduler(total_steps=total_steps)
    return student


def build_teacher(model_name: str) -> TeacherModel:
    return TeacherModel(model_name=model_name, dtype="bfloat16")


def init_vllm_transfer(
    rollout_worker_url: str,
    rollout_worker_world_size: int,
    train_world_size: int,
    master_process: bool,
    all_group: Any,
    nccl_port: int = 29600,
):
    """Set up the NCCL weight-transfer channel between student ranks and the vLLM worker.

    Only the master process opens the TCPStore and triggers the remote rendezvous;
    all other ranks wait at the barrier. Returns the model_update_group (None on
    non-master ranks — only the master needs it for sync_weights_to_vllm_inplace).
    """
    wait_for_rollout_worker(rollout_worker_url)

    master_addr = os.environ.get("NCCL_MASTER_ADDR", "127.0.0.1")
    transfer_world_size = train_world_size + rollout_worker_world_size

    if master_process:
        remote_vllm_init_weight_transfer(
            rollout_worker_url,
            master_address=master_addr,
            master_port=nccl_port,
            rank_offset=train_world_size,
            world_size=transfer_world_size,
        )
        model_update_group = NCCLWeightTransferEngine.trainer_init({
            "master_address": master_addr,
            "master_port": nccl_port,
            "world_size": transfer_world_size,
        })
    else:
        model_update_group = None

    dist.barrier(group=all_group)
    return model_update_group
