import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from opd.common import compute_init, print0, autodetect_device_type
from opd.fsdp.model import StudentModel, TeacherModel
from opd.generator.rollout import remote_vllm_init_weight_transfer, wait_for_rollout_worker
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine


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
    device_type = autodetect_device_type() if device_type_arg == "" else device_type_arg
    _, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

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
