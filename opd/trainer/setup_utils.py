import math
import os
from typing import Any, Literal

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import OmegaConf

from opd.fsdp.model import StudentModel, TeacherModel
from opd.generator.rollout import remote_vllm_init_weight_transfer, wait_for_rollout_worker
from opd.trainer.models import DistributedContext
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


def init_distributed(device_type_arg: str, train_world_size: int) -> DistributedContext:
    """Init torch.distributed and partition ranks into student and teacher sets.

    Ranks 0..train_world_size-1 are student (FSDP) ranks.
    Rank train_world_size is the teacher rank.
    """
    if device_type_arg == "":
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
        # vLLM initializes dummy weights
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


def assert_prompts_divisible(prompts_per_step: int, train_world_size: int) -> None:
    """Raises if `prompts_per_step` doesn't divide evenly across student ranks.

    Args:
        prompts_per_step: Number of distinct prompts sampled per step.
        train_world_size: Number of student (FSDP) ranks.

    Raises:
        AssertionError: If `prompts_per_step % train_world_size != 0`.
    """
    assert prompts_per_step % train_world_size == 0, (
        f"prompts_per_step ({prompts_per_step}) must be divisible by "
        f"train_world_size ({train_world_size})"
    )


def topk_selector_for(algorithm: str) -> Literal["student", "teacher"]:
    """Returns which policy should select the top-K vocab indices for `algorithm`.

    `reverse_kl`/`jsd`/`mopd_pg_loss` weight the divergence by the student
    distribution, so only tokens with student mass matter — the student
    selects. `forward_kl`/`mopd_loss` weight by the teacher distribution, so
    the teacher selects instead.

    Args:
        algorithm: Key into `opd.loss.ALGORITHMS`.

    Returns:
        `"student"` or `"teacher"`.
    """
    return "teacher" if algorithm in ("forward_kl", "mopd_loss") else "student"


def broadcast_n_minibatches(
    is_student: bool,
    num_sequences: int,
    train_batch_size: int,
    device: torch.device,
    all_group: Any,
) -> tuple[int, torch.Tensor | None]:
    """Computes this epoch's minibatch count and shuffle permutation, then broadcasts the count.

    The teacher rank has no rollout batch of its own but must still know how
    many minibatch iterations to participate in, so the student-computed
    count is broadcast to it.

    Args:
        is_student: Whether this rank is a student (FSDP) rank.
        num_sequences: Number of sequences in the student's rollout batch
          (ignored on non-student ranks).
        train_batch_size: Sequences per minibatch.
        device: Device to allocate the broadcast tensor on.
        all_group: Process group spanning both student and teacher ranks.

    Returns:
        `(n_mb, perm)`: the number of minibatches this epoch, and a random
        permutation over `[0, num_sequences)` used to slice minibatches.
        `perm` is `None` on non-student ranks, which have no batch to slice.
    """
    if is_student:
        n_mb = math.ceil(num_sequences / train_batch_size)
        perm = torch.randperm(num_sequences, device=device)
        n_mb_t = torch.tensor([n_mb], dtype=torch.long, device=device)
    else:
        perm = None
        n_mb_t = torch.zeros(1, dtype=torch.long, device=device)
    dist.broadcast(n_mb_t, src=0, group=all_group)
    return int(n_mb_t.item()), perm


def maybe_save_checkpoint(
    student: StudentModel, save_dir: str, save_every: int, step: int
) -> None:
    """Saves a checkpoint every `save_every` steps, if checkpointing is enabled.

    Args:
        student: The student model to save.
        save_dir: Directory checkpoints are written under
          (`{save_dir}/step_{step+1}`).
        save_every: Save every this many steps; 0 disables checkpointing.
        step: Zero-indexed training step.
    """
    if save_every > 0 and (step + 1) % save_every == 0:
        save_path = f"{save_dir}/step_{step + 1}"
        student.save_model(save_path)
        print0(f"Saved checkpoint to {save_path}")
