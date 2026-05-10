from typing import Dict, cast

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import FSDPWoker
from .data_parallelism import prepare_dp_model


class TeacherModel:
    """
    Teacher model running on its own dedicated GPU process.

    Loaded on the current CUDA device; the process that instantiates this
    class must be the sole owner of the GPU (i.e. a dedicated teacher rank
    in the torchrun world, separate from all FSDP student ranks).

    Args:
        model_name: HuggingFace model identifier or local path.
        dtype: Parameter dtype, e.g. ``"bfloat16"``.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
    ):
        self._dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to("cuda")

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw logits ``(B, T, V)`` for top-K KL distillation."""
        self.model.eval()
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


class StudentModel(FSDPWoker):
    """
    Student model for on-policy distillation training using FSDP only.

    Args:
        config: Training configuration (must contain ``model_name``, ``dtype``,
            ``enable_gradient_checkpointing``, and ``optimizer`` keys).
        data_parallel_size: Number of student ranks across which FSDP shards.
        process_group: The process group that spans exactly the student ranks.
            FSDP and all student-side collectives use this group, so teacher
            ranks (which live at higher global ranks in the same torchrun world)
            are never involved in student-only barriers or all-reduces.
    """

    def __init__(
        self,
        config: DictConfig,
        data_parallel_size: int,
        process_group=None,
    ):
        assert dist.get_world_size(group=process_group) == data_parallel_size, (
            f"process_group size ({dist.get_world_size(group=process_group)}) must equal "
            f"data_parallel_size ({data_parallel_size})."
        )

        self.config = config
        self.train = True
        self.data_parallel_size = data_parallel_size
        self._process_group = process_group

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        with self._init_weight_context():
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=getattr(torch, config.dtype),
                trust_remote_code=True,
                attn_implementation=getattr(
                    config, "attn_implementation", "flash_attention_2"
                ),
            )

        self._prepare_model_optimizer()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _init_weight_context(self, use_meta_tensor: bool = True):
        # Global rank 0 is always a student rank; load weights there on CPU.
        if any([
            dist.get_rank() == 0,
            getattr(self.config, "offload_model", False),
            not use_meta_tensor,
        ]):
            return torch.device("cpu")
        return init_empty_weights()

    def _prepare_model_optimizer(self):
        if self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # sync_module_states=True broadcasts rank-0 weights to all FSDP ranks.
        # Pass process_group (not device_mesh) so FSDP stays within student ranks.
        self.model = prepare_dp_model(
            self.model,
            self.config.dtype,
            sync_module_states=True,
            process_group=self._process_group,
            sharding_strategy=getattr(self.config, "sharding_strategy", "FULL_SHARD"),
        )

        optimizer_config = cast(
            Dict, OmegaConf.to_container(self.config.optimizer, resolve=True)
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), **optimizer_config
        )
        self._load_model_to_device("cpu")

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.data_parallel_size * loss

    def save_model(self, save_dir: str):
        # Override to barrier within student_group only; teacher ranks must not
        # participate in this barrier (they don't call save_model).
        state_dict = self._get_model_state_dict(full_state_dict=True)
        if dist.get_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
            self.model.module.save_pretrained(save_dir, state_dict=state_dict)
        dist.barrier(group=self._process_group)
