from typing import Dict, List, Optional, cast

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import FSDPWoker
from .data_parallelism import prepare_dp_model


class TeacherModel:
    """
    Teacher model for computing reference log-probabilities.

    No tensor or sequence parallelism — each participant rank holds a full
    model replica and runs inference independently.  Non-participant ranks
    return ``None`` from all inference methods.

    Args:
        model_name: HuggingFace model identifier or local path.
        gpu_ids: Global ranks that will own the teacher weights.
        dtype: Parameter dtype, e.g. ``"bfloat16"``.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        dtype: str = "bfloat16",
    ):
        self._dtype = getattr(torch, dtype)
        self._is_participant = dist.get_rank() in set(gpu_ids)

        if not self._is_participant:
            self.model = None
            return

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to("cuda")

    @torch.no_grad()
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Returns per-token log-probabilities ``(B, T-1)`` aligned with
        ``input_ids[:, 1:]``.  Non-participant ranks return ``None``.
        """
        if not self._is_participant:
            return None

        assert self.model is not None
        self.model.eval()
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits                                         # (B, T, V)
        log_probs = logits[:, :-1].log_softmax(dim=-1)  # (B, T-1, V)
        return log_probs.gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1) * attention_mask[:, 1:]           # (B, T-1)

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Returns raw logits ``(B, T, V)``.  Non-participant ranks return ``None``.
        Used with ``compute_topk_logprobs_for_distillation`` for top-K KL distillation.
        """
        if not self._is_participant:
            return None

        assert self.model is not None
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
        data_parallel_size: Number of ranks across which FSDP shards parameters.
    """

    def __init__(
        self,
        config: DictConfig,
        data_parallel_size: int,
    ):
        assert dist.get_world_size() == data_parallel_size, (
            f"world_size ({dist.get_world_size()}) must equal "
            f"data_parallel_size ({data_parallel_size})."
        )

        self.config = config
        self.train = True
        self.data_parallel_size = data_parallel_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp",),
            mesh_shape=(data_parallel_size,),
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
        self.model = prepare_dp_model(
            self.model,
            self.config.dtype,
            sync_module_states=True,
            device_mesh=self.device_mesh,
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
