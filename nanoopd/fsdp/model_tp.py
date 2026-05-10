from typing import Dict, List, Optional, cast

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from accelerate import init_empty_weights
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import FSDPWoker
from .data_parallelism import prepare_dp_model
from .tensor_parallelism import prepare_tp_model


class TeacherModel:
    """
    Teacher model for computing reference log-probabilities via TP+SP.

    Occupies a designated set of GPU ranks (``gpu_ids``).  All other ranks
    participate in the collective DeviceMesh initialisation but do not load
    the model and receive ``None`` from :py:meth:`get_logprobs`.

    Args:
        model_name: HuggingFace model identifier or local path.
        gpu_ids: Global ranks that will own the teacher weights.
            ``len(gpu_ids)`` must equal ``tensor_parallel_size``.
        tensor_parallel_size: Number of GPUs per TP group.
        dtype: Parameter dtype, e.g. ``"bfloat16"``.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        tensor_parallel_size: int,
        dtype: str = "bfloat16",
    ):
        assert len(gpu_ids) == tensor_parallel_size, (
            f"len(gpu_ids) ({len(gpu_ids)}) must equal "
            f"tensor_parallel_size ({tensor_parallel_size})."
        )
        self._dtype = getattr(torch, dtype)
        self._is_participant = dist.get_rank() in set(gpu_ids)

        # DeviceMesh.__init__ calls dist.new_group(), which is collective —
        # every rank in the world must call this, even non-participants.
        self.tp_mesh = DeviceMesh(
            "cuda",
            torch.tensor(gpu_ids),
            mesh_dim_names=("tp",),
        )

        if not self._is_participant:
            self.model = None
            return

        # TP rank 0 loads weights from disk; other TP ranks use meta tensors
        # and receive the parameters via broadcast inside prepare_tp_model.
        is_tp_rank_0 = self.tp_mesh.get_local_rank() == 0
        ctx = torch.device("cpu") if is_tp_rank_0 else init_empty_weights()
        with ctx:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self._dtype, trust_remote_code=True
            )

        if tensor_parallel_size > 1:
            prepare_tp_model(self.model, self.tp_mesh)
        else:
            self.model = self.model.to("cuda")  # type: ignore

    @torch.no_grad()
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Returns per-token log-probabilities of shape ``(B, T-1)`` aligned with
        ``input_ids[:, 1:]``.  Non-participant ranks return ``None``.
        """
        if not self._is_participant:
            return None

        assert self.model is not None
        self.model.eval()  # type: ignore[union-attr]
        logits = self.model(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits                                        # (B, T, V)
        log_probs = logits[:, :-1].log_softmax(dim=-1)  # (B, T-1, V)
        return log_probs.gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1) * attention_mask[:, 1:]           # (B, T-1)


class StudentModel(FSDPWoker):
    """
    Student model for on-policy distillation training using FSDP × (TP + SP).

    Device mesh layout (``data_parallel_size × tensor_parallel_size`` GPUs):

    .. code-block:: text

        ┌─────────────────────────────┐
        │  dp axis  (FSDP sharding)   │  ← data_parallel_size ranks
        │  tp axis  (TP + SP)         │  ← tensor_parallel_size ranks
        └─────────────────────────────┘

    Args:
        config: Training configuration (must contain ``model_name``, ``dtype``,
            ``enable_gradient_checkpointing``, and ``optimizer`` keys).
        data_parallel_size: Number of ranks across which FSDP shards parameters.
        tensor_parallel_size: Number of ranks per TP+SP group.
    """

    def __init__(
        self,
        config: DictConfig,
        data_parallel_size: int,
        tensor_parallel_size: int,
    ):
        world_size = dist.get_world_size()
        assert world_size == data_parallel_size * tensor_parallel_size, (
            f"world_size ({world_size}) must equal "
            f"data_parallel_size ({data_parallel_size}) × "
            f"tensor_parallel_size ({tensor_parallel_size})."
        )

        # Attributes expected by FSDPWoker helpers.
        self.config = config
        self.train = True
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        # 2-D mesh: dp (FSDP) × tp (tensor + sequence parallel).
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(data_parallel_size, tensor_parallel_size),
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
        """
        Returns a context manager for model initialisation.

        - Rank 0, or TP rank 0 when TP > 1: load on CPU (has real weights).
        - All other ranks: initialise on meta device (weights broadcast later).
        - Offload-model mode: always CPU so offloading can proceed immediately.
        """
        if any([
            dist.get_rank() == 0,
            self.tensor_parallel_size > 1
                and self.device_mesh["tp"].get_local_rank() == 0,
            getattr(self.config, "offload_model", False),
            not use_meta_tensor,
        ]):
            return torch.device("cpu")
        return init_empty_weights()

    def _prepare_model_optimizer(self):
        if self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.tensor_parallel_size > 1:
            prepare_tp_model(self.model, self.device_mesh["tp"])

        # sync_module_states broadcasts weights from rank 0 to all FSDP ranks.
        # When TP > 1 the weights are already distributed by prepare_tp_model,
        # so no additional broadcast is needed.
        self.model = prepare_dp_model(
            self.model,
            self.config.dtype,
            sync_module_states=(self.tensor_parallel_size == 1),
            device_mesh=self.device_mesh["dp"],
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
