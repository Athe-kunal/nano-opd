"""Base class for FSDP-wrapped models: device mesh, offload, checkpointing."""

from typing import Any, ContextManager, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate import init_empty_weights
from omegaconf import DictConfig, OmegaConf
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, get_scheduler

from .data_parallelism import prepare_dp_model


class FSDPWorker:
    """Shared FSDP plumbing: device mesh setup, offload, and checkpointing.

    Subclasses (`StudentModel`, and any future FSDP-wrapped model) build
    `self.model` and call `_prepare_model_optimizer` to wrap it in FSDP.
    This class does not construct the underlying `nn.Module` itself.

    Attributes:
        config: The model's `DictConfig` (model name, dtype, sharding
          strategy, offload flags, etc).
        train: Whether this worker's model is being trained (builds an
          optimizer and scheduler) or only used for inference.
        tokenizer: Tokenizer matching `config.model_name`.
        model_device_mesh: 3D mesh (ddp, fsdp, tp) used to construct the
          model under FSDP.
        device_mesh: 2D mesh (dp, tp) used for loss scaling and future
          tensor-parallel work.
    """

    def __init__(self, config: DictConfig, train: bool):
        """Initializes device meshes and loads the tokenizer.

        Args:
            config: Model config with `model_name`, `ddp_size`, `tp_size`,
              and related fields.
            train: Whether to prepare this worker for training.

        Raises:
            AssertionError: If the distributed world size does not divide
              evenly into `ddp_size * tp_size`.
        """
        self.config = config
        self.train = train

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        world_size = dist.get_world_size()
        assert world_size % (config.ddp_size * config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {config.ddp_size} * tp_size {config.tp_size}."
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(
                config.ddp_size,
                world_size // (config.ddp_size * config.tp_size),
                config.tp_size
            )
        )
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(
                world_size // config.tp_size,
                config.tp_size
            )
        )

    def _init_weight_context(self, use_meta_tensor: bool = True) -> ContextManager:
        """Returns the context to construct the model under.

        Meta-device construction (`init_empty_weights()`) avoids allocating
        real memory for a model that will be re-sharded by FSDP anyway, but
        only one rank per shard group may skip real allocation — the others
        need real (CPU) tensors to broadcast from or to offload to.

        Args:
            use_meta_tensor: If False, always construct on CPU.

        Returns:
            `torch.device("cpu")` if this rank needs real tensors, otherwise
            the `init_empty_weights()` meta-device context manager.
        """
        # TODO: why offloading is incompatible with initialization on meta device?
        if any([
            dist.get_rank() == 0,
            self.device_mesh["tp"].size() > 1 and self.device_mesh["tp"].get_local_rank() == 0,
            getattr(self.config, "offload_model", False),
            not use_meta_tensor
        ]):
            return torch.device("cpu")
        return init_empty_weights()

    def _prepare_model_optimizer(self) -> None:
        """Wraps `self.model` in FSDP and builds the optimizer, if training.

        Raises:
            NotImplementedError: If `config.tp_size > 1` (tensor parallelism
              is not implemented in this codebase).
        """
        if self.train and self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            # Tensor parallelism is not implemented: the device mesh above
            # reserves a "tp" dimension, but no code shards layers across
            # it. Fail loudly here instead of only at `--tp-size 1` time.
            raise NotImplementedError(
                "tp_size > 1 requires tensor-parallel model sharding, which "
                "this codebase does not implement. Set --tp-size 1."
            )

        self.model = prepare_dp_model(
            self.model,
            self.config.dtype,
            self.config.tp_size == 1,
            self.model_device_mesh["ddp", "fsdp"]
        )

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_config
            )

        self._load_model_to_device("cpu")

    def prepare_scheduler(self, total_steps: int) -> None:
        """Builds `self.scheduler` from `config.scheduler` for `total_steps` rollouts.

        Args:
            total_steps: Number of rollout collection steps the training run
              will take; scaled by `config.update_per_rollout` to get the
              total optimizer-step count the LR schedule should span.
        """
        num_training_steps = total_steps * getattr(
            self.config, "update_per_rollout", 1
        )
        scheduler_config = OmegaConf.to_container(self.config.scheduler)
        scheduler_name = scheduler_config.pop("name")
        num_warmup_steps = int(
            scheduler_config.pop("warmup_ratio") * num_training_steps
        )
        self.scheduler = get_scheduler(
            scheduler_name,
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=scheduler_config
        )
    
    def _load_model_to_device(self, device: Union[torch.device, str]) -> None:
        """Moves FSDP's flat parameter shards to `device`, a no-op unless offloading is enabled.

        Args:
            device: Target device for the model's flat parameter shards.
        """
        if not getattr(self.config, "offload_model", False):
            return

        torch.cuda.empty_cache()
        _lazy_init(self.model, self.model)
        for handle in self.model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            handle.flat_param_to(device, non_blocking=True)
            flat_param._local_shard = flat_param.data
        torch.cuda.empty_cache()

    def _load_optimizer_to_device(self, device: Union[torch.device, str]) -> None:
        """Moves optimizer state tensors to `device`, a no-op unless offloading is enabled.

        Args:
            device: Target device for the optimizer's state tensors.
        """
        if not getattr(self.config, "offload_optimizer", False):
            return

        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(
                            device, non_blocking=True
                        )

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales `loss` by the data-parallel world size.

        FSDP averages gradients across the DP group on backward, but the
        loss here is already a per-rank mean; scaling by DP size before
        `.backward()` cancels that averaging so gradients reflect the sum
        (not mean) across ranks. See
        https://github.com/ChenmienTan/RL2/issues/11.
        """
        return self.device_mesh["dp"].size() * loss

    def _optimizer_step(self) -> float:
        """Clips gradients, steps the optimizer and scheduler, returns the grad norm.

        Returns:
            The pre-clipping global gradient L2 norm.
        """
        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )
        self._load_optimizer_to_device(
            torch.cuda.current_device()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._load_optimizer_to_device("cpu")
        self.scheduler.step()
        return grad_norm.item()

    def _get_model_state_dict(
        self, full_state_dict: bool = False
    ) -> dict[str, Any]:
        """Gathers the model's state dict, temporarily loading it onto the GPU if offloaded.

        Args:
            full_state_dict: If True, gather the full (unsharded) state
              dict on rank 0 for saving; if False, keep it sharded.

        Returns:
            The model's state dict.
        """
        options = StateDictOptions(
            full_state_dict=full_state_dict,
            cpu_offload=True
        )
        self._load_model_to_device(torch.cuda.current_device())
        state_dict = get_model_state_dict(self.model, options=options)
        self._load_model_to_device("cpu")
        return state_dict

    def _get_ckpt(self) -> dict[str, dict[str, Any]]:
        """Returns the optimizer and scheduler state dicts for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

    def load_ckpt(self, checkpoint_id: str) -> None:
        """Restores optimizer and scheduler state from a distributed checkpoint.

        Args:
            checkpoint_id: Path passed to `torch.distributed.checkpoint.load`.
        """
        ckpt = self._get_ckpt()
        dcp.load(ckpt, checkpoint_id=checkpoint_id)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

    def save_ckpt(self, save_dir: str) -> None:
        """Saves the model plus optimizer/scheduler state for resumable training.

        Args:
            save_dir: Directory to save into; the model goes to
              `{save_dir}/model` and optimizer/scheduler state to
              `{save_dir}/optimizer_scheduler`.
        """
        self.save_model(f"{save_dir}/model")
        dcp.save(
            self._get_ckpt(),
            checkpoint_id=f"{save_dir}/optimizer_scheduler"
        )

    def save_model(self, save_dir: str) -> None:
        """Saves the full (unsharded) model and tokenizer, rank 0 only.

        Args:
            save_dir: Directory to save the HuggingFace-format model and
              tokenizer into.
        """
        state_dict = self._get_model_state_dict(full_state_dict=True)
        if dist.get_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
            self.model.module.save_pretrained(
                save_dir, state_dict=state_dict
            )

        dist.barrier()