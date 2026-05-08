from typing import ContextManager, Dict, Union, List, Optional, Any
from omegaconf import OmegaConf, DictConfig
from accelerate import init_empty_weights
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import _lazy_init
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict
)
from transformers import get_scheduler, AutoTokenizer
from .data_parallelism import prepare_dp_model
from .tensor_parallelism import prepare_tp_model

class FSDPWoker:
    def __init__(self, config: DictConfig, train: bool):

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
    def _init_weight_context(self, use_meta_tensor: bool = True) -> ContextManager:
        # TODO: why offloading is incompatible with initialization on meta device?
        if any([
            dist.get_rank() == 0,
            self.device_mesh["tp"].size() > 1 and self.device_mesh["tp"].get_local_rank() == 0,
            getattr(self.config, "offload_model", False),
            not use_meta_tensor
        ]):
            return torch.device("cpu")
        return init_empty_weights()
    
    def _prepare_model_optimizer(self):

        if self.train and self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

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
    
    def prepare_scheduler(self, total_steps: int):

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
    
    