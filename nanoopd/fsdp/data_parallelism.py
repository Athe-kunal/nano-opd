import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Keys match ShardingStrategy enum names; descriptions align with PyTorch FSDP docs.
DP_SHARDING_STRATEGIES: dict[str, ShardingStrategy] = {
    # Parameters, gradients, and optimizer states sharded; unshard parameters around forward/backward,
    # reduce-scatter gradients after backward; optimizer states updated locally per rank.
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    # Gradients and optimizer states sharded during computation; parameters sharded outside compute;
    # unshard before forward, no reshard after forward until after backward (see PyTorch for no_sync).
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    # Like DDP: replicate params/grads/optimizer states; all-reduce grads after backward.
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    # FULL_SHARD within a node, replicate parameters across nodes (cheaper cross-node comms).
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    # SHARD_GRAD_OP within a node, replicate across nodes; may save forward all-gathers vs HYBRID_SHARD.
    "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

def _param_init_fn(module: nn.Module):
    # FSDP already walks the module tree itself when applying wrapping policies. 
    # If your init function also recursed into children, you'd be double-initializing 
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

def prepare_dp_model(
    model: nn.Module,
    dtype: str,
    sync_module_states: bool,
    device_mesh: dist.DeviceMesh,
    sharding_strategy: str = "HYBRID_SHARD",
) -> FSDP:
    def _get_module_cls_from_name(name: str) -> type[nn.Module]:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__
    
    transformer_layer_cls = {
        _get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )
    dtype: torch.dtype = getattr(torch, dtype)
    mixed_precision = MixedPrecision(
        param_dtype = dtype,
        # gradient all reduce
        reduce_dtype=dtype,
        # batch norm, running stats
        buffer_dtype=dtype
    )
    try:
        strat = DP_SHARDING_STRATEGIES[sharding_strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sharding_strategy {sharding_strategy!r}; "
            f"expected one of {sorted(DP_SHARDING_STRATEGIES)}."
        ) from exc
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strat,
        mixed_precision=mixed_precision,
        param_init_fn=_param_init_fn,
        sync_module_states=sync_module_states,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device()
    )