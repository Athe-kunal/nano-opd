"""FSDP data-parallel wrapping helpers: sharding strategies and model prep."""

import functools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
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


def _param_init_fn(module: nn.Module) -> None:
    """Materializes one meta-device submodule in place, for FSDP's param_init_fn hook.

    FSDP already walks the module tree itself when applying wrapping
    policies, so this must not recurse into children or they would be
    double-initialized.
    """
    module.to_empty(device=torch.cuda.current_device(), recurse=False)


def prepare_dp_model(
    model: nn.Module,
    dtype: str,
    sync_module_states: bool,
    device_mesh: dist.DeviceMesh = None,
    process_group=None,
    sharding_strategy: str = "HYBRID_SHARD",
) -> FSDP:
    """Wraps `model` in FSDP with the given mixed-precision and sharding config.

    Args:
        model: The module to wrap. Its `_no_split_modules` attribute names
          the transformer layer classes FSDP should treat as atomic
          wrapping units.
        dtype: Name of a `torch` dtype (e.g. "bfloat16") used for
          parameters, gradient reduction, and buffers.
        sync_module_states: If True, broadcast rank-0 parameter values to
          all ranks on construction (needed when other ranks were
          initialized on the meta device).
        device_mesh: Optional device mesh FSDP should shard over.
        process_group: Optional process group FSDP should shard over.
        sharding_strategy: Key into `DP_SHARDING_STRATEGIES`.

    Returns:
        The FSDP-wrapped model.

    Raises:
        ValueError: If `sharding_strategy` is not a known strategy name.
    """
    def _get_module_cls_from_name(name: str) -> type[nn.Module] | None:
        """Returns the first submodule class in `model` named `name`, if any."""
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__
        return None

    transformer_layer_cls = {
        _get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )
    torch_dtype: torch.dtype = getattr(torch, dtype)
    mixed_precision = MixedPrecision(
        param_dtype=torch_dtype,
        # gradient all reduce
        reduce_dtype=torch_dtype,
        # batch norm, running stats
        buffer_dtype=torch_dtype,
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
        process_group=process_group,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )