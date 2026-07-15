"""
Rollout / batch utilities for RL training.

Generation runs in a separate vLLM worker process. The trainer calls the
`*_remote` helpers (HTTP to the worker) and pushes weights into the worker
in-place via NCCL using `sync_weights_to_vllm_inplace`. The worker itself
uses `generate_rollouts` from this module.
"""

import json
import time
from loguru import logger
from typing import Any
import urllib.error
import urllib.request
from vllm import SamplingParams
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine, NCCLTrainerSendWeightsArgs

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def generate_rollouts(
    vllm_engine: Any,
    tokenizer: Any,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Generates `num_samples` completions per prompt using vLLM.

    Args:
        vllm_engine: A vLLM `LLM`-like engine exposing `.generate`.
        tokenizer: Tokenizer used to determine the EOS stop string.
        prompts: Prompt strings to generate completions for.
        num_samples: Number of completions to sample per prompt.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature: Sampling temperature.
        top_k: Sampling top-k cutoff.

    Returns:
        A flat list of dicts, one per (prompt, sample) pair, ordered such
        that the N samples for prompt i occupy positions [i*N, (i+1)*N).
    """
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_new_tokens,
        logprobs=1,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )
    outputs = vllm_engine.generate(prompts, sampling_params)
    results: list[dict[str,Any]] = []
    for output in outputs:
        prompt_text = output.prompt
        for completion in output.outputs:
            inference_logprobs = [
                lp_dict[token_id].logprob
                for token_id, lp_dict in zip(completion.token_ids, completion.logprobs)
            ]
            results.append({
                "prompt": prompt_text,
                "response": completion.text,
                "prompt_ids": list(output.prompt_token_ids),
                "response_ids": list(completion.token_ids),
                "inference_logprobs": inference_logprobs,
            })
    return results



def _remote_json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600,
    op_name: str | None = None,
) -> Any:
    """Sends a JSON HTTP request to the rollout worker and returns the parsed response.

    Args:
        base_url: Base URL of the rollout worker.
        method: HTTP method, e.g. "GET" or "POST".
        path: URL path to request, appended to `base_url`.
        payload: JSON body to send, if any.
        timeout: Request timeout in seconds.
        op_name: If given, treat this as a "start/finish/init"-style
          operation that must return `{"ok": true, ...}` — raise
          `RuntimeError` naming `op_name` if the response is missing or
          reports failure. This collapses the repeated `if not resp or not
          resp.get("ok"): raise ...` check duplicated across callers.

    Returns:
        The parsed JSON response, or `None` if the body was empty.

    Raises:
        RuntimeError: If the HTTP request fails, or (when `op_name` is set)
          if the response is missing or does not report `ok: true`.
    """
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"remote rollout request failed: {e.code} {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"remote rollout request failed: {e}") from e

    parsed = json.loads(body) if body else None
    if op_name is not None and not (parsed and parsed.get("ok")):
        raise RuntimeError(f"rollout worker {op_name} failed: {parsed}")
    return parsed


def wait_for_rollout_worker(base_url: str, timeout_s: float = 300) -> dict[str, Any]:
    """Polls the rollout worker until it reports healthy.

    Args:
        base_url: Base URL of the rollout worker.
        timeout_s: How long to keep polling before giving up.

    Returns:
        The worker's health-check response payload.

    Raises:
        RuntimeError: If the worker does not report healthy within `timeout_s`.
    """
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            payload = _remote_json_request(base_url, "GET", "/health", timeout=10)
            if payload and payload.get("ok"):
                return payload
        except Exception as e:  # pragma: no cover - best-effort polling
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"rollout worker at {base_url} did not become healthy within {timeout_s}s"
        + (f"; last error: {last_err}" if last_err else "")
    )


def generate_rollouts_remote(
    base_url: str,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Generates rollouts via a separate rollout worker process.

    Args:
        base_url: Base URL of the rollout worker.
        prompts: Prompt strings to generate completions for.
        num_samples: Number of completions to sample per prompt.
        max_new_tokens: Maximum number of tokens to generate per completion.
        temperature: Sampling temperature.
        top_k: Sampling top-k cutoff.

    Returns:
        A flat list of rollout dicts, in the same order as `generate_rollouts`.
    """
    payload = {
        "prompts": prompts,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    resp = _remote_json_request(base_url, "POST", "/generate", payload=payload, timeout=1800)
    return resp["rollouts"]


def prepare_batch(
    rollouts: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_len: int,
    max_response_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Packs a list of rollouts into padded training tensors.

    Args:
        rollouts: Rollout dicts as produced by `generate_rollouts`.
        tokenizer: Tokenizer supplying the pad token id.
        max_prompt_len: Maximum allowed prompt length; raises if exceeded.
        max_response_len: Response length to truncate completions to.
        device: Device to place the returned tensors on.

    Returns:
        A dict with `input_ids`, `attention_mask`, `response_mask`, and
        `inference_logprobs` tensors, each of shape `[B, T]`.

    Raises:
        ValueError: If any rollout's prompt exceeds `max_prompt_len`.
    """
    input_ids_list: list[int] = []
    response_mask_list: list[int] = []
    inference_logprobs_list: list[float] = []
    for rollout in rollouts:
        prompt_ids = rollout["prompt_ids"]
        response_ids = rollout["response_ids"]
        if len(prompt_ids) > max_prompt_len:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) > max_prompt_len ({max_prompt_len}). "
                "Increase --max-prompt-len."
            )
        response_ids = response_ids[:max_response_len]
        full_ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        input_ids_list.append(full_ids)
        response_mask_list.append(mask)

        # Align inference logprobs with the full sequence: 0.0 for prompt positions,
        # then the per-response-token vLLM logprobs (truncated if the response was truncated).
        inf_lp = rollout.get("inference_logprobs", [])
        inf_lp_aligned = [0.0] * len(prompt_ids) + list(inf_lp)[: len(response_ids)]
        inference_logprobs_list.append(inf_lp_aligned)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m + [0] * (max_len - len(m)) for m in response_mask_list]
    attn_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]
    padded_inf_lp = [lp + [0.0] * (max_len - len(lp)) for lp in inference_logprobs_list]

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=device),
        "response_mask": torch.tensor(padded_masks, dtype=torch.float, device=device),
        "inference_logprobs": torch.tensor(padded_inf_lp, dtype=torch.float32, device=device),
    }

def _iter_fsdp_full_params(model: torch.nn.Module):
    """Yields `(name, param)` for each parameter, gathering FSDP shards first."""
    # summon_full_params temporarily gathers shards and exposes parameters
    # under their original (non-flattened) names instead of FSDP's _flat_param.
    with FSDP.summon_full_params(model, writeback=False, recurse=True):
        for name, param in model.named_parameters():
            yield name, param.detach().clone()


def _iter_model_parameters(model: torch.nn.Module, fsdp: bool):
    """Yields `(name, param)` for each parameter, unsharding first if `fsdp`."""
    if fsdp:
        yield from _iter_fsdp_full_params(model)
        return
    yield from model.named_parameters()


def collect_weight_metadata(model: torch.nn.Module, fsdp: bool = False) -> dict[str, Any]:
    """Collects parameter names, dtypes, and shapes for a vLLM weight-update request.

    Args:
        model: The model to collect metadata from.
        fsdp: Whether `model` is FSDP-wrapped (its shards are gathered first).

    Returns:
        A dict with parallel `names`, `dtype_names`, and `shapes` lists.
    """
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    # Consume the generator fully so summon_full_params context stays open
    # for the duration of metadata collection.
    for name, param in list(_iter_model_parameters(model, fsdp=fsdp)):
        names.append(name)
        dtype_names.append(str(param.dtype).split(".")[-1])
        shapes.append(list(param.shape))
    return {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
    }

def remote_vllm_start_update_weights(
    base_url: str, metadata: dict[str, Any], packed: bool
) -> dict[str, Any]:
    """Tells the rollout worker to begin receiving an in-place weight update."""
    payload = {
        "names": metadata["names"],
        "dtype_names": metadata["dtype_names"],
        "shapes": metadata["shapes"],
        "packed": packed,
        "is_checkpoint_format": True,
    }
    logger.info(f"Starting in-place vLLM weight update with {packed=}, {len(metadata['names'])=}")
    return _remote_json_request(
        base_url, "POST", "/update_weights_start", payload=payload, timeout=1800,
        op_name="update-start",
    )


def remote_vllm_finish_update_weights(base_url: str) -> dict[str, Any]:
    """Tells the rollout worker the in-place weight update finished sending."""
    logger.info("Waiting for vLLM weight update to complete.")
    return _remote_json_request(
        base_url, "POST", "/update_weights_finish", payload={}, timeout=1800,
        op_name="update-finish",
    )


def sync_weights_to_vllm_inplace(
    train_model: torch.nn.Module,
    base_url: str,
    model_update_group: Any,
    *,
    packed: bool = True,
    fsdp: bool = False,
) -> None:
    """Syncs trainer weights into the running vLLM worker without checkpoints.

    Args:
        train_model: The trainer's model (FSDP-wrapped if `fsdp` is True).
        base_url: Base URL of the rollout worker.
        model_update_group: NCCL process group used for the weight transfer.
        packed: Whether to pack parameters for transfer (see
          `NCCLWeightTransferEngine`).
        fsdp: Whether `train_model` is FSDP-wrapped.
    """

    # For FSDP, keep the FSDP wrapper so summon_full_params can unshard params
    # with their original names. For non-FSDP (e.g. DDP), unwrap .module.
    if not fsdp and hasattr(train_model, "module"):
        train_model = train_model.module

    metadata = collect_weight_metadata(train_model, fsdp=fsdp)
    remote_vllm_start_update_weights(base_url, metadata, packed=packed)

    param_iterator = _iter_model_parameters(train_model, fsdp=fsdp)
    logger.info(f"Sending trainer weights via NCCL with {packed=}, {fsdp=}.")
    NCCLWeightTransferEngine.trainer_send_weights(
        param_iterator,
        NCCLTrainerSendWeightsArgs(group=model_update_group, packed=packed),
    )

    remote_vllm_finish_update_weights(base_url)
    logger.info("Completed in-place vLLM weight update.")

def remote_vllm_init_weight_transfer(
    base_url: str,
    *,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
) -> dict[str, Any]:
    """Tells the rollout worker to join the NCCL weight-transfer group."""
    payload = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": rank_offset,
        "world_size": world_size,
    }
    logger.info(
        "Initializing vLLM weight transfer engine with "
        f"{master_address=}, {master_port=}, {rank_offset=}, {world_size=}."
    )
    return _remote_json_request(
        base_url, "POST", "/init_weight_transfer", payload=payload, timeout=1800,
        op_name="weight-transfer init",
    )


