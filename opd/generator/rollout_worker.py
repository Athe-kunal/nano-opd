"""
Standalone rollout worker for strict-synchronous RL training.

This process owns a single vLLM async engine on a dedicated GPU. The trainer
talks to it over localhost HTTP:

  - GET  /health
  - POST /generate              {prompts, num_samples, max_new_tokens, temperature, top_k}
  - POST /init_weight_transfer  {master_address, master_port, rank_offset, world_size}
  - POST /update_weights_start  {names, dtype_names, shapes, packed, ...}
  - POST /update_weights_finish {}

The trainer keeps semantics strict by:
  1. generating step-t rollouts from weights W_t
  2. updating the policy to W_{t+1}
  3. pushing W_{t+1} into this worker in-place via NCCL (start/finish)
  4. only then starting step t+1

The engine is vLLM's `AsyncLLM`, not the offline `LLM` class: its internal
scheduler batches whatever requests are concurrently in flight, so multiple
concurrent `/generate` calls (e.g. many single-problem eval requests) get
real continuous-batching throughput instead of needing to be serialized by
the caller (the offline `LLM.generate()` is a single blocking batch call,
not safe to invoke concurrently from multiple callers).
"""

import argparse
import asyncio
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from transformers import AutoTokenizer
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest, WeightTransferUpdateRequest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from opd.generator.rollout import generate_rollouts_async


class RolloutState:
    """Owns the async vLLM engine and in-place weight-update state for one worker process.

    Weight updates run as a background asyncio task so the HTTP handler that
    kicked them off can return immediately; `start_update_weights` and
    `finish_update_weights` together bracket one such update. Generation is
    paused (via `_generation_ready`) for the duration of an update, since the
    async engine gives no inherent guarantee that a concurrent `generate()`
    call is safe while weights are being swapped in-place.

    Attributes:
        tokenizer: Tokenizer for the served model.
        model_path: Path or HF id of the served model.
        engine: The vLLM `AsyncLLM` engine instance.
    """

    def __init__(self, engine: AsyncLLM, tokenizer: Any, model_path: str) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_path = model_path
        self._generation_ready = asyncio.Event()
        self._generation_ready.set()
        self._update_task: asyncio.Task | None = None

    async def wait_for_generation_slot(self) -> None:
        """Waits until no in-place weight update is in progress."""
        await self._generation_ready.wait()

    async def _run_update(self, request: WeightTransferUpdateRequest) -> None:
        """Applies a weight update. Runs as a background task; errors surface in `finish_update_weights`."""
        await self.engine.update_weights(request)

    def start_update_weights(self, payload: dict[str, Any]) -> None:
        """Begins an in-place weight update as a background task.

        Raises:
            RuntimeError: If a weight update is already in progress.
        """
        if self._update_task is not None and not self._update_task.done():
            raise RuntimeError("weight update already in progress")
        logger.info(f"Applying in-place vLLM update with {len(payload['names'])=}, {payload['packed']=}.")
        self._generation_ready.clear()
        self._update_task = asyncio.create_task(
            self._run_update(WeightTransferUpdateRequest(update_info=payload))
        )

    async def init_weight_transfer(self, payload: dict[str, Any]) -> None:
        """Joins the NCCL/IPC weight-transfer group in the background.

        Runs as a background task so the HTTP response returns immediately;
        the trainer must call its side of the rendezvous concurrently, or
        this would deadlock.
        """
        logger.info(
            "Initializing worker weight transfer with "
            f"{payload['master_address']=}, {payload['master_port']=}, "
            f"{payload['rank_offset']=}, {payload['world_size']=}."
        )
        asyncio.create_task(
            self.engine.init_weight_transfer_engine(WeightTransferInitRequest(init_info=payload))
        )

    async def finish_update_weights(self) -> None:
        """Awaits the in-progress weight update and resets the KV cache.

        Raises:
            RuntimeError: If no update was started, or if the update task
              raised an error while applying weights.
        """
        if self._update_task is None:
            raise RuntimeError("no weight update has been started")
        task, self._update_task = self._update_task, None
        try:
            await task
        except Exception as exc:
            self._generation_ready.set()
            raise RuntimeError(f"in-place weight update failed: {exc}") from exc
        await self.engine.reset_prefix_cache()
        self._generation_ready.set()
        logger.info("In-place vLLM update completed and prefix cache reset.")


app = FastAPI()


def _state(request: Request) -> RolloutState:
    return request.app.state.rollout


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    state = _state(request)
    return JSONResponse({"ok": True, "model_path": state.model_path})


@app.post("/generate")
async def generate(request: Request) -> JSONResponse:
    state = _state(request)
    payload = await request.json()
    await state.wait_for_generation_slot()
    rollouts = await generate_rollouts_async(
        state.engine,
        state.tokenizer,
        payload["prompts"],
        payload["num_samples"],
        payload["max_new_tokens"],
        payload["temperature"],
        payload["top_k"],
    )
    return JSONResponse({"ok": True, "rollouts": rollouts})


@app.post("/update_weights_start")
async def update_weights_start(request: Request) -> JSONResponse:
    state = _state(request)
    payload = await request.json()
    state.start_update_weights(payload)
    return JSONResponse({"ok": True, "status": "started"})


@app.post("/init_weight_transfer")
async def init_weight_transfer(request: Request) -> JSONResponse:
    state = _state(request)
    payload = await request.json()
    await state.init_weight_transfer(payload)
    return JSONResponse({"ok": True, "status": "initialized"})


@app.post("/update_weights_finish")
async def update_weights_finish(request: Request) -> JSONResponse:
    state = _state(request)
    await state.finish_update_weights()
    return JSONResponse({"ok": True, "status": "completed"})


async def _run(args: argparse.Namespace) -> None:
    """Builds the async engine and `RolloutState`, then serves the HTTP API until stopped."""
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if args.weight_transfer_backend:
        # Backend can be IPC (colocate trainer and inference) or nccl (different gpus).
        # Do NOT set load_format="dummy" — the trainer pushes weights at end-of-step,
        # so step 0's rollout would otherwise serve random weights, and the
        # reload-after-dummy escape hatch raises NotImplementedError because
        # vLLM's DummyModelLoader has no get_all_weights.
        engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend=args.weight_transfer_backend)

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
    app.state.rollout = RolloutState(engine, tokenizer, args.model)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    print(f"rollout worker listening on http://{args.host}:{args.port} model={args.model}", flush=True)
    await server.serve()


def main() -> None:
    """Parses CLI args and runs the rollout worker's asyncio event loop."""
    parser = argparse.ArgumentParser(description="opd rollout worker")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8047)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--weight-transfer-backend", type=str, default="nccl",
                        choices=["nccl", "ipc"], help="Backend for inplace weight transfer (nccl or ipc)")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
