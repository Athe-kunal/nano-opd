"""
FastAPI code execution server with a multiprocessing backend.

Each request submits (code, tests) and gets back structured results.
The actual execution runs in a ProcessPoolExecutor so the event loop
is never blocked and all requests are in-flight concurrently.

Start:
    python -m opd.envs.code_executor_server          # default port 8001
    python -m opd.envs.code_executor_server --port 9000 --workers 8

Then point livecodebench.py at it:
    export CODE_EXECUTOR_URL=http://localhost:8001
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from opd.envs.code_execution import execute_all

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TestsPayload(BaseModel):
    inputs: list[Any]
    outputs: list[Any]
    testtype: str = "stdio"        # "stdio" | "functional"
    fn_name: str = ""
    time_limit: float = 6.0


class ExecuteRequest(BaseModel):
    code: str
    tests: TestsPayload


class TestResult(BaseModel):
    status: str                    # "pass" | "wrong_answer" | "runtime_error" | "timeout"
    input: str | None = None
    expected: str | None = None
    actual: str | None = None
    stderr: str | None = None
    time_limit: float | None = None


class ExecuteResponse(BaseModel):
    results: list[TestResult]


# ---------------------------------------------------------------------------
# FastAPI app — `execute_all` (imported above) runs inside worker processes
# ---------------------------------------------------------------------------

app = FastAPI(title="Code Executor")
_pool: ProcessPoolExecutor | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _pool
    workers = int(os.environ.get("EXECUTOR_WORKERS", os.cpu_count() or 4))
    _pool = ProcessPoolExecutor(max_workers=workers)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _pool:
        _pool.shutdown(wait=False)


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    loop = asyncio.get_running_loop()
    tests_dict = req.tests.model_dump()
    results = await loop.run_in_executor(_pool, execute_all, req.code, tests_dict)
    return ExecuteResponse(results=[TestResult(**r) for r in results])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code execution server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--workers", type=int, default=None,
                        help="Process pool size (default: cpu_count)")
    args = parser.parse_args()

    if args.workers:
        os.environ["EXECUTOR_WORKERS"] = str(args.workers)

    uvicorn.run("opd.envs.code_executor_server:app", host=args.host, port=args.port)
