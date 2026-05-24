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
import json
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

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
# Execution logic (runs inside worker processes)
# ---------------------------------------------------------------------------

def _run_one_stdio(code: str, inp: Any, expected: Any, time_limit: float) -> dict:
    stdin_text = inp if isinstance(inp, str) else "\n".join(str(x) for x in inp)
    expected_text = expected if isinstance(expected, str) else str(expected)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_text,
            capture_output=True, text=True, timeout=time_limit,
        )
        if proc.returncode != 0:
            return {"status": "runtime_error", "input": stdin_text, "stderr": proc.stderr.strip()}
        got = proc.stdout.strip()
        if got == expected_text.strip():
            return {"status": "pass"}
        return {"status": "wrong_answer", "input": stdin_text,
                "expected": expected_text, "actual": got}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "input": stdin_text, "time_limit": time_limit}
    except Exception as e:
        return {"status": "runtime_error", "input": stdin_text, "stderr": str(e)}


def _run_one_functional(code: str, fn_name: str, inp: Any, expected: Any, time_limit: float) -> dict:
    driver = textwrap.dedent(f"""
import json, sys
{code}

_inp = json.loads(sys.stdin.read())
_result = {fn_name}(*_inp)
print(json.dumps(_result))
""").lstrip()
    inp_str = json.dumps(inp)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            input=inp_str,
            capture_output=True, text=True, timeout=time_limit,
        )
        if proc.returncode != 0:
            return {"status": "runtime_error", "input": inp_str, "stderr": proc.stderr.strip()}
        got = json.loads(proc.stdout.strip())
        if got == expected:
            return {"status": "pass"}
        return {"status": "wrong_answer", "input": inp_str,
                "expected": json.dumps(expected), "actual": json.dumps(got)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "input": inp_str, "time_limit": time_limit}
    except (json.JSONDecodeError, Exception) as e:
        return {"status": "runtime_error", "input": inp_str, "stderr": str(e)}


def _execute_all(code: str, tests: dict) -> list[dict]:
    """Runs in a worker process. Executes all test cases sequentially."""
    inputs     = tests["inputs"]
    outputs    = tests["outputs"]
    testtype   = tests.get("testtype", "stdio")
    fn_name    = tests.get("fn_name", "")
    time_limit = tests.get("time_limit", 6.0)

    results = []
    for inp, expected in zip(inputs, outputs):
        if testtype == "functional" and fn_name:
            results.append(_run_one_functional(code, fn_name, inp, expected, time_limit))
        else:
            results.append(_run_one_stdio(code, inp, expected, time_limit))
    return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Code Executor")
_pool: ProcessPoolExecutor | None = None


@app.on_event("startup")
async def _startup():
    global _pool
    import os
    workers = int(os.environ.get("EXECUTOR_WORKERS", os.cpu_count() or 4))
    _pool = ProcessPoolExecutor(max_workers=workers)


@app.on_event("shutdown")
async def _shutdown():
    if _pool:
        _pool.shutdown(wait=False)


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    loop = asyncio.get_running_loop()
    tests_dict = req.tests.model_dump()
    results = await loop.run_in_executor(_pool, _execute_all, req.code, tests_dict)
    return ExecuteResponse(results=[TestResult(**r) for r in results])


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Code execution server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--workers", type=int, default=None,
                        help="Process pool size (default: cpu_count)")
    args = parser.parse_args()

    if args.workers:
        os.environ["EXECUTOR_WORKERS"] = str(args.workers)

    uvicorn.run("opd.envs.code_executor_server:app", host=args.host, port=args.port)
