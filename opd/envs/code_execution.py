"""Subprocess-based Python test execution engine, shared by two call sites.

`code_executor_server.py` dispatches each request to this engine inside a
worker process (via a `ProcessPoolExecutor`), and `livecodebench.py` falls
back to calling it in-process when `CODE_EXECUTOR_URL` is not set. Keeping
one copy means a correctness or sandboxing fix here applies to both.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any


def _run_subprocess(program: str, stdin_text: str, time_limit: float) -> tuple[str, str]:
    """Runs `program` as a subprocess, feeding `stdin_text` on stdin.

    Shared by `run_one_stdio` and `run_one_functional` — both spawn a
    `python -c <program>` subprocess with the same timeout/error handling;
    only what the program is and how its output is compared differ.

    Args:
        program: Python source to run via `python -c`.
        stdin_text: Text piped to the subprocess's stdin.
        time_limit: Seconds to allow before killing the subprocess.

    Returns:
        `("ok", stdout)` on a clean exit, `("timeout", str(time_limit))` if
        it ran past `time_limit`, or `("runtime_error", stderr_or_message)`
        on a nonzero exit or any other exception.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            input=stdin_text,
            capture_output=True, text=True, timeout=time_limit,
        )
        if proc.returncode != 0:
            return "runtime_error", proc.stderr.strip()
        return "ok", proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return "timeout", str(time_limit)
    except Exception as e:
        return "runtime_error", str(e)


def run_one_stdio(
    code: str, inp: Any, expected: Any, time_limit: float
) -> dict[str, Any]:
    """Runs `code` as a subprocess, feeding `inp` on stdin.

    Args:
        code: Python source to execute.
        inp: Test input; joined with newlines onto stdin if not already a string.
        expected: Expected stdout, compared after stripping whitespace.
        time_limit: Seconds to allow before killing the subprocess.

    Returns:
        A result dict with `status` one of "pass", "wrong_answer",
        "runtime_error", or "timeout", plus context fields for failures.
    """
    stdin_text = inp if isinstance(inp, str) else "\n".join(str(x) for x in inp)
    expected_text = expected if isinstance(expected, str) else str(expected)
    status, result = _run_subprocess(code, stdin_text, time_limit)
    if status == "runtime_error":
        return {"status": "runtime_error", "input": stdin_text, "stderr": result}
    if status == "timeout":
        return {"status": "timeout", "input": stdin_text, "time_limit": time_limit}
    if result == expected_text.strip():
        return {"status": "pass"}
    return {"status": "wrong_answer", "input": stdin_text,
            "expected": expected_text, "actual": result}


def run_one_functional(
    code: str, fn_name: str, inp: Any, expected: Any, time_limit: float
) -> dict[str, Any]:
    """Runs `code` in a subprocess that calls `fn_name(*inp)` and compares the result.

    Args:
        code: Python source defining `fn_name`.
        fn_name: Name of the function to call with the unpacked `inp` args.
        inp: Arguments to pass to `fn_name`, JSON-serializable.
        expected: Expected return value, compared by JSON round-trip equality.
        time_limit: Seconds to allow before killing the subprocess.

    Returns:
        A result dict with `status` one of "pass", "wrong_answer",
        "runtime_error", or "timeout", plus context fields for failures.
    """
    driver = textwrap.dedent(f"""
        import json, sys
        {code}

        _inp = json.loads(sys.stdin.read())
        _result = {fn_name}(*_inp)
        print(json.dumps(_result))
    """).lstrip()
    inp_str = json.dumps(inp)
    status, result = _run_subprocess(driver, inp_str, time_limit)
    if status == "runtime_error":
        return {"status": "runtime_error", "input": inp_str, "stderr": result}
    if status == "timeout":
        return {"status": "timeout", "input": inp_str, "time_limit": time_limit}
    got = json.loads(result)
    if got == expected:
        return {"status": "pass"}
    return {"status": "wrong_answer", "input": inp_str,
            "expected": json.dumps(expected), "actual": json.dumps(got)}


def execute_all(code: str, tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Runs every test case in `tests` against `code` sequentially.

    Args:
        code: Python source to test.
        tests: Dict with `inputs`, `outputs`, and optionally `testtype`
          ("stdio" or "functional"), `fn_name`, and `time_limit`.

    Returns:
        One result dict per test case, in the same order as `inputs`.
    """
    inputs     = tests["inputs"]
    outputs    = tests["outputs"]
    testtype   = tests.get("testtype", "stdio")
    fn_name    = tests.get("fn_name", "")
    time_limit = tests.get("time_limit", 6.0)

    results = []
    for inp, expected in zip(inputs, outputs):
        if testtype == "functional" and fn_name:
            results.append(run_one_functional(code, fn_name, inp, expected, time_limit))
        else:
            results.append(run_one_stdio(code, inp, expected, time_limit))
    return results
