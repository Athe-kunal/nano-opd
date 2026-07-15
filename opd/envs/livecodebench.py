from __future__ import annotations

import base64
import json
import os
import pickle
import re
import zlib
from datetime import datetime
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, run_pass_at_k_eval
from opd.envs.code_execution import execute_all

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ```.

{problem}"""


def _parse_signature(starter_code: str) -> str:
    after_def = starter_code.split("def ")[1]
    return "def " + (after_def.split("Input\n")[0] if "Input\n" in after_def else after_def).strip()


def _translate_private_test_cases(encoded_data, fn_name: str) -> str:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    tests = json.loads(original_data)
    return json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "testtype": tests[0]["testtype"],
        "fn_name": fn_name,
        "time_limit": TIME_LIMIT,
    }, ensure_ascii=False)


def load_livecodebench(dataset_split: str, until: datetime | None = None) -> Dataset:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        revision="refs/pr/6"
    )

    if dataset_split == "train":
        ds = ds.filter(lambda ex: ex["contest_date"] < LCB_TRAIN_CUTOFF)
    else:
        ds = ds.filter(lambda ex: ex["contest_date"] >= LCB_TEST_CUTOFF)

    if until is not None:
        ds = ds.filter(lambda ex: ex["contest_date"] < until)

    def format_prompt(ex):
        problem = ex["question_content"]
        if ex["starter_code"].strip() != "":
            problem += f"\n\nYour solution should have the following signature: ```python\n{_parse_signature(ex['starter_code'])}\n```"

        fn_name = ""
        if ex["metadata"].strip() != "":
            metadata = json.loads(ex["metadata"])
            fn_name = metadata.get("func_name", "")

        return {
            "kind": "code",
            "dataset": "livecodebench",
            "description": problem,
            "prompt": CODE_PROMPT.format(problem=problem),
            "tests": _translate_private_test_cases(ex["private_test_cases"], fn_name=fn_name),
        }

    processed_shards = []
    for i in range(4):
        shard = ds.shard(num_shards=4, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)

    return concatenate_datasets(processed_shards)


def _extract_code(response: str) -> str | None:
    """Return the last Python code block from a model response, or None."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _execute_tests(code: str, tests: dict) -> list[dict]:
    """Dispatches to the FastAPI executor server (CODE_EXECUTOR_URL), if set.

    Otherwise falls back to running locally via
    `opd.envs.code_execution.execute_all`.
    """
    if os.environ.get("CODE_EXECUTOR_URL"):
        # httpx is only needed for this optional remote-executor path, so it
        # isn't a hard top-level dependency of this module.
        import httpx
        url = os.environ["CODE_EXECUTOR_URL"].rstrip("/") + "/execute"
        resp = httpx.post(url, json={"code": code, "tests": tests}, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["results"]

    return execute_all(code, tests)


def _all_tests_pass(code: str, tests: dict) -> bool:
    return all(r["status"] == "pass" for r in _execute_tests(code, tests))


def _cap(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + "..."


def _cap_lines(s: str, max_chars: int, max_lines: int) -> str:
    lines = s.splitlines()
    truncated = lines[:max_lines]
    joined = "\n".join(_cap(ln, max_chars) for ln in truncated)
    if len(lines) > max_lines:
        joined += f"\n... ({len(lines) - max_lines} more lines)"
    return joined


def format_test_feedback(results: list[dict], n_total: int, max_tests_to_show: int = 2, max_length: int = 2000) -> str:
    """
    Produce compact, model-readable feedback from structured test results.

    Priority rules:
    - Errors and timeouts are shown first; wrong-answer cases are dropped when
      any error/timeout exists (a crash is more actionable than a wrong answer).
    - Among wrong-answer cases, shorter inputs are shown first.
    - Each field is character-capped; total output is hard-capped at max_length.
    """
    n_pass = sum(1 for r in results if r["status"] == "pass")
    failures = [r for r in results if r["status"] != "pass"]

    if not failures:
        return f"All {n_total} tests passed ✅"

    header = f"{n_pass}/{n_total} tests passed."

    priority = [r for r in failures if r["status"] in ("runtime_error", "timeout")]
    candidates = priority if priority else sorted(
        [r for r in failures if r["status"] == "wrong_answer"],
        key=lambda r: len(r.get("input", "")) + len(r.get("actual", "")),
    )
    shown = candidates[:max_tests_to_show]

    parts = [header]
    for i, r in enumerate(shown, 1):
        idx = results.index(r) + 1
        if r["status"] == "timeout":
            inp_fmt = _cap_lines(r.get("input", ""), max_chars=250, max_lines=8)
            parts.append(f"\nTest {idx}: ❌ Time limit exceeded ({r.get('time_limit', TIME_LIMIT)}s)\nInput:\n{inp_fmt}")
        elif r["status"] == "runtime_error":
            inp_fmt = _cap_lines(r.get("input", ""), max_chars=250, max_lines=8)
            stderr_fmt = _cap_lines(r.get("stderr", ""), max_chars=300, max_lines=10)
            parts.append(f"\nTest {idx}: ❌ Runtime error\nInput:\n{inp_fmt}\nStderr:\n{stderr_fmt}")
        else:  # wrong_answer
            inp_fmt = _cap_lines(r.get("input", ""), max_chars=250, max_lines=8)
            exp_fmt = _cap(r.get("expected", ""), 250)
            act_fmt = _cap(r.get("actual", ""), 250)
            parts.append(
                f"\nTest {idx}: ❌ Wrong answer\nInput:\n{inp_fmt}\nExpected: {exp_fmt}\nActual:   {act_fmt}"
            )

    if len(candidates) > max_tests_to_show:
        parts.append(f"\n... and {len(candidates) - max_tests_to_show} more failing test(s) not shown.")

    full = "\n".join(parts)
    return full if len(full) <= max_length else full[:max_length] + "..."


class LiveCodeBenchEnv(OPDEnvBase):
    """
    skyrl_gym environment for LiveCodeBench problems.

    Each instance wraps a single (prompt, tests) pair. The reward is 1.0 if
    all test cases pass, else 0.0. get_privileged_information returns per-test execution
    results for SDPO self-distillation. evaluate runs the LCB test split.
    """

    def __init__(self, prompt: str, tests: dict[str, Any]) -> None:
        super().__init__(kind="code", dataset="livecodebench")
        self.prompt = prompt
        self.tests = tests

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return [{"role": "user", "content": self.prompt}], {}

    def _run_tests(self, action: str) -> list[dict] | None:
        """Run all test cases for this action. Returns structured result dicts, or None if no code found."""
        code = _extract_code(action)
        if code is None:
            return None
        return _execute_tests(code, self.tests)

    def compute_reward(self, action: str) -> tuple[float, bool]:
        results = self._run_tests(action)
        if results is None:
            return 0.0, True
        return (1.0 if all(r["status"] == "pass" for r in results) else 0.0), True

    def get_privileged_information(self, action: str) -> str:
        results = self._run_tests(action)
        if results is None:
            return "❌ No code block found in response"
        return format_test_feedback(results, n_total=len(results))

    @classmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("tokenizer", None)
        return run_livecodebench_eval(rollout_worker_url=rollout_worker_url, step=step, **kwargs)

    @classmethod
    def load(cls, dataset_split: str = "train", until: datetime | None = None) -> list[LiveCodeBenchEnv]:
        ds = load_livecodebench(dataset_split=dataset_split, until=until)
        envs = []
        for row in ds:
            tests = json.loads(row["tests"]) if isinstance(row["tests"], str) else row["tests"]
            envs.append(cls(prompt=row["prompt"], tests=tests))
        return envs


def run_livecodebench_eval(
    rollout_worker_url: str,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on LiveCodeBench test split (problems after LCB_TEST_CUTOFF)."""
    test_ds = load_livecodebench(dataset_split="test")
    problems = [dict(r) for r in test_ds]
    prompts = [p["prompt"] for p in problems]

    def is_correct(i: int, r: dict) -> bool:
        tests = json.loads(problems[i]["tests"]) if isinstance(problems[i]["tests"], str) else problems[i]["tests"]
        code = _extract_code(r["response"])
        return code is not None and _all_tests_pass(code, tests)

    return run_pass_at_k_eval(
        rollout_worker_url, prompts, range(len(problems)),
        eval_k, eval_max_tokens, step, temperature, top_k,
        is_correct=is_correct, metrics_key=f"eval/pass@{eval_k}", log_prefix="lcb eval",
    )
