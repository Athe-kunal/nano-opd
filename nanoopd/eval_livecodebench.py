from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from typing import Any

import wandb

from nanoopd.common import print0
from nanoopd.data.livecodebench import (
    TIME_LIMIT,
    _run_functional,
    _run_stdio,
    load_livecodebench,
)
from nanoopd.eval_aime import pass_at_k
from nanoopd.rollout import generate_rollouts_remote


def _extract_code(response: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def _all_tests_pass(code: str, tests: dict) -> bool:
    inputs = tests["inputs"]
    outputs = tests["outputs"]
    fn_name = tests.get("fn_name", "")
    testtype = tests.get("testtype", "stdin")
    time_limit = tests.get("time_limit", TIME_LIMIT)

    if testtype == "functional" and fn_name:
        results = _run_functional(code, fn_name, inputs, outputs, time_limit)
    else:
        results = _run_stdio(code, inputs, outputs, time_limit)

    return all(r == "✅" for r in results)


def run_eval(
    rollout_worker_url: str,
    tokenizer,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on LiveCodeBench test split (problems after cutoff).

    pass@k is 1 for a problem if at least one of the k completions has all
    tests passing, computed via the unbiased estimator.
    """
    from nanoopd.data.livecodebench import CODE_PROMPT

    test_ds = load_livecodebench(dataset_split="test")
    problems = [dict(r) for r in test_ds]

    prompts = [CODE_PROMPT.format(problem=p["description"]) for p in problems]
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    per_problem = []
    for i, prob in enumerate(problems):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        tests = json.loads(prob["tests"]) if isinstance(prob["tests"], str) else prob["tests"]
        n_correct = 0
        for r in batch:
            code = _extract_code(r["response"])
            if code is not None and _all_tests_pass(code, tests):
                n_correct += 1
        per_problem.append(
            {
                "problem_idx": i,
                "n_correct": n_correct,
                "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
            }
        )

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {f"eval/pass@{eval_k}": overall}

    print0(f"[lcb eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(
            f"  problem {r['problem_idx']:03d}: {r['n_correct']}/{eval_k}"
            f"  pass@{eval_k}={r['pass_at_k']:.3f}"
        )

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
