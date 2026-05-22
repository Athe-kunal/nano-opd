from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from typing import Any

import wandb
from datasets import Dataset, concatenate_datasets, load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.common import print0
from opd.envs.base import OPDEnvBase
from opd.eval.eval_aime import pass_at_k
from opd.generator.rollout import generate_rollouts_remote

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ```.

{problem}"""


def _parse_signature(starter_code: str) -> str:
    after_def = starter_code.split("def ")[1]
    return "def " + (after_def.split("Input\n")[0] if "Input\n" in after_def else after_def).strip()


def _translate_private_test_cases(encoded_data, fn_name: str) -> str:
    import base64, pickle, zlib
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


def _run_functional(code: str, fn_name: str, inputs: list, outputs: list, time_limit: int) -> list[str]:
    """Execute code as a function call per test case. Returns per-test feedback strings."""
    results = []
    for inp, expected in zip(inputs, outputs):
        driver = textwrap.dedent(f"""
import json, sys
{code}

_inp = json.loads(sys.stdin.read())
_result = {fn_name}(*_inp)
print(json.dumps(_result))
""")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                input=json.dumps(inp),
                capture_output=True, text=True, timeout=time_limit,
            )
            if proc.returncode != 0:
                results.append(f"❌ Runtime error: {proc.stderr.strip()}")
                continue
            got = json.loads(proc.stdout.strip())
            if got == expected:
                results.append("✅")
            else:
                results.append(f"❌ Expected {expected!r}, got {got!r}")
        except subprocess.TimeoutExpired:
            results.append(f"❌ Time limit exceeded ({time_limit}s)")
        except Exception as e:
            results.append(f"❌ {e}")
    return results


def _run_stdio(code: str, inputs: list, outputs: list, time_limit: int) -> list[str]:
    """Execute code with stdin per test case. Returns per-test feedback strings."""
    results = []
    for inp, expected in zip(inputs, outputs):
        stdin_text = inp if isinstance(inp, str) else "\n".join(str(x) for x in inp)
        expected_text = expected if isinstance(expected, str) else str(expected)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                input=stdin_text,
                capture_output=True, text=True, timeout=time_limit,
            )
            if proc.returncode != 0:
                results.append(f"❌ Runtime error: {proc.stderr.strip()}")
                continue
            got = proc.stdout.strip()
            if got == expected_text.strip():
                results.append("✅")
            else:
                results.append(f"❌ Expected {expected_text!r}, got {got!r}")
        except subprocess.TimeoutExpired:
            results.append(f"❌ Time limit exceeded ({time_limit}s)")
        except Exception as e:
            results.append(f"❌ {e}")
    return results


def _all_tests_pass(code: str, tests: dict) -> bool:
    fn_name = tests.get("fn_name", "")
    testtype = tests.get("testtype", "stdin")
    time_limit = tests.get("time_limit", TIME_LIMIT)
    if testtype == "functional" and fn_name:
        results = _run_functional(code, fn_name, tests["inputs"], tests["outputs"], time_limit)
    else:
        results = _run_stdio(code, tests["inputs"], tests["outputs"], time_limit)
    return all(r == "✅" for r in results)


class LiveCodeBenchEnv(OPDEnvBase):
    """
    skyrl_gym environment for LiveCodeBench problems.

    Each instance wraps a single (prompt, tests) pair. The reward is 1.0 if
    all test cases pass, else 0.0. get_feedback returns per-test execution
    results for SDPO self-distillation. evaluate runs the LCB test split.
    """

    def __init__(self, prompt: str, tests: dict[str, Any]) -> None:
        super().__init__(kind="code", dataset="livecodebench")
        self.prompt = prompt
        self.tests = tests

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return [{"role": "user", "content": self.prompt}], {}

    def _run_tests(self, action: str) -> list[str] | None:
        """Run all test cases for this action. Returns per-test results, or None if no code found."""
        code = _extract_code(action)
        if code is None:
            return None
        fn_name = self.tests.get("fn_name", "")
        testtype = self.tests.get("testtype", "stdin")
        time_limit = self.tests.get("time_limit", TIME_LIMIT)
        if testtype == "functional" and fn_name:
            return _run_functional(code, fn_name, self.tests["inputs"], self.tests["outputs"], time_limit)
        return _run_stdio(code, self.tests["inputs"], self.tests["outputs"], time_limit)

    def compute_reward(self, action: str) -> tuple[float, bool]:
        results = self._run_tests(action)
        if results is None:
            return 0.0, True
        return (1.0 if all(r == "✅" for r in results) else 0.0), True

    def get_feedback(self, action: str) -> str:
        results = self._run_tests(action)
        if results is None:
            return "❌ No code block found in response"
        return "\n".join(f"Test {i + 1}: {r}" for i, r in enumerate(results))

    @classmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return run_eval(rollout_worker_url=rollout_worker_url, step=step, **kwargs)

    @classmethod
    def load(cls, dataset_split: str = "train", until: datetime | None = None) -> list[LiveCodeBenchEnv]:
        ds = load_livecodebench(dataset_split=dataset_split, until=until)
        envs = []
        for row in ds:
            tests = json.loads(row["tests"]) if isinstance(row["tests"], str) else row["tests"]
            envs.append(cls(prompt=row["prompt"], tests=tests))
        return envs


def run_eval(
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
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    per_problem = []
    for i, prob in enumerate(problems):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        tests = json.loads(prob["tests"]) if isinstance(prob["tests"], str) else prob["tests"]
        n_correct = sum(
            1 for r in batch
            if (code := _extract_code(r["response"])) is not None and _all_tests_pass(code, tests)
        )
        per_problem.append({
            "problem_idx": i,
            "n_correct": n_correct,
            "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
        })

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {f"eval/pass@{eval_k}": overall}

    print0(f"[lcb eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(f"  problem {r['problem_idx']:03d}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
