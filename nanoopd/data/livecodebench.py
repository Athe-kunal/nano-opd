import os, copy, re, subprocess, sys, tempfile, textwrap
import numpy as np
import base64, json, pickle, zlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Sequence
from datasets import concatenate_datasets, load_dataset, Dataset

from nanoopd.data.base import FeedBackExample, SelfDistillationDatasetbase

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6
PERCENTAGE_TO_KEEP = 0.5

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ```.

{problem}"""


def _parse_signature(starter_code: str) -> str:
    after_def = starter_code.split("def ")[1]
    return "def " + (after_def.split("Input\n")[0] if "Input\n" in after_def else after_def).strip()


def _translate_private_test_cases(encoded_data, fn_name: str):
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
            "problem": problem,
            "prompt": CODE_PROMPT.format(problem=problem),
            "tests": _translate_private_test_cases(ex["private_test_cases"], fn_name=fn_name),
        }

    processed_shards = []
    for i in range(4):
        shard = ds.shard(num_shards=4, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)

    return concatenate_datasets(processed_shards)


def sample_tests(example):
    """Keep 50% of tests for the train set."""
    tests = json.loads(example["tests"])
    inputs, outputs = tests["inputs"], tests["outputs"]

    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    keep_indices = np.sort(np.random.choice(num_tests, size=keep_count, replace=False))

    reduced_tests = copy.deepcopy(tests)
    reduced_tests["inputs"] = [inputs[i] for i in keep_indices]
    reduced_tests["outputs"] = [outputs[i] for i in keep_indices]

    example["tests"] = json.dumps(reduced_tests)
    return example


def split_and_save(ds: Dataset, output_dir: str):
    """
    test.json  → full test suite per problem (for evaluation)
    train.json → 50% of tests per problem (for RL training)
    """
    np.random.seed(0)
    os.makedirs(output_dir, exist_ok=True)

    ds.to_json(os.path.join(output_dir, "test.json"))

    ds_reduced = ds.map(sample_tests)
    ds_reduced.to_json(os.path.join(output_dir, "train.json"))

    print(f"Test set:  {len(ds)} problems, full tests")
    print(f"Train set: {len(ds_reduced)} problems, {PERCENTAGE_TO_KEEP:.0%} of tests kept")


def _extract_code(response: str) -> str | None:
    """Return the last Python code block from a model response, or None."""
    # Match ```python ... ``` or ``` ... ```
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


class LiveCodeSelfDistillationDataset(SelfDistillationDatasetbase):

    def save_dataset(self, hf_name: str, path: str) -> None:
        ds = load_livecodebench(dataset_split="train")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(path)

    def get_feedback(self, result: Sequence[FeedBackExample]) -> list[FeedBackExample]:
        updated = []
        for ex in result:
            if ex.answer is None or ex.metadata is None:
                updated.append(replace(ex, feedback="❌ Missing model response or test metadata"))
                continue

            code = _extract_code(ex.answer)
            if code is None:
                updated.append(replace(ex, feedback="❌ No code block found in response"))
                continue

            tests = json.loads(ex.metadata) if isinstance(ex.metadata, str) else ex.metadata
            inputs = tests["inputs"]
            outputs = tests["outputs"]
            fn_name = tests.get("fn_name", "")
            testtype = tests.get("testtype", "stdin")
            time_limit = tests.get("time_limit", TIME_LIMIT)

            if testtype == "functional" and fn_name:
                per_test = _run_functional(code, fn_name, inputs, outputs, time_limit)
            else:
                per_test = _run_stdio(code, inputs, outputs, time_limit)

            feedback = "\n".join(f"Test {i + 1}: {r}" for i, r in enumerate(per_test))
            updated.append(replace(ex, feedback=feedback))

        return updated


if __name__ == "__main__":
    ds = load_livecodebench(dataset_split="test", until=datetime(2025, 5, 1))
    split_and_save(ds, output_dir="datasets/lcb_v6")
    # → datasets/lcb_v6/test.json   (full tests, for eval)
    # → datasets/lcb_v6/train.json  (50% tests, for RL training)