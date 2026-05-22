from __future__ import annotations

import json
import re
from typing import Any, Optional, List, Dict

from math import comb
import wandb
from datasets import Dataset, load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.common import print0
from opd.envs.base import OPDEnvBase
from opd.generator.rollout import generate_rollouts_remote

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


SYSTEM_PROMPT = """
Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer>
"""


def format_choices(choices: Dict[str, str]) -> str:
    texts, labels = choices['text'], choices['label']
    return "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])


def _format_row(row: dict) -> dict:
    return {
        "dataset": "sciknoweval",
        "system": SYSTEM_PROMPT,
        "prompt": row['question'] + "\n\n" + format_choices(row['choices']) + "\nPlease reason step by step.",
        "answer_key": row['answerKey'],
        "description": row['question'],
        "kind": "mcq",
    }


def _extract_answer(response: str) -> str | None:
    """Return the letter inside the last <answer>...</answer> tag, or None."""
    matches = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", response)
    return matches[-1].strip().upper() if matches else None


def load_sciknoweval(
    domains: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
) -> Dataset:
    ds = load_dataset("hicai-zju/SciKnowEval", split='test')

    if domains:
        ds = ds.filter(lambda x: x['domain'] in domains)
    if levels:
        ds = ds.filter(lambda x: x['details']['level'] in levels)
    if types:
        ds = ds.filter(lambda x: x['type'] in types)

    ds = ds.map(_format_row, remove_columns=ds.column_names, load_from_cache_file=False)

    # The raw dataset generates multiple MCQ variants from the same question stem.
    # Deduplicate by description so each underlying question appears only once.
    seen: set[str] = set()
    def _is_unique(row):
        if row["description"] in seen:
            return False
        seen.add(row["description"])
        return True

    return ds.filter(_is_unique)


class SciKnowEvalEnv(OPDEnvBase):
    """
    skyrl_gym environment for SciKnowEval MCQ problems.

    Each instance wraps a single (prompt, answer_key) pair. The reward is 1.0
    if the model's <answer> tag matches the ground-truth letter, else 0.0.
    get_feedback reveals the correct answer for SDPO self-distillation.
    evaluate runs the held-out 10% test split via the rollout worker.
    """

    def __init__(self, prompt: str, answer_key: str, system: str = SYSTEM_PROMPT) -> None:
        super().__init__(kind="mcq", dataset="sciknoweval")
        self.prompt = prompt
        self.answer_key = answer_key.strip().upper()
        self.system = system

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        messages: ConversationType = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.prompt},
        ]
        return messages, {}

    def compute_reward(self, action: str) -> tuple[float, bool]:
        pred = _extract_answer(action)
        return (1.0 if pred == self.answer_key else 0.0), True

    def get_feedback(self, action: str) -> str:
        pred = _extract_answer(action)
        if pred == self.answer_key:
            return f"Correct! The answer is {self.answer_key}."
        return f"Incorrect. The correct answer is {self.answer_key}."

    @classmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return run_eval(rollout_worker_url=rollout_worker_url, step=step, **kwargs)

    @classmethod
    def load(
        cls,
        domains: Optional[List[str]] = None,
        levels: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        test_size: float = 0.0,
        seed: int = 42,
    ) -> list[SciKnowEvalEnv]:
        ds = load_sciknoweval(domains=domains, levels=levels, types=types)
        if test_size > 0:
            ds = ds.train_test_split(test_size=test_size, seed=seed)["train"]
        return [cls(prompt=row["prompt"], answer_key=row["answer_key"]) for row in ds]


def run_eval(
    rollout_worker_url: str,
    tokenizer: Any,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on SciKnowEval test split (10% held-out, seed=42)."""
    full_ds = load_sciknoweval()
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    test_rows = [dict(r) for r in split["test"]]

    prompts = [r["prompt"] for r in test_rows]
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    per_problem = []
    for i, row in enumerate(test_rows):
        answer_key = row["answer_key"].strip().upper()
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        n_correct = sum(_extract_answer(r["response"]) == answer_key for r in batch)
        per_problem.append({
            "problem_idx": i,
            "answer_key": answer_key,
            "n_correct": n_correct,
            "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
        })

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {f"eval/pass@{eval_k}": overall}

    print0(f"[sciknoweval eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(
            f"  problem {r['problem_idx']:04d} (ans={r['answer_key']}): "
            f"{r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}"
        )

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
