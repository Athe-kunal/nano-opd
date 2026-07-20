from __future__ import annotations

import json
import re
from typing import Any

import wandb
from datasets import Dataset, load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, build_system_user_conversation, pass_at_k
from opd.generator.rollout import generate_rollouts_remote
from opd.trainer.setup_utils import print0

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


def format_choices(choices: dict[str, list[str]]) -> str:
    texts, labels = choices['text'], choices['label']
    return "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])


def _format_row(row: dict) -> dict:
    choices_text = format_choices(row['choices'])
    return {
        "dataset": "sciknoweval",
        "system": SYSTEM_PROMPT,
        "prompt": row['question'] + "\n\n" + choices_text + "\nPlease reason step by step.",
        "answer_key": row['answerKey'],
        "choices_text": choices_text,
        "description": row['question'],
        "kind": "mcq",
    }


def _extract_answer(response: str) -> str | None:
    """Return the letter inside the last <answer>...</answer> tag, or None."""
    matches = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", response)
    return matches[-1].strip().upper() if matches else None


def load_sciknoweval(
    domains: list[str] | None = None,
    levels: list[str] | None = None,
    types: list[str] | None = None,
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
    def _is_unique(row: dict[str, Any]) -> bool:
        if row["description"] in seen:
            return False
        seen.add(row["description"])
        return True

    return ds.filter(_is_unique)


def load_sciknoweval_split(
    test_size: float,
    seed: int = 42,
    domains: list[str] | None = None,
    levels: list[str] | None = None,
    types: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, test_rows) as plain lists. Split is deterministic."""
    ds = load_sciknoweval(domains=domains, levels=levels, types=types)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return [dict(r) for r in split["train"]], [dict(r) for r in split["test"]]


class SciKnowEvalEnv(OPDEnvBase):
    """
    skyrl_gym environment for SciKnowEval MCQ problems.

    Each instance wraps a single (prompt, answer_key) pair. The reward is 1.0
    if the model's <answer> tag matches the ground-truth letter, else 0.0.
    get_privileged_information reveals the correct answer for SDPO self-distillation.
    evaluate runs the held-out 10% test split via the rollout worker.
    """

    def __init__(self, prompt: str, answer_key: str, choices_text: str = "", system: str = SYSTEM_PROMPT) -> None:
        super().__init__(kind="mcq", dataset="sciknoweval")
        self.prompt = prompt
        self.answer_key = answer_key.strip().upper()
        self.choices_text = choices_text
        self.system = system

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return build_system_user_conversation(self.system, self.prompt), {}

    def compute_reward(self, action: str) -> tuple[float, bool]:
        pred = _extract_answer(action)
        return (1.0 if pred == self.answer_key else 0.0), True

    def get_privileged_information(self, action: str) -> str:
        if _extract_answer(action) is None:
            return "Format error: response must contain an <answer>A/B/C/D</answer> tag with your final answer."
        pred = _extract_answer(action)
        options = f"\n{self.choices_text}" if self.choices_text else ""
        return f"The options are:{options}\nYour answer {pred} is incorrect. The correct answer is {self.answer_key}."

    @classmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("tokenizer", None)
        test_size = kwargs.pop("test_size", 0.1)
        _, test_rows = load_sciknoweval_split(test_size=test_size)
        return run_sciknow_eval(
            rollout_worker_url=rollout_worker_url,
            step=step,
            test_rows=test_rows,
            eval_k=kwargs["eval_k"],
            eval_max_tokens=kwargs["eval_max_tokens"],
            temperature=kwargs.get("temperature", 0.6),
            top_k=kwargs.get("top_k", -1),
        )

    @classmethod
    def load(
        cls,
        test_size: float = 0.01,
        domains: list[str] | None = None,
        levels: list[str] | None = None,
        types: list[str] | None = None,
        seed: int = 42,
    ) -> list[SciKnowEvalEnv]:
        train_rows, _ = load_sciknoweval_split(test_size=test_size, seed=seed, domains=domains, levels=levels, types=types)
        return [cls(prompt=row["prompt"], answer_key=row["answer_key"], choices_text=row.get("choices_text", "")) for row in train_rows]


def run_sciknow_eval(
    rollout_worker_url: str,
    test_rows: list[dict],
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on the pre-split SciKnowEval test rows."""
    prompts = [r["prompt"] for r in test_rows]
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    total_correct = 0
    per_problem = []
    for i, row in enumerate(test_rows):
        answer_key = row["answer_key"].strip().upper()
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        n_correct = sum(_extract_answer(r["response"]) == answer_key for r in batch)
        total_correct += n_correct
        per_problem.append({
            "problem_idx": i,
            "answer_key": answer_key,
            "n_correct": n_correct,
            "accuracy": n_correct / eval_k,
        })

    total_responses = len(test_rows) * eval_k
    accuracy = total_correct / total_responses
    metrics = {"eval/sciknoweval/accuracy": accuracy}

    print0(f"[sciknoweval eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(
            f"  problem {r['problem_idx']:04d} (ans={r['answer_key']}): "
            f"{r['n_correct']}/{eval_k}  acc={r['accuracy']:.3f}"
        )

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
