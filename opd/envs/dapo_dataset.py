from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, build_system_user_conversation

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer in \\boxed{}."
)


def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last brace-balanced \\boxed{...} in text, or None."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def _canon(s: str) -> str:
    return " ".join(str(s).strip().split())


def check_answer(pred: str | None, answer: int | float | str) -> bool:
    if pred is None:
        return False
    pred_s = _canon(pred)
    ans_s = _canon(str(answer))
    if pred_s == ans_s:
        return True
    try:
        return float(pred_s) == float(ans_s)
    except (ValueError, TypeError):
        return False


def load_dapo_math(
    dataset_id: str = "open-r1/DAPO-Math-17k-Processed",
    config_name: str = "all",
    split: str = "train",
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_id, config_name, split=split)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        row: Mapping[str, Any]
        prompt_text = (row.get("prompt") or "").strip()
        reward_model = row.get("reward_model") or {}
        top_extra = row.get("extra_info") or {}
        nested_extra = reward_model.get("extra_info") or {}
        raw_id = top_extra.get("index") or nested_extra.get("index")
        row_label = f"dapo_math/{raw_id}" if raw_id and str(raw_id).strip() else f"dapo_math/row_{i}"

        answer = ((row.get("solution") or "").strip()
                  or (reward_model.get("ground_truth") or "").strip())

        if not prompt_text or not answer:
            logger.warning(f"Skipping row {i} ({row_label}): empty prompt or answer")
            continue

        records.append({
            "prompt": prompt_text,
            "answer": answer,
        })
    return records


class DapoMathEnv(OPDEnvBase):
    """
    skyrl_gym environment for the DAPO-Math dataset.

    Each instance wraps a single (prompt, answer) pair. The reward is 1.0 if
    the model's final \\boxed{} answer matches the ground truth, else 0.0.
    get_privileged_information reveals the ground-truth answer for SDPO self-distillation.
    evaluate runs the AIME 2025 benchmark via the rollout worker.
    """

    def __init__(self, prompt: str, answer: str) -> None:
        super().__init__(kind="math", dataset="dapo")
        self.prompt = prompt
        self.answer = answer

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return build_system_user_conversation(DEFAULT_SYSTEM_PROMPT, self.prompt), {}

    def compute_reward(self, action: str) -> tuple[float, bool]:
        pred = extract_last_boxed(action)
        correct = check_answer(pred, self.answer)
        return (1.0 if correct else 0.0), True

    def get_privileged_information(self, action: str) -> str:
        if extract_last_boxed(action) is None:
            return "Format error: response must contain a \\boxed{...} expression with your final answer."
        pred = extract_last_boxed(action)
        return f"Your answer \\boxed{{{pred}}} is incorrect. The correct answer is \\boxed{{{self.answer}}}."

    @classmethod
    def evaluate(cls, rollout_worker_url: str, step: int, **kwargs: Any) -> dict[str, Any]:
        # Local import: opd.eval.eval_math imports check_answer/extract_last_boxed
        # from this module, so importing it at module level would be circular.
        from opd.eval.eval_math import run_eval

        kwargs.pop("tokenizer", None)
        kwargs.pop("test_size", None)
        return run_eval(
            rollout_worker_url=rollout_worker_url,
            eval_k=kwargs["eval_k"],
            eval_max_tokens=kwargs["eval_max_tokens"],
            step=step,
            eval_datasets=kwargs.get("eval_datasets", "aime_2025,aime_2024,hmmt_2025"),
            temperature=kwargs.get("temperature", 0.6),
            top_k=kwargs.get("top_k", -1),
        )

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> list[DapoMathEnv]:
        return [cls(prompt=r["prompt"], answer=r["answer"]) for r in records]

    @classmethod
    def load(
        cls,
        dataset_id: str = "open-r1/DAPO-Math-17k-Processed",
        config_name: str = "all",
        split: str = "train",
    ) -> list[DapoMathEnv]:
        return cls.from_records(load_dapo_math(dataset_id, config_name, split))
