from __future__ import annotations

import re
from typing import Any

from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, build_system_user_conversation, load_arrow_split

_BASE_URL = "https://github.com/idanshen/Self-Distillation/raw/main/data/science_data"

_SYSTEM_PROMPT = (
    "You are a knowledgeable science assistant. "
    "Answer the question below clearly and concisely."
)


def load_sdft_science(split: str = "train") -> list[dict]:
    """Load science_data from the Self-Distillation GitHub repo.

    Each row has `messages` (list of {role, content}) and `output_text`.
    We extract the last user message as the question and output_text as
    the demonstration for SDFT's demonstration-conditioned teacher.

    Args:
        split: "train" or "eval"

    Returns:
        List of {"question": str, "demonstration": str} dicts.
    """
    ds = load_arrow_split(_BASE_URL, split)

    records = []
    for row in ds:
        messages = row["messages"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        question = user_messages[-1] if user_messages else messages[-1]["content"]
        records.append({"question": question, "demonstration": row["output_text"]})
    return records


def load_sdft_science_eval() -> list[dict]:
    """Load science MCQ eval split for pass@k evaluation.

    The eval split has a different schema from train: each row has `prompt`
    (the MCQ question) and `answer` (correct letter A/B/C/D).
    """
    ds = load_arrow_split(_BASE_URL, "eval")
    return [{"question": row["prompt"], "answer": row["answer"]} for row in ds]


def grade_science_response(response: str, answer: str) -> bool:
    """True if response contains the correct MCQ letter.

    Looks for <answer>X</answer> first; falls back to the last standalone
    A/B/C/D letter in the response.
    """
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", response, re.IGNORECASE)
    if m:
        return m.group(1).upper() == answer.upper()
    letters = re.findall(r"\b([A-D])\b", response)
    return bool(letters) and letters[-1].upper() == answer.upper()


class SdftScienceEnv(OPDEnvBase):
    """Environment wrapper for the Self-Distillation science dataset.

    SDFT trains via KL distillation, not reward maximization, so compute_reward
    always returns 1.0. get_privileged_information returns the worked
    demonstration (doesn't depend on the student's action, unlike SDPO envs).
    """

    def __init__(self, question: str, demonstration: str):
        super().__init__(kind="science", dataset="sdft_science")
        self.question = question
        self.demonstration = demonstration

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return build_system_user_conversation(_SYSTEM_PROMPT, self.question), {}

    def compute_reward(self, response: str) -> tuple[float, bool]:
        return 1.0, True

    def get_privileged_information(self, action: str) -> str:
        return self.demonstration

    @classmethod
    def load(cls, split: str = "train") -> list["SdftScienceEnv"]:
        return [
            cls(question=r["question"], demonstration=r["demonstration"])
            for r in load_sdft_science(split=split)
        ]
