from __future__ import annotations

from typing import Any

from datasets import load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase

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
    url = f"{_BASE_URL}/{split}_data/data-00000-of-00001.arrow"
    ds = load_dataset("arrow", data_files={split: url}, split=split)

    records = []
    for row in ds:
        messages = row["messages"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        question = user_messages[-1] if user_messages else messages[-1]["content"]
        records.append({"question": question, "demonstration": row["output_text"]})
    return records


class SdftScienceEnv(OPDEnvBase):
    """Environment wrapper for the Self-Distillation science dataset.

    SDFT trains via KL distillation, not reward maximization, so compute_reward
    always returns 1.0 and get_feedback returns an empty string. The env class
    exists to satisfy the OPDEnvBase interface and enable mid-training evaluation
    if needed.
    """

    def __init__(self, question: str, demonstration: str):
        super().__init__(kind="science", dataset="sdft_science")
        self.question = question
        self.demonstration = demonstration

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        conversation = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
        ]
        return conversation, {}

    def compute_reward(self, response: str) -> tuple[float, bool]:
        return 1.0, True

    def get_feedback(self, response: str) -> str:
        return ""

    @classmethod
    def load(cls, split: str = "train") -> list["SdftScienceEnv"]:
        return [
            cls(question=r["question"], demonstration=r["demonstration"])
            for r in load_sdft_science(split=split)
        ]
