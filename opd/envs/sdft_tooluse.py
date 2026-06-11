from __future__ import annotations

from typing import Any

from datasets import load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase

_BASE_URL = "https://github.com/idanshen/Self-Distillation/raw/main/data/tooluse_data"

_SYSTEM_PROMPT = (
    "You are an AI assistant that can use tools to complete tasks. "
    "For each step, output an Action and Action_Input on separate lines."
)


def _format_demonstration(golden_answer: list[dict]) -> str:
    """Format a golden_answer sequence into a readable tool-call demonstration.

    Each entry has "Action" and "Action_Input" keys. The result is a numbered
    sequence so the teacher can show the student the expected tool-use pattern.
    """
    lines = []
    for i, step in enumerate(golden_answer, 1):
        action = step.get("Action", "").strip()
        action_input = step.get("Action_Input", "").strip()
        lines.append(f"Step {i}:")
        lines.append(f"Action: {action}")
        lines.append(f"Action Input: {action_input}")
    return "\n".join(lines)


def load_sdft_tooluse(split: str = "train") -> list[dict]:
    """Load tooluse_data from the Self-Distillation GitHub repo.

    Each row has `prompt`, `instruction`, and `golden_answer` (list of
    {Action, Action_Input}). The question combines prompt + instruction;
    the demonstration is the formatted golden_answer sequence.

    Args:
        split: "train" or "eval"

    Returns:
        List of {"question": str, "demonstration": str} dicts.
    """
    url = f"{_BASE_URL}/{split}_data/data-00000-of-00001.arrow"
    ds = load_dataset("arrow", data_files={split: url}, split=split)

    records = []
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        instruction = (row.get("instruction") or "").strip()
        question = f"{prompt}\n\n{instruction}".strip() if instruction else prompt
        demonstration = _format_demonstration(row.get("golden_answer") or [])
        records.append({"question": question, "demonstration": demonstration})
    return records


class SdftToolUseEnv(OPDEnvBase):
    """Environment wrapper for the Self-Distillation tool-use dataset.

    SDFT trains via KL distillation, not reward maximization, so compute_reward
    always returns 1.0 and get_feedback returns an empty string.
    """

    def __init__(self, question: str, demonstration: str):
        super().__init__(kind="tooluse", dataset="sdft_tooluse")
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
    def load(cls, split: str = "train") -> list["SdftToolUseEnv"]:
        return [
            cls(question=r["question"], demonstration=r["demonstration"])
            for r in load_sdft_tooluse(split=split)
        ]
