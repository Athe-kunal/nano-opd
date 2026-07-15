from __future__ import annotations

import re
from typing import Any

from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, build_system_user_conversation, load_arrow_split

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


def _build_question(row: dict) -> str:
    """Combines a row's `prompt` and `instruction` fields into one question string.

    `instruction` is folded in (on its own paragraph) only when present —
    some rows carry no instruction beyond the prompt itself.
    """
    prompt = (row.get("prompt") or "").strip()
    instruction = (row.get("instruction") or "").strip()
    return f"{prompt}\n\n{instruction}".strip() if instruction else prompt


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
    ds = load_arrow_split(_BASE_URL, split)

    records = []
    for row in ds:
        demonstration = _format_demonstration(row.get("golden_answer") or [])
        records.append({"question": _build_question(row), "demonstration": demonstration})
    return records


def load_sdft_tooluse_eval() -> list[dict]:
    """Load tooluse eval split for pass@k evaluation.

    Unlike load_sdft_tooluse, keeps golden_answer as a raw list[dict] so the
    grader can compare action names directly.
    """
    ds = load_arrow_split(_BASE_URL, "eval")
    records: list[dict] = []
    for row in ds:
        records.append({"question": _build_question(row), "golden_answer": row.get("golden_answer") or []})
    return records


def grade_tooluse_response(response: str, golden_answer: list[dict]) -> bool:
    """True if the action names in the response match the expected sequence.

    Parses every "Action: <name>" line from the response (case-insensitive)
    and compares the sequence to the golden_answer action names.
    Action_Input is not checked — tool selection is the primary signal.
    """
    parsed = re.findall(r"(?i)^Action:\s*(.+)", response, re.MULTILINE)
    expected = [step["Action"].strip() for step in golden_answer]
    return [p.strip() for p in parsed] == expected


class SdftToolUseEnv(OPDEnvBase):
    """Environment wrapper for the Self-Distillation tool-use dataset.

    SDFT trains via KL distillation, not reward maximization, so compute_reward
    always returns 1.0. get_privileged_information returns the worked
    demonstration (doesn't depend on the student's action, unlike SDPO envs).
    """

    def __init__(self, question: str, demonstration: str):
        super().__init__(kind="tooluse", dataset="sdft_tooluse")
        self.question = question
        self.demonstration = demonstration

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return build_system_user_conversation(_SYSTEM_PROMPT, self.question), {}

    def compute_reward(self, response: str) -> tuple[float, bool]:
        return 1.0, True

    def get_privileged_information(self, action: str) -> str:
        return self.demonstration

    @classmethod
    def load(cls, split: str = "train") -> list["SdftToolUseEnv"]:
        return [
            cls(question=r["question"], demonstration=r["demonstration"])
            for r in load_sdft_tooluse(split=split)
        ]
