from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase, build_system_user_conversation
from opd.envs.dapo_dataset import check_answer, extract_last_boxed

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer in \\boxed{}."
)

DATASET_ID = "siyanzhao/Openthoughts_math_30k_opsd"


class OPSDMathEnv(OPDEnvBase):
    """
    skyrl_gym environment for the Openthoughts OPSD math dataset.

    Each instance wraps a single (problem, solution) pair.
    - problem: the math problem shown to both student and teacher
    - solution: the reference reasoning chain, returned via
      get_privileged_information so the training loop can build the
      teacher's privileged prompt (Figure 2 of the OPSD paper). Unlike SDPO
      envs, this doesn't depend on the student's action — the teacher is
      frozen and never conditioned on the student's own attempt.

    Reward: 1.0 if the model's \\boxed{} answer matches the reference, else 0.0.
    """

    def __init__(self, problem: str, solution: str) -> None:
        super().__init__(kind="math", dataset="opsd_math")
        self.problem = problem
        self.solution = solution

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return build_system_user_conversation(DEFAULT_SYSTEM_PROMPT, self.problem), {}

    def compute_reward(self, action: str) -> tuple[float, bool]:
        pred = extract_last_boxed(action)
        correct = check_answer(pred, extract_last_boxed(self.solution) or self.solution)
        return (1.0 if correct else 0.0), True

    def get_privileged_information(self, action: str) -> str:
        return self.solution

    @classmethod
    def load(
        cls,
        split: str = "train",
        dataset_id: str = DATASET_ID,
    ) -> list[OPSDMathEnv]:
        ds = load_dataset(dataset_id, split=split)
        envs = []
        for i, row in enumerate(ds):
            problem = (row.get("problem") or "").strip()
            solution = (row.get("solution") or "").strip()
            if not problem or not solution:
                logger.warning(f"Skipping row {i}: empty problem or solution")
                continue
            envs.append(cls(problem=problem, solution=solution))
        return envs
