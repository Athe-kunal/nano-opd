from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset
from skyrl_gym.envs.base_text_env import ConversationType

from opd.envs.base import OPDEnvBase
from opd.envs.dapo_dataset import extract_last_boxed, check_answer

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
    - solution: the reference reasoning chain, returned as feedback so the
      training loop can build the teacher's privileged prompt (Figure 2 of
      the OPSD paper).

    Reward: 1.0 if the model's \\boxed{} answer matches the reference, else 0.0.
    """

    def __init__(self, problem: str, solution: str) -> None:
        super().__init__(kind="math", dataset="opsd_math")
        self.problem = problem
        self.solution = solution

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        messages: ConversationType = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": self.problem},
        ]
        return messages, {}

    def compute_reward(self, action: str) -> tuple[float, bool]:
        pred = extract_last_boxed(action)
        correct = check_answer(pred, extract_last_boxed(self.solution) or self.solution)
        return (1.0 if correct else 0.0), True

    def get_feedback(self, action: str) -> str:
        # The full reference solution — used by train_opsd.py to build the
        # teacher's privileged prompt via _build_teacher_messages.
        return self.solution

    @classmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from opd.eval.eval_aime_2025 import run_eval
        return run_eval(
            rollout_worker_url=rollout_worker_url,
            tokenizer=tokenizer,
            step=step,
            **kwargs,
        )

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
