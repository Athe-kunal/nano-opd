import abc
from typing import Any

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


class OPDEnvBase(BaseTextEnv):
    """
    Base skyrl_gym environment for OPD datasets.

    Subclasses must implement:
      - init:           build opening conversation from prompt (dataset-specific)
      - compute_reward: per-step correctness signal for the RL reward
      - get_feedback:   SDPO feedback string injected into step metadata
      - evaluate:       dataset-level benchmark eval (e.g. AIME pass@k), logs to wandb
    """

    def __init__(self, kind: str, dataset: str) -> None:
        super().__init__()
        self.kind = kind
        self.dataset = dataset

    @abc.abstractmethod
    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        """Build the opening conversation. Dataset-specific."""
        ...

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward, done = self.compute_reward(action)
        feedback = self.get_feedback(action)
        self.turns += 1
        return BaseTextEnvStepOutput(
            observations=[{"role": "assistant", "content": action}],
            reward=reward,
            done=done or self.turns >= self.max_turns,
            metadata={"feedback": feedback, "kind": self.kind, "dataset": self.dataset},
            postprocessed_action=None,
        )

    @abc.abstractmethod
    def compute_reward(self, action: str) -> tuple[float, bool]:
        """Return (reward, done) for a single model completion."""
        ...

    @abc.abstractmethod
    def get_feedback(self, action: str) -> str:
        """
        Return a feedback string for self-policy distillation.
        Injected into step metadata so the training loop can use it as an
        additional signal when constructing the distillation target.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def evaluate(
        cls,
        rollout_worker_url: str,
        step: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run dataset-level evaluation (e.g. AIME pass@k, LCB pass@1) using the
        already weight-synced rollout worker. Log results to wandb and return a
        metrics dict keyed like {"eval/pass@8": 0.42}.
        """
        ...
