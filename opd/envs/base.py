import abc
from math import comb
from typing import Any

from datasets import Dataset, load_dataset
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: Number of samples generated per problem.
        c: Number of correct samples among the `n`.
        k: The "pass@k" cutoff to estimate.

    Returns:
        The probability that at least one of `k` samples drawn (without
        replacement) from the `n` generated would be correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def build_system_user_conversation(system: str, user_content: str) -> ConversationType:
    """Builds the two-message [system, user] conversation most envs' `init` returns.

    Args:
        system: The system prompt.
        user_content: The user-turn content (the problem/question text).

    Returns:
        A `[{"role": "system", ...}, {"role": "user", ...}]` conversation.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def load_arrow_split(base_url: str, split: str) -> Dataset:
    """Loads one arrow-format split from a GitHub-hosted data directory.

    Used by the SDFT science/tool-use loaders, whose train and eval splits
    both live at `{base_url}/{split}_data/data-00000-of-00001.arrow`.

    Args:
        base_url: Base URL of the dataset's data directory.
        split: Split name (e.g. "train", "eval"); also the shard
          subdirectory name (`{split}_data`).

    Returns:
        The loaded HuggingFace `Dataset` for this split.
    """
    url = f"{base_url}/{split}_data/data-00000-of-00001.arrow"
    return load_dataset("arrow", data_files={split: url}, split=split)


class OPDEnvBase(BaseTextEnv):
    """
    Base skyrl_gym environment for OPD datasets.

    Subclasses must implement:
      - init:           build opening conversation from prompt (dataset-specific)
      - compute_reward: per-step correctness signal for the RL reward
      - get_privileged_information: ground truth injected into step metadata,
                        used to condition a richer teacher. For SDPO envs
                        (Dapo/LiveCodeBench/SciKnowEval) this is feedback text
                        derived from grading `action` (the student's own
                        attempt). For OPSD/SDFT envs this is a fixed piece of
                        ground truth — a reference solution or worked
                        demonstration — that doesn't depend on `action` at all.

    Subclasses may override:
      - evaluate:       mid-training eval via the live rollout worker (online envs only).
                        Math datasets use eval_math.py post-training instead — they
                        inherit the default no-op here.
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
        privileged_information = self.get_privileged_information(action)
        self.turns += 1
        return BaseTextEnvStepOutput(
            observations=[{"role": "assistant", "content": action}],
            reward=reward,
            done=done or self.turns >= self.max_turns,
            metadata={
                "privileged_information": privileged_information,
                "kind": self.kind,
                "dataset": self.dataset,
            },
            postprocessed_action=None,
        )

    @abc.abstractmethod
    def compute_reward(self, action: str) -> tuple[float, bool]:
        """Return (reward, done) for a single model completion."""
        ...

    @abc.abstractmethod
    def get_privileged_information(self, action: str) -> str:
        """Return ground truth used to condition a richer teacher.

        SDPO envs grade `action` and return a feedback string (e.g. "your
        answer was X, the correct answer is Y"). OPSD/SDFT envs ignore
        `action` and return a fixed piece of ground truth (a reference
        solution or worked demonstration) that doesn't depend on what the
        student produced.

        Args:
            action: The student's completion. Ignored by envs whose
              privileged information doesn't depend on it.

        Returns:
            The privileged-information string (empty if none applies).
        """
        ...

    @classmethod
    def evaluate(
        cls,
        _rollout_worker_url: str,
        _step: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Mid-training eval using the live rollout worker. Override in envs that
        support online evaluation (e.g. livecodebench, sciknoweval). Math envs
        (dapo_math, opsd_math) leave this as a no-op and use eval_math.py
        post-training via the shell scripts instead.
        """
        return {}
