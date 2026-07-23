import abc
import json
from collections.abc import Callable, Sequence
from math import comb
from typing import Any

import wandb
from datasets import Dataset, load_dataset
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

from opd.generator.rollout import generate_rollouts_remote
from opd.trainer.setup_utils import print0


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


def run_pass_at_k_eval(
    rollout_worker_url: str,
    prompts: list[str],
    problem_ids: Sequence[Any],
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float,
    top_k: int,
    is_correct: Callable[[int, dict], bool],
    metrics_key: str,
    log_prefix: str,
) -> dict[str, Any]:
    """Generates `eval_k` rollouts per prompt, scores each, and logs a pass@k metric.

    Shared by `eval_math._run_eval_one_dataset` and
    `livecodebench.run_livecodebench_eval` — both generate rollouts, slice
    them into per-problem batches of size `eval_k`, score each with a
    dataset-specific correctness predicate, average pass@k over problems,
    and log/print the result. Only the predicate, the problem IDs used in
    per-row logging, and the metric key/log prefix differ per caller.

    Args:
        rollout_worker_url: Base URL of the running rollout worker.
        prompts: One prompt string per problem.
        problem_ids: One display ID per problem (e.g. a dataset's own ID
          column, or just `range(len(prompts))`), parallel to `prompts`.
        eval_k: Samples per problem, for pass@k.
        eval_max_tokens: Max generation length.
        step: Trainer step, used for logging and the W&B x-axis.
        temperature: Sampling temperature.
        top_k: Sampling top-k (-1 disables it).
        is_correct: `(problem_index, rollout) -> bool`, called once per
          generated rollout.
        metrics_key: The single key under which the averaged pass@k is
          logged. Follow the `eval/{dataset_name}/pass@{k}` convention
          shared by every eval site (e.g. `"eval/livecodebench/pass@4"`).
        log_prefix: Prefix for the `print0` summary line, e.g. `"lcb eval"`
          -> `"[lcb eval step=3] ..."`.

    Returns:
        `{metrics_key: average pass@k over all problems}`.
    """
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    per_problem = []
    for i, problem_id in enumerate(problem_ids):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        n_correct = sum(is_correct(i, r) for r in batch)
        per_problem.append({
            "problem_idx": problem_id,
            "n_correct": n_correct,
            "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
        })

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {metrics_key: overall}

    print0(f"[{log_prefix} step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(f"  problem {r['problem_idx']}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics


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
