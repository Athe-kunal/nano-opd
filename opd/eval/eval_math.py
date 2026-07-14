"""Post-training math evaluation on AIME 2024/2025 and HMMT 2025 (pass@k).

Same pattern as `opd.envs.sciknoweval.run_sciknow_eval`: rollouts come from
the already-running, weight-synced rollout worker via `generate_rollouts_remote`,
not a standalone vLLM engine.
"""

from __future__ import annotations

import asyncio
from typing import Any

from datasets import load_dataset

from opd.envs.base import run_pass_at_k_eval
from opd.envs.dapo_dataset import check_answer, extract_last_boxed

# DAPO-style math benchmarks this evaluator supports. Not valid for
# livecodebench, whose scoring requires running generated code against
# test cases rather than comparing a single boxed answer.
DATASET_CONFIGS = {
    "aime_2024": {
        "hf_path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "id",
    },
    "aime_2025": {
        "hf_path": "MathArena/aime_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
    },
    "hmmt_2025": {
        "hf_path": "MathArena/hmmt_feb_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
    },
}


async def _run_eval_one_dataset(
    dataset_name: str,
    rollout_worker_url: str,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float,
    top_k: int,
) -> dict[str, float]:
    """Evaluates the already weight-synced rollout worker on one benchmark.

    Args:
        dataset_name: Key into `DATASET_CONFIGS`.
        rollout_worker_url: Base URL of the running rollout worker.
        eval_k: Samples per problem, for pass@k.
        eval_max_tokens: Max generation length.
        step: Trainer step, used for logging and the W&B x-axis.
        temperature: Sampling temperature.
        top_k: Sampling top-k (-1 disables it).

    Returns:
        A dict with a single `eval/{dataset_name}/pass@{eval_k}` entry.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    problems: list[dict] = list(
        await asyncio.to_thread(load_dataset, cfg["hf_path"], split=cfg["split"])
    )
    prompts = [p[cfg["problem_col"]] for p in problems]
    problem_ids = [p.get(cfg["id_col"], i) for i, p in enumerate(problems)]

    def is_correct(i: int, r: dict) -> bool:
        return check_answer(extract_last_boxed(r["response"]), problems[i][cfg["answer_col"]])

    return await asyncio.to_thread(
        run_pass_at_k_eval,
        rollout_worker_url, prompts, problem_ids,
        eval_k, eval_max_tokens, step, temperature, top_k,
        is_correct=is_correct,
        metrics_key=f"eval/{dataset_name}/pass@{eval_k}",
        log_prefix=f"{dataset_name} eval",
    )


def run_eval(
    rollout_worker_url: str,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    eval_datasets: str = "aime_2025,aime_2024,hmmt_2025",
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluates on one or more DAPO-style math benchmarks (AIME/HMMT), concurrently.

    Args:
        rollout_worker_url: Base URL of the running rollout worker.
        eval_k: Samples per problem, for pass@k.
        eval_max_tokens: Max generation length.
        step: Trainer step, used for logging and the W&B x-axis.
        eval_datasets: Comma-separated `DATASET_CONFIGS` keys, e.g.
          "aime_2025,aime_2024,hmmt_2025".
        temperature: Sampling temperature.
        top_k: Sampling top-k (-1 disables it).

    Returns:
        Combined `eval/{dataset_name}/pass@{eval_k}` metrics from every
        dataset in `eval_datasets`.
    """

    async def _run_all() -> list[dict[str, float]]:
        return await asyncio.gather(*[
            _run_eval_one_dataset(
                dataset_name=name.strip(),
                rollout_worker_url=rollout_worker_url,
                eval_k=eval_k,
                eval_max_tokens=eval_max_tokens,
                step=step,
                temperature=temperature,
                top_k=top_k,
            )
            for name in eval_datasets.split(",")
        ])

    metrics: dict[str, Any] = {}
    for dataset_metrics in asyncio.run(_run_all()):
        metrics.update(dataset_metrics)
    return metrics
