"""Post-training math evaluation on AIME 2024/2025 and HMMT 2025 (pass@k).

Same pattern as `opd.envs.sciknoweval.run_sciknow_eval`: rollouts come from
the already-running, weight-synced rollout worker via `generate_rollouts_remote`,
not a standalone vLLM engine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm as async_tqdm

from opd.envs.base import pass_at_k
from opd.envs.dapo_dataset import check_answer, extract_last_boxed
from opd.generator.rollout import generate_rollouts_remote
from opd.trainer.setup_utils import print0

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


async def _load_dataset_problems(dataset_name: str) -> tuple[list[dict], list[str], list[Any]]:
    """Loads one benchmark's problems, prompts, and display IDs.

    Args:
        dataset_name: Key into `DATASET_CONFIGS`.

    Returns:
        `(problems, prompts, problem_ids)`.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    problems: list[dict] = list(
        await asyncio.to_thread(load_dataset, cfg["hf_path"], split=cfg["split"])
    )
    prompts = [p[cfg["problem_col"]] for p in problems]
    problem_ids = [p.get(cfg["id_col"], i) for i, p in enumerate(problems)]
    return problems, prompts, problem_ids


async def _generate_one_problem(
    rollout_worker_url: str,
    prompt: str,
    eval_k: int,
    eval_max_tokens: int,
    temperature: float,
    top_k: int,
    pbar: async_tqdm,
) -> list[dict[str, Any]]:
    """Generates `eval_k` rollouts for one problem, ticking `pbar` on completion.

    One `/generate` request per problem (rather than one batched request per
    dataset) so the progress bar reflects real per-example completion.
    """
    rollouts = await asyncio.to_thread(
        generate_rollouts_remote,
        rollout_worker_url, [prompt], eval_k, eval_max_tokens, temperature, top_k,
    )
    pbar.update(1)
    return rollouts


def _score_dataset(
    dataset_name: str,
    problems: list[dict],
    problem_ids: list[Any],
    per_problem_rollouts: list[list[dict]],
    eval_k: int,
    step: int,
) -> dict[str, float]:
    """Scores one benchmark's generated rollouts and logs a pass@k summary.

    Same scoring/logging shape as `opd.envs.base.run_pass_at_k_eval`, kept
    separate here since generation runs per-problem (see
    `_generate_one_problem`) rather than through that function's single
    batched call.

    Args:
        dataset_name: Key into `DATASET_CONFIGS`.
        problems: This dataset's raw problem rows.
        problem_ids: Display ID per problem, parallel to `problems`.
        per_problem_rollouts: `eval_k` rollouts per problem, parallel to `problems`.
        eval_k: Samples per problem, for pass@k.
        step: Trainer step, used for logging and the W&B x-axis.

    Returns:
        `{"eval/{dataset_name}/pass@{eval_k}": average pass@k over all problems}`.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    metrics_key = f"eval/{dataset_name}/pass@{eval_k}"

    per_problem = []
    for i, (problem_id, rollouts) in enumerate(zip(problem_ids, per_problem_rollouts)):
        n_correct = sum(
            check_answer(extract_last_boxed(r["response"]), problems[i][cfg["answer_col"]])
            for r in rollouts
        )
        per_problem.append({
            "problem_idx": problem_id,
            "n_correct": n_correct,
            "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
        })

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {metrics_key: overall}

    print0(f"[{dataset_name} eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(f"  problem {r['problem_idx']}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics


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

    Generation is issued one request per problem (not one batched request per
    dataset), so the progress bar's granularity is individual examples —
    `sum(len(dataset) for dataset in eval_datasets)` total ticks.

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
    dataset_names = [name.strip() for name in eval_datasets.split(",")]

    async def _run_all() -> dict[str, Any]:
        loaded = await asyncio.gather(*[_load_dataset_problems(name) for name in dataset_names])
        total_examples = sum(len(prompts) for _, prompts, _ in loaded)

        with async_tqdm(total=total_examples, desc="math eval examples") as pbar:
            per_dataset_rollouts = await asyncio.gather(*[
                asyncio.gather(*[
                    _generate_one_problem(
                        rollout_worker_url, prompt, eval_k, eval_max_tokens, temperature, top_k, pbar,
                    )
                    for prompt in prompts
                ])
                for _, prompts, _ in loaded
            ])

        metrics: dict[str, Any] = {}
        for name, (problems, _prompts, problem_ids), rollouts in zip(dataset_names, loaded, per_dataset_rollouts):
            metrics.update(_score_dataset(name, problems, problem_ids, rollouts, eval_k, step))
        return metrics

    return asyncio.run(_run_all())
