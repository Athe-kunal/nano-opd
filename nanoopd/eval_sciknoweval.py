from __future__ import annotations

import json
import re
from typing import Any

import wandb

from nanoopd.common import print0
from nanoopd.data.sciknoweval import load_sciknoweval
from nanoopd.eval_aime import pass_at_k
from nanoopd.rollout import generate_rollouts_remote


def _extract_answer(response: str) -> str | None:
    """Return the letter inside the last <answer>...</answer> tag, or None."""
    matches = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", response)
    return matches[-1].strip().upper() if matches else None


def run_eval(
    rollout_worker_url: str,
    tokenizer,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on SciKnowEval test split (10% held-out, seed=42).

    pass@k: unbiased estimator over whether any of the k responses contains
    the correct answer letter inside <answer>...</answer>.
    """
    from nanoopd.data.base import InputExample

    full_ds = load_sciknoweval()
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    test_rows = [dict(r) for r in split["test"]]

    prompts = [r["prompt"] for r in test_rows]
    rollouts = generate_rollouts_remote(
        rollout_worker_url, prompts, eval_k, eval_max_tokens, temperature, top_k
    )

    per_problem = []
    for i, row in enumerate(test_rows):
        answer_key = row["tests"]["answerKey"].strip().upper()
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        n_correct = sum(
            _extract_answer(r["response"]) == answer_key for r in batch
        )
        per_problem.append(
            {
                "problem_idx": i,
                "answer_key": answer_key,
                "n_correct": n_correct,
                "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
            }
        )

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {f"eval/pass@{eval_k}": overall}

    print0(f"[sciknoweval eval step={step}] {json.dumps(metrics)}")
    for r in per_problem:
        print0(
            f"  problem {r['problem_idx']:04d} (ans={r['answer_key']}): "
            f"{r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}"
        )

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
