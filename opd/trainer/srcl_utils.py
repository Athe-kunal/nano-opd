"""Self-Refined Concise Learning (SRCL) helpers for train_dce.py.

SRCL asks the current policy to greedily rewrite its own rollout into a shorter,
reflection-free solution (without seeing the reference), keeps only rewrites that
pass a verifier, and trains the student with cross-entropy on them
(arXiv 2609.30652, Section 3.2 and Appendices M-N).
"""

import re
from typing import Any

import torch
import torch.nn.functional as F

from opd.fsdp.model import StudentModel
from opd.generator.rollout import generate_rollouts_remote, prepare_batch
from opd.trainer.models import DistributedContext

_REWRITE_TEMPLATE = (
    "SOURCE ROLLOUT\n{rollout}\nEND SOURCE ROLLOUT\n\n"
    "PROBLEM\n{problem}\nEND PROBLEM\n\n"
    "Rewrite the source rollout into the shortest direct, self-contained, correct "
    "solution to the problem. Preserve only reasoning needed to derive the final "
    "answer. Remove every failed branch, retry, repeated calculation, and reflection "
    "phrase such as Wait, reconsider, actually, correction, or start over. Do not "
    "mention the source rollout or omitted text. Do not add analysis about rewriting. "
    "End with exactly one final \\boxed{{...}} answer. Output only the clean solution."
)
_REFLECTION_RE = re.compile(r"\b(wait|actually|reconsider|correction|double-check|start over)\b", re.I)
_MIN_REWRITE_TOKENS = 32


def refine_rollouts(
    cfg: Any,
    tokenizer: Any,
    examples: list[Any],
    rollouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[bool]]:
    """Greedily rewrites each rollout and flags the rewrites that pass the filters.

    A rewrite is accepted if it is shorter than its source rollout, terminates
    before `max_new_tokens`, has at least 32 tokens, contains no reflection
    phrase, and its boxed answer matches the reference.

    Args:
        cfg: Config with `rollout_worker_url` and `max_new_tokens`.
        tokenizer: Tokenizer used to apply the chat template.
        examples: One environment per rollout, in the same order.
        rollouts: On-policy rollouts (one per example).

    Returns:
        `(srcl_rollouts, accepted)`. `srcl_rollouts[i]` pairs the original
        student prompt ids with the rewrite's response ids; `accepted[i]` says
        whether that rewrite is a valid SRCL target.
    """
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": _REWRITE_TEMPLATE.format(
                rollout=r["response"], problem=ex.problem)}],
            tokenize=False, add_generation_prompt=True,
            # Non-thinking mode, so the rewrite is a clean solution rather than a <think> trace.
            enable_thinking=False,
        )
        for ex, r in zip(examples, rollouts)
    ]
    rewrites = generate_rollouts_remote(
        cfg.rollout_worker_url, prompts, num_samples=1,
        max_new_tokens=cfg.max_new_tokens, temperature=0.0, top_k=-1,
    )
    accepted = [
        _MIN_REWRITE_TOKENS <= len(c["response_ids"]) < min(len(r["response_ids"]), cfg.max_new_tokens)
        and not _REFLECTION_RE.search(c["response"])
        and ex.compute_reward(c["response"])[0] > 0
        for ex, r, c in zip(examples, rollouts, rewrites)
    ]
    srcl_rollouts = [
        {"prompt_ids": r["prompt_ids"], "response_ids": c["response_ids"]}
        for r, c in zip(rollouts, rewrites)
    ]
    return srcl_rollouts, accepted


def srcl_backward(
    cfg: Any,
    ctx: DistributedContext,
    student: StudentModel,
    srcl_rollouts: list[dict[str, Any]],
    accepted: list[bool],
) -> float:
    """Backprops `srcl_weight * CE` over accepted rewrite tokens (Eq. 6) and returns the loss.

    Rejected rows stay in the batch with a zero mask rather than being dropped,
    so every FSDP rank runs the same number of forward/backward passes. Gradients
    accumulate into the next optimizer step.
    """
    batch = prepare_batch(
        srcl_rollouts, student.tokenizer, cfg.max_prompt_len, cfg.max_response_len, ctx.device,
    )
    accepted_mask = torch.tensor(accepted, dtype=torch.float, device=ctx.device)
    mask = batch["response_mask"] * accepted_mask[:, None]
    num_tokens = mask.sum().clamp(min=1)  # N_k; clamp keeps the all-rejected case a 0 loss.

    num_seqs, _ = batch["input_ids"].shape
    total_loss = 0.0
    for start in range(0, num_seqs, cfg.train_batch_size):
        rows = slice(start, start + cfg.train_batch_size)
        ids = batch["input_ids"][rows]
        logits = student.get_logits(ids, batch["attention_mask"][rows])
        # Position t predicts token t+1; select response positions before the fp32 cast to save memory.
        target_mask = mask[rows][:, 1:].bool()
        nll = F.cross_entropy(
            logits[:, :-1][target_mask].float(), ids[:, 1:][target_mask], reduction="sum",
        )
        loss = cfg.srcl_weight * nll / num_tokens
        student._scale_loss(loss).backward()
        total_loss += loss.item()
    return total_loss
