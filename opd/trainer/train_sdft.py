"""Self-Distillation Fine-Tuning (SDFT) training loop — EMA self-teacher, demonstration-conditioned."""

import json
import random
import time
from typing import Literal

import torch

from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, log_eval_metrics, should_use_wandb
from opd.trainer.self_distillation_utils import self_distill_minibatch
from opd.trainer.setup_utils import (
    accum_window_size,
    assert_prompts_divisible,
    build_student_from_args,
    build_teacher,
    compute_cleanup,
    generate_rollouts_for_prompts,
    init_distributed,
    load_config,
    print0,
    print_run_banner,
    topk_selector_for,
)
from opd.trainer.models import MinibatchTensors, StepAccumulator
from opd.trainer.trainer_utils import build_trainer
from opd.trainer.sync_teacher import SYNC_METHODS, build_syncer
from opd.envs.dataset import distributed_opd_loader
from opd.envs.sdft_science import SdftScienceEnv, load_sdft_science, load_sdft_science_eval, grade_science_response
from opd.envs.sdft_tooluse import SdftToolUseEnv, load_sdft_tooluse, load_sdft_tooluse_eval, grade_tooluse_response

_ENV_CLS = {"science": SdftScienceEnv, "tooluse": SdftToolUseEnv}
_EVAL_LOADERS = {"science": load_sdft_science_eval, "tooluse": load_sdft_tooluse_eval}
_GRADERS = {
    "science": lambda response, ex: grade_science_response(response, ex["answer"]),
    "tooluse": lambda response, ex: grade_tooluse_response(response, ex["golden_answer"]),
}


# ---------------------------------------------------------------------------
# Teacher prompt construction
# ---------------------------------------------------------------------------

# Template from SDFT paper Section 3. The teacher sees the question AND a
# worked demonstration; this in-context conditioning shifts the teacher's
# distribution toward the expert behaviour while keeping it anchored to the
# pretrained base — closer to the base model than SFT, but task-capable.
_TEACHER_TEMPLATE = (
    "{question}\n"
    "This is an example for a response to the question:\n"
    "{demonstration}\n"
    "Now answer with a response of your own, including the thinking process:"
)


def _build_teacher_messages(
    student_messages: list[dict],
    demonstration: str,
) -> list[dict]:
    """Construct the demonstration-conditioned teacher prompt (SDFT paper, Section 3).

    The student sees: system (optional) + user(question)
    The teacher sees: system (optional) + user(question + demonstration template)

    We splice the demonstration into the *last* user turn so the chat template
    renders correctly regardless of whether there is a system message.
    """
    user_content = student_messages[-1]["content"]
    teacher_user = _TEACHER_TEMPLATE.format(
        question=user_content,
        demonstration=demonstration,
    )
    teacher_messages = list(student_messages[:-1])
    teacher_messages.append({"role": "user", "content": teacher_user})
    return teacher_messages



# ---------------------------------------------------------------------------
# Loss masking helper
# ---------------------------------------------------------------------------


def apply_token_skip_mask(compact_mask: torch.Tensor, num_skip: int) -> torch.Tensor:
    """Zero out the first `num_skip` response tokens in every sequence.

    SDFT (Section 5, Learned Artifacts) finds that the teacher sometimes
    outputs preambles like "Based on the text..." which the student learns
    to mimic even without the demonstration context. Masking the first few
    response tokens suppresses this artifact.

    Args:
        compact_mask: [B, R_max] float mask (1 = real response token).
        num_skip:     number of leading response tokens to exclude from loss.

    Returns:
        Modified [B, R_max] mask with first `num_skip` positions set to 0.
    """
    if num_skip <= 0:
        return compact_mask
    skip = torch.zeros_like(compact_mask)
    _, seq_len = compact_mask.shape
    if seq_len > num_skip:
        skip[:, num_skip:] = 1.0
    return compact_mask * skip


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_pass_at_k(
    eval_dataset: list[dict],
    eval_size: int,
    cfg,
    student,
    grader,
    k: int = 4,
) -> tuple[float, float]:
    """Pass@k on the held-out eval split. Runs on master rank only (no collectives).

    Generates k rollouts per question, grades each with `grader(response, example)`,
    and returns (pass@k, avg@k) where:
      pass@k  = fraction of questions with at least 1 correct rollout
      avg@k   = average fraction of correct rollouts per question
    """
    rng = random.Random(cfg.seed)
    idxs = list(range(len(eval_dataset)))
    rng.shuffle(idxs)
    examples = [eval_dataset[i] for i in idxs[:eval_size]]

    # Eval loaders supply a full [system, user] conversation — the system turn
    # carries the output-format instruction the graders parse.
    prompts = [student.tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=True,
    ) for ex in examples]

    rollouts = generate_rollouts_for_prompts(cfg, prompts, num_samples=k)

    pass_count = 0
    total_correct = 0
    for i, ex in enumerate(examples):
        group = rollouts[i * k : (i + 1) * k]
        correct = [grader(r["response"], ex) for r in group]
        pass_count += any(correct)
        total_correct += sum(correct)

    n = len(examples)
    return pass_count / n, total_correct / (n * k)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_sdft_dataset(cfg) -> list[SdftScienceEnv | SdftToolUseEnv]:
    """Load the SDFT training dataset from a built-in source or a custom JSONL.

    Returns a list of env instances consumed by distributed_opd_loader, each
    exposing .question and get_privileged_information() (the demonstration).
    The --dataset-path flag takes precedence over --dataset so users can drop
    in arbitrary JSONL files without changing code; --dataset still selects
    which env class (science vs. tooluse) wraps those records.
    """
    env_cls = _ENV_CLS[cfg.dataset]
    if cfg.dataset_path:
        with open(cfg.dataset_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif cfg.dataset == "science":
        records = load_sdft_science(split="train")
    elif cfg.dataset == "tooluse":
        records = load_sdft_tooluse(split="train")
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
    if cfg.dataset_limit > 0:
        records = records[: cfg.dataset_limit]
    return [env_cls(question=r["question"], demonstration=r["demonstration"]) for r in records]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Config — see opd/examples/sdft.yaml for the full set of hyperparameters
    # (grouped and commented) and `load_config`'s docstring for CLI override syntax.
    cfg = load_config(default_config_path="opd/examples/sdft.yaml")

    use_wandb = should_use_wandb()

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD/SDPO:
    #   ranks 0..train_world_size-1  ->  student (FSDP)
    #   rank  train_world_size       ->  teacher (plain nn.Module, EMA of student)
    ctx = init_distributed(cfg.device_type, cfg.train_world_size)

    print0(
        f"Model: {cfg.student_model}  (student = teacher, synced via {cfg.sync_method})"
    )
    print_run_banner(ctx, cfg)
    print0(f"Loss token skip: {cfg.num_loss_tokens_to_skip}")

    init_wandb(
        cfg.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": cfg.student_model,
            "algorithm": cfg.algorithm,
            "distill_top_k": cfg.distill_top_k,
            "sync_method": cfg.sync_method,
            "ema_alpha": cfg.ema_alpha,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "prompts_per_step": cfg.prompts_per_step,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "epochs": cfg.epochs,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "num_loss_tokens_to_skip": cfg.num_loss_tokens_to_skip,
        },
    )

    assert_prompts_divisible(cfg.prompts_per_step, ctx.train_world_size)

    # -------------------------------------------------------------------------
    # Model setup
    if ctx.is_student:
        student = build_student_from_args(cfg, ctx)

    if ctx.is_teacher:
        # Same checkpoint as the student; weights will be EMA-synced after each step.
        teacher = build_teacher(cfg.student_model)

    # -------------------------------------------------------------------------
    # Teacher syncer
    syncer_kwargs: dict = {}
    if cfg.sync_method == "ema":
        syncer_kwargs["alpha"] = cfg.ema_alpha
    elif cfg.sync_method == "trust_region":
        if ctx.is_teacher:
            syncer_kwargs["initial_params"] = [
                p.data.clone() for p in teacher.model.parameters()
            ]
        else:
            syncer_kwargs["initial_params"] = []
        syncer_kwargs["beta"] = cfg.trust_region_beta
    elif cfg.sync_method == "hard_sync":
        syncer_kwargs["sync_every_n_steps"] = cfg.hard_sync_every_n

    syncer = build_syncer(cfg.sync_method, **syncer_kwargs)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    # SDFT uses reverse KL by default: KL(π_student || π_teacher).
    # For reverse KL, the student selects the top-K indices (the student-weighted
    # sum means we only need tokens where the student has mass).
    loss_fn = ALGORITHMS[cfg.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(cfg.algorithm)
    top_k = cfg.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup + trainer construction
    trainer = build_trainer(
        cfg, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        use_wandb,
    )

    # -------------------------------------------------------------------------
    # Dataset (student ranks only)
    eval_dataset: list[dict] = []
    grader = None
    if ctx.is_student:
        dataset = build_sdft_dataset(cfg)
        if cfg.eval_every > 0:
            # Custom JSONL datasets (--dataset-path) have no held-out eval split.
            eval_dataset = [] if cfg.dataset_path else _EVAL_LOADERS[cfg.dataset]()
            grader = _GRADERS[cfg.dataset]
        loader = distributed_opd_loader(
            dataset, cfg.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=cfg.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Per-minibatch exchange + loss + backward.
    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        window_size = accum_window_size(mb, cfg.grad_accum_steps)

        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if ctx.is_student else None, teacher=teacher if ctx.is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=cfg.student_chunk_size, teacher_chunk_size=cfg.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=cfg.algorithm == "mopd_pg_loss",
            tis_clip=cfg.tis_clip, divisor=window_size,
            mask_fn=lambda m: apply_token_skip_mask(m, cfg.num_loss_tokens_to_skip),
        )

    # -------------------------------------------------------------------------
    # Training loop
    for step in range(cfg.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        rollouts = None
        if ctx.is_student:
            examples, _ = next(loader_iter)  # list[SdftScienceEnv | SdftToolUseEnv], state dict

            # Student prompt: question only — π_θ(y|x)
            prompts = [
                student.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex.question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in examples
            ]

            # SDFT: single on-policy trajectory per prompt
            rollouts = generate_rollouts_for_prompts(cfg, prompts, num_samples=1)

            # Attach demonstration-conditioned teacher prompt to each rollout.
            # The teacher sees: question + worked demonstration -> richer context
            # than the student, which sees only the question.
            for i, ex in enumerate(examples):
                r = rollouts[i]  # single rollout per prompt (num_samples=1)
                student_msgs = [{"role": "user", "content": ex.question}]
                teacher_msgs = _build_teacher_messages(
                    student_msgs, ex.get_privileged_information(r["response"])
                )
                r["teacher_prompt"] = student.tokenizer.apply_chat_template(
                    teacher_msgs, tokenize=False, add_generation_prompt=True
                )

            if step == 0:
                print0(
                    f"[debug step=0] example teacher prompt snippet:\n"
                    f"{rollouts[0]['teacher_prompt'][:300]}",
                    flush=True,
                )

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=True)

        # -- EMA sync: after each epoch's optimizer step(s), teacher tracks
        # student. This is the core SDFT mechanism: teacher φ ← α·θ + (1−α)·φ
        trainer.step(
            step, t0, batch, teacher_batch, do_minibatch,
            has_teacher_batch=True,
            accum_steps=cfg.grad_accum_steps,
            syncer=syncer,
            teacher_sync_scope="epoch",
        )

        # -- Pass@k eval (master rank only) --
        if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0 and ctx.master_process and eval_dataset:
            k = cfg.eval_k
            pass_at_k, avg_at_k = eval_pass_at_k(eval_dataset, len(eval_dataset), cfg, student, grader, k=k)
            print0(f"[eval step={step+1}] pass@{k}={pass_at_k:.3f} avg@{k}={avg_at_k:.3f} (n={len(eval_dataset)})")
            log_eval_metrics(
                {
                    f"eval/{cfg.dataset}/pass@{k}": pass_at_k,
                    f"eval/{cfg.dataset}/avg@{k}": avg_at_k,
                },
                step + 1, ctx.master_process, use_wandb,
            )

        trainer.barrier()

    compute_cleanup()
    finish_wandb(ctx.master_process, use_wandb)
