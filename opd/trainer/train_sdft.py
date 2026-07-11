"""Self-Distillation Fine-Tuning (SDFT) training loop — EMA self-teacher, demonstration-conditioned."""

import argparse
import json
import random
import time
from typing import Literal

import torch

from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, log_eval_metrics, should_use_wandb
from opd.trainer.self_distillation_utils import self_distill_minibatch
from opd.trainer.setup_utils import (
    assert_prompts_divisible,
    build_student,
    build_teacher,
    compute_cleanup,
    init_distributed,
    init_vllm_transfer,
    print0,
    topk_selector_for,
)
from opd.trainer.trainer_utils import MinibatchTensors, StepAccumulator, Trainer
from opd.trainer.sync_teacher import SYNC_METHODS, build_syncer
from opd.generator.rollout import generate_rollouts_remote
from opd.envs.dataset import distributed_opd_loader
from opd.envs.sdft_science import load_sdft_science, load_sdft_science_eval, grade_science_response
from opd.envs.sdft_tooluse import load_sdft_tooluse, load_sdft_tooluse_eval, grade_tooluse_response


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
    teacher_messages = list(student_messages[:-1])  # preserve system message if any
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
    if compact_mask.shape[1] > num_skip:
        skip[:, num_skip:] = 1.0
    return compact_mask * skip


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_pass_at_k(
    eval_dataset: list[dict],
    eval_size: int,
    args,
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
    rng = random.Random(args.seed)
    idxs = list(range(len(eval_dataset)))
    rng.shuffle(idxs)
    examples = [eval_dataset[i] for i in idxs[:eval_size]]

    prompts = [student.tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["question"]}], tokenize=False, add_generation_prompt=True,
    ) for ex in examples]

    rollouts = generate_rollouts_remote(
        args.rollout_worker_url, prompts=prompts, num_samples=k,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
    )

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


def build_sdft_eval_dataset(args) -> list[dict]:
    """Load the held-out eval split for pass@k evaluation.

    Returns [] for custom JSONL datasets (no eval split available).
    """
    if args.dataset_path:
        return []
    if args.dataset == "science":
        return load_sdft_science_eval()
    if args.dataset == "tooluse":
        return load_sdft_tooluse_eval()
    return []


def build_sdft_grader(args):
    """Return a grader callable (response, example) -> bool for the dataset."""
    if args.dataset == "science":
        return lambda response, ex: grade_science_response(response, ex["answer"])
    if args.dataset == "tooluse":
        return lambda response, ex: grade_tooluse_response(response, ex["golden_answer"])
    return lambda response, ex: False


def build_sdft_dataset(args) -> list[dict]:
    """Load the SDFT training dataset from a built-in source or a custom JSONL.

    Returns a list of {"question": str, "demonstration": str} dicts consumed
    by distributed_opd_loader. The --dataset-path flag takes precedence over
    --dataset so users can drop in arbitrary JSONL files without changing code.
    """
    if args.dataset_path:
        with open(args.dataset_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif args.dataset == "science":
        records = load_sdft_science(split="train")
    elif args.dataset == "tooluse":
        records = load_sdft_tooluse(split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")
    if args.dataset_limit > 0:
        records = records[: args.dataset_limit]
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(
        description="Self-Distillation Fine-Tuning (SDFT) — on-policy distillation "
        "from a demonstration-conditioned self-teacher."
    )
    # Model
    parser.add_argument(
        "--student-model",
        type=str,
        required=True,
        help="HuggingFace model ID or path. Used for both student and (EMA) teacher.",
    )
    parser.add_argument(
        "--train-world-size",
        type=int,
        required=True,
        help="Number of student (FSDP) ranks. The teacher occupies "
        "rank train_world_size in the torchrun world.",
    )
    # Dataset — built-in sources pulled from the Self-Distillation GitHub repo,
    # or a custom JSONL on disk.
    parser.add_argument(
        "--dataset",
        type=str,
        default="science",
        choices=["science", "tooluse"],
        help="Built-in dataset to pull from the Self-Distillation GitHub repo. "
        "'science': science Q&A with worked demonstrations. "
        "'tooluse': tool-use tasks with golden Action/Action_Input sequences.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Optional override: path to a custom JSONL, one record "
        'per line {"question": ..., "demonstration": ...}. '
        "Takes precedence over --dataset when set.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=0,
        help="Cap the number of training examples (0 = use all).",
    )
    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        default="reverse_kl",
        choices=list(ALGORITHMS.keys()),
        help="Distillation loss. SDFT paper uses reverse_kl (Eq. 1).",
    )
    parser.add_argument(
        "--distill-top-k",
        type=int,
        default=100,
        help="Top-K vocab for KL distillation. Larger K is more faithful "
        "but uses more memory and bandwidth. NOTE: the SDFT paper's "
        "analytic per-token KL estimator (Appendix A.1) sums over the FULL "
        "vocabulary V; top-K is a nano-opd memory/bandwidth approximation, "
        "not part of the paper. Raise K toward V to reduce the truncation bias.",
    )
    parser.add_argument("--student-chunk-size", type=int, default=-1)
    parser.add_argument("--teacher-chunk-size", type=int, default=-1)
    parser.add_argument(
        "--tis-clip",
        type=float,
        default=0.0,
        help="TIS importance-weight clip C (0 disables). Corrects for "
        "log-prob gap between vLLM inference and training forward pass.",
    )
    # Loss masking — suppress 'Based on the text...' preamble artifacts
    parser.add_argument(
        "--num-loss-tokens-to-skip",
        type=int,
        default=5,
        help="Mask the first N response tokens from the distillation loss. "
        "SDFT paper (Section 5, 'Learned Artifacts') masks 'the first few "
        "tokens' to suppress preamble artifacts (e.g. 'Based on the text...') "
        "the student mimics from the demonstration-conditioned teacher; the "
        "paper does not give an exact count, so N=5 is a nano-opd default.",
    )
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--rollout-worker-url", type=str, default="http://127.0.0.1:8047"
    )
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument(
        "--prompts-per-step",
        type=int,
        default=8,
        help="Number of distinct (question, demonstration) pairs per step. "
        "Unlike SDPO, there is no num-samples multiplier: each pair "
        "produces exactly one on-policy rollout.",
    )
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Each minibatch is split into this many "
        "micro batches; gradients accumulate before a single optimizer step. "
        "Effective batch size = train-batch-size * grad-accum-steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Optimizer steps taken on the SAME rollout batch before collecting "
        "new rollouts (PPO-style inner reuse). WARNING: this is NOT the same "
        "as the SDFT paper's 'epochs' (Tables 3/4: 2 for Skill Learning, 4 for "
        "Knowledge Acquisition), which are full passes over the dataset that "
        "RE-ROLL fresh on-policy trajectories each pass. epochs>1 here reuses "
        "stale rollouts and is mildly off-policy (--tis-clip partially corrects "
        "it). To replicate the paper's epochs, raise --num-steps instead.",
    )
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-response-len", type=int, default=1536)
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    # Teacher sync
    parser.add_argument(
        "--sync-method",
        type=str,
        default="ema",
        choices=list(SYNC_METHODS.keys()),
        help="How the EMA teacher tracks the student. SDFT paper uses EMA "
        "(alpha in {0.01, 0.02, 0.05} per Table 3).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.02,
        help="[ema] teacher ← α·student + (1−α)·teacher. "
        "Small α → stable but lagging teacher. "
        "SDFT paper sweeps {0.01, 0.02, 0.05}.",
    )
    parser.add_argument(
        "--trust-region-beta",
        type=float,
        default=0.05,
        help="[trust_region] teacher ← β·student + (1−β)·initial_weights.",
    )
    parser.add_argument(
        "--hard-sync-every-n",
        type=int,
        default=100,
        help="[hard_sync] Full copy every N optimizer steps.",
    )
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="sdft_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0, help="Pass@k eval on held-out split every N steps (0 = disabled).")
    parser.add_argument("--eval-k", type=int, default=4, help="Number of rollouts per question for pass@k evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = should_use_wandb()

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD/SDPO:
    #   ranks 0..train_world_size-1  →  student (FSDP)
    #   rank  train_world_size       →  teacher (plain nn.Module, EMA of student)
    ctx = init_distributed(args.device_type, args.train_world_size)
    ddp_rank = ctx.ddp_rank
    ddp_world_size = ctx.ddp_world_size
    device = ctx.device
    train_world_size = ctx.train_world_size
    is_student = ctx.is_student
    is_teacher = ctx.is_teacher
    master_process = ctx.master_process
    student_group = ctx.student_group
    all_group = ctx.all_group

    print0(
        f"Model: {args.student_model}  (student = teacher, synced via {args.sync_method})"
    )
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(
        f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}"
    )
    print0(f"Loss token skip: {args.num_loss_tokens_to_skip}")

    init_wandb(
        args.run_name, master_process, use_wandb,
        config={
            "student_model": args.student_model,
            "algorithm": args.algorithm,
            "distill_top_k": args.distill_top_k,
            "sync_method": args.sync_method,
            "ema_alpha": args.ema_alpha,
            "lr": args.lr,
            "num_steps": args.num_steps,
            "prompts_per_step": args.prompts_per_step,
            "train_batch_size": args.train_batch_size,
            "epochs": args.epochs,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "num_loss_tokens_to_skip": args.num_loss_tokens_to_skip,
        },
    )

    assert_prompts_divisible(args.prompts_per_step, train_world_size)

    # -------------------------------------------------------------------------
    # Model setup
    if is_student:
        student = build_student(
            args.student_model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            sharding_strategy=args.sharding_strategy,
            train_world_size=train_world_size,
            student_group=student_group,
            total_steps=args.num_steps * args.epochs,
        )

    if is_teacher:
        # Same checkpoint as the student; weights will be EMA-synced after each step.
        teacher = build_teacher(args.student_model)

    # -------------------------------------------------------------------------
    # Teacher syncer
    syncer_kwargs: dict = {}
    if args.sync_method == "ema":
        syncer_kwargs["alpha"] = args.ema_alpha
    elif args.sync_method == "trust_region":
        if is_teacher:
            syncer_kwargs["initial_params"] = [
                p.data.clone() for p in teacher.model.parameters()
            ]
        else:
            syncer_kwargs["initial_params"] = []
        syncer_kwargs["beta"] = args.trust_region_beta
    elif args.sync_method == "hard_sync":
        syncer_kwargs["sync_every_n_steps"] = args.hard_sync_every_n

    syncer = build_syncer(args.sync_method, **syncer_kwargs)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    # SDFT uses reverse KL by default (paper Eq. 1): KL(π_student || π_teacher).
    # For reverse KL, the student selects the top-K indices (the student-weighted
    # sum means we only need tokens where the student has mass).
    loss_fn = ALGORITHMS[args.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(args.algorithm)
    top_k = args.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup
    model_update_group = init_vllm_transfer(
        args.rollout_worker_url,
        rollout_worker_world_size=args.rollout_worker_world_size,
        train_world_size=train_world_size,
        master_process=master_process,
        all_group=all_group,
    )

    trainer = Trainer(
        args, ctx,
        student if is_student else None, teacher if is_teacher else None,
        model_update_group, use_wandb,
    )

    # -------------------------------------------------------------------------
    # Dataset (student ranks only)
    eval_dataset: list[dict] = []
    grader = None
    if is_student:
        dataset = build_sdft_dataset(args)
        if args.eval_every > 0:
            eval_dataset = build_sdft_eval_dataset(args)
            grader = build_sdft_grader(args)
        loader = distributed_opd_loader(
            dataset, args.prompts_per_step, train_world_size, ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Per-minibatch exchange + loss + backward.
    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        # How many minibatches are in this accumulation window? The last
        # window may be smaller than G if n_mb % G != 0 — same adaptive-window
        # divisor SDPO uses, so a short final window isn't under-weighted.
        G = args.grad_accum_steps
        window_start = (mb.mb_idx // G) * G
        window_size = min(window_start + G, mb.n_mb) - window_start

        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if is_student else None, teacher=teacher if is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=args.student_chunk_size, teacher_chunk_size=args.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=args.algorithm == "mopd_pg_loss",
            tis_clip=args.tis_clip, divisor=window_size,
            mask_fn=lambda m: apply_token_skip_mask(m, args.num_loss_tokens_to_skip),
        )

    # -------------------------------------------------------------------------
    # Training loop
    for step in range(args.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        rollouts = None
        if is_student:
            examples, _ = next(loader_iter)  # list of {question, demonstration}, state dict

            # Student prompt: question only — π_θ(y|x)
            prompts = [
                student.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in examples
            ]

            rollouts = generate_rollouts_remote(
                args.rollout_worker_url,
                prompts=prompts,
                num_samples=1,  # SDFT: single on-policy trajectory per prompt
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Attach demonstration-conditioned teacher prompt to each rollout.
            # The teacher sees: question + worked demonstration → richer context
            # than the student, which sees only the question.
            for i, ex in enumerate(examples):
                r = rollouts[i]  # single rollout per prompt (num_samples=1)
                student_msgs = [{"role": "user", "content": ex["question"]}]
                teacher_msgs = _build_teacher_messages(
                    student_msgs, ex["demonstration"]
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
            accum_steps=args.grad_accum_steps,
            syncer=syncer,
            teacher_sync_scope="epoch",
        )

        # -- Pass@k eval (master rank only) --
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0 and master_process and eval_dataset:
            k = args.eval_k
            pass_at_k, avg_at_k = eval_pass_at_k(eval_dataset, len(eval_dataset), args, student, grader, k=k)
            print0(f"[eval step={step+1}] pass@{k}={pass_at_k:.3f} avg@{k}={avg_at_k:.3f} (n={len(eval_dataset)})")
            log_eval_metrics(
                {f"eval/pass_at_{k}": pass_at_k, f"eval/avg_at_{k}": avg_at_k},
                step + 1, master_process, use_wandb,
            )

        trainer.barrier()

    compute_cleanup()
    finish_wandb(master_process, use_wandb)
