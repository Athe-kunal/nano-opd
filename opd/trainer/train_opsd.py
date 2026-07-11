"""On-Policy Self-Distillation (OPSD) training loop — frozen reference-solution teacher."""

import argparse
import time
from typing import Literal

from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
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
from opd.envs.opsd_dataset import OPSDMathEnv
from opd.envs.dataset import distributed_opd_loader
from opd.generator.rollout import generate_rollouts_remote


# ---------------------------------------------------------------------------
# Teacher prompt construction (Figure 2 of the OPSD paper)
# ---------------------------------------------------------------------------

# The teacher sees the problem AND the ground-truth reference solution y*.
# This follows Figure 2 of the paper exactly: after reading the reference
# solution the teacher is asked to solve the problem in its own way — this
# rationalization is done implicitly through a single forward pass (no
# generation), so the teacher never actually produces new tokens here.
_STUDENT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # not passed through .format()

_TEACHER_TEMPLATE = (
    "{problem}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n"
    "{solution}\n"
    "=== Reference Solution End ===\n\n"
    "After reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def _build_teacher_messages(
    student_messages: list[dict],
    solution: str,
) -> list[dict]:
    """Construct the reference-conditioned teacher prompt (OPSD paper, Figure 2).

    Student sees: system (optional) + user(problem)
    Teacher sees: system (optional) + user(problem + reference solution template)

    Splicing into the last user turn preserves the chat template structure
    regardless of whether a system message is present.
    """
    problem_content = student_messages[-1]["content"]
    teacher_user = _TEACHER_TEMPLATE.format(
        problem=problem_content,
        solution=solution,
    )
    teacher_messages = list(student_messages[:-1])   # preserve system message if any
    teacher_messages.append({"role": "user", "content": teacher_user})
    return teacher_messages



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(
        description="On-Policy Self-Distillation (OPSD) — a single LLM acts as "
                    "both student (sees problem only) and teacher (sees problem + "
                    "reference solution). The teacher is the frozen initial policy."
    )
    # Model — student and teacher start from the same checkpoint; teacher is frozen
    parser.add_argument("--student-model", type=str, required=True,
                        help="HuggingFace model ID or path. Used for both student "
                             "(updated) and teacher (frozen initial policy).")
    parser.add_argument("--train-world-size", type=int, required=True,
                        help="Number of student (FSDP) ranks. The teacher occupies "
                             "rank train_world_size in the torchrun world.")
    # Dataset — siyanzhao/Openthoughts_math_30k_opsd (hardcoded)
    parser.add_argument("--dataset-id", type=str,
                        default="siyanzhao/Openthoughts_math_30k_opsd",
                        help="HuggingFace dataset ID for OPSD training.")
    parser.add_argument("--dataset-split", type=str, default="train",
                        help="HuggingFace split to load.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="forward_kl",
                        choices=list(ALGORITHMS.keys()),
                        help="Distillation loss. OPSD paper (Table 3) finds forward KL "
                             "KL(p_T || p_S) consistently outperforms reverse KL and JSD.")
    parser.add_argument("--distill-top-k", type=int, default=100,
                        help="Top-K vocab for KL distillation. Larger K is more faithful "
                             "but uses more memory and bandwidth.")
    parser.add_argument("--student-chunk-size", type=int, default=-1)
    parser.add_argument("--teacher-chunk-size", type=int, default=-1)
    parser.add_argument("--tis-clip", type=float, default=0.0,
                        help="TIS importance-weight clip C (0 disables). Corrects for "
                             "log-prob gap between vLLM inference and training forward pass.")
    parser.add_argument("--kl-clip", type=float, default=0.0,
                        help="Per-token pointwise KL clip τ (0 disables). Clips each "
                             "token's divergence contribution to prevent stylistic tokens "
                             "from dominating the gradient signal (OPSD paper Section 3.2 "
                             "and Figure 4). Strongly recommended — the paper shows this "
                             "prevents performance collapse on Qwen3-1.7B.")
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100,
                        help="OPSD paper converges within 100 gradient update steps.")
    parser.add_argument("--prompts-per-step", type=int, default=8,
                        help="Number of distinct (problem, solution) pairs per step. "
                             "Each pair produces exactly one on-policy rollout (num_samples=1).")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Optimizer steps per rollout batch before collecting new rollouts.")
    parser.add_argument("--max-prompt-len", type=int, default=512,
                        help="Hard cap on prompt tokens. Raises if exceeded.")
    parser.add_argument("--max-response-len", type=int, default=1536,
                        help="Cap on response tokens. Truncates silently if exceeded.")
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="LR scheduler. 'constant' disables warmup/decay entirely.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Fraction of total steps used for LR warmup.")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="opsd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = should_use_wandb()

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD/SDFT:
    #   ranks 0..train_world_size-1  →  student (FSDP, updated by optimizer)
    #   rank  train_world_size       →  teacher (plain nn.Module, frozen initial policy)
    ctx = init_distributed(args.device_type, args.train_world_size)
    ddp_rank            = ctx.ddp_rank
    ddp_world_size      = ctx.ddp_world_size
    device              = ctx.device
    train_world_size    = ctx.train_world_size
    is_student          = ctx.is_student
    is_teacher          = ctx.is_teacher
    master_process      = ctx.master_process
    student_group       = ctx.student_group
    all_group           = ctx.all_group

    print0(f"Model: {args.student_model}  (teacher = frozen initial policy)")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {device}  Student ranks: {train_world_size}  Total world: {ddp_world_size}")
    if args.kl_clip > 0.0:
        print0(f"Per-token KL clip: {args.kl_clip}")

    init_wandb(
        args.run_name, master_process, use_wandb,
        config={
            "student_model": args.student_model,
            "algorithm": args.algorithm,
            "distill_top_k": args.distill_top_k,
            "kl_clip": args.kl_clip,
            "lr": args.lr,
            "num_steps": args.num_steps,
            "prompts_per_step": args.prompts_per_step,
            "train_batch_size": args.train_batch_size,
            "epochs": args.epochs,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
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
            scheduler_name=args.scheduler,
            warmup_ratio=args.warmup_ratio,
        )

    if is_teacher:
        # Frozen initial policy — weights are never updated after this load.
        # The paper (Section 4.1) finds that fixing the teacher to the initial
        # policy stabilises training and acts as an implicit regulariser that
        # prevents excessive deviation from the pretrained distribution.
        teacher = build_teacher(args.student_model)
        print(f"[teacher] Loaded initial policy from {args.student_model} (frozen)", flush=True)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    # OPSD paper (Table 3) recommends forward KL: KL(p_T || p_S).
    # For forward KL the teacher selects the top-K indices (the teacher-weighted
    # sum means we need tokens where the teacher has non-negligible probability).
    loss_fn = ALGORITHMS[args.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(args.algorithm)
    top_k = args.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only)
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
    if is_student:
        dataset     = OPSDMathEnv.load(split=args.dataset_split, dataset_id=args.dataset_id)
        loader      = distributed_opd_loader(
            dataset, args.prompts_per_step, train_world_size, ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Per-minibatch exchange + loss + backward. The teacher p_T(· | x, y*)
    # conditions on both the problem and the reference solution. Gradients
    # must NOT flow through the teacher — it acts as a fixed target
    # distribution. 
    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        # Per-token pointwise KL clipping (OPSD paper Section 3.2). Stylistic
        # tokens can exhibit much higher KL than math tokens, dominating the
        # gradient signal. Clipping each token's divergence contribution to τ
        # stabilises training and prevents performance collapse, especially
        # for smaller models (Figure 4).
        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if is_student else None, teacher=teacher if is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=args.student_chunk_size, teacher_chunk_size=args.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=args.algorithm == "mopd_pg_loss",
            tis_clip=args.tis_clip, divisor=mb.n_mb,
            kl_clip=args.kl_clip if args.kl_clip > 0.0 else None,
        )

    # -------------------------------------------------------------------------
    # Training loop — all ranks iterate together
    for step in range(args.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        rollouts = None
        if is_student:
            examples, _ = next(loader_iter)   # list[OPSDMathEnv], state_dict

            # Student prompt: problem only — p_S(· | x)
            prompts = [
                student.tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex.problem + _STUDENT_SUFFIX}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in examples
            ]

            rollouts = generate_rollouts_remote(
                args.rollout_worker_url,
                prompts=prompts,
                num_samples=1,        # OPSD: single on-policy trajectory per prompt
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Attach reference-conditioned teacher prompt to each rollout.
            # The teacher sees: problem + ground-truth solution y* → richer
            # context than the student (problem only), following Figure 2.
            for i, ex in enumerate(examples):
                r            = rollouts[i]    # one rollout per prompt (num_samples=1)
                student_msgs = [{"role": "user", "content": ex.problem + _STUDENT_SUFFIX}]
                teacher_msgs = _build_teacher_messages(
                    [{"role": "user", "content": ex.problem}], ex.solution
                )
                r["teacher_prompt"] = student.tokenizer.apply_chat_template(
                    teacher_msgs, tokenize=False, add_generation_prompt=True
                )

            if step == 0:
                print0(
                    f"[debug step=0] teacher prompt snippet:\n"
                    f"{rollouts[0]['teacher_prompt'][:400]}",
                    flush=True,
                )

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=True)

        trainer.step(step, t0, batch, teacher_batch, do_minibatch, has_teacher_batch=True)

        trainer.barrier()

    compute_cleanup()
    finish_wandb(master_process, use_wandb)
