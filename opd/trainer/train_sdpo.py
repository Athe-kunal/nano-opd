"""Self-Distillation Policy Optimization (SDPO) training loop — EMA self-teacher."""

import argparse
import time
from typing import Literal

import torch

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
from opd.trainer.sync_teacher import SYNC_METHODS, build_syncer
from opd.generator.rollout import generate_rollouts_remote
from opd.envs.dataset import distributed_opd_loader, build_opd_dataset
from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

_ENV_CLS = {
    "dapo_math": DapoMathEnv,
    "livecodebench": LiveCodeBenchEnv,
    "sciknoweval": SciKnowEvalEnv,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_teacher_messages(init_messages, env_output, successful_rollout):
    """Construct the self-teacher prompt following Table 2 of the SDPO paper.

    The teacher sees the original question augmented with:
      - a successful rollout from the group (if any) as a correct reference
      - environment feedback from the current failed attempt (if it failed)

    Conditioning on this richer context lets the same model evaluate its own
    original response from a hindsight perspective, assigning dense logit-level
    credit without an external teacher.
    """
    user_content = init_messages[-1]["content"]
    parts = [user_content]
    has_extra = False
    if successful_rollout is not None:
        parts.append(f"\nCorrect solution:\n{successful_rollout}")
        has_extra = True
    if env_output:
        parts.append(
            f"\nThe following is feedback from your unsuccessful earlier attempt:\n{env_output}"
        )
        has_extra = True
    if has_extra:
        parts.append("\nCorrectly solve the original question.")

    teacher_messages = list(init_messages[:-1])  # preserve system message if any
    teacher_messages.append({"role": "user", "content": "\n".join(parts)})
    return teacher_messages, has_extra


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # CLI
    parser = argparse.ArgumentParser(description="Self-Distillation Policy Optimization (SDPO)")
    # Model — student and teacher share the same checkpoint; teacher is synced
    # to the student after every optimizer step via the chosen sync method.
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--train-world-size", type=int, required=True,
                        help="Number of student (FSDP) ranks. The teacher occupies "
                             "rank train_world_size in the torchrun world.")
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="jsd", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--distill-top-k", type=int, default=100,
                        help="Top-K vocab for KL distillation")
    parser.add_argument("--student-chunk-size", type=int, default=-1)
    parser.add_argument("--teacher-chunk-size", type=int, default=-1)
    parser.add_argument("--tis-clip", type=float, default=0.0,
                        help="TIS importance-weight clip C (0 disables)")
    # Generation
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Completions per prompt (group size G in Algorithm 1)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rollout-worker-url", type=str, default="http://127.0.0.1:8047")
    parser.add_argument("--rollout-worker-world-size", type=int, default=1)
    # Training
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Optimizer step every N minibatches. "
                             "Effective batch size = train_batch_size * grad_accum_steps.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=512,
                        help="Hard cap on prompt tokens. Raises if exceeded.")
    parser.add_argument("--max-response-len", type=int, default=1536,
                        help="Cap on response tokens. Truncates silently if exceeded.")
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    # Teacher sync — controls how the self-teacher tracks the student
    parser.add_argument("--sync-method", type=str, default="ema",
                        choices=list(SYNC_METHODS.keys()),
                        help="How the self-teacher's weights follow the student after each step.")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="[ema] teacher ← α·student + (1−α)·teacher. "
                             "Small α → stable but lagging teacher.")
    parser.add_argument("--trust-region-beta", type=float, default=0.05,
                        help="[trust_region] teacher ← β·student + (1−β)·initial_weights. "
                             "Anchors the teacher to the pre-trained distribution.")
    parser.add_argument("--hard-sync-every-n", type=int, default=100,
                        help="[hard_sync] Full copy every N optimizer steps.")
    # Runtime
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--run-name", type=str, default="dummy")
    parser.add_argument("--save-dir", type=str, default="opd_checkpoints")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-k", type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=4096)
    parser.add_argument("--sciknoweval-test-size", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, required=True, choices=list(_ENV_CLS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    use_wandb = should_use_wandb()

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD:
    #   ranks 0..train_world_size-1  →  student (FSDP)
    #   rank  train_world_size       →  teacher (plain nn.Module, same model)
    ctx = init_distributed(args.device_type, args.train_world_size)

    print0(f"Model: {args.student_model}  (student = teacher, synced via {args.sync_method})")
    print0(f"Algorithm: {args.algorithm}  distill-top-k: {args.distill_top_k}")
    print0(f"Device: {ctx.device}  Student ranks: {ctx.train_world_size}  Total world: {ctx.ddp_world_size}")

    init_wandb(
        args.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": args.student_model,
            "algorithm": args.algorithm,
            "distill_top_k": args.distill_top_k,
            "sync_method": args.sync_method,
            "ema_alpha": args.ema_alpha,
            "trust_region_beta": args.trust_region_beta,
            "hard_sync_every_n": args.hard_sync_every_n,
            "lr": args.lr,
            "num_steps": args.num_steps,
            "prompts_per_step": args.prompts_per_step,
            "train_batch_size": args.train_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "epochs": args.epochs,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    )

    assert_prompts_divisible(args.prompts_per_step, ctx.train_world_size)

    # -------------------------------------------------------------------------
    # Model setup
    if ctx.is_student:
        student = build_student(
            args.student_model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing,
            sharding_strategy=args.sharding_strategy,
            train_world_size=ctx.train_world_size,
            student_group=ctx.student_group,
            total_steps=args.num_steps * args.epochs,
            scheduler_name=args.scheduler,
            warmup_ratio=args.warmup_ratio,
        )

    if ctx.is_teacher:
        # Same checkpoint as the student; weights will be synced after each step.
        teacher = build_teacher(args.student_model)

    # -------------------------------------------------------------------------
    # Teacher syncer — instantiated on all ranks so hyperparameters are visible,
    # but step() is only called on the teacher rank inside sync_student_to_teacher.
    syncer_kwargs: dict = {}
    if args.sync_method == "ema":
        syncer_kwargs["alpha"] = args.ema_alpha
    elif args.sync_method == "trust_region":
        if ctx.is_teacher:
            # Snapshot the initial weights as the regularization anchor.
            syncer_kwargs["initial_params"] = [
                p.data.clone() for p in teacher.model.parameters()
            ]
        else:
            syncer_kwargs["initial_params"] = []   # unused on student ranks
        syncer_kwargs["beta"] = args.trust_region_beta
    elif args.sync_method == "hard_sync":
        syncer_kwargs["sync_every_n_steps"] = args.hard_sync_every_n
    # "on_policy" takes no kwargs

    syncer = build_syncer(args.sync_method, **syncer_kwargs)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    loss_fn = ALGORITHMS[args.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(args.algorithm)
    top_k = args.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only)
    model_update_group = init_vllm_transfer(
        args.rollout_worker_url,
        rollout_worker_world_size=args.rollout_worker_world_size,
        train_world_size=ctx.train_world_size,
        master_process=ctx.master_process,
        all_group=ctx.all_group,
    )

    trainer = Trainer(
        args, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        model_update_group, use_wandb,
    )

    # -------------------------------------------------------------------------
    # Dataset (student ranks only)
    if ctx.is_student:
        dataset = build_opd_dataset(args.dataset, eval_test_size=args.sciknoweval_test_size, seed=args.seed)
        loader = distributed_opd_loader(
            dataset, args.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=args.seed
        )
        loader_iter = iter(loader)

    # -------------------------------------------------------------------------
    # Per-minibatch exchange + loss + backward. The self-teacher uses the same
    # weights as the student but sees a richer context: the question +
    # feedback + original response. stopgrad (no_grad, inside
    # minibatch_exchange) prevents gradients from flowing through the teacher
    # back into the student's computation graph.
    #
    # SDPO-specific: the loss divisor is the actual size of the current
    # grad-accumulation window (the last window may be smaller than G if
    # n_mb % G != 0), and an extra self-distillation mask excludes rollouts
    # where the teacher had no augmented context (teacher == student prompt,
    # so the KL signal is meaningless there).
    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        G = args.grad_accum_steps
        window_start = (mb.mb_idx // G) * G
        window_size = min(window_start + G, mb.n_mb) - window_start

        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if ctx.is_student else None, teacher=teacher if ctx.is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=args.student_chunk_size, teacher_chunk_size=args.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=args.algorithm == "mopd_pg_loss",
            tis_clip=args.tis_clip, divisor=window_size,
            extra_mask=mb.extra["sd_mask"].unsqueeze(1),
        )

    # -------------------------------------------------------------------------
    # Training loop — all ranks iterate together
    for step in range(args.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        rollouts = None
        if ctx.is_student:
            examples, _ = next(loader_iter)
            prompts = [
                student.tokenizer.apply_chat_template(
                    env.init([])[0], tokenize=False, add_generation_prompt=True
                )
                for env in examples
            ]
            rollouts = generate_rollouts_remote(
                args.rollout_worker_url,
                prompts=prompts,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Build feedback-augmented teacher prompts for each rollout.
            # For each question, if any rollout succeeded, pass it as a correct
            # reference for the failed ones.
            for i, env in enumerate(examples):
                group = rollouts[i * args.num_samples : (i + 1) * args.num_samples]
                rewards = [env.compute_reward(r["response"])[0] for r in group]
                successful_text = next(
                    (group[j]["response"] for j, rw in enumerate(rewards) if rw > 0), None
                )
                init_msgs, _ = env.init([])
                if i == 0:
                    print0(f"[debug step={step}] rewards={rewards} has_success={successful_text is not None}", flush=True)
                for j, r in enumerate(group):
                    env_output   = env.get_privileged_information(r["response"]) if rewards[j] == 0 else ""
                    # successful attempts pass their own response as the correct
                    # solution; failed attempts pass a different successful rollout (if any).
                    success_hint = r["response"] if rewards[j] > 0 else successful_text
                    teacher_msgs, has_distillation = _build_teacher_messages(init_msgs, env_output, success_hint)
                    r["has_distillation"] = has_distillation
                    r["teacher_prompt"] = student.tokenizer.apply_chat_template(
                        teacher_msgs, tokenize=False, add_generation_prompt=True
                    )

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=True)

        # 1 for rollouts where the teacher received augmented context (solution or feedback),
        # 0 for rollouts where teacher == student context (distillation signal is meaningless).
        sd_mask = None
        if ctx.is_student:
            sd_mask = torch.tensor(
                [r["has_distillation"] for r in rollouts], dtype=torch.float, device=ctx.device
            )  # [N]

        trainer.step(
            step, t0, batch, teacher_batch, do_minibatch,
            has_teacher_batch=True,
            accum_steps=args.grad_accum_steps,
            extra_tensors={"sd_mask": sd_mask} if ctx.is_student else None,
            extra_specs={"sd_mask": torch.float},
            syncer=syncer,
            teacher_sync_scope="step",
        )

        trainer.barrier()

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            if ctx.master_process:
                _ENV_CLS[args.dataset].evaluate(
                    rollout_worker_url=args.rollout_worker_url,
                    step=step + 1,
                    tokenizer=student.tokenizer,
                    eval_k=args.eval_k,
                    eval_max_tokens=args.eval_max_tokens,
                    test_size=args.sciknoweval_test_size,
                )
            # All ranks wait so non-master ranks don't race ahead into the next
            # step's collectives while rank 0 is still running eval.
            trainer.barrier()

    compute_cleanup()
    finish_wandb(ctx.master_process, use_wandb)
