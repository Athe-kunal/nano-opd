"""Self-Distillation Policy Optimization (SDPO) training loop — EMA self-teacher."""

import time
from typing import Literal

import torch

from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
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
    # Config — see opd/examples/sdpo.yaml for the full set of hyperparameters
    # (grouped and commented) and `load_config`'s docstring for CLI override syntax.
    cfg = load_config(default_config_path="opd/examples/sdpo.yaml")

    use_wandb = should_use_wandb()

    # -------------------------------------------------------------------------
    # Distributed init — same rank split as OPD:
    #   ranks 0..train_world_size-1  →  student (FSDP)
    #   rank  train_world_size       →  teacher (plain nn.Module, same model)
    ctx = init_distributed(cfg.device_type, cfg.train_world_size)

    print0(f"Model: {cfg.student_model}  (student = teacher, synced via {cfg.sync_method})")
    print_run_banner(ctx, cfg)

    init_wandb(
        cfg.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": cfg.student_model,
            "algorithm": cfg.algorithm,
            "distill_top_k": cfg.distill_top_k,
            "sync_method": cfg.sync_method,
            "ema_alpha": cfg.ema_alpha,
            "trust_region_beta": cfg.trust_region_beta,
            "hard_sync_every_n": cfg.hard_sync_every_n,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "prompts_per_step": cfg.prompts_per_step,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "epochs": cfg.epochs,
            "num_samples": cfg.num_samples,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
        },
    )

    assert_prompts_divisible(cfg.prompts_per_step, ctx.train_world_size)

    # -------------------------------------------------------------------------
    # Model setup
    if ctx.is_student:
        student = build_student_from_args(cfg, ctx)

    if ctx.is_teacher:
        # Same checkpoint as the student; weights will be synced after each step.
        teacher = build_teacher(cfg.student_model)

    # -------------------------------------------------------------------------
    # Teacher syncer — instantiated on all ranks so hyperparameters are visible,
    # but step() is only called on the teacher rank inside sync_student_to_teacher.
    syncer_kwargs: dict = {}
    if cfg.sync_method == "ema":
        syncer_kwargs["alpha"] = cfg.ema_alpha
    elif cfg.sync_method == "trust_region":
        if ctx.is_teacher:
            # Snapshot the initial weights as the regularization anchor.
            syncer_kwargs["initial_params"] = [
                p.data.clone() for p in teacher.model.parameters()
            ]
        else:
            syncer_kwargs["initial_params"] = []   # unused on student ranks
        syncer_kwargs["beta"] = cfg.trust_region_beta
    elif cfg.sync_method == "hard_sync":
        syncer_kwargs["sync_every_n_steps"] = cfg.hard_sync_every_n
    # "on_policy" takes no kwargs

    syncer = build_syncer(cfg.sync_method, **syncer_kwargs)

    # -------------------------------------------------------------------------
    # Loss function and top-K selection
    loss_fn = ALGORITHMS[cfg.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(cfg.algorithm)
    top_k = cfg.distill_top_k

    # -------------------------------------------------------------------------
    # vLLM weight-transfer setup (student ranks only) + trainer construction
    trainer = build_trainer(
        cfg, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        use_wandb,
    )

    # -------------------------------------------------------------------------
    # Dataset (student ranks only)
    if ctx.is_student:
        dataset = build_opd_dataset(cfg.dataset, eval_test_size=cfg.sciknoweval_test_size, seed=cfg.seed)
        loader = distributed_opd_loader(
            dataset, cfg.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=cfg.seed
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
        window_size = accum_window_size(mb, cfg.grad_accum_steps)

        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if ctx.is_student else None, teacher=teacher if ctx.is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=cfg.student_chunk_size, teacher_chunk_size=cfg.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=cfg.algorithm == "mopd_pg_loss",
            tis_clip=cfg.tis_clip, divisor=window_size,
            extra_mask=mb.extra["sd_mask"].unsqueeze(1),
        )

    # -------------------------------------------------------------------------
    # Training loop — all ranks iterate together
    for step in range(cfg.num_steps):
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
            rollouts = generate_rollouts_for_prompts(cfg, prompts, cfg.num_samples)

            # Build feedback-augmented teacher prompts for each rollout.
            # For each question, if any rollout succeeded, pass it as a correct
            # reference for the failed ones.
            for i, env in enumerate(examples):
                group = rollouts[i * cfg.num_samples : (i + 1) * cfg.num_samples]
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
            accum_steps=cfg.grad_accum_steps,
            extra_tensors={"sd_mask": sd_mask} if ctx.is_student else None,
            extra_specs={"sd_mask": torch.float},
            syncer=syncer,
            teacher_sync_scope="step",
        )

        trainer.barrier()

        if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
            if ctx.master_process:
                _ENV_CLS[cfg.dataset].evaluate(
                    rollout_worker_url=cfg.rollout_worker_url,
                    step=step + 1,
                    tokenizer=student.tokenizer,
                    eval_k=cfg.eval_k,
                    eval_max_tokens=cfg.eval_max_tokens,
                    test_size=cfg.sciknoweval_test_size,
                )
            # All ranks wait so non-master ranks don't race ahead into the next
            # step's collectives while rank 0 is still running eval.
            trainer.barrier()

    compute_cleanup()
    finish_wandb(ctx.master_process, use_wandb)
