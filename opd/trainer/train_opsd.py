import time
from typing import Literal

from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
from opd.trainer.self_distillation_utils import self_distill_minibatch
from opd.trainer.setup_utils import (
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
from opd.envs.opsd_dataset import OPSDMathEnv
from opd.envs.dataset import distributed_opd_loader


# The teacher sees the problem AND the ground-truth reference solution y*.
# After reading the reference solution the teacher is asked to solve the problem in its own way — this
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
    teacher_messages = list(student_messages[:-1])
    teacher_messages.append({"role": "user", "content": teacher_user})
    return teacher_messages

if __name__ == "__main__":

    # Config — see opd/examples/opsd.yaml for the full set of hyperparameters
    # (grouped and commented) and `load_config`'s docstring for CLI override syntax.
    cfg = load_config(default_config_path="opd/examples/opsd.yaml")

    use_wandb = should_use_wandb()

    # Distributed init — same rank split as OPD/SDFT:
    #   ranks 0..train_world_size-1  →  student (FSDP, updated by optimizer)
    #   rank  train_world_size       →  teacher (plain nn.Module, frozen initial policy)
    ctx = init_distributed(cfg.device_type, cfg.train_world_size)

    print0(f"Model: {cfg.student_model}  (teacher = frozen initial policy)")
    print_run_banner(ctx, cfg)
    if cfg.kl_clip > 0.0:
        print0(f"Per-token KL clip: {cfg.kl_clip}")

    init_wandb(
        cfg.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": cfg.student_model,
            "algorithm": cfg.algorithm,
            "distill_top_k": cfg.distill_top_k,
            "kl_clip": cfg.kl_clip,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "prompts_per_step": cfg.prompts_per_step,
            "train_batch_size": cfg.train_batch_size,
            "epochs": cfg.epochs,
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
        teacher = build_teacher(cfg.student_model)
        print(f"[teacher] Loaded initial policy from {cfg.student_model} (frozen)", flush=True)


    loss_fn = ALGORITHMS[cfg.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(cfg.algorithm)
    top_k = cfg.distill_top_k

    trainer = build_trainer(
        cfg, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        use_wandb,
    )

    if ctx.is_student:
        dataset     = OPSDMathEnv.load(split=cfg.dataset_split, dataset_id=cfg.dataset_id)
        loader      = distributed_opd_loader(
            dataset, cfg.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=cfg.seed
        )
        loader_iter = iter(loader)


    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        # Per-token pointwise KL clipping. Stylistic
        # tokens can exhibit much higher KL than math tokens, dominating the
        # gradient signal. Clipping each token's divergence contribution to τ
        # stabilises training and prevents performance collapse, especially for smaller models.
        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if ctx.is_student else None, teacher=teacher if ctx.is_teacher else None,
            select_topk_by=select_topk_by, top_k=top_k,
            student_chunk_size=cfg.student_chunk_size, teacher_chunk_size=cfg.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=cfg.algorithm == "mopd_pg_loss",
            tis_clip=cfg.tis_clip, divisor=mb.n_mb,
            kl_clip=cfg.kl_clip if cfg.kl_clip > 0.0 else None,
        )

    # Training loop — all ranks iterate together
    for step in range(cfg.num_steps):
        t0 = time.time()

        # -- Rollout generation (student ranks only) --
        rollouts = None
        if ctx.is_student:
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

            # OPSD: single on-policy trajectory per prompt
            rollouts = generate_rollouts_for_prompts(cfg, prompts, num_samples=1)

            # Attach reference-conditioned teacher prompt to each rollout.
            # The teacher sees: problem + ground-truth solution y* → richer
            # context than the student (problem only).
            for i, ex in enumerate(examples):
                r            = rollouts[i]    # one rollout per prompt (num_samples=1)
                student_msgs = [{"role": "user", "content": ex.problem + _STUDENT_SUFFIX}]
                teacher_msgs = _build_teacher_messages(
                    [{"role": "user", "content": ex.problem}],
                    ex.get_privileged_information(r["response"]),
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
    finish_wandb(ctx.master_process, use_wandb)
