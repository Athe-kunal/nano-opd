"""Dynamic Co-Evolution + Self-Refined Concise Learning (DCE+SRCL) training loop.

OPSD with three changes (arXiv 2609.30652): the gold-conditioned teacher is
re-synced to the student every step, the reference solution is placed in the
assistant turn, and an SRCL cross-entropy loss on verified concise rewrites is
added to the distillation loss.
"""

import time
from typing import Literal

from opd.envs.dataset import distributed_opd_loader
from opd.envs.opsd_dataset import OPSDMathEnv
from opd.loss import ALGORITHMS
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
from opd.trainer.models import MinibatchTensors, StepAccumulator
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
from opd.trainer.srcl_utils import refine_rollouts, srcl_backward
from opd.trainer.sync_teacher import build_syncer
from opd.trainer.trainer_utils import build_trainer

_STUDENT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# Teacher and student share the same user turn; the reference solution is
# prefilled in the assistant turn so the model conditions on it as its own
# earlier reasoning (paper Figure 7, "assistant-side").
_ASSISTANT_PREFILL = (
    "Here is a reference solution: {solution}\n"
    "After understanding the reference solution, please try to solve this problem "
    "using your own approach below:\n"
)


if __name__ == "__main__":
    cfg = load_config(default_config_path="opd/examples/dce.yaml")
    use_wandb = should_use_wandb()

    # Same rank split as OPSD: student ranks (FSDP) + one teacher rank.
    ctx = init_distributed(cfg.device_type, cfg.train_world_size)

    print0(f"Model: {cfg.student_model}  (teacher synced every {cfg.teacher_sync_every_n} step(s))")
    print_run_banner(ctx, cfg)

    init_wandb(
        cfg.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": cfg.student_model,
            "algorithm": cfg.algorithm,
            "distill_top_k": cfg.distill_top_k,
            "kl_clip": cfg.kl_clip,
            "teacher_sync_every_n": cfg.teacher_sync_every_n,
            "srcl_enabled": cfg.srcl_enabled,
            "srcl_weight": cfg.srcl_weight,
            "lr": cfg.lr,
            "num_steps": cfg.num_steps,
            "prompts_per_step": cfg.prompts_per_step,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "epochs": cfg.epochs,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
        },
    )

    assert_prompts_divisible(cfg.prompts_per_step, ctx.train_world_size)

    if ctx.is_student:
        student = build_student_from_args(cfg, ctx)
    if ctx.is_teacher:
        teacher = build_teacher(cfg.student_model)

    # Dynamic co-evolution: with n=1 the teacher becomes the updated student after every step.
    syncer = build_syncer("hard_sync", sync_every_n_steps=cfg.teacher_sync_every_n)

    loss_fn = ALGORITHMS[cfg.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(cfg.algorithm)

    trainer = build_trainer(
        cfg, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        use_wandb,
    )

    if ctx.is_student:
        dataset = OPSDMathEnv.load(split=cfg.dataset_split, dataset_id=cfg.dataset_id)
        loader = distributed_opd_loader(
            dataset, cfg.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=cfg.seed
        )
        loader_iter = iter(loader)

    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        self_distill_minibatch(
            mb, acc,
            ctx=ctx, student=student if ctx.is_student else None, teacher=teacher if ctx.is_teacher else None,
            select_topk_by=select_topk_by, top_k=cfg.distill_top_k,
            student_chunk_size=cfg.student_chunk_size, teacher_chunk_size=cfg.teacher_chunk_size,
            loss_fn=loss_fn, is_pg=cfg.algorithm == "mopd_pg_loss",
            tis_clip=cfg.tis_clip, divisor=accum_window_size(mb, cfg.grad_accum_steps),
            kl_clip=cfg.kl_clip if cfg.kl_clip > 0.0 else None,
        )

    for step in range(cfg.num_steps):
        t0 = time.time()

        rollouts = srcl_rollouts = accepted = None
        if ctx.is_student:
            examples, _ = next(loader_iter)
            tokenizer = student.tokenizer
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": ex.problem + _STUDENT_SUFFIX}],
                    tokenize=False, add_generation_prompt=True,
                )
                for ex in examples
            ]
            rollouts = generate_rollouts_for_prompts(cfg, prompts, num_samples=1)

            for prompt, ex, r in zip(prompts, examples, rollouts):
                reference_ids = tokenizer.encode(ex.solution, add_special_tokens=False)
                reference = tokenizer.decode(reference_ids[: cfg.max_reference_tokens])
                r["teacher_prompt"] = prompt + _ASSISTANT_PREFILL.format(solution=reference)

            if cfg.srcl_enabled:
                srcl_rollouts, accepted = refine_rollouts(cfg, tokenizer, examples, rollouts)
                print0(f"[step={step}] SRCL accepted {sum(accepted)}/{len(accepted)} rewrites")

            if step == 0:
                print0(f"[debug step=0] teacher prompt snippet:\n{rollouts[0]['teacher_prompt'][-600:]}")

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=True)

        # SRCL grads accumulate into the first optimizer step of trainer.step, so with
        # epochs=1 and one accumulation window this is the paper's joint DCE+SRCL update.
        if ctx.is_student and cfg.srcl_enabled:
            srcl_loss = srcl_backward(cfg, ctx, student, srcl_rollouts, accepted)
            print0(f"[step={step}] SRCL loss {srcl_loss:.4f}")

        trainer.step(
            step, t0, batch, teacher_batch, do_minibatch,
            has_teacher_batch=True, accum_steps=cfg.grad_accum_steps,
            syncer=syncer, teacher_sync_scope="step",
        )

        trainer.barrier()

    compute_cleanup()
    finish_wandb(ctx.master_process, use_wandb)
