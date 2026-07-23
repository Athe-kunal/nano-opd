import time
from typing import Literal

import torch

from opd.loss import ALGORITHMS, compute_tis_weights
from opd.fsdp.algorithms import student_logprobs_at_indices
from opd.trainer.distillation_utils import (
    fetch_teacher_sampled_logprob,
    fetch_teacher_topk,
    mopd_pg_loss_and_backward,
    topk_kl_loss_and_backward,
)
from opd.trainer.models import MopdPGExchange, TopKExchange
from opd.trainer.logging_utils import finish_wandb, init_wandb, should_use_wandb
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
from opd.metrics import compute_topk_health_metrics
from opd.envs.dataset import distributed_opd_loader, build_opd_dataset
from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

_ENV_CLS = {
    "dapo_math": DapoMathEnv,
    "livecodebench": LiveCodeBenchEnv,
    "sciknoweval": SciKnowEvalEnv,
}

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # Config — see opd/examples/opd.yaml for the full set of hyperparameters
    # (grouped and commented) and `load_config`'s docstring for CLI override syntax.
    cfg = load_config(default_config_path="opd/examples/opd.yaml")

    use_wandb = should_use_wandb()

    ctx = init_distributed(cfg.device_type, cfg.train_world_size)

    print0(f"Student: {cfg.student_model}")
    print0(f"Teacher: {cfg.teacher_model}")
    print_run_banner(ctx, cfg)

    init_wandb(
        cfg.run_name, ctx.master_process, use_wandb,
        config={
            "student_model": cfg.student_model,
            "teacher_model": cfg.teacher_model,
            "algorithm": cfg.algorithm,
            "distill_top_k": cfg.distill_top_k,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "num_steps": cfg.num_steps,
            "prompts_per_step": cfg.prompts_per_step,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "epochs": cfg.epochs,
            "num_samples": cfg.num_samples,
            "max_new_tokens": cfg.max_new_tokens,
            "max_prompt_len": cfg.max_prompt_len,
            "max_response_len": cfg.max_response_len,
            "temperature": cfg.temperature,
            "sharding_strategy": cfg.sharding_strategy,
        },
    )

    assert_prompts_divisible(cfg.prompts_per_step, ctx.train_world_size)

    # -----------------------------------------------------------------------------
    # Model setup
    if ctx.is_student:
        student = build_student_from_args(cfg, ctx)

    if ctx.is_teacher:
        teacher = build_teacher(cfg.teacher_model)

    # -----------------------------------------------------------------------------
    # Loss function and top-K selection
    loss_fn = ALGORITHMS[cfg.algorithm]
    select_topk_by: Literal["student", "teacher"] = topk_selector_for(cfg.algorithm)


    trainer = build_trainer(
        cfg, ctx,
        student if ctx.is_student else None, teacher if ctx.is_teacher else None,
        use_wandb,
    )

    if ctx.is_student:
        dataset = build_opd_dataset(cfg.dataset)
        loader = distributed_opd_loader(
            dataset, cfg.prompts_per_step, ctx.train_world_size, ctx.ddp_rank, seed=cfg.seed
        )
        loader_iter = iter(loader)

    def _topk_and_tis(
        mb: MinibatchTensors,
        student_logits: torch.Tensor | None, teacher_logits: torch.Tensor | None,
        B: int, T: int,
    ) -> tuple[TopKExchange, torch.Tensor, torch.Tensor] | None:
        """Shared top-K fetch + TIS-weight computation for both minibatch steps.

        `fetch_teacher_topk` is a collective — every rank must call it,
        whether or not its output feeds the loss (for `mopd_pg_loss` it's a
        diagnostic only; for the top-K KL family it's the loss's support).
        Returns `None` on the teacher rank, which has nothing left to do
        after the collective.
        """
        topk = fetch_teacher_topk(
            ctx=ctx,
            select_topk_by=select_topk_by,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            B=B, T=T,
            top_k=cfg.distill_top_k,
            student_chunk_size=cfg.student_chunk_size,
            teacher_chunk_size=cfg.teacher_chunk_size,
        )

        if not ctx.is_student:
            return None

        sampled_ids = mb.mb_ids[:, 1:]                             # [B, T-1]
        tis_weights = compute_tis_weights(
            student_logits, sampled_ids, mb.mb_inf_lp[:, 1:], cfg.tis_clip
        )
        return topk, sampled_ids, tis_weights

    def _mopd_pg_minibatch(
        mb: MinibatchTensors, acc: StepAccumulator,
        student_logits: torch.Tensor | None, teacher_logits: torch.Tensor | None,
        B: int, T: int, divisor: int,
    ) -> None:
        """MOPD policy-gradient step (mopd_pg_loss): sampled-token-only exchange.

        Unlike `_topk_kl_minibatch`, the loss needs only the teacher's
        log-prob at the student's own sampled token
        (`fetch_teacher_sampled_logprob`) — no shared top-K support at all.
        `_topk_and_tis`'s fetch exists purely to compute the
        overlap/entropy-gap health-metric diagnostics; its output never
        feeds the loss.
        """
        t_logprob = fetch_teacher_sampled_logprob(
            ctx=ctx,
            teacher_logits=teacher_logits,
            token_ids=mb.mb_ids[:, 1:],
            B=B, T=T,
            teacher_chunk_size=cfg.teacher_chunk_size,
        )
        result = _topk_and_tis(mb, student_logits, teacher_logits, B, T)
        if result is None:
            return
        topk_for_health_metrics, sampled_ids, tis_weights = result

        s_logprob = student_logprobs_at_indices(
            student_logits, sampled_ids.unsqueeze(-1), cfg.student_chunk_size
        ).squeeze(-1)                                          # [B, T-1], grad-carrying
        shift_mask = mb.mb_mask[:, 1:]                         # [B, T-1]

        pg = MopdPGExchange(
            s_logprob=s_logprob, t_logprob=t_logprob,
            s_compact_mask=shift_mask, t_compact_mask=shift_mask,
        )
        loss = mopd_pg_loss_and_backward(
            student=student, pg=pg, loss_fn=loss_fn,
            tis_weights=tis_weights, divisor=divisor,
        )
        acc.add_loss(loss)

        with torch.no_grad():
            ratio, adv, ent_gap = compute_topk_health_metrics(
                student_logits, topk_for_health_metrics, cfg.student_chunk_size
            )
            acc.add_health_metrics(ratio, adv, ent_gap)

    def _topk_kl_minibatch(
        mb: MinibatchTensors, acc: StepAccumulator,
        student_logits: torch.Tensor | None, teacher_logits: torch.Tensor | None,
        B: int, T: int, divisor: int,
    ) -> None:
        """Shared step for the top-K KL family: reverse_kl, forward_kl, jsd, mopd_loss.

        All four share the same exchange -> loss -> backward -> health-metric
        shape; only `loss_fn` and `select_topk_by` (picked once from
        `cfg.algorithm`, see `topk_selector_for`) differ between them.
        """
        result = _topk_and_tis(mb, student_logits, teacher_logits, B, T)
        if result is None:
            return
        topk, _sampled_ids, tis_weights = result

        shift_mask = mb.mb_mask[:, 1:]                             # [B, T-1]

        loss = topk_kl_loss_and_backward(
            student=student, student_logits=student_logits, topk=topk, loss_fn=loss_fn,
            response_mask=shift_mask, tis_weights=tis_weights, divisor=divisor,
            student_chunk_size=cfg.student_chunk_size,
        )
        acc.add_loss(loss)

        with torch.no_grad():
            ratio, adv, ent_gap = compute_topk_health_metrics(
                student_logits, topk, cfg.student_chunk_size
            )
            acc.add_health_metrics(ratio, adv, ent_gap)

    def do_minibatch(mb: MinibatchTensors, acc: StepAccumulator) -> None:
        batch_size, seq_len = mb.mb_ids.shape
        B, T = batch_size, seq_len - 1
        # last token logits are not required, hence indexed till -1
        student_logits = student.get_logits(mb.mb_ids, mb.mb_attn)[:, :-1] if ctx.is_student else None
        teacher_logits = teacher.get_logits(mb.mb_ids, mb.mb_attn)[:, :-1] if ctx.is_teacher else None

        divisor = accum_window_size(mb, cfg.grad_accum_steps)

        if cfg.algorithm == "mopd_pg_loss":
            _mopd_pg_minibatch(mb, acc, student_logits, teacher_logits, B, T, divisor)
        else:
            _topk_kl_minibatch(mb, acc, student_logits, teacher_logits, B, T, divisor)

    for step in range(cfg.num_steps):
        t0 = time.time()
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

        batch, teacher_batch = trainer.prepare_batches(rollouts, has_teacher_batch=False)

        trainer.step(
            step, t0, batch, teacher_batch, do_minibatch,
            has_teacher_batch=False, accum_steps=cfg.grad_accum_steps,
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
                )

    compute_cleanup()
    finish_wandb(ctx.master_process, use_wandb)
