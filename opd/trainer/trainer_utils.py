"""Shared training-loop plumbing for OPD/SDPO/OPSD/SDFT.

This module owns exactly the mechanics that are identical "after the
dataloader is initiated": broadcasting minibatches to the teacher rank,
stepping the optimizer, syncing weights to vLLM, logging, and checkpointing.

Explicitly NOT owned here — these stay in each script, since they're what
makes each algorithm unique:
  - rollout generation and teacher-prompt construction
  - the actual exchange/loss/backward logic inside a minibatch (supplied as a
    `minibatch_fn` callback)
  - eval (the three scripts that have it use incompatible mechanisms)
  - model construction (`build_student`/`build_teacher`/`build_syncer`)

Correctness note: every `torch.distributed` collective here must run on all
ranks in the same order. `has_teacher_batch` and `extra_specs` must be plain
Python literals, identical on every rank at the call site — never inferred
from whether some rank-local variable (like a teacher batch dict) happens to
be `None`, since on the teacher rank several of these variables don't exist as
data at all (they're built only inside `if ctx.is_student:` blocks upstream).
"""

import time
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.distributed as dist

from opd.fsdp.model import StudentModel, TeacherModel
from opd.generator.rollout import prepare_batch, sync_weights_to_vllm_inplace
from opd.trainer.logging_utils import log_step_metrics
from opd.trainer.distillation_utils import (
    RankBroadcaster,
    prepare_teacher_batch,
    sync_student_to_teacher,
)
from opd.trainer.models import DistributedContext, MinibatchTensors, StepAccumulator
from opd.trainer.setup_utils import broadcast_n_minibatches, maybe_save_checkpoint
from opd.trainer.sync_teacher import TeacherSyncer

MinibatchFn = Callable[[MinibatchTensors, StepAccumulator], None]


class Trainer:
    """Owns the training-loop mechanics shared by OPD/SDPO/OPSD/SDFT.

    Attributes:
        args: The script's parsed CLI args.
        ctx: Distributed rank/topology info for this run.
        student: The student model wrapper (student ranks only; `None` on the
          teacher rank).
        teacher: The teacher model (teacher rank only; `None` on student ranks).
        model_update_group: NCCL process group for vLLM weight sync.
        use_wandb: Whether wandb logging is enabled for this run.
    """

    def __init__(
        self,
        args: Any,
        ctx: DistributedContext,
        student: StudentModel | None,
        teacher: TeacherModel | None,
        model_update_group: Any,
        use_wandb: bool,
    ) -> None:
        self.args = args
        self.ctx = ctx
        self.student = student
        self.teacher = teacher
        self.model_update_group = model_update_group
        self.use_wandb = use_wandb

    def prepare_batches(
        self,
        rollouts: list[dict[str, Any]] | None,
        has_teacher_batch: bool,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """Builds the student batch (and teacher batch, if `has_teacher_batch`) and sets train mode.

        Args:
            rollouts: Rollout dicts from `generate_rollouts_remote` (student
              ranks only; ignored — may be `None` — on the teacher rank,
              which never generates rollouts).
            has_teacher_batch: Whether this script builds a separate,
              differently-shaped teacher batch (SDPO/OPSD/SDFT) or runs the
              teacher forward on the same tensors as the student (OPD). A
              fixed literal for a given script, identical on every rank.

        Returns:
            `(batch, teacher_batch)`. Both `None` on the teacher rank;
            `teacher_batch` is also `None` on student ranks when
            `has_teacher_batch` is False.
        """
        if not self.ctx.is_student:
            return None, None
        batch = prepare_batch(
            rollouts,
            tokenizer=self.student.tokenizer,
            max_prompt_len=self.args.max_prompt_len,
            max_response_len=self.args.max_response_len,
            device=self.ctx.device,
        )
        teacher_batch = None
        if has_teacher_batch:
            teacher_batch = prepare_teacher_batch(
                rollouts, tokenizer=self.student.tokenizer, device=self.ctx.device,
            )
        self.student.model.train()
        return batch, teacher_batch

    def run_epoch(
        self,
        batch: dict[str, torch.Tensor] | None,
        teacher_batch: dict[str, torch.Tensor] | None,
        minibatch_fn: MinibatchFn,
        acc: StepAccumulator,
        has_teacher_batch: bool,
        accum_steps: int | None = None,
        extra_tensors: dict[str, torch.Tensor] | None = None,
        extra_specs: dict[str, torch.dtype] | None = None,
    ) -> None:
        """Runs one epoch's worth of minibatches: broadcast, exchange (via `minibatch_fn`), optimizer step.

        Args:
            batch: Student batch from `prepare_batches` (`None` on the teacher rank).
            teacher_batch: Teacher batch from `prepare_batches`, or `None` if
              `has_teacher_batch` is False.
            minibatch_fn: Called once per minibatch as `minibatch_fn(mb, acc)`;
              owns all algorithm-specific exchange/loss/backward logic and is
              responsible for its own `acc.add_loss(...)`/
              `acc.add_health_metrics(...)` calls.
            acc: Accumulator to pass through to `minibatch_fn` (create once
              per outer step, reused across every `run_epoch` call in that
              step — never reset per-epoch).
            has_teacher_batch: Whether to broadcast/pass teacher tensors — a
              fixed per-script literal, identical on every rank (see module
              docstring: never derived from `teacher_batch is not None`).
            accum_steps: Step the optimizer every this many minibatches (and
              always on the last one). Defaults to stepping once, after all
              minibatches (pass `args.grad_accum_steps` for gradient
              accumulation).
            extra_tensors: Full per-example tensors (student ranks only) to
              slice by the same permutation as `mb_ids` and broadcast each
              minibatch (e.g. SDPO's `{"sd_mask": sd_mask}`). Must have an
              entry for every key in `extra_specs`.
            extra_specs: `{name: dtype}` for each entry in `extra_tensors`,
              used to allocate a placeholder on non-owning ranks. A fixed
              literal, identical on every rank (e.g. SDPO passes
              `{"sd_mask": torch.float}`; other scripts pass `None`).
        """
        ctx = self.ctx
        rb = RankBroadcaster(ctx)
        num_sequences = batch["input_ids"].shape[0] if ctx.is_student else 0
        n_mb, perm = broadcast_n_minibatches(
            ctx.is_student, num_sequences, self.args.train_batch_size, ctx.device, ctx.all_group,
        )
        accum_steps = accum_steps or n_mb
        extra_specs = extra_specs or {}

        for mb_idx in range(n_mb):
            if ctx.is_student:
                start = mb_idx * self.args.train_batch_size
                idx = perm[start : start + self.args.train_batch_size]
                mb_ids = batch["input_ids"][idx]
                mb_attn = batch["attention_mask"][idx]
                mb_mask = batch["response_mask"][idx]
                mb_inf_lp = batch["inference_logprobs"][idx]
                if has_teacher_batch:
                    t_mb_ids = teacher_batch["input_ids"][idx]
                    t_mb_attn = teacher_batch["attention_mask"][idx]
                    t_mb_mask = teacher_batch["response_mask"][idx]
                else:
                    t_mb_ids = t_mb_attn = t_mb_mask = None
            else:
                mb_ids = mb_attn = mb_mask = mb_inf_lp = None
                t_mb_ids = t_mb_attn = t_mb_mask = None

            mb_ids, mb_attn = rb.broadcast_minibatch(mb_ids, mb_attn)
            if has_teacher_batch:
                t_mb_ids, t_mb_attn, t_mb_mask = rb.broadcast_teacher_inputs(
                    t_mb_ids, t_mb_attn, t_mb_mask
                )

            extra: dict[str, torch.Tensor] = {}
            for name, dtype in extra_specs.items():
                if ctx.is_student:
                    val = extra_tensors[name][idx]
                else:
                    val = torch.zeros(mb_ids.shape[0], dtype=dtype, device=ctx.device)
                dist.broadcast(val, src=0, group=ctx.all_group)
                extra[name] = val

            mb = MinibatchTensors(
                mb_ids=mb_ids, mb_attn=mb_attn, mb_mask=mb_mask, mb_inf_lp=mb_inf_lp,
                t_mb_ids=t_mb_ids, t_mb_attn=t_mb_attn, t_mb_mask=t_mb_mask,
                mb_idx=mb_idx, n_mb=n_mb, extra=extra,
            )
            minibatch_fn(mb, acc)

            if ctx.is_student and ((mb_idx + 1) % accum_steps == 0 or mb_idx == n_mb - 1):
                self.student._optimizer_step()

    def sync_teacher(self, syncer: TeacherSyncer, step_idx: int) -> None:
        """Broadcasts student weights to the teacher rank and applies `syncer`.

        See `distillation_utils.sync_student_to_teacher`.
        """
        sync_student_to_teacher(
            student_fsdp_model=self.student.model if self.ctx.is_student else None,
            teacher=self.teacher if self.ctx.is_teacher else None,
            syncer=syncer,
            global_step=step_idx,
            ctx=self.ctx,
        )

    def finish_step(
        self,
        step_idx: int,
        t0: float,
        batch: dict[str, torch.Tensor] | None,
        acc: StepAccumulator,
    ) -> None:
        """Syncs weights to vLLM, logs metrics, and checkpoints — student ranks only.

        Args:
            step_idx: Zero-indexed training step.
            t0: `time.time()` at the start of this step, for the logged duration.
            batch: Student batch (for `tokens = batch["input_ids"].numel()`);
              `None` on the teacher rank, never dereferenced there.
            acc: This step's accumulated loss/health metrics.
        """
        if not self.ctx.is_student:
            return
        sync_weights_to_vllm_inplace(
            self.student.model, self.args.rollout_worker_url, self.model_update_group, fsdp=True,
        )
        dt = time.time() - t0
        avg_loss, overlap_ratio, overlap_advantage, entropy_gap = acc.normalized()
        current_lr = (
            self.student.scheduler.get_last_lr()[0]
            if self.student.scheduler is not None
            else self.args.lr
        )
        tokens = batch["input_ids"].numel()
        log_step_metrics(
            step_idx, self.args.num_steps, avg_loss, current_lr, tokens, dt,
            overlap_ratio, overlap_advantage, entropy_gap,
            self.ctx.master_process, self.use_wandb,
        )
        maybe_save_checkpoint(self.student, self.args.save_dir, self.args.save_every, step_idx)

    def step(
        self,
        step_idx: int,
        t0: float,
        batch: dict[str, torch.Tensor] | None,
        teacher_batch: dict[str, torch.Tensor] | None,
        minibatch_fn: MinibatchFn,
        has_teacher_batch: bool,
        accum_steps: int | None = None,
        extra_tensors: dict[str, torch.Tensor] | None = None,
        extra_specs: dict[str, torch.dtype] | None = None,
        syncer: TeacherSyncer | None = None,
        teacher_sync_scope: Literal["epoch", "step"] = "step",
    ) -> StepAccumulator:
        """Runs all `args.epochs` passes over `batch`, then syncs/logs/checkpoints.

        This is the main entry point — call it once per outer training step.

        Args:
            step_idx: Zero-indexed training step.
            t0: `time.time()` at the start of this step.
            batch: Student batch from `prepare_batches`.
            teacher_batch: Teacher batch from `prepare_batches`.
            minibatch_fn: Per-minibatch callback — see `run_epoch`.
            has_teacher_batch: See `run_epoch`.
            accum_steps: See `run_epoch`.
            extra_tensors: See `run_epoch`.
            extra_specs: See `run_epoch`.
            syncer: If given, sync the student into this self-teacher syncer
              at the cadence named by `teacher_sync_scope`. `None` (the
              default) means never sync a self-teacher — OPD's separate
              frozen teacher and OPSD's frozen initial policy both pass
              `None`. A fixed per-script literal.
            teacher_sync_scope: `"epoch"` syncs after every epoch's minibatch
              loop (SDFT); `"step"` syncs once, after all epochs finish
              (SDPO). Ignored if `syncer` is `None`.

        Returns:
            The `StepAccumulator` for this step (already used internally to
            log; returned in case the caller wants the raw numbers too).
        """
        acc = StepAccumulator()
        for _epoch_idx in range(self.args.epochs):
            self.run_epoch(
                batch, teacher_batch, minibatch_fn, acc, has_teacher_batch,
                accum_steps=accum_steps, extra_tensors=extra_tensors, extra_specs=extra_specs,
            )
            if syncer is not None and teacher_sync_scope == "epoch":
                self.sync_teacher(syncer, step_idx)
        if syncer is not None and teacher_sync_scope == "step":
            self.sync_teacher(syncer, step_idx)
        self.finish_step(step_idx, t0, batch, acc)
        return acc

    def barrier(self) -> None:
        """Blocks until every rank (student and teacher) reaches this point."""
        dist.barrier(group=self.ctx.all_group)
