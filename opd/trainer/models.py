"""Shared dataclasses used across opd/trainer/*.py.

Moved here verbatim from setup_utils.py, trainer_utils.py, and
distillation_utils.py so all four training scripts and their shared
plumbing import these bundles from one place.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class DistributedContext:
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: torch.device
    train_world_size: int
    teacher_global_rank: int
    is_student: bool
    is_teacher: bool
    master_process: bool
    student_group: Any   # dist.ProcessGroup
    all_group: Any       # dist.ProcessGroup


@dataclass(slots=True)
class MinibatchTensors:
    """Bundle passed to each script's `minibatch_fn`.

    `mb_mask`/`mb_inf_lp` are student-only and are `None` on the teacher rank
    — never touch them without a `ctx.is_student` guard. `t_mb_*` are `None`
    when the script has no separate teacher batch (OPD).
    """
    mb_ids: torch.Tensor
    mb_attn: torch.Tensor
    mb_mask: torch.Tensor | None
    mb_inf_lp: torch.Tensor | None
    t_mb_ids: torch.Tensor | None
    t_mb_attn: torch.Tensor | None
    t_mb_mask: torch.Tensor | None
    mb_idx: int
    n_mb: int
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class StepAccumulator:
    """Loss/health-metric accumulator for one outer training step.

    Create exactly one of these per step, before the epoch loop, and never
    reset it per-epoch — this matches where `total_loss`/`n_batches`/etc. are
    initialized in all four training scripts today.
    """
    total_loss: float = 0.0
    n_batches: int = 0
    overlap_ratio: float = 0.0
    overlap_advantage: float = 0.0
    entropy_gap: float = 0.0

    def add_loss(self, loss: torch.Tensor) -> None:
        self.total_loss += loss.item()
        self.n_batches += 1

    def add_health_metrics(self, ratio: float, advantage: float, entropy_gap: float) -> None:
        self.overlap_ratio += ratio
        self.overlap_advantage += advantage
        self.entropy_gap += entropy_gap

    def normalized(self) -> tuple[float, float, float, float]:
        """Returns `(avg_loss, overlap_ratio, overlap_advantage, entropy_gap)`, each divided by `max(n_batches, 1)`."""
        n = max(self.n_batches, 1)
        return (
            self.total_loss / n,
            self.overlap_ratio / n,
            self.overlap_advantage / n,
            self.entropy_gap / n,
        )


@dataclass(slots=True)
class MopdPGExchange:
    """Result of exchange_mopd_pg_packed: sampled-token log-probs for the MOPD-PG loss."""
    s_logprob: torch.Tensor | None       # [B, R_max], grad-carrying (student ranks only)
    t_logprob: torch.Tensor | None       # [B, R_max], no-grad
    s_compact_mask: torch.Tensor | None  # [B, R_max] (student ranks only)
    t_compact_mask: torch.Tensor | None  # [B, R_max]


@dataclass(slots=True)
class TopKExchange:
    """Result of fetch_teacher_topk: the top-K distributions used to compute the KL loss."""
    topk_idx: torch.Tensor
    t_logprobs: torch.Tensor
    t_compact_mask: torch.Tensor
    student_topk_idx: torch.Tensor
    teacher_topk_idx: torch.Tensor
    t_logprobs_at_student: torch.Tensor
    teacher_own_logprobs: torch.Tensor


@dataclass(slots=True)
class MinibatchExchangeResult:
    """Result of minibatch_exchange: exactly one of `topk`/`pg` is set, per `is_pg`.

    Packed student tensors (`s_resp`, `s_compact_mask`) are only produced in
    the top-K path (`is_pg=False`) — the PG path skips packing the full
    logits and leaves them `None`.
    """
    is_pg: bool
    topk: TopKExchange | None
    pg: MopdPGExchange | None
    s_resp: torch.Tensor | None
    s_compact_mask: torch.Tensor | None
    s_shift_mask: torch.Tensor | None
    student_logits: torch.Tensor | None
