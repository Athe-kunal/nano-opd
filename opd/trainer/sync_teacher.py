"""Teacher weight synchronization methods for Self-Distillation Policy Optimization (SDPO).

Four sync strategies are provided. Each takes the current student parameters and
updates the teacher parameters in-place. Call the chosen strategy after every
optimizer step, before the next rollout.

  EMA             — exponential moving average toward student (recommended default)
  TrustRegion     — blend toward initial weights instead of old teacher
  HardSync        — periodic full copy every N steps
  OnPolicy        — live copy (teacher = student); for inference only, not training
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TeacherSyncer(ABC):
    """Abstract base for all teacher sync strategies."""

    @abstractmethod
    def step(
        self,
        student_params: Iterable[torch.Tensor],
        teacher_params: Iterable[nn.Parameter],
        global_step: int,
    ) -> None:
        """Update teacher parameters in-place given current student parameters.

        `student_params` is often a plain tensor iterable (e.g. broadcast
        receive buffers), not necessarily `nn.Parameter` — only `.data`
        access is required, which both support.
        """


# ---------------------------------------------------------------------------
# 1. EMA — Exponential Moving Average
# ---------------------------------------------------------------------------

class EMASyncer(TeacherSyncer):
    """ϕ ← α · θ + (1 - α) · ϕ

    The teacher lags the student by roughly 1/α steps. With the default
    α = 0.05 that is ~20 steps of smoothing, giving a stable but adaptive
    teaching signal.

    Args:
        alpha: EMA update rate in (0, 1]. Higher -> faster adaptation, more noise.
               Lower -> more stability, teacher lags further behind student.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def step(
        self,
        student_params: Iterable[torch.Tensor],
        teacher_params: Iterable[nn.Parameter],
        global_step: int,
    ) -> None:
        with torch.no_grad():
            for theta, phi in zip(student_params, teacher_params):
                phi.data.mul_(1 - self.alpha).add_(theta.data, alpha=self.alpha)


# ---------------------------------------------------------------------------
# 2. Trust-Region Mixing — blend toward initial weights
# ---------------------------------------------------------------------------

class TrustRegionSyncer(TeacherSyncer):
    """ϕ ← β · θ + (1 - β) · ϕ₀

    Unlike EMA, the teacher is always pulled back toward the *initial* weights
    ϕ₀, not toward the previous teacher. This prevents the teacher from drifting
    too far from the pre-trained distribution and acts as a global anchor.

    Prefer over EMA when retaining pre-trained priors in the teacher matters and
    the task distribution is stable throughout training.

    Args:
        beta: Mixing coefficient in (0, 1]. Same role as EMA's alpha but controls
              blending with the initial weights instead of the old teacher.
        initial_params: Snapshot of the model parameters before any RL training.
                        Pass a list of tensors captured with
                        ``[p.data.clone() for p in model.parameters()]``.
    """

    def __init__(
        self,
        beta: float,
        initial_params: list[torch.Tensor],
    ) -> None:
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        self.beta = beta
        # Store on CPU by default; move lazily to match teacher device on first step.
        self._phi0: list[torch.Tensor] = [p.clone() for p in initial_params]

    def step(
        self,
        student_params: Iterable[torch.Tensor],
        teacher_params: Iterable[nn.Parameter],
        global_step: int,
    ) -> None:
        with torch.no_grad():
            for theta, phi, phi0 in zip(student_params, teacher_params, self._phi0):
                phi0_dev = phi0.to(phi.device)
                phi.data.copy_(self.beta * theta.data + (1 - self.beta) * phi0_dev)


# ---------------------------------------------------------------------------
# 3. Hard Sync — periodic full copy
# ---------------------------------------------------------------------------

class HardSyncSyncer(TeacherSyncer):
    """Every N steps: ϕ ← θ  (full copy of student weights).

    Creates discontinuous jumps in the teaching signal — common in discrete-
    action RL (DQN target networks) but generally not preferred for LM
    distillation where smoothness matters. Provided for completeness and
    ablation experiments.

    Args:
        sync_every_n_steps: Copy student -> teacher every this many optimizer steps.
    """

    def __init__(self, sync_every_n_steps: int = 100) -> None:
        if sync_every_n_steps < 1:
            raise ValueError(f"sync_every_n_steps must be >= 1, got {sync_every_n_steps}")
        self.sync_every_n_steps = sync_every_n_steps

    def step(
        self,
        student_params: Iterable[torch.Tensor],
        teacher_params: Iterable[nn.Parameter],
        global_step: int,
    ) -> None:
        # global_step is 0-indexed; sync at step 0 and then every N steps.
        if global_step % self.sync_every_n_steps != 0:
            return
        with torch.no_grad():
            for theta, phi in zip(student_params, teacher_params):
                phi.data.copy_(theta.data)


# ---------------------------------------------------------------------------
# 4. On-Policy (live copy) — teacher IS the student
# ---------------------------------------------------------------------------

class OnPolicySyncer(HardSyncSyncer):
    """ϕ ← θ at every step (teacher = live student).

    Mathematically just `HardSyncSyncer` with `sync_every_n_steps=1`: the
    step-count gate always passes, so the copy runs every step.

    Only safe for the *inference* pass where the teacher conditions on feedback
    context. Do NOT use this during training — any gradient update immediately
    corrupts the teaching signal, leading to feedback loops and divergence.

    Included so the full method space can be explored in ablations. For
    production training use EMASyncer instead.
    """

    def __init__(self) -> None:
        super().__init__(sync_every_n_steps=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SYNC_METHODS = {
    "ema":          EMASyncer,
    "trust_region": TrustRegionSyncer,
    "hard_sync":    HardSyncSyncer,
    "on_policy":    OnPolicySyncer,
}


def build_syncer(method: str, **kwargs) -> TeacherSyncer:
    """Instantiate a TeacherSyncer by name.

    Args:
        method: One of "ema", "trust_region", "hard_sync", "on_policy".
        **kwargs: Forwarded to the syncer's __init__.

    Example::

        syncer = build_syncer("ema", alpha=0.05)
        syncer = build_syncer("trust_region", beta=0.1, initial_params=phi0)
        syncer = build_syncer("hard_sync", sync_every_n_steps=50)
        syncer = build_syncer("on_policy")
    """
    if method not in SYNC_METHODS:
        raise ValueError(f"Unknown sync method '{method}'. Choose from {list(SYNC_METHODS)}")
    return SYNC_METHODS[method](**kwargs)
