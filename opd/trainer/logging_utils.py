import os
from typing import Any

from opd.trainer.setup_utils import print0


def should_use_wandb() -> bool:
    """Whether wandb logging is enabled for this run (via the USE_WANDB env var; default on)."""
    return os.environ.get("USE_WANDB", "1").strip().lower() not in ("0", "false", "no")


def init_wandb(
    run_name: str,
    master_process: bool,
    use_wandb: bool,
    config: dict[str, Any],
) -> None:
    """Initializes a wandb run on the master process, if wandb logging is enabled.

    Args:
        run_name: Name for the wandb run.
        master_process: Only rank 0 should initialize a run.
        use_wandb: Whether wandb logging is enabled (see `should_use_wandb`).
        config: Hyperparameter config dict to log.
    """
    if not (master_process and use_wandb):
        return
    import wandb
    wandb.init(project="nano-opd", name=run_name, config=config)


def log_step_metrics(
    step: int,
    num_steps: int,
    avg_loss: float,
    current_lr: float,
    tokens: int,
    dt: float,
    overlap_ratio: float,
    overlap_advantage: float,
    entropy_gap: float,
    master_process: bool,
    use_wandb: bool,
) -> None:
    """Prints a one-line step summary and logs the same metrics to wandb.

    Called on student ranks only, after averaging a step's per-minibatch
    losses and distillation health metrics (see `compute_topk_health_metrics`).

    Args:
        step: Zero-indexed training step.
        num_steps: Total number of training steps, for the printed progress
          fraction.
        avg_loss: Mean distillation loss over this step's minibatches.
        current_lr: Current learning rate.
        tokens: Total tokens processed in this step's rollout batch.
        dt: Wall-clock seconds this step took.
        overlap_ratio: Mean student/teacher top-K overlap ratio.
        overlap_advantage: Mean overlap-token advantage.
        entropy_gap: Mean student/teacher entropy gap.
        master_process: Whether this is the rank-0 process (only it logs to
          wandb).
        use_wandb: Whether wandb logging is enabled for this run.
    """
    print0(
        f"step {step + 1:4d}/{num_steps} | loss {avg_loss:.4f} "
        f"| lr {current_lr:.2e} | tokens {tokens} | dt {dt:.1f}s "
        f"| overlap {overlap_ratio:.3f} "
        f"| adv {overlap_advantage:.4f} "
        f"| ent_gap {entropy_gap:.4f}"
    )
    if master_process and use_wandb:
        import wandb
        wandb.log(
            {
                "train/loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/step_time_s": dt,
                "train/tokens_per_step": tokens,
                "metrics/overlap_ratio": overlap_ratio,
                "metrics/overlap_token_advantage": overlap_advantage,
                "metrics/entropy_gap": entropy_gap,
            },
            step=step + 1,
        )


def log_eval_metrics(
    metrics: dict[str, float],
    step: int,
    master_process: bool,
    use_wandb: bool,
) -> None:
    """Logs eval metrics to wandb, if enabled.

    Generalizes the ad-hoc eval `wandb.log(...)` call scripts like SDFT need
    (whose eval mechanism doesn't go through an env class's `.evaluate()`,
    unlike OPD/SDPO/OPSD).

    Args:
        metrics: Metric name -> value, logged as-is.
        step: Training step to log against (wandb's x-axis).
        master_process: Only rank 0 should log.
        use_wandb: Whether wandb logging is enabled for this run.
    """
    if not (master_process and use_wandb):
        return
    import wandb
    wandb.log(metrics, step=step)


def finish_wandb(master_process: bool, use_wandb: bool) -> None:
    """Closes the wandb run on the master process, if one was opened."""
    if master_process and use_wandb:
        import wandb
        wandb.finish()
