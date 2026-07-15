# nano-opd — CLAUDE.md

**nano-opd** is a library for on-policy distillation (OPD) of language models. The training loop is directly readable in `train_opd.py`, `train_sdpo.py`, `train_opsd.py`, and `train_sdft.py` — no framework hides it.

## Architecture overview

```
trainer (FSDP, train_opd.py / train_sdpo.py / train_opsd.py / train_sdft.py)
    │
    ├── student model (FSDP-wrapped, updated by optimizer)
    ├── teacher model (frozen or EMA-updated)
    │
    └──► rollout worker (separate process, vLLM HTTP server)
              └── generates completions under current student policy
```

The trainer and rollout worker are separate processes communicating over HTTP. After each training step, the trainer pushes updated student weights into the vLLM worker via NCCL (no checkpoint files involved).

## The four training scripts

### `train_opd.py` — On-Policy Distillation with a separate teacher

The student and a frozen external teacher run in the same torchrun job but on different ranks. Ranks `0..train_world_size-1` are student ranks (FSDP); rank `train_world_size` is the teacher rank. They communicate via `dist.broadcast`.

Rather than broadcasting the full `[B, T, V]` logit tensor (V ≈ 128,000 for a large vocab), only the top-K slice is broadcast — a ~1000× reduction, and it lets NCCL broadcast the teacher log-probs directly GPU-to-GPU in one collective.

### `train_sdpo.py` — Self-Policy Distillation (SDPO)

No separate teacher model — the teacher is an EMA copy of the student, living as a plain `nn.Module` on every rank. Every rank runs both student and teacher forward passes locally, so there's no cross-rank broadcast of teacher outputs needed. The teacher is always chasing the student (analogous to target networks in DQN-style deep RL), so the distillation target is never too far ahead.

### `train_opsd.py` — On-Policy Self-Distillation (OPSD)

Also a self-teacher setup (same checkpoint for student and teacher, same rank split as SDPO), but the teacher is the **frozen initial policy**, conditioned on the problem plus the ground-truth reference solution. The student only ever sees the problem, and the teacher is never updated during training — it acts as a fixed regularizer anchoring the student to the pretrained distribution.

The paper (Table 3) finds forward KL — `KL(p_teacher || p_student)` — consistently outperforms reverse KL and JSD here, so it's the default. `--kl-clip` does per-token pointwise KL clipping, which stops stylistic tokens (which can have much higher KL than content tokens) from dominating the gradient signal — the paper shows this prevents performance collapse on small models.

### `train_sdft.py` — Self-Distillation Fine-Tuning (SDFT)

Another self-teacher setup: the teacher is an EMA copy of the student (like SDPO), but instead of environment feedback, the teacher is conditioned on the question plus a worked demonstration pulled from a dataset (science Q&A or tool-use). Default loss is reverse KL.

The demonstration-conditioned teacher tends to produce preamble artifacts (e.g. "Based on the text...") that the student would otherwise learn to mimic even without ever seeing the demonstration itself. `--num-loss-tokens-to-skip` masks the first few response tokens from the loss to suppress this ("Learned Artifacts", paper Section 5).

## Loss functions (`opd/loss.py`)

All losses operate on top-K truncated distributions, `[B, T, K]` where K = `distill-top-k`.

- **Reverse KL** — `KL(p_student || p_teacher)`. Student is in the first argument: mode-seeking, penalizes the student for spreading mass where the teacher doesn't. The student selects the top-K indices (only tokens with student mass matter).
- **Forward KL** — `KL(p_teacher || p_student)`. Teacher is in the first argument: mean-seeking, penalizes the student for missing mass the teacher assigns. The teacher selects the top-K indices.
- **JSD** — `α·KL(p_student || M) + (1−α)·KL(p_teacher || M)` where `M = α·p_student + (1−α)·p_teacher`. Bounded in `[0, log 2]` regardless of distribution support, so more numerically stable than either pure KL. `jsd_alpha=0` = forward KL, `jsd_alpha=1` = reverse KL. SDPO's default.
- **`renormalize=True`** (default) — after `exp()` of top-K log-probs, renormalize to sum to 1 over K, since top-K truncation makes the raw sum < 1. Without it the KL formula compares a proper distribution against a sub-distribution.

## Key parameters

- **`distill-top-k`** (default 100) — top-K vocab truncation for the KL computation. Memory optimization (`[B, T, K]` not `[B, T, V]`) and signal-quality improvement (tail of the distribution is mostly noise).
- **`num-samples`** (default 4) — completions generated per prompt at each step. More samples reduce variance in the distillation signal at the cost of rollout compute.
- **`prompts-per-step`** (default 8) — distinct prompts sampled per step. `prompts-per-step × num-samples` = total rollouts per step; must be divisible by `train-world-size`.
- **`epochs`** (default 1) — optimizer steps on one rollout batch before collecting new rollouts. `>1` is PPO-style reuse and risks off-policy drift.
- **`temperature`** (default 1.0) — rollout sampling temperature. Don't use 0 (greedy) — distillation needs diversity in the student's on-policy distribution.
- **`ema-alpha`** (SDPO only, default 0.05) — `teacher ← α·student + (1−α)·teacher`. Small α = stable but lagging teacher.
- **`ema-sync-method`** (SDPO only) — `ema` (standard) or `trust_region` (`teacher ← α·student + (1−α)·ϕ₀`, blends with initial weights instead of the previous teacher — regularizes toward the pretrained distribution).
- **`max-grad-norm`** (default 1.0) — global L2 gradient-norm clip.
- **`student-chunk-size` / `teacher-chunk-size`** (default -1, no chunking) — process the `[B, T, V]` top-K selection in T-slices to reduce peak memory; only needed on OOM.
- **`sharding-strategy`** (default `FULL_SHARD`) — FSDP mode. `FULL_SHARD` shards params+grads+optimizer (max memory efficiency, most comms); `SHARD_GRAD_OP` shards only grads+optimizer state.

## Data flow for one training step

```
1. Sample prompts from dataset (student rank 0, then scatter)
2. Generate rollouts via vLLM worker (current student policy)
3. Prepare batch: tokenize, pad, build response_mask
4. For each minibatch:
   a. Student forward pass → student_logits [B, T, V]
   b. Select top-K indices (by student or teacher, depending on algorithm)
   c. Teacher forward pass → teacher log-probs at top-K indices
   d. Compute loss (reverse KL / forward KL / JSD)
   e. loss.backward(), accumulate gradients
5. Optimizer step
6. (SDPO only) EMA update: teacher ← α·student + (1-α)·teacher
7. Sync updated student weights → vLLM worker via NCCL
```

`response_mask` is 1 for response tokens, 0 for prompt tokens — loss is only computed over response tokens since the prompt is given, not generated by the student. The shift by 1 (`[:, 1:]` on mask, `[:, :-1]` on logits) aligns token t's logits (which predict token t+1) with token t+1's mask entry.

## Config

Hyperparameters for each training script live in `opd/examples/{opd,sdpo,opsd,sdft}.yaml` (OmegaConf), not argparse. Edit the YAML directly to change a run; `opd/examples/train_*.sh` are thin launcher scripts that read the YAML and handle process orchestration (vLLM worker startup, GPU placement). See each YAML's header comment for CLI override syntax.

## Files at a glance

| File | Purpose |
|------|---------|
| `opd/loss.py` | Reverse KL, forward KL, JSD loss functions |
| `opd/trainer/train_opd.py` | OPD training loop (external frozen teacher) |
| `opd/trainer/train_sdpo.py` | SDPO training loop (EMA self-teacher, feedback conditioned) |
| `opd/trainer/train_opsd.py` | OPSD training loop (frozen initial-policy teacher, reference-solution conditioned) |
| `opd/trainer/train_sdft.py` | SDFT training loop (EMA self-teacher, demonstration conditioned) |
| `opd/trainer/setup_utils.py` | Distributed init, device/seed setup, model construction, vLLM weight-transfer setup, `load_config` |
| `opd/trainer/distillation_utils.py` | Shared minibatch-exchange helpers (top-K exchange, PG exchange, response packing, broadcast utilities) |
| `opd/trainer/sync_teacher.py` | Self-teacher sync strategies (EMA, trust-region, hard-sync, on-policy) |
| `opd/generator/rollout.py` | vLLM HTTP client, batch preparation, weight sync |
| `opd/generator/rollout_worker.py` | The vLLM HTTP server that runs in a separate process |
| `opd/fsdp/algorithms.py` | Top-K log-prob selection (chunked, distributed) |
| `opd/fsdp/model.py` | Student (FSDP-wrapped) and Teacher model classes |
| `opd/envs/dataset.py` | Dataset loading and distributed data loader |
| `opd/envs/dapo_dataset.py` | DAPO math dataset exporter |
| `opd/eval/eval_math.py` | AIME 2024/2025 and HMMT 2025 evaluation (pass@k) |

## Entry points for hacking

- **New loss function**: add to `loss.py`, register in `ALGORITHMS`, add a choice to `--algorithm` in the training scripts.
- **New dataset**: add a file in `opd/envs/`, wire it into `build_opd_dataset()`.
- **Different EMA schedule**: `EMASyncer` in `opd/trainer/sync_teacher.py` is self-contained.
- **Different top-K selection**: `fsdp/algorithms.py` has the top-K helpers in isolation.
