# nano-opd — CLAUDE.md

## What this library is

**nano-opd** is a **pedagogical and hackable** library for on-policy distillation (OPD) of language models. Every design choice is meant to be readable, modifiable, and understandable. There is no magic framework hiding the training loop — the loop is right there in `train_opd.py` and `train_sdpo.py`, readable top-to-bottom.

If you are here to learn how on-policy distillation works, you are in the right place. If you are here to swap in a new loss function, a different teacher schedule, or a new dataset, you can do that in a few lines.

## How Claude should interact with the user

This project is **pedagogical first**. When the user asks about any part of the code:

1. **Quiz before explaining.** Ask the user what they think first. For example, if they ask "what does `distill-top-k` do?", ask them to guess before explaining. This is not gatekeeping — it builds genuine understanding.
2. **Explain the WHY behind every parameter and design choice**, not just the what. Parameters exist for algorithmic reasons; those reasons should be surfaced.
3. **Explain the math** behind every loss function and algorithm in plain language, then connect it to the code.
4. **Flag misconceptions** gently but directly if the user's framing suggests a wrong mental model.
5. **Suggest experiments.** After explaining a concept, offer a concrete thing the user can change in the code to see the effect.
6. **Act as a typing partner, not an autonomous agent.** When making code changes, do a few lines or one function at a time. Explain each change in detail before or as you make it. Never rewrite large blocks in one shot. The user should feel like they are coding alongside you, not watching you work.

---

## Architecture overview

```
trainer (FSDP, train_opd.py / train_sdpo.py)
    │
    ├── student model (FSDP-wrapped, updated by optimizer)
    ├── teacher model (frozen or EMA-updated)
    │
    └──► rollout worker (separate process, vLLM HTTP server)
              └── generates completions under current student policy
```

The trainer and rollout worker are **separate processes** communicating over HTTP. After each training step, the trainer pushes updated student weights into the vLLM worker via NCCL (no checkpoint files involved).

---

## The two training scripts

### `train_opd.py` — On-Policy Distillation with a separate teacher

The student and a frozen external teacher run in the **same torchrun job** but on different ranks. Ranks `0..train_world_size-1` are student ranks (FSDP); rank `train_world_size` is the teacher rank. They communicate via `dist.broadcast`.

**Why separate ranks instead of separate processes?** It lets NCCL broadcast the tiny `[B, T, K]` teacher log-prob tensor directly from the teacher GPU to all student GPUs in one collective, avoiding serialization overhead. The key insight: instead of broadcasting the full `[B, T, V]` logit tensor (V ≈ 128,000 for a large vocab), we only broadcast the top-K slice — a ~1000× reduction.

### `train_sdpo.py` — Self-Policy Distillation (SDPO)

There is **no separate teacher model**. The teacher is an EMA copy of the student, living as a plain `nn.Module` on every rank. Every rank runs both student and teacher forward passes locally — no cross-rank broadcast of teacher outputs needed.

**Why is this interesting?** The teacher is always chasing the student, so the distillation target is never too far ahead. This is analogous to how target networks work in deep RL (DQN-style).

---

## Loss functions (`opd/loss.py`)

All losses operate on **top-K truncated distributions**, not the full vocabulary. The tensors have shape `[B, T, K]` where K = `distill-top-k`.

### Reverse KL — `KL(p_student || p_teacher)`

```
loss = Σ_k p_student(k) · [log p_student(k) − log p_teacher(k)]
```

**The student is in the first argument.** It asks: "where does the student put probability mass that the teacher does not?" This is **mode-seeking** — the student is penalized for being spread out in places where the teacher is concentrated. In practice, reverse KL tends to produce sharper, more confident outputs.

**Top-K selection:** the student selects the K indices (tokens) used. This is natural for reverse KL because the student-weighted sum means we only need tokens where the student has non-negligible probability.

### Forward KL — `KL(p_teacher || p_student)`

```
loss = Σ_k p_teacher(k) · [log p_teacher(k) − log p_student(k)]
```

**The teacher is in the first argument.** It asks: "where does the teacher put probability mass that the student misses?" This is **mean-seeking** — the student is penalized for assigning low probability to tokens the teacher likes. It tends to produce more diverse, coverage-seeking behavior.

**Top-K selection:** the teacher selects the K indices. This is correct because the teacher-weighted sum means we need tokens where the teacher has non-negligible probability.

### JSD — Jensen-Shannon Divergence (SDPO default)

```
M = α · p_student + (1−α) · p_teacher
JSD = α · KL(p_student || M) + (1−α) · KL(p_teacher || M)
```

With `jsd_alpha=0.5` (default), this is the symmetric JSD — it penalizes both the student for deviating from the mixture and the teacher for deviating from the mixture. JSD is always bounded in [0, log 2] regardless of distribution support, making training more numerically stable than either KL variant alone.

**Key insight:** JSD interpolates smoothly between forward KL (`jsd_alpha=0.0`) and reverse KL (`jsd_alpha=1.0`). The SDPO paper uses symmetric JSD as its default because it avoids the mode-seeking / coverage problems of the pure KL variants.

### `renormalize=True` (default)

After taking `exp()` of top-K log-probs, the probabilities are renormalized to sum to 1 over the K tokens. Why? Because the top-K truncation makes the sum < 1. Without renormalization, the KL formula would be comparing a proper distribution against a sub-distribution, introducing a systematic bias. With renormalization, we are computing the KL between two conditional distributions restricted to the top-K support.

---

## Key parameters explained

### `distill-top-k` (default: 100)

Instead of computing KL over the full vocabulary V (e.g., 128,000 tokens), we only use the top-K tokens. This is both a **memory optimization** (broadcast tensor is `[B, T, K]` not `[B, T, V]`) and a **signal quality improvement** (the tail of the distribution contributes negligible KL mass but adds noise).

Setting K too low loses coverage of valid alternatives. Setting K too high wastes memory and communication bandwidth. 100 is a practical sweet spot for most models.

### `num-samples` (default: 4)

How many completions to generate per prompt at each step. More samples give a better estimate of the student's current distribution and reduce variance in the distillation signal. But each sample adds rollout cost. In GRPO/DAPO-style RL this parameter controls the group size for advantage estimation — here it controls the diversity of sequences the distillation loss is computed over.

### `prompts-per-step` (default: 8)

How many distinct prompts to sample from the dataset per training step. Combined with `num-samples`, the total rollouts per step = `prompts-per-step × num-samples`. This must be divisible by `train-world-size` so each student rank gets an equal slice.

### `epochs` (default: 1)

How many optimizer steps to take on a single rollout batch before collecting new rollouts. Setting `epochs > 1` is the on-policy equivalent of PPO's multiple epochs — it reuses rollout data but risks off-policy drift because the rollouts were generated by an earlier checkpoint of the student.

### `temperature` (default: 1.0)

Sampling temperature for rollout generation. Temperature=1.0 means sampling from the true model distribution. Lower temperatures make the model greedier (less diverse rollouts). Higher temperatures increase diversity but may produce lower-quality completions. For distillation, it is important to not use temperature=0 (greedy) because we need diversity in the student's on-policy distribution.

### `ema-alpha` (SDPO only, default: 0.05)

The rate at which the EMA teacher tracks the student:
```
teacher ← α · student + (1−α) · teacher
```
A small α (e.g. 0.05) means the teacher changes slowly — more stable distillation target, but the teacher lags behind the student significantly. A large α means the teacher tracks the student closely — less lag, but potentially unstable if the student changes quickly.

### `ema-sync-method` (SDPO only)

- **`ema`**: standard exponential moving average, as above. The teacher drifts toward the student over time.
- **`trust_region`**: instead of blending with the previous teacher, blend with the **initial weights** ϕ₀:
  ```
  teacher ← α · student + (1−α) · ϕ₀
  ```
  This prevents the teacher from drifting too far from the initialization, acting as a regularizer that keeps both student and teacher anchored to the pretrained distribution.

### `max-grad-norm` (default: 1.0)

Gradient clipping threshold. Clips the global L2 norm of all gradients to this value before the optimizer step. This prevents catastrophic gradient spikes during distillation, which can happen when the student and teacher distributions diverge suddenly.

### `student-chunk-size` / `teacher-chunk-size` (default: -1, no chunking)

When processing long sequences, the `[B, T, V]` logit tensor can be too large to hold in GPU memory for top-K selection. Chunking along the T dimension processes the sequence in slices, reducing peak memory at the cost of more kernel launches. `-1` means process the full T at once. Start with chunking only if you hit OOM errors.

### `sharding-strategy` (default: `FULL_SHARD`)

FSDP (Fully Sharded Data Parallel) sharding mode. `FULL_SHARD` shards both parameters and gradients across all student ranks — maximum memory efficiency but highest communication cost. `SHARD_GRAD_OP` only shards gradients and optimizer states. Choose based on your GPU memory budget.

---

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

The `response_mask` is critical: it is 1 for response tokens and 0 for prompt tokens. The loss is only computed over response tokens because the prompt is given (not generated by the student), and distilling the prompt region would just train the student to match the teacher on the fixed input context — not useful.

The shift by 1 (`[:, 1:]` on mask, `[:, :-1]` on logits) aligns token t's logits (which predict token t+1) with token t+1's mask entry.

---

## Files at a glance

| File | Purpose |
|------|---------|
| `opd/loss.py` | Reverse KL, forward KL, JSD loss functions |
| `opd/train_opd.py` | OPD training loop (external frozen teacher) |
| `opd/train_sdpo.py` | SDPO training loop (EMA self-teacher) |
| `opd/rollout.py` | vLLM HTTP client, batch preparation, weight sync |
| `opd/rollout_worker.py` | The vLLM HTTP server that runs in a separate process |
| `opd/fsdp/algorithms.py` | Top-K log-prob selection (chunked, distributed) |
| `opd/fsdp/model.py` | Student (FSDP-wrapped) and Teacher model classes |
| `opd/data/dataset.py` | Dataset loading and distributed data loader |
| `opd/data/dapo.py` | DAPO math dataset exporter |
| `opd/eval_aime.py` | AIME evaluation (pass@k) |

## Good entry points for hacking

- **New loss function**: add a function to `loss.py`, register it in `ALGORITHMS`, add a choice to the `--algorithm` argparse argument in both training scripts.
- **New dataset**: add a file in `opd/data/`, make it produce `Example` objects (see `dataset.py`), wire it into `build_opd_dataset()`.
- **Different EMA schedule**: the `update_teacher_ema` function in `train_sdpo.py` is self-contained — swap in a warmup, a cyclical schedule, or a per-layer alpha.
- **Different top-K selection**: `fsdp/algorithms.py` contains the top-K helpers in isolation — easy to swap for top-p, nucleus sampling-style selection, or importance-weighted sampling.
