# nano-opd

**nano-opd** is a hackable library for on-policy distillation: you can swap models, data, and training knobs without fighting a monolithic stack.

This follows my YouTube [video](https://youtu.be/iiYUccK4nJU?si=_-b0qU_1IYCTsW4j) on the same topic.

## Algorithms

| Algorithm | Teacher | Paper |
|---|---|---|
| **OPD** | separate frozen model | [On-Policy Distillation of Language Models](https://thinkingmachines.ai/blog/on-policy-distillation/) |
| **SDPO** | EMA of student, feedback-conditioned | [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) |
| **OPSD** | frozen initial policy, reference-solution-conditioned | [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) |
| **SDFT** | EMA of student, demonstration-conditioned | [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) |

Losses: reverse KL, forward KL, JSD, `mopd`, `mopd_pg` — all top-K truncated (`opd/loss.py`).

## Exporting training data (`dapo_dataset.py`)

The DAPO exporter writes JSONL rows that match the `Example` type used by the trainer's dataset code. From the repo root (with `PYTHONPATH` set as you normally would for this package), you can run:

**English** (default Hugging Face config):

```bash
python -m opd.envs.dapo -o datasets/dapo_math.jsonl
```

## Training scripts and config

Each algorithm has a launcher under `opd/examples/`: `train_opd.sh`, `train_sdpo.sh`, `train_opsd.sh`, `train_sdft.sh`. Every launcher takes an optional **run tag** as its first argument (default: `default`). Artifacts go under `opd_BASE_DIR` (default: `.opd` under the repo), in `<algorithm>/<tag>/` (logs, checkpoints).

**All hyperparameters — model names, algorithm, learning rate, batch sizes, eval cadence, teacher-sync settings, and so on — live in a YAML file**, not in the shell script or as environment variables. Each launcher has a matching config:

| Launcher | Config |
|---|---|
| `opd/examples/train_opd.sh` | `opd/examples/opd.yaml` |
| `opd/examples/train_sdpo.sh` | `opd/examples/sdpo.yaml` |
| `opd/examples/train_opsd.sh` | `opd/examples/opsd.yaml` |
| `opd/examples/train_sdft.sh` | `opd/examples/sdft.yaml` |

To change a hyperparameter, edit the YAML directly. To point a launcher at a different config file, set `CONFIG_YAML`:

```bash
CONFIG_YAML=/path/to/my_opd.yaml bash opd/examples/train_opd.sh my_run
```

Values marked `???` in a YAML must be overridden — in the file or via a `key=value` override — before the trainer runs. `train_world_size` is hardware-dependent; the launchers always supply it from `train_gpus` (see below), overriding whatever the YAML holds.

You can also invoke a training script directly, without a launcher, passing the config path and any overrides as trailing `key=value` args:

```bash
python opd/trainer/train_opd.py opd/examples/opd.yaml train_world_size=1 lr=1e-5 num_steps=50
```

### GPUs

GPU placement is in the YAML, as three comma-separated lists:

- **`rollout_gpus`** — vLLM rollout worker. Multiple IDs set rollout tensor-parallel size.
- **`train_gpus`** — FSDP student training, one rank each. Sets `train_world_size`.
- **`teacher_gpus`** — teacher forward passes. Exactly one GPU.

Rules (enforced by the launcher):

- `train_gpus` must not overlap `rollout_gpus` or `teacher_gpus`.
- `rollout_gpus` and `teacher_gpus` may reuse the same IDs, to colocate the teacher with vLLM.
- `CUDA_VISIBLE_DEVICES` set -> YAML GPU fields are 0-based indices into it (`CUDA_VISIBLE_DEVICES=2,3` + `rollout_gpus: "0"` = physical GPU 2). Unset -> they are physical IDs.

### Rollout worker (in the YAML)

- **`rollout_host`** / **`rollout_port`** — Where the trainer talks to the vLLM HTTP server.
- **`rollout_gpu_mem_util`** — vLLM GPU memory fraction.
- **`weight_transfer_backend`** — Backend for weight sync to the worker (e.g. `nccl`).
- **`use_wandb`** — Whether to log the run to Weights & Biases.

### Training loop, batching, and objective (in the YAML)

- **`num_steps`** — Optimization steps.
- **`prompts_per_step`** — How many prompts to roll out per step (on-policy data volume).
- **`num_samples`** — Samples per prompt (OPD/SDPO only; OPSD/SDFT roll out one trajectory per prompt).
- **`train_batch_size`** — Sequences per gradient-accumulation microbatch.
- **`lr`** — Learning rate.
- **`max_new_tokens`** — Generation cap for rollouts.
- **`max_prompt_len`** / **`max_response_len`** — Truncation ceilings for training sequences.
- **`algorithm`** — Distillation loss variant (`reverse_kl`, `forward_kl`, `jsd`, `mopd_loss`, `mopd_pg_loss`; default differs per script).
- **`distill_top_k`** — Top-K vocab used when matching the teacher distribution.

See the comments in each `opd/examples/*.yaml` for the full list — every field is documented inline, grouped by Model / Algorithm / Generation / Training / Runtime.

### Distillation health metrics (logged to W&B)

Three token-level metrics are logged under `metrics/` each step:

- **`overlap_ratio`** — fraction of tokens shared between student and teacher top-K sets. Rising over training indicates the student is finding the teacher's high-probability region. Stagnant overlap is a sign of a failing run.
- **`overlap_token_advantage`** — within the shared tokens, measures whether the student's probability mass matches the teacher's. A value near zero means good calibration; negative means the student is overconfident relative to the teacher.
- **`entropy_gap`** — absolute difference in entropy between teacher and student distributions at each token position. A narrowing gap means the student is matching the teacher's confidence level. A persistent gap means the student has collapsed to sharper modes than the teacher.

### Checkpointing and eval (in the YAML)

- **`save_every`** — Checkpoint frequency.
- **`eval_every`** / **`eval_k`** / **`eval_max_tokens`** — How often to run eval and generation limits for eval.

### Post-training math eval (in the YAML; `opd`/`sdpo`/`opsd` only)

After training finishes, these launchers run a one-off AIME/HMMT eval (`opd.eval.eval_math.run_eval`) against the still-running, weight-synced rollout worker — no separate vLLM engine or checkpoint loading needed.

- **`skip_eval`** — Set `true` to bypass this step entirely.
- **`run_eval`** — Whether to run it (only meaningful for math datasets).
- **`posttrain_eval_datasets`** — Comma-separated eval sets, e.g. `aime_2025,aime_2024,hmmt_2025`.
- **`posttrain_eval_max_new_tokens`** / **`posttrain_eval_temperature`** / **`posttrain_eval_top_k`** — Generation settings for eval rollouts.
- **`posttrain_eval_val_n`** — Samples per problem for pass@k.
- **`posttrain_eval_wandb_project`** / **`posttrain_eval_wandb_run_name`** — Optional separate W&B project/run for the eval; `posttrain_eval_wandb_run_name` defaults to `${run_name}` via OmegaConf interpolation.
- **`posttrain_eval_step`** — Step number logged alongside eval results; defaults to `${num_steps}`.

### FSDP (in the YAML)

- **`sharding_strategy`** — FSDP sharding mode (e.g. `FULL_SHARD`).

### Chunking and OOM (in the YAML)

- **`student_chunk_size`** / **`teacher_chunk_size`** — chunk the top-K log-prob gather along T; `-1` = don't chunk.

Chunking does **not** shrink peak memory on the student. That peak is the `[B, T, V]` logits tensor plus its gradient (`train_batch_size × seq_len × vocab × 2 bytes`, doubled for backward), which the forward materializes regardless. To actually cut it, in order of leverage:

1. Lower **`max_new_tokens`** / **`max_response_len`** — `T` also sets rollout wall-clock, so this is the only knob that helps speed too.
2. Lower **`train_batch_size`**.
3. Add GPUs to **`train_gpus`** — `FULL_SHARD` is a no-op at one rank (FSDP silently falls back to `NO_SHARD`); a second rank shards params/grads/optimizer state.

### Running

Edit the YAML (hyperparameters and GPU placement both live there), then:

```bash
bash opd/examples/train_opd.sh my_run
```
