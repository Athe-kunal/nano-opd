# nano-opd

**nano-opd** is a hackable library for on-policy distillation: you can swap models, data, and training knobs without fighting a monolithic stack.

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

Values in the YAML marked `???` (e.g. `train_world_size`) must be overridden — either in the file or via a `key=value` override — before the trainer will run. `train_world_size` is left `???` deliberately: it's hardware-dependent and the launcher scripts always supply it, computed from the GPU list you pass in (see below).

You can also invoke a training script directly, without a launcher, passing the config path and any overrides as trailing `key=value` args:

```bash
python opd/trainer/train_opd.py opd/examples/opd.yaml train_world_size=1 lr=1e-5 num_steps=50
```

### GPUs

The one thing the YAML doesn't hold is GPU placement — it's deployment-specific, so it stays as environment variables on the launcher. Comma-separated physical IDs. **`TRAIN_GPUS` must not overlap** with **`ROLLOUT_GPUS`** or **`TEACHER_GPUS`** (the launcher checks this). **`ROLLOUT_GPUS`** and **`TEACHER_GPUS`** **may reuse the same IDs** if you want to **colocate** the vLLM rollout worker and the teacher on the same GPUs; give them disjoint IDs if you want those roles on separate devices.

Example (student on GPU 3; teacher and rollouts both on GPU 2):

```bash
ROLLOUT_GPUS=2 TRAIN_GPUS=3 TEACHER_GPUS=2 bash opd/examples/train_opd.sh
```

- **`ROLLOUT_GPUS`** — GPUs for the vLLM rollout worker (student sampling). Multiple IDs set tensor-parallel size for rollout.
- **`TRAIN_GPUS`** — GPUs for FSDP student training (`torchrun` ranks for the student).
- **`TEACHER_GPUS`** — GPUs for the teacher forward passes (can match `ROLLOUT_GPUS` to colocate with vLLM).

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

### Chunking (optional, in the YAML)

- **`student_chunk_size`** / **`teacher_chunk_size`** — Chunk sizes for long-sequence handling; `-1` means "don't chunk." Only needed if you hit OOM errors.

### Running

Edit the YAML for hyperparameters; use GPU environment variables for placement, for example:

```bash
TRAIN_GPUS=0,1 bash opd/examples/train_opd.sh my_run
```
