# nano-opd

**nano-opd** is a hackable library for on-policy distillation: you can swap models, data, and training knobs without fighting a monolithic stack.

## Exporting training data (`dapo.py`)

The DAPO exporter writes JSONL rows that match the `Example` type used by the trainer's dataset code. From the repo root (with `PYTHONPATH` set as you normally would for this package), you can run:

**English** (default Hugging Face config):

```bash
python -m nanoopd.data.dapo -o datasets/dapo_math.jsonl
```

## Main parameters in `train.sh`

The script takes an optional **run tag** as the first argument (default: `default`). Artifacts go under `NANOCHAT_BASE_DIR` (default: `.nanoopd` under the repo), in `opd/<tag>/` (logs, checkpoints).

### Models

- **`STUDENT_MODEL`** / **`TEACHER_MODEL`** — Hugging Face IDs (or paths) for the small student and the teacher used for distillation.

### GPUs

Comma-separated physical IDs. **`TRAIN_GPUS` must not overlap** with **`ROLLOUT_GPUS`** or **`TEACHER_GPUS`** (the launcher checks this). **`ROLLOUT_GPUS`** and **`TEACHER_GPUS`** **may reuse the same IDs** if you want to **colocate** the vLLM rollout worker and the teacher on the same GPUs; give them disjoint IDs if you want those roles on separate devices.

Example (student on GPU 3; teacher and rollouts both on GPU 2):

```bash
ROLLOUT_GPUS=2 TRAIN_GPUS=3 TEACHER_GPUS=2 bash nanoopd/train.sh
```

- **`ROLLOUT_GPUS`** — GPUs for the vLLM rollout worker (student sampling). Multiple IDs set tensor-parallel size for rollout.
- **`TRAIN_GPUS`** — GPUs for FSDP student training (`torchrun` ranks for the student).
- **`TEACHER_GPUS`** — GPUs for the teacher forward passes (can match `ROLLOUT_GPUS` to colocate with vLLM).

### Rollout worker

- **`ROLLOUT_HOST`** / **`ROLLOUT_PORT`** — Where the trainer talks to the vLLM HTTP server.
- **`ROLLOUT_GPU_MEM_UTIL`** — vLLM GPU memory fraction.
- **`WEIGHT_TRANSFER_BACKEND`** — Backend for weight sync to the worker (e.g. `nccl`).

### Training loop and batching

- **`NUM_STEPS`** — Optimization steps.
- **`PROMPTS_PER_STEP`** — How many prompts to roll out per step (on-policy data volume).
- **`NUM_SAMPLES`** — Samples per prompt (for variance / ranking in the objective).
- **`TRAIN_BATCH_SIZE`** — Per-step training microbatch behavior (as wired into `train.py`).
- **`LR`** — Learning rate.

### Sequence and objective

- **`MAX_NEW_TOKENS`** — Generation cap for rollouts.
- **`MAX_SEQ_LEN`** — Truncation / packing ceiling for training sequences.
- **`ALGORITHM`** — Distillation loss variant (default: `reverse_kl`).
- **`DISTILL_TOP_K`** — Top-k used when matching teacher distribution (semantics follow `train.py`).

### Checkpointing and eval

- **`SAVE_EVERY`** — Checkpoint frequency.
- **`EVAL_EVERY`** / **`EVAL_K`** / **`EVAL_MAX_TOKENS`** — How often to run eval and generation limits for eval.

### FSDP

- **`SHARDING_STRATEGY`** — FSDP sharding mode (e.g. `FULL_SHARD`).

### Chunking (optional)

- **`STUDENT_CHUNK_SIZE`** / **`TEACHER_CHUNK_SIZE`** — Chunk sizes for long-sequence handling; `-1` typically means “don’t chunk” (see how `train.py` interprets them).

### Running

Override variables as environment variables when invoking the script, for example:

```bash
STUDENT_MODEL=... TRAIN_GPUS=0,1 bash nanoopd/train.sh my_run
```
