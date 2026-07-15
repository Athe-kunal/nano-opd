#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$OPD_DIR/.." && pwd)"
BASE_DIR="${opd_BASE_DIR:-$REPO_ROOT/.opd}"

CONFIG_YAML="${CONFIG_YAML:-$OPD_DIR/examples/sdpo.yaml}"

# Reads a single key out of $CONFIG_YAML.
read_cfg() {
  uv run --extra gpu --directory "$REPO_ROOT" python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('$CONFIG_YAML')['$1'])
"
}
# In SDPO the teacher IS the student (EMA copy) — only one checkpoint needed.
STUDENT_MODEL="$(read_cfg student_model)"
DATASET="$(read_cfg dataset)"
NUM_STEPS="$(read_cfg num_steps)"

# GPU assignment (comma-separated physical GPU IDs)
#   TRAIN_GPUS    – student FSDP training ranks (0..N-1)
#   TEACHER_GPUS  – dedicated teacher rank (exactly 1 GPU, rank N in torchrun world)
#   ROLLOUT_GPUS  – vLLM rollout worker (may share with TEACHER_GPUS)
#   TRAIN_GPUS must not overlap ROLLOUT_GPUS or TEACHER_GPUS.
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
TEACHER_GPUS="${TEACHER_GPUS:-0}"

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8048}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"

USE_WANDB="${USE_WANDB:-1}"

# Post-training math evaluation (opd.eval.eval_math.run_eval).
# Reuses the already-running, weight-synced rollout worker — no separate
# vLLM engine, checkpoint loading, or LoRA needed.
# Set SKIP_EVAL=1 to bypass evaluation entirely regardless of dataset.
# Auto-enable only for math datasets; override with RUN_EVAL=0 or RUN_EVAL=1.
SKIP_EVAL="${SKIP_EVAL:-0}"
if [[ "$DATASET" == "dapo_math" || "$DATASET" == "opsd_math" ]]; then
  RUN_EVAL="${RUN_EVAL:-1}"
else
  RUN_EVAL="${RUN_EVAL:-0}"
fi
EVAL_DATASETS="${EVAL_DATASETS:-aime_2025,aime_2024,hmmt_2025}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-38912}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_VAL_N="${EVAL_VAL_N:-6}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-$TAG}"
EVAL_STEP="${EVAL_STEP:-$NUM_STEPS}"

RUN_DIR="$BASE_DIR/sdpo/$TAG"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$REPO_ROOT"
export opd_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT
export USE_WANDB

# ---------------------------------------------------------------------------
# Parse GPU lists
IFS=, read -r -a TRAIN_GPU_LIST   <<< "$TRAIN_GPUS"
IFS=, read -r -a ROLLOUT_GPU_LIST <<< "$ROLLOUT_GPUS"
IFS=, read -r -a TEACHER_GPU_LIST <<< "$TEACHER_GPUS"

TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
ROLLOUT_TP="${#ROLLOUT_GPU_LIST[@]}"
TEACHER_NPROC="${#TEACHER_GPU_LIST[@]}"
TOTAL_NPROC=$(( TRAIN_NPROC + TEACHER_NPROC ))

# Exactly one teacher rank is supported; the broadcast src is hard-coded to train_world_size.
if (( TEACHER_NPROC != 1 )); then
  echo "TEACHER_GPUS must specify exactly 1 GPU (got $TEACHER_NPROC: $TEACHER_GPUS). Multi-teacher rank is not supported." >&2
  exit 1
fi

# TRAIN_GPUS must be disjoint from ROLLOUT_GPUS and TEACHER_GPUS (rollout/teacher may share)
for tgpu in "${TRAIN_GPU_LIST[@]}"; do
  for rgpu in "${ROLLOUT_GPU_LIST[@]}"; do
    if [[ "$tgpu" == "$rgpu" ]]; then
      echo "TRAIN_GPUS ($TRAIN_GPUS) and ROLLOUT_GPUS ($ROLLOUT_GPUS) must not overlap" >&2
      exit 1
    fi
  done
  for cgpu in "${TEACHER_GPU_LIST[@]}"; do
    if [[ "$tgpu" == "$cgpu" ]]; then
      echo "TRAIN_GPUS ($TRAIN_GPUS) and TEACHER_GPUS ($TEACHER_GPUS) must not overlap" >&2
      exit 1
    fi
  done
done

# ---------------------------------------------------------------------------
WORKER_PID=""

kill_subtree_sig() {
  local sig=$1
  local pid=$2
  local children
  children=$(pgrep -P "$pid" 2>/dev/null) || true
  for child in $children; do
    kill_subtree_sig "$sig" "$child"
  done
  kill -s "$sig" "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -z "${WORKER_PID:-}" ]]; then
    return
  fi
  echo "[launcher] stopping rollout worker (pid/pgid=$WORKER_PID)"
  kill -TERM -- "-${WORKER_PID}" 2>/dev/null || true
  kill_subtree_sig TERM "$WORKER_PID"
  local _i=0
  while kill -0 "$WORKER_PID" 2>/dev/null && ((_i < 15)); do
    sleep 1
    _i=$((_i + 1))
  done
  kill -KILL -- "-${WORKER_PID}" 2>/dev/null || true
  kill_subtree_sig KILL "$WORKER_PID"
  wait "$WORKER_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

mkdir -p "$RUN_DIR" "$SAVE_DIR"

echo "[launcher] run tag         : $TAG"
echo "[launcher] run dir         : $RUN_DIR"
echo "[launcher] config          : $CONFIG_YAML"
echo "[launcher] student model   : $STUDENT_MODEL  (teacher = EMA of student, same checkpoint)"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC student ranks)"
echo "[launcher] teacher GPUs    : $TEACHER_GPUS  ($TEACHER_NPROC teacher rank)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"

# ---------------------------------------------------------------------------
# Start vLLM rollout worker (initialized with the student checkpoint)
echo "[launcher] starting rollout worker -> $WORKER_LOG"
setsid env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
  uv run --extra gpu --directory "$REPO_ROOT" python "$OPD_DIR/generator/rollout_worker.py" \
    --model "$STUDENT_MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEM_UTIL" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    --weight-transfer-backend "$WEIGHT_TRANSFER_BACKEND" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

echo "[launcher] waiting for rollout worker health"
HEALTH_URL="http://$ROLLOUT_HOST:$ROLLOUT_PORT/health"
for _ in $(seq 300); do
  if curl -sf "$HEALTH_URL" | grep -q '"ok": *true'; then
    echo "[launcher] rollout worker healthy"
    break
  fi
  sleep 1
done
curl -sf "$HEALTH_URL" | grep -q '"ok": *true' \
  || { echo "[launcher] rollout worker did not become healthy" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Start student trainer + teacher
# torchrun world: ranks 0..(TRAIN_NPROC-1) = student FSDP, rank TRAIN_NPROC = teacher.
# CUDA_VISIBLE_DEVICES maps logical indices to physical GPUs in order.
echo "[launcher] starting SDPO trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,$TEACHER_GPUS" \
  uv run --extra gpu --directory "$REPO_ROOT" torchrun --standalone --nproc_per_node="$TOTAL_NPROC" \
    "$OPD_DIR/trainer/train_sdpo.py" "$CONFIG_YAML" \
    train_world_size="$TRAIN_NPROC" \
    rollout_worker_url="http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    rollout_worker_world_size="$ROLLOUT_TP" \
    save_dir="$SAVE_DIR" \
    run_name="$TAG" \
    2>&1 | tee "$TRAIN_LOG"

# ---------------------------------------------------------------------------
# Post-training evaluation with opd.eval.eval_math.run_eval — talks to the
# still-running, weight-synced rollout worker over HTTP, so no separate vLLM
# engine or checkpoint loading happens here.
if [[ "$RUN_EVAL" == "1" && "$SKIP_EVAL" != "1" ]]; then
  echo "[launcher] running post-training eval on: $EVAL_DATASETS"
  uv run --extra gpu --directory "$REPO_ROOT" python - <<PYEOF 2>&1 | tee "$RUN_DIR/eval.log"
import wandb

from opd.eval.eval_math import run_eval

if "$EVAL_WANDB_PROJECT":
    wandb.init(project="$EVAL_WANDB_PROJECT", name="$EVAL_WANDB_RUN_NAME")

run_eval(
    rollout_worker_url="http://$ROLLOUT_HOST:$ROLLOUT_PORT",
    eval_k=$EVAL_VAL_N,
    eval_max_tokens=$EVAL_MAX_NEW_TOKENS,
    step=$EVAL_STEP,
    eval_datasets="$EVAL_DATASETS",
    temperature=$EVAL_TEMPERATURE,
    top_k=$EVAL_TOP_K,
)

if wandb.run is not None:
    wandb.finish()
PYEOF
fi
