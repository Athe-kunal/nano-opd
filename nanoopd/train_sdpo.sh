#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

# Repo root is one level above this script (nanoopd/train_sdpo.sh -> nano-opd/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${NANOOPD_BASE_DIR:-$ROOT_DIR/.nanoopd}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# GPU assignment (comma-separated physical GPU IDs).
# TRAIN_GPUS and ROLLOUT_GPUS must not overlap.
# TEACHER_GPU_ID: physical GPU for the EMA teacher (empty = same as student).
# Setting it to a rollout GPU keeps trainer VRAM free; set ROLLOUT_GPU_MEM_UTIL=0.5
# so vLLM and the teacher share the rollout GPU's memory.
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
TEACHER_GPU_ID="${TEACHER_GPU_ID:-}"  # e.g. TEACHER_GPU_ID=1 to colocate with rollout

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"

USE_WANDB="${USE_WANDB:-1}"

NUM_STEPS="${NUM_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LR="${LR:-1e-6}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

# Algorithm: reverse_kl | forward_kl | jsd (default: jsd — symmetric JSD, SDPO paper default)
ALGORITHM="${ALGORITHM:-jsd}"
DISTILL_TOP_K="${DISTILL_TOP_K:-100}"
STUDENT_CHUNK_SIZE="${STUDENT_CHUNK_SIZE:--1}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:--1}"
JSD_ALPHA="${JSD_ALPHA:-0.5}"  # 0.5=symmetric JSD, 0.0=forward KL, 1.0=reverse KL

# EMA teacher sync
# ema:          ϕ ← α·θ + (1-α)·ϕ          (recommended)
# trust_region: ϕ ← α·θ + (1-α)·ϕ₀         (anchors to initial weights)
EMA_ALPHA="${EMA_ALPHA:-0.05}"
EMA_SYNC_METHOD="${EMA_SYNC_METHOD:-ema}"

EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_K="${EVAL_K:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"

# FSDP sharding strategy — choose one of:
#   FULL_SHARD          params+grads+optimizer sharded; unshard around fwd/bwd
#   SHARD_GRAD_OP       params sharded; keep unsharded after fwd until bwd done
#   NO_SHARD            replicate everything (like DDP)
#   HYBRID_SHARD        FULL_SHARD within a node, replicate across nodes
#   _HYBRID_SHARD_ZERO2 SHARD_GRAD_OP within a node, replicate across nodes
SHARDING_STRATEGY="${SHARDING_STRATEGY:-NO_SHARD}"

RUN_DIR="$BASE_DIR/sdpo/$TAG"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$ROOT_DIR"
export NANOOPD_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT
export USE_WANDB

# ---------------------------------------------------------------------------
# Parse GPU lists
IFS=, read -r -a TRAIN_GPU_LIST   <<< "$TRAIN_GPUS"
IFS=, read -r -a ROLLOUT_GPU_LIST <<< "$ROLLOUT_GPUS"

TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
ROLLOUT_TP="${#ROLLOUT_GPU_LIST[@]}"

# TRAIN_GPUS must be disjoint from ROLLOUT_GPUS
for tgpu in "${TRAIN_GPU_LIST[@]}"; do
  for rgpu in "${ROLLOUT_GPU_LIST[@]}"; do
    if [[ "$tgpu" == "$rgpu" ]]; then
      echo "TRAIN_GPUS ($TRAIN_GPUS) and ROLLOUT_GPUS ($ROLLOUT_GPUS) must not overlap" >&2
      exit 1
    fi
  done
done

# Build the trainer's CUDA_VISIBLE_DEVICES and teacher device index.
# If TEACHER_GPU_ID is set, append it so the trainer process can access it;
# the teacher's CUDA device index is then TRAIN_NPROC (after all train ranks).
if [[ -n "${TEACHER_GPU_ID:-}" ]]; then
  TRAINER_VISIBLE="${TRAIN_GPUS},${TEACHER_GPU_ID}"
  TEACHER_DEVICE_IDX="${TRAIN_NPROC}"
else
  TRAINER_VISIBLE="${TRAIN_GPUS}"
  TEACHER_DEVICE_IDX="-1"
fi

# ---------------------------------------------------------------------------
WORKER_PID=""

# Send signal to pid and all descendants (deepest first). Handles vLLM/workers
# that are not in the setsid process group.
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
echo "[launcher] student model   : $STUDENT_MODEL"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC ranks)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"
echo "[launcher] algorithm       : $ALGORITHM  (jsd-alpha=$JSD_ALPHA)"
echo "[launcher] EMA             : alpha=$EMA_ALPHA  method=$EMA_SYNC_METHOD"
echo "[launcher] sharding        : $SHARDING_STRATEGY"

# ---------------------------------------------------------------------------
# Start vLLM rollout worker
echo "[launcher] starting rollout worker -> $WORKER_LOG"
setsid env CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
  uv run --extra gpu python "$ROOT_DIR/nanoopd/rollout_worker.py" \
    --model "$STUDENT_MODEL" \
    --host "$ROLLOUT_HOST" \
    --port "$ROLLOUT_PORT" \
    --gpu-memory-utilization "$ROLLOUT_GPU_MEM_UTIL" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    --weight-transfer-backend "$WEIGHT_TRANSFER_BACKEND" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID="$!"

# Wait for the worker to become healthy
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
# Start student trainer (all ranks are student ranks — no separate teacher rank)
echo "[launcher] starting trainer -> $TRAIN_LOG"
echo "[launcher] teacher GPU      : ${TEACHER_GPU_ID:-<same as student>}  (device idx=${TEACHER_DEVICE_IDX})"
CUDA_VISIBLE_DEVICES="$TRAINER_VISIBLE" \
  uv run --extra gpu torchrun --standalone --nproc_per_node="$TRAIN_NPROC" \
    nanoopd/train_sdpo.py \
    --teacher-gpu-id "$TEACHER_DEVICE_IDX" \
    --student-model "$STUDENT_MODEL" \
    --algorithm "$ALGORITHM" \
    --distill-top-k "$DISTILL_TOP_K" \
    --student-chunk-size "$STUDENT_CHUNK_SIZE" \
    --teacher-chunk-size "$TEACHER_CHUNK_SIZE" \
    --jsd-alpha "$JSD_ALPHA" \
    --ema-alpha "$EMA_ALPHA" \
    --ema-sync-method "$EMA_SYNC_METHOD" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-worker-world-size "$ROLLOUT_TP" \
    --num-steps "$NUM_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --num-samples "$NUM_SAMPLES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lr "$LR" \
    --sharding-strategy "$SHARDING_STRATEGY" \
    --save-dir "$SAVE_DIR" \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --eval-k "$EVAL_K" \
    --eval-max-tokens "$EVAL_MAX_TOKENS" \
    --run-name "$TAG" \
    2>&1 | tee "$TRAIN_LOG"
