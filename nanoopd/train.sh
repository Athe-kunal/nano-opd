#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

# Repo root is one level above this script (nanoopd/train.sh -> nano-opd/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${NANOOPD_BASE_DIR:-$ROOT_DIR/.nanoopd}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-open-thoughts/OpenThinker3-7B}"

# GPU assignment (comma-separated physical GPU IDs)
#   TRAIN_GPUS must not overlap ROLLOUT_GPUS or TEACHER_GPUS (enforced below).
#   ROLLOUT_GPUS and TEACHER_GPUS may be the same list to colocate vLLM + teacher.
#   ROLLOUT_GPUS  – vLLM rollout worker
#   TRAIN_GPUS    – student FSDP training ranks
#   TEACHER_GPUS  – teacher model ranks
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-3}"
TEACHER_GPUS="${TEACHER_GPUS:-2}"

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"

USE_WANDB="${USE_WANDB:-1}"

NUM_STEPS="${NUM_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LR="${LR:-5e-7}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
DATASET="${DATASET:-dapo_math}"
ALGORITHM="${ALGORITHM:-reverse_kl}"
DISTILL_TOP_K="${DISTILL_TOP_K:-100}"
STUDENT_CHUNK_SIZE="${STUDENT_CHUNK_SIZE:--1}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:--1}"
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

RUN_DIR="$BASE_DIR/opd/$TAG"
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
  # setsid makes WORKER_PID the session/process-group leader; kill the whole group first.
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
echo "[launcher] teacher model   : $TEACHER_MODEL"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC student ranks)"
echo "[launcher] teacher GPUs    : $TEACHER_GPUS  ($TEACHER_NPROC teacher ranks)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"
echo "[launcher] algorithm       : $ALGORITHM"
echo "[launcher] sharding        : $SHARDING_STRATEGY"

# ---------------------------------------------------------------------------
# Start vLLM rollout worker
# setsid: new session so WORKER_PID is a process-group leader; cleanup can
# signal the whole group (uv, Python, vLLM workers) with kill -TERM -- -PID.
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
# Start student trainer + teacher
# torchrun world: ranks 0..(TRAIN_NPROC-1) = student, ranks TRAIN_NPROC..(TOTAL_NPROC-1) = teacher.
# CUDA_VISIBLE_DEVICES maps logical device indices to physical GPUs in that order.
echo "[launcher] starting trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,$TEACHER_GPUS" \
  uv run --extra gpu torchrun --standalone --nproc_per_node="$TOTAL_NPROC" \
    nanoopd/train_opd.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --train-world-size "$TRAIN_NPROC" \
    --dataset "$DATASET" \
    --algorithm "$ALGORITHM" \
    --distill-top-k "$DISTILL_TOP_K" \
    --student-chunk-size "$STUDENT_CHUNK_SIZE" \
    --teacher-chunk-size "$TEACHER_CHUNK_SIZE" \
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
