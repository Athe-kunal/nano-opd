#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

# Repo root is one level above this script (nanoopd/train.sh -> nano-opd/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/.nanoopd}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-open-thoughts/OpenThinker3-7B}"

# GPU assignment (comma-separated physical GPU IDs, must not overlap)
#   ROLLOUT_GPUS  – vLLM rollout worker
#   TRAIN_GPUS    – student FSDP training ranks
#   TEACHER_GPU   – single GPU (must be in TRAIN_GPUS) that loads the teacher
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
TEACHER_GPU="${TEACHER_GPU:-2}"

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.9}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"

NUM_STEPS="${NUM_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LR="${LR:-1e-6}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-3072}"
ALGORITHM="${ALGORITHM:-reverse_kl}"
DISTILL_TOP_K="${DISTILL_TOP_K:-100}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_K="${EVAL_K:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
# FSDP sharding strategy — choose one of:
#   FULL_SHARD          params+grads+optimizer sharded; unshard around fwd/bwd
#   SHARD_GRAD_OP       params sharded; keep unsharded after fwd until bwd done
#   NO_SHARD            replicate everything (like DDP)
#   HYBRID_SHARD        FULL_SHARD within a node, replicate across nodes
#   _HYBRID_SHARD_ZERO2 SHARD_GRAD_OP within a node, replicate across nodes
SHARDING_STRATEGY="${SHARDING_STRATEGY:-FULL_SHARD}"

RUN_DIR="$BASE_DIR/opd/$TAG"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$ROOT_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT

# ---------------------------------------------------------------------------
# Validate GPU assignments
IFS=, read -r -a TRAIN_GPU_LIST   <<< "$TRAIN_GPUS"
IFS=, read -r -a ROLLOUT_GPU_LIST <<< "$ROLLOUT_GPUS"

TRAIN_NPROC="${#TRAIN_GPU_LIST[@]}"
ROLLOUT_TP="${#ROLLOUT_GPU_LIST[@]}"

# Rollout and train GPUs must not overlap
for tgpu in "${TRAIN_GPU_LIST[@]}"; do
  for rgpu in "${ROLLOUT_GPU_LIST[@]}"; do
    if [[ "$tgpu" == "$rgpu" ]]; then
      echo "TRAIN_GPUS ($TRAIN_GPUS) and ROLLOUT_GPUS ($ROLLOUT_GPUS) must not overlap" >&2
      exit 1
    fi
  done
done

# Compute the rank ID of TEACHER_GPU within TRAIN_GPUS
TEACHER_RANK_ID=""
for i in "${!TRAIN_GPU_LIST[@]}"; do
  if [[ "${TRAIN_GPU_LIST[$i]}" == "$TEACHER_GPU" ]]; then
    TEACHER_RANK_ID="$i"
    break
  fi
done
if [[ -z "$TEACHER_RANK_ID" ]]; then
  echo "TEACHER_GPU ($TEACHER_GPU) must be one of TRAIN_GPUS ($TRAIN_GPUS)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
WORKER_PID=""

kill_subtree() {
  local pid=$1
  local children
  children=$(pgrep -P "$pid" 2>/dev/null) || true
  for child in $children; do
    kill_subtree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  if [[ -n "$WORKER_PID" ]]; then
    echo "[launcher] killing rollout worker subtree (root pid=$WORKER_PID)"
    kill_subtree "$WORKER_PID"
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "$RUN_DIR" "$SAVE_DIR"

echo "[launcher] run tag         : $TAG"
echo "[launcher] run dir         : $RUN_DIR"
echo "[launcher] student model   : $STUDENT_MODEL"
echo "[launcher] teacher model   : $TEACHER_MODEL  (GPU $TEACHER_GPU → rank $TEACHER_RANK_ID)"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC ranks)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"
echo "[launcher] algorithm       : $ALGORITHM"
echo "[launcher] sharding        : $SHARDING_STRATEGY"

# ---------------------------------------------------------------------------
# Start vLLM rollout worker
echo "[launcher] starting rollout worker -> $WORKER_LOG"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPUS" \
  python "$ROOT_DIR/nanoopd/rollout_worker.py" \
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
# Start student trainer
echo "[launcher] starting trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  torchrun --standalone --nproc_per_node="$TRAIN_NPROC" \
    nanoopd/train.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-gpu-id "$TEACHER_RANK_ID" \
    --algorithm "$ALGORITHM" \
    --distill-top-k "$DISTILL_TOP_K" \
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
