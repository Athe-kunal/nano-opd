#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-default}"

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$OPD_DIR/.." && pwd)"
BASE_DIR="${opd_BASE_DIR:-$REPO_ROOT/.opd}"

# In SDPO the teacher IS the student — only one model checkpoint needed.
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

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

NUM_STEPS="${NUM_STEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-15872}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DATASET="${DATASET:-sciknoweval}"

# JSD is the SDPO default (symmetric, bounded in [0, log2], more stable than pure KL).
ALGORITHM="${ALGORITHM:-jsd}"
DISTILL_TOP_K="${DISTILL_TOP_K:-100}"
STUDENT_CHUNK_SIZE="${STUDENT_CHUNK_SIZE:--1}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:--1}"
TIS_CLIP="${TIS_CLIP:-0.0}"

# Teacher sync — controls how the self-teacher tracks the student after each step.
#   ema          : teacher ← α·student + (1−α)·teacher  (smooth, recommended)
#   trust_region : teacher ← β·student + (1−β)·initial_weights  (anchored to init)
#   hard_sync    : full copy every N steps (DQN-style, less smooth)
#   on_policy    : teacher = live student (ablation only, can diverge)
SYNC_METHOD="${SYNC_METHOD:-ema}"
EMA_ALPHA="${EMA_ALPHA:-0.05}"
TRUST_REGION_BETA="${TRUST_REGION_BETA:-0.05}"
HARD_SYNC_EVERY_N="${HARD_SYNC_EVERY_N:-5}"

EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_K="${EVAL_K:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
SCIKNOWEVAL_TEST_SIZE="${SCIKNOWEVAL_TEST_SIZE:-0.01}"  # fraction held out for eval (sciknoweval only)
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SEED="${SEED:-0}"

# Post-training math evaluation (eval_math.py)
# Set SKIP_EVAL=1 to bypass evaluation entirely regardless of dataset.
# Auto-enable only for math datasets; override with RUN_EVAL=0 or RUN_EVAL=1.
SKIP_EVAL="${SKIP_EVAL:-0}"
if [[ "$DATASET" == "dapo_math" || "$DATASET" == "opsd_math" ]]; then
  RUN_EVAL="${RUN_EVAL:-1}"
else
  RUN_EVAL="${RUN_EVAL:-0}"
fi
EVAL_DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-38912}"
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-1}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-}"           # blank → auto (0.95 thinking / 0.8 non-thinking)
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_MIN_P="${EVAL_MIN_P:-0.0}"
EVAL_PRESENCE_PENALTY="${EVAL_PRESENCE_PENALTY:-0.0}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"  # blank → use all problems in dataset
EVAL_SMOKE_TEST="${EVAL_SMOKE_TEST:-0}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.9}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-}"  # blank → auto (40960 thinking / 32768 non-thinking)
EVAL_VAL_N="${EVAL_VAL_N:-6}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-$TAG}"
EVAL_STEP="${EVAL_STEP:-}"             # blank → omit --step (no x-axis pin in W&B)

# FSDP sharding strategy — choose one of:
#   FULL_SHARD          params+grads+optimizer sharded; unshard around fwd/bwd
#   SHARD_GRAD_OP       params sharded; keep unsharded after fwd until bwd done
#   NO_SHARD            replicate everything (like DDP)
#   HYBRID_SHARD        FULL_SHARD within a node, replicate across nodes
#   _HYBRID_SHARD_ZERO2 SHARD_GRAD_OP within a node, replicate across nodes
SHARDING_STRATEGY="${SHARDING_STRATEGY:-NO_SHARD}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

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
echo "[launcher] student model   : $STUDENT_MODEL  (teacher = EMA of student, same checkpoint)"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC student ranks)"
echo "[launcher] teacher GPUs    : $TEACHER_GPUS  ($TEACHER_NPROC teacher rank)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"
echo "[launcher] algorithm       : $ALGORITHM"
echo "[launcher] sync method     : $SYNC_METHOD  (ema_alpha=$EMA_ALPHA  tr_beta=$TRUST_REGION_BETA  hard_n=$HARD_SYNC_EVERY_N)"
echo "[launcher] sharding        : $SHARDING_STRATEGY"

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
# Build optional flags
EXTRA_FLAGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  EXTRA_FLAGS+=(--gradient-checkpointing)
fi

# ---------------------------------------------------------------------------
# Start student trainer + teacher
# torchrun world: ranks 0..(TRAIN_NPROC-1) = student FSDP, rank TRAIN_NPROC = teacher.
# CUDA_VISIBLE_DEVICES maps logical indices to physical GPUs in order.
echo "[launcher] starting SDPO trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,$TEACHER_GPUS" \
  uv run --extra gpu --directory "$REPO_ROOT" torchrun --standalone --nproc_per_node="$TOTAL_NPROC" \
    "$OPD_DIR/trainer/train_sdpo.py" \
    --student-model "$STUDENT_MODEL" \
    --train-world-size "$TRAIN_NPROC" \
    --dataset "$DATASET" \
    --algorithm "$ALGORITHM" \
    --distill-top-k "$DISTILL_TOP_K" \
    --student-chunk-size "$STUDENT_CHUNK_SIZE" \
    --teacher-chunk-size "$TEACHER_CHUNK_SIZE" \
    --tis-clip "$TIS_CLIP" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-worker-world-size "$ROLLOUT_TP" \
    --num-steps "$NUM_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --num-samples "$NUM_SAMPLES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --epochs "$EPOCHS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-len "$MAX_PROMPT_LEN" \
    --max-response-len "$MAX_RESPONSE_LEN" \
    --temperature "$TEMPERATURE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --sync-method "$SYNC_METHOD" \
    --ema-alpha "$EMA_ALPHA" \
    --trust-region-beta "$TRUST_REGION_BETA" \
    --hard-sync-every-n "$HARD_SYNC_EVERY_N" \
    --scheduler "$SCHEDULER" \
    --warmup-ratio "$WARMUP_RATIO" \
    --sharding-strategy "$SHARDING_STRATEGY" \
    --save-dir "$SAVE_DIR" \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --eval-k "$EVAL_K" \
    --eval-max-tokens "$EVAL_MAX_TOKENS" \
    --sciknoweval-test-size "$SCIKNOWEVAL_TEST_SIZE" \
    --seed "$SEED" \
    --run-name "$TAG" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee "$TRAIN_LOG"

# ---------------------------------------------------------------------------
# Post-training evaluation with eval_math.py
if [[ "$RUN_EVAL" == "1" && "$SKIP_EVAL" != "1" ]]; then
  FINAL_CKPT="$SAVE_DIR/final"
  CKPT_ARG=()
  [[ -d "$FINAL_CKPT" ]] && CKPT_ARG=(--checkpoint_dir "$FINAL_CKPT")

  EVAL_EXTRA_FLAGS=()
  [[ "$EVAL_ENABLE_THINKING" == "0" ]] && EVAL_EXTRA_FLAGS+=(--no_thinking)
  [[ "$EVAL_SMOKE_TEST" == "1" ]]      && EVAL_EXTRA_FLAGS+=(--smoke_test)
  [[ -n "$EVAL_TOP_P" ]]               && EVAL_EXTRA_FLAGS+=(--top_p "$EVAL_TOP_P")
  [[ -n "$EVAL_NUM_SAMPLES" ]]         && EVAL_EXTRA_FLAGS+=(--num_samples "$EVAL_NUM_SAMPLES")
  [[ -n "$EVAL_MAX_MODEL_LEN" ]]       && EVAL_EXTRA_FLAGS+=(--max_model_len "$EVAL_MAX_MODEL_LEN")
  [[ -n "$EVAL_WANDB_PROJECT" ]]       && EVAL_EXTRA_FLAGS+=(--wandb_project "$EVAL_WANDB_PROJECT" --wandb_run_name "$EVAL_WANDB_RUN_NAME")
  [[ -n "$EVAL_STEP" ]]                && EVAL_EXTRA_FLAGS+=(--step "$EVAL_STEP")

  echo "[launcher] running post-training eval on: $EVAL_DATASETS"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS},${TEACHER_GPUS}" \
    uv run --extra gpu --directory "$REPO_ROOT" python "$OPD_DIR/eval/eval_math.py" \
      --base_model "$STUDENT_MODEL" \
      "${CKPT_ARG[@]}" \
      --datasets $EVAL_DATASETS \
      --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --temperature "$EVAL_TEMPERATURE" \
      --top_k "$EVAL_TOP_K" \
      --min_p "$EVAL_MIN_P" \
      --presence_penalty "$EVAL_PRESENCE_PENALTY" \
      --val_n "$EVAL_VAL_N" \
      --gpu_memory_utilization "$EVAL_GPU_MEM_UTIL" \
      --tensor_parallel_size "$EVAL_TENSOR_PARALLEL_SIZE" \
      "${EVAL_EXTRA_FLAGS[@]}" \
      2>&1 | tee "$RUN_DIR/eval.log"
fi
