#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Self-Distillation Fine-Tuning (SDFT) launcher.
#   Shenfeld, Damani, Hübotter, Agrawal — "Self-Distillation Enables Continual
#   Learning", arXiv:2601.19897.
#
# SDFT distills, on-policy, from a DEMONSTRATION-CONDITIONED self-teacher:
#   - student sees only the question x        → π_θ(y|x)
#   - teacher is an EMA copy of the student, conditioned in-context on a worked
#     demonstration c                          → π(y|x,c)
#   - loss is the reverse KL  D_KL(π_θ(·|x) ‖ π(·|x,c))   (paper Eq. 1)
#
# Like SDPO there is only ONE checkpoint (teacher = EMA of student), so the
# GPU-split / rollout-worker plumbing mirrors train_sdpo.sh.
# ---------------------------------------------------------------------------

TAG="${1:-default}"

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$OPD_DIR/.." && pwd)"
BASE_DIR="${opd_BASE_DIR:-$REPO_ROOT/.opd}"

# In SDFT the teacher IS the student (EMA copy) — only one model checkpoint needed.
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# Dataset. Built-in options pulled from the Self-Distillation GitHub repo:
#   science   – science Q&A with worked demonstrations (paper's Science Q&A task)
#   tooluse   – tool-use tasks with golden Action/Action_Input sequences
# Set DATASET_PATH to a JSONL on disk to override (one {"question","demonstration"} per line).
DATASET="${DATASET:-tooluse}"  # science|tooluse
DATASET_LIMIT="${DATASET_LIMIT:-0}"          # 0 = use all rows
# Custom data (optional): point DATASET_PATH at your own {question, demonstration}
# JSONL to override the built-in dataset.
DATASET_PATH="${DATASET_PATH:-}"

# GPU assignment (comma-separated physical GPU IDs)
#   TRAIN_GPUS    – student FSDP training ranks (0..N-1)
#   TEACHER_GPUS  – dedicated teacher rank (exactly 1 GPU, rank N in torchrun world)
#   ROLLOUT_GPUS  – vLLM rollout worker (may share with TEACHER_GPUS)
#   TRAIN_GPUS must not overlap ROLLOUT_GPUS or TEACHER_GPUS.
ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
TEACHER_GPUS="${TEACHER_GPUS:-0}"

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8047}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
WEIGHT_TRANSFER_BACKEND="${WEIGHT_TRANSFER_BACKEND:-nccl}"

USE_WANDB="${USE_WANDB:-1}"

NUM_STEPS="${NUM_STEPS:-30}"
SAVE_EVERY="${SAVE_EVERY:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# NOTE: --epochs here is PPO-style reuse of the SAME rollout batch, NOT the
# paper's dataset-pass epochs (which re-roll fresh on-policy trajectories).
# To replicate the paper's "2 epochs Skill / 4 epochs Knowledge", raise
# NUM_STEPS instead and keep EPOCHS=1. See the --epochs help in train_sdft.py.
EPOCHS="${EPOCHS:-1}"

LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# SDFT: exactly one on-policy rollout per (question, demonstration) pair — there
# is no num-samples multiplier (paper Appendix A.1 uses a single trajectory).
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1536}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}"   # 0 = MAX_PROMPT_LEN + MAX_RESPONSE_LEN
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"

# reverse_kl is the SDFT default (paper Eq. 1: KL(π_student ‖ π_teacher)).
ALGORITHM="${ALGORITHM:-forward_kl}"
DISTILL_TOP_K="${DISTILL_TOP_K:-100}"
STUDENT_CHUNK_SIZE="${STUDENT_CHUNK_SIZE:--1}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:--1}"
TIS_CLIP="${TIS_CLIP:-0.0}"

# Mask the first N response tokens to suppress "Based on the text..." preamble
# artifacts the student inherits from the demonstration-conditioned teacher
# (paper Section 5, "Learned Artifacts" — paper says "first few tokens").
NUM_LOSS_TOKENS_TO_SKIP="${NUM_LOSS_TOKENS_TO_SKIP:-5}"

# Teacher sync — how the EMA self-teacher tracks the student after each step.
#   ema          : teacher ← α·student + (1−α)·teacher  (paper default)
#   trust_region : teacher ← β·student + (1−β)·initial_weights  (anchored to init)
#   hard_sync    : full copy every N steps (DQN-style, less smooth)
SYNC_METHOD="${SYNC_METHOD:-ema}"
EMA_ALPHA="${EMA_ALPHA:-0.02}"          # paper sweeps {0.01, 0.02, 0.05} (Tables 3/4)
TRUST_REGION_BETA="${TRUST_REGION_BETA:-0.05}"
HARD_SYNC_EVERY_N="${HARD_SYNC_EVERY_N:-100}"

SEED="${SEED:-0}"

# Held-out evaluation on the dataset's eval split (science/tooluse only; ignored for custom JSONL).
# Set EVAL_EVERY=0 to disable.
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_K="${EVAL_K:-4}"

# FSDP sharding strategy — choose one of:
#   FULL_SHARD          params+grads+optimizer sharded; unshard around fwd/bwd
#   SHARD_GRAD_OP       params sharded; keep unsharded after fwd until bwd done
#   NO_SHARD            replicate everything (like DDP)
#   HYBRID_SHARD        FULL_SHARD within a node, replicate across nodes
#   _HYBRID_SHARD_ZERO2 SHARD_GRAD_OP within a node, replicate across nodes
SHARDING_STRATEGY="${SHARDING_STRATEGY:-NO_SHARD}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

RUN_DIR="$BASE_DIR/sdft/$TAG"
SAVE_DIR="$RUN_DIR/checkpoints"
WORKER_LOG="$RUN_DIR/rollout_worker.log"
TRAIN_LOG="$RUN_DIR/train.log"

export PYTHONPATH="$REPO_ROOT"
export opd_BASE_DIR="$BASE_DIR"
export ROLLOUT_HOST
export ROLLOUT_PORT
export USE_WANDB

# ---------------------------------------------------------------------------
# Sanity: if DATASET_PATH is set it must exist; otherwise a built-in dataset is used.
if [[ -n "$DATASET_PATH" && ! -f "$DATASET_PATH" ]]; then
  echo "DATASET_PATH does not exist: $DATASET_PATH" >&2
  echo "Provide a JSONL file, one record per line:" >&2
  echo '  {"question": "...", "demonstration": "...worked example..."}' >&2
  echo "and re-run, e.g.  DATASET_PATH=/path/to/data.jsonl bash $0 $TAG" >&2
  exit 1
fi

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
echo "[launcher] student model   : $STUDENT_MODEL  (teacher = EMA of student, demonstration-conditioned)"
echo "[launcher] dataset         : $DATASET_PATH"
echo "[launcher] train GPUs      : $TRAIN_GPUS  ($TRAIN_NPROC student ranks)"
echo "[launcher] teacher GPUs    : $TEACHER_GPUS  ($TEACHER_NPROC teacher rank)"
echo "[launcher] rollout GPUs    : $ROLLOUT_GPUS  (tp=$ROLLOUT_TP)"
echo "[launcher] algorithm       : $ALGORITHM  (top-k=$DISTILL_TOP_K)"
echo "[launcher] sync method     : $SYNC_METHOD  (ema_alpha=$EMA_ALPHA  tr_beta=$TRUST_REGION_BETA  hard_n=$HARD_SYNC_EVERY_N)"
echo "[launcher] token skip      : $NUM_LOSS_TOKENS_TO_SKIP"
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
echo "[launcher] starting SDFT trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,$TEACHER_GPUS" \
  uv run --extra gpu --directory "$REPO_ROOT" torchrun --standalone --nproc_per_node="$TOTAL_NPROC" \
    "$OPD_DIR/trainer/train_sdft.py" \
    --student-model "$STUDENT_MODEL" \
    --train-world-size "$TRAIN_NPROC" \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --dataset-limit "$DATASET_LIMIT" \
    --algorithm "$ALGORITHM" \
    --distill-top-k "$DISTILL_TOP_K" \
    --student-chunk-size "$STUDENT_CHUNK_SIZE" \
    --teacher-chunk-size "$TEACHER_CHUNK_SIZE" \
    --tis-clip "$TIS_CLIP" \
    --num-loss-tokens-to-skip "$NUM_LOSS_TOKENS_TO_SKIP" \
    --rollout-worker-url "http://$ROLLOUT_HOST:$ROLLOUT_PORT" \
    --rollout-worker-world-size "$ROLLOUT_TP" \
    --num-steps "$NUM_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --epochs "$EPOCHS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-len "$MAX_PROMPT_LEN" \
    --max-response-len "$MAX_RESPONSE_LEN" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --sync-method "$SYNC_METHOD" \
    --ema-alpha "$EMA_ALPHA" \
    --trust-region-beta "$TRUST_REGION_BETA" \
    --hard-sync-every-n "$HARD_SYNC_EVERY_N" \
    --sharding-strategy "$SHARDING_STRATEGY" \
    --save-dir "$SAVE_DIR" \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --eval-k "$EVAL_K" \
    --seed "$SEED" \
    --run-name "$TAG" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee "$TRAIN_LOG"
