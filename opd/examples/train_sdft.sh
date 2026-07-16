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
#   - loss is the reverse KL  D_KL(π_θ(·|x) ‖ π(·|x,c))
#
# Like SDPO there is only ONE checkpoint (teacher = EMA of student), so the
# GPU-split / rollout-worker plumbing mirrors train_sdpo.sh.
# ---------------------------------------------------------------------------

TAG="${1:-default}"

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$OPD_DIR/.." && pwd)"
BASE_DIR="${opd_BASE_DIR:-$REPO_ROOT/.opd}"

CONFIG_YAML="${CONFIG_YAML:-$OPD_DIR/examples/sdft.yaml}"

# Resolve the values this script needs before the trainer starts (model
# name, dataset path, rollout worker settings) straight out of $CONFIG_YAML.
eval "$(uv run --extra gpu --directory "$REPO_ROOT" python -c "
import shlex
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG_YAML')
for key in (
    'student_model', 'dataset_path',
    'train_gpus', 'rollout_gpus', 'teacher_gpus',
    'rollout_host', 'rollout_port', 'rollout_gpu_mem_util', 'weight_transfer_backend',
    'use_wandb',
):
    print(f'{key.upper()}={shlex.quote(str(cfg[key]))}')
")"

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
# WORLD_GPUS restricts which physical GPUs this launcher may use, e.g.
# `CUDA_VISIBLE_DEVICES=2,3 bash opd/examples/train_sdft.sh`. train_gpus/
# rollout_gpus/teacher_gpus in the YAML are LOCAL indices into this list
# (0-based), not physical IDs — with CUDA_VISIBLE_DEVICES=2,3, rollout_gpus:
# "0" means physical GPU 2 and train_gpus: "1" means physical GPU 3. If
# CUDA_VISIBLE_DEVICES isn't set, the YAML's GPU fields are used as physical
# IDs directly (no remapping).
WORLD_GPUS="${CUDA_VISIBLE_DEVICES:-}"
IFS=, read -r -a WORLD_GPU_LIST <<< "$WORLD_GPUS"

# Translates a comma-separated list of local indices (from the YAML) into
# physical GPU IDs via $WORLD_GPU_LIST.
to_physical_gpus() {
  local indices="$1"
  if [[ -z "$WORLD_GPUS" ]]; then
    echo "$indices"
    return
  fi
  local idx_arr physical=()
  IFS=, read -r -a idx_arr <<< "$indices"
  for idx in "${idx_arr[@]}"; do
    physical+=("${WORLD_GPU_LIST[$idx]}")
  done
  local IFS=,
  echo "${physical[*]}"
}

TRAIN_GPUS="$(to_physical_gpus "$TRAIN_GPUS")"
ROLLOUT_GPUS="$(to_physical_gpus "$ROLLOUT_GPUS")"
TEACHER_GPUS="$(to_physical_gpus "$TEACHER_GPUS")"

# ---------------------------------------------------------------------------
# Sanity: if dataset_path is set in the YAML it must exist; otherwise a
# built-in dataset (the YAML's `dataset` key) is used.
if [[ -n "$DATASET_PATH" && ! -f "$DATASET_PATH" ]]; then
  echo "dataset_path does not exist: $DATASET_PATH" >&2
  echo "Provide a JSONL file, one record per line:" >&2
  echo '  {"question": "...", "demonstration": "...worked example..."}' >&2
  echo "and re-run, e.g.  dataset_path=/path/to/data.jsonl in $CONFIG_YAML" >&2
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
echo "[launcher] config          : $CONFIG_YAML"
echo "[launcher] world GPUs      : ${WORLD_GPUS:-<none, YAML fields are physical IDs>}"
echo "[launcher] student model   : $STUDENT_MODEL  (teacher = EMA of student, demonstration-conditioned)"
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
echo "[launcher] starting SDFT trainer -> $TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS,$TEACHER_GPUS" \
  uv run --extra gpu --directory "$REPO_ROOT" torchrun --standalone --nproc_per_node="$TOTAL_NPROC" \
    "$OPD_DIR/trainer/train_sdft.py" "$CONFIG_YAML" \
    train_world_size="$TRAIN_NPROC" \
    rollout_worker_world_size="$ROLLOUT_TP" \
    save_dir="$SAVE_DIR" \
    run_name="$TAG" \
    2>&1 | tee "$TRAIN_LOG"
