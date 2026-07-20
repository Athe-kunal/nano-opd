#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Runs all four training scripts (OPD, SDPO, OPSD, SDFT) back to back, each
# using its opd/examples/*.yaml as-is:
#   - OPD, SDPO: dataset: dapo_math
#   - OPSD:      always trains on its own paper dataset (dataset_id), no
#                dapo_math option
#   - SDFT:      left on its own default dataset (dataset: science) — not
#                changed to a math dataset
# All four already have num_steps: 100 and the same post-training eval
# config (posttrain_eval_datasets: aime_2025,aime_2024,hmmt_2025,
# posttrain_eval_max_new_tokens: 38912). Post-training eval auto-runs for
# OPD/SDPO/OPSD (math training data); SDFT's post-training eval is skipped
# (its training data isn't math), matching its default behavior.
#
# Runs sequentially (one training job at a time) since all four share the
# same GPUs by default. Set CUDA_VISIBLE_DEVICES to restrict which physical
# GPUs are used, same as running any of these scripts individually.
#
# Usage:
#   bash opd/examples/run_all.sh [tag_prefix]
#   CUDA_VISIBLE_DEVICES=2,3 bash opd/examples/run_all.sh my_sweep
# ---------------------------------------------------------------------------

OPD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG_PREFIX="${1:-dapo_math_run}"
# train_opd.sh train_sdpo.sh train_opsd.sh 
SCRIPTS=(train_sdft.sh)

for script in "${SCRIPTS[@]}"; do
  tag="${TAG_PREFIX}_${script%.sh}"
  echo
  echo "==============================================================================="
  echo "[run_all] starting $script (tag=$tag)"
  echo "==============================================================================="
  bash "$OPD_DIR/examples/$script" "$tag"
done

echo
echo "[run_all] all four runs completed."
