#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v278_renderloop_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CSV="${CANDIDATE_CSV:-$ROOT/exp/stageB/logs/377_stageB_v277_verified_support_ab_v277_verified_support_ab_20260517_202148_bjt/candidates/actual_radii/actual_radii_accepted_candidates.csv}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_stageB_v278_renderloop_selector_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v278_renderloop_selector_${RUN_ID}}"

mkdir -p "$EXP_ROOT" "$LOG_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" tools/run_377_stageB_v278_renderloop_selector.py \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU" \
  --base-exp "$BASE_EXP" \
  --base-ckpt "$BASE_CKPT" \
  --candidate-csv "$CANDIDATE_CSV" \
  --dataset-root "$DATA_ROOT" \
  --exp-root "$EXP_ROOT" \
  --log-dir "$LOG_DIR" \
  --run-id "$RUN_ID" \
  --max-candidates "${MAX_CANDIDATES:-24}" \
  --max-accept "${MAX_ACCEPT:-8}" \
  --append-iter "${APPEND_ITER:-136411}" \
  --child-opacity-factor "${CHILD_OPACITY_FACTOR:-0.80}" \
  --child-scale-factor "${CHILD_SCALE_FACTOR:-0.55}" \
  --child-scale-max "${CHILD_SCALE_MAX:-0.008}" \
  --min-inner-gain "${MIN_INNER_GAIN:-0.05}" \
  --min-hard-gain "${MIN_HARD_GAIN:-0.000001}" \
  --max-outer-worsen "${MAX_OUTER_WORSEN:-0.0}" \
  --max-fg-worsen "${MAX_FG_WORSEN:-0.0}" \
  --max-boundary-worsen "${MAX_BOUNDARY_WORSEN:-0.0}" \
  --max-edge-worsen "${MAX_EDGE_WORSEN:-0.0}" \
  --train-iterations "${TRAIN_ITERATIONS:-200}" \
  --train-checkpoint-steps "${TRAIN_CHECKPOINT_STEPS:-100,200}" \
  "$@"
