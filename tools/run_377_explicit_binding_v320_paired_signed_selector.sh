#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v320_paired_signed_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TRAIN_FRAMES_DENSE_SPEC="${TRAIN_FRAMES_DENSE_SPEC:-[0,570,1]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v320_paired_signed_selector_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v320_paired_signed_selector_${RUN_ID}}"

mkdir -p "$EXP_ROOT" "$LOG_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

TRAIN_FLAG="--no-do-train"
if [ "${DO_TRAIN:-false}" = "1" ] || [ "${DO_TRAIN:-false}" = "true" ]; then
  TRAIN_FLAG="--do-train"
fi

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_SECONDS="${EST_SECONDS:-5400}"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$(($(date +%s) + EST_SECONDS))" '+%F %T BJT')"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC

Goal:
  v320 turns the current main-cause finding into a selector mechanism:
  do not evaluate one-sided inner-grow or outer-protect edits. Build paired
  signed component sets from the same camera/frame residual field and accept
  only sets that improve raw inner missing and outer leak together under the
  real render-in-loop do-no-harm gate.
EOF

"$PYTHON_BIN" tools/run_377_explicit_binding_v320_paired_signed_selector.py \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU" \
  --base-exp "$BASE_EXP" \
  --base-ckpt "$BASE_CKPT" \
  --component-csv "$COMPONENT_CSV" \
  --point-csv "$POINT_CSV" \
  --dataset-root "$DATA_ROOT" \
  --exp-root "$EXP_ROOT" \
  --log-dir "$LOG_DIR" \
  --run-id "$RUN_ID" \
  --train-views-spec "$TRAIN_VIEWS_SPEC" \
  --train-frames-spec "$TRAIN_FRAMES_SPEC" \
  --train-frames-dense-spec "$TRAIN_FRAMES_DENSE_SPEC" \
  --test-views-spec "$TEST_VIEWS_SPEC" \
  --test-frames-spec "$TEST_FRAMES_SPEC" \
  --max-candidates "${MAX_CANDIDATES:-24}" \
  --max-accept "${MAX_ACCEPT:-4}" \
  --max-inner-per-image "${MAX_INNER_PER_IMAGE:-4}" \
  --max-outer-per-image "${MAX_OUTER_PER_IMAGE:-4}" \
  --paired-outer-neighbors "${PAIRED_OUTER_NEIGHBORS:-2}" \
  --max-image-groups "${MAX_IMAGE_GROUPS:-8}" \
  --image-pair-size "${IMAGE_PAIR_SIZE:-2}" \
  --global-per-image "${GLOBAL_PER_IMAGE:-4}" \
  --no-include-full-candidate \
  --score-full-reference \
  --center-strength "${CENTER_STRENGTH:-0.45}" \
  --outer-px "${OUTER_PX:-0.35}" \
  --max-points-per-action "${MAX_POINTS_PER_ACTION:-96}" \
  --min-inner-gain "${MIN_INNER_GAIN:-0.05}" \
  --min-outer-gain "${MIN_OUTER_GAIN:-0.01}" \
  --min-hard-gain "${MIN_HARD_GAIN:-0.000001}" \
  --max-outer-worsen "${MAX_OUTER_WORSEN:-0.0}" \
  --max-fg-worsen "${MAX_FG_WORSEN:-0.0}" \
  --max-boundary-worsen "${MAX_BOUNDARY_WORSEN:-0.0}" \
  --max-edge-worsen "${MAX_EDGE_WORSEN:-0.0}" \
  "$TRAIN_FLAG" \
  --train-iterations "${TRAIN_ITERATIONS:-100}" \
  --train-checkpoint-steps "${TRAIN_CHECKPOINT_STEPS:-100}" \
  "$@"

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
