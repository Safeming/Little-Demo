#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v331_target_raw_component_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

TRAIN_VIEWS_CSV="${TRAIN_VIEWS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TARGET_VIEWS_CSV="${TARGET_VIEWS_CSV:-21,22,23}"
TRAIN_FRAMES_CSV="${TRAIN_FRAMES_CSV:-0,570,60}"
TARGET_FRAMES_CSV="${TARGET_FRAMES_CSV:-0,570,60}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TRAIN_FRAMES_DENSE_SPEC="${TRAIN_FRAMES_DENSE_SPEC:-[0,570,1]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

MAX_CANDIDATES="${MAX_CANDIDATES:-14}"
MAX_ACCEPT="${MAX_ACCEPT:-3}"
MAX_INNER_PER_IMAGE="${MAX_INNER_PER_IMAGE:-4}"
MAX_OUTER_PER_IMAGE="${MAX_OUTER_PER_IMAGE:-4}"
PAIRED_OUTER_NEIGHBORS="${PAIRED_OUTER_NEIGHBORS:-2}"
IMAGE_PAIR_SIZE="${IMAGE_PAIR_SIZE:-2}"
MAX_IMAGE_GROUPS="${MAX_IMAGE_GROUPS:-4}"
GLOBAL_PER_IMAGE="${GLOBAL_PER_IMAGE:-8}"

DO_TRAIN="${DO_TRAIN:-true}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-50}"
TRAIN_FEATURE_LR="${TRAIN_FEATURE_LR:-0.000005}"
TRAIN_TEXTURE_LR="${TRAIN_TEXTURE_LR:-0.0}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v331_target_raw_component_selector_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v331_target_raw_component_selector_${RUN_ID}}"
COMPONENT_DIR="$LOG_DIR/component"
TARGET_COMPONENT_CSV="$COMPONENT_DIR/target_raw_components.csv"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$COMPONENT_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
POINT_CSV=$POINT_CSV
TARGET_COMPONENT_CSV=$TARGET_COMPONENT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DO_TRAIN=$DO_TRAIN

Goal:
  v331 selects a strict-pass subset from v330 target raw components.
  The hard blocker is edge_delta<=0 while preserving fg/boundary/inner/outer/hard gains.
  Color/SH short training is only launched if the selected no-train subset is strict_pass.
EOF

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

TRAIN_FLAG="--no-do-train"
case "$(printf '%s' "$DO_TRAIN" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) TRAIN_FLAG="--do-train" ;;
esac

log_event "component_start" "$TARGET_COMPONENT_CSV"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v330_target_raw_component_csv.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --dataset-root "$DATA_ROOT" \
  --out-dir "$COMPONENT_DIR" \
  --out-csv "$TARGET_COMPONENT_CSV" \
  --train-views "$TRAIN_VIEWS_CSV" \
  --target-views "$TARGET_VIEWS_CSV" \
  --train-frames "$TRAIN_FRAMES_CSV" \
  --target-frames "$TARGET_FRAMES_CSV" \
  --min-component-area 20 \
  --max-components-per-direction 16 \
  --top-points 8 \
  --point-pad-px 10 \
  > "$LOG_DIR/component.log" 2>&1
log_event "component_done" "$TARGET_COMPONENT_CSV"

log_event "selector_start" "$TARGET_COMPONENT_CSV"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/run_377_explicit_binding_v320_paired_signed_selector.py \
  --python-bin "$PYTHON_BIN" \
  --gpu "$GPU" \
  --base-exp "$BASE_EXP" \
  --base-ckpt "$BASE_CKPT" \
  --base-iter 136410 \
  --component-csv "$TARGET_COMPONENT_CSV" \
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
  --min-component-area 20 \
  --max-candidates "$MAX_CANDIDATES" \
  --max-accept "$MAX_ACCEPT" \
  --max-inner-per-image "$MAX_INNER_PER_IMAGE" \
  --max-outer-per-image "$MAX_OUTER_PER_IMAGE" \
  --paired-outer-neighbors "$PAIRED_OUTER_NEIGHBORS" \
  --image-pair-size "$IMAGE_PAIR_SIZE" \
  --max-image-groups "$MAX_IMAGE_GROUPS" \
  --global-per-image "$GLOBAL_PER_IMAGE" \
  --no-include-full-candidate \
  --score-full-reference \
  --center-strength 0.45 \
  --outer-px 0.35 \
  --max-points-per-action 96 \
  --min-inner-gain 0.05 \
  --min-outer-gain 0.0 \
  --min-hard-gain 0.000001 \
  --max-outer-worsen 0.0 \
  --max-fg-worsen 0.0 \
  --max-boundary-worsen 0.0 \
  --max-edge-worsen 0.0 \
  --no-allow-probe \
  "$TRAIN_FLAG" \
  --train-iterations "$TRAIN_ITERS" \
  --train-checkpoint-steps "$TRAIN_CHECKPOINT_STEPS" \
  --train-feature-lr "$TRAIN_FEATURE_LR" \
  --train-texture-lr "$TRAIN_TEXTURE_LR" \
  > "$LOG_DIR/selector.log" 2>&1
log_event "selector_done" "$LOG_DIR/selection_summary.tsv"

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event "done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "TARGET_COMPONENT_CSV=$TARGET_COMPONENT_CSV"
echo "SELECTED_CSV=$LOG_DIR/selected_components.csv"
echo "SUMMARY=$LOG_DIR/selection_summary.tsv"
echo "TRAIN_SUMMARY=$LOG_DIR/train_summary.tsv"
echo "END_BJT=$END_BJT"
