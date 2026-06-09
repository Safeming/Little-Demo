#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v291_signed2_component_cov_residual_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
TEST_COMPONENT_CSV="${TEST_COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
TEST_POINT_CSV="${TEST_POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"

AUDIT_DIR="${AUDIT_DIR:-$ROOT/exp/stageB/logs/377_stageB_v291_train_component_audit_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v291_signed2_component_cov_residual_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v291_signed2_component_cov_residual_${RUN_ID}}"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$AUDIT_DIR" "$EXP_ROOT" "$LOG_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$TEST_COMPONENT_CSV" "$TEST_POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
DATA_ROOT=$DATA_ROOT
TEST_COMPONENT_CSV=$TEST_COMPONENT_CSV
TEST_POINT_CSV=$TEST_POINT_CSV
AUDIT_DIR=$AUDIT_DIR
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v291 keeps the v281/v282 dynamic screen actuator baseline but changes the
  trainable checkpoint residual from one shared scalar to two signed channels:
  channel 0 only affects outer shrink, channel 1 only affects inner grow.
  The actuator is component-required and component-signature gated, without
  v290 top-id-only sparsity.
EOF

log_event "train_component_audit_start" "$AUDIT_DIR"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/audit_377_stageB_v274_contributors.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --out-dir "$AUDIT_DIR" \
  --dataset-root "$DATA_ROOT" \
  --eval-views "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20" \
  --eval-frames "0,570,30" \
  --top-frames 80 \
  --candidate-points 192 \
  --top-points 320 \
  --max-components-per-frame 10 \
  --component-top-points 16 \
  --min-component-area 12 \
  > "$LOG_DIR/train_component_audit.log" 2>&1
log_event "train_component_audit_done" "$AUDIT_DIR"

TRAIN_COMPONENT_CSV="$AUDIT_DIR/component_contributors.csv"
TRAIN_POINT_CSV="$AUDIT_DIR/point_contributors_all.csv"
for required in "$TRAIN_COMPONENT_CSV" "$TRAIN_POINT_CSV"; do
  if [ ! -s "$required" ]; then
    echo "missing generated audit output: $required" >&2
    exit 3
  fi
done

log_event "v291_train_start" "$EXP_ROOT"
RUN_ID="$RUN_ID" \
PYTHON_BIN="$PYTHON_BIN" \
GPU="$GPU" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
DATA_ROOT="$DATA_ROOT" \
BASE_EXP="$BASE_EXP" \
BASE_CKPT="$BASE_CKPT" \
COMPONENT_CSV="$TEST_COMPONENT_CSV" \
POINT_CSV="$TEST_POINT_CSV" \
TRAIN_COMPONENT_CSV="$TRAIN_COMPONENT_CSV" \
TRAIN_POINT_CSV="$TRAIN_POINT_CSV" \
COMPONENT_REQUIRED=true \
COMPONENT_SIGNATURE_ENABLE=true \
COMPONENT_TOP_IDS_ENABLE=false \
COMPONENT_TOP_IDS_ONLY=false \
TRAIN_COMPONENT_REQUIRED=true \
TRAIN_COMPONENT_SIGNATURE_ENABLE=true \
TRAIN_COMPONENT_TOP_IDS_ENABLE=false \
TRAIN_COMPONENT_TOP_IDS_ONLY=false \
BOUNDARY_COV_RESIDUAL_CHANNELS=2 \
BOUNDARY_COV_SIGNED_TWO_CHANNEL=1 \
EXP_ROOT="$EXP_ROOT" \
LOG_DIR="$LOG_DIR/v288_core" \
TRAIN_ITERS="${TRAIN_ITERS:-80}" \
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-40,80}" \
bash tools/run_377_explicit_binding_v288_boundary_cov_residual.sh \
  > "$LOG_DIR/v288_core_stdout.log" 2>&1
log_event "v291_train_done" "$EXP_ROOT"

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "TRAIN_COMPONENT_CSV=$TRAIN_COMPONENT_CSV"
  echo "TRAIN_POINT_CSV=$TRAIN_POINT_CSV"
  echo "SUMMARY=$LOG_DIR/v288_core/summary.tsv"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "AUDIT_DIR=$AUDIT_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$LOG_DIR/v288_core/summary.tsv"
echo "END_BJT=$END_BJT"
