#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v274_contributor_audit_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

OUT_DIR="${OUT_DIR:-$ROOT/exp/stageB/logs/377_stageB_v274_contributor_audit_${RUN_ID}}"
EVENTS="$OUT_DIR/events.tsv"

mkdir -p "$OUT_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"

cat > "$OUT_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
OUT_DIR=$OUT_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v274 no-train contributor audit.
  Attribute top raw explicit_binding boundary residuals to individual Gaussian ids.
  Produce over-contributor shrink candidates and under-supported grow candidates for v275.
EOF

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
)

log_event "audit_start" "$OUT_DIR"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/audit_377_stageB_v274_contributors.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --out-dir "$OUT_DIR" \
  --dataset-root "$DATA_ROOT" \
  --eval-views "21,22,23" \
  --eval-frames "0,570,60" \
  --top-frames 12 \
  --render-support-threshold 0.025 \
  --close-kernel 5 \
  --band-width 7 \
  --search-band-width 24 \
  --residual-dilate 1 \
  --outer-radius-scale 1.20 \
  --inner-radius-scale 1.75 \
  --min-radius 1.5 \
  --max-radius 18.0 \
  --opacity-power 0.50 \
  --radius-power 0.35 \
  --min-component-area 18 \
  --max-components-per-frame 8 \
  --component-top-points 8 \
  --top-points 160 \
  --candidate-points 96 \
  --render-scaling-modifier 1.0 \
  --compute-cov3d-python \
  --camera-geometry \
  > "$OUT_DIR/audit.log" 2>&1

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$OUT_DIR/contributor_audit_summary.json"
  echo "CANDIDATES=$OUT_DIR/v275_candidate_point_sets.json"
} >> "$OUT_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "OUT_DIR=$OUT_DIR"
echo "SUMMARY=$OUT_DIR/contributor_audit_summary.json"
echo "CANDIDATES=$OUT_DIR/v275_candidate_point_sets.json"
echo "END_BJT=$END_BJT"
