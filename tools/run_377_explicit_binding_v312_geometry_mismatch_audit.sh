#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v312_geometry_mismatch_audit_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

TRAIN_VIEWS_CSV="${TRAIN_VIEWS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TRAIN_FRAMES_CSV="${TRAIN_FRAMES_CSV:-0,570,60}"
EVAL_VIEWS_CSV="${EVAL_VIEWS_CSV:-21,22,23}"
EVAL_FRAMES_CSV="${EVAL_FRAMES_CSV:-0,570,60}"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v312_geometry_mismatch_audit_${RUN_ID}}"
BASELINE_OUT="$LOG_DIR/baseline"
ADOPTED_OUT="$LOG_DIR/adopted"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$LOG_DIR" "$BASELINE_OUT" "$ADOPTED_OUT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-1200}"
EST_END_EPOCH="$((START_EPOCH + EST_SECONDS))"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$EST_END_EPOCH" '+%F %T BJT')"

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" "$START_BJT" "$EST_END_BJT" <<'PY' || true
import json
import sys
import time
from pathlib import Path

path, run_id, gpu, phase, detail, start_bjt, est_end_bjt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_bjt": start_bjt,
    "est_end_bjt": est_end_bjt,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
  write_status "$1" "$2"
}

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
DATA_ROOT=$DATA_ROOT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
TRAIN_VIEWS_CSV=$TRAIN_VIEWS_CSV
TRAIN_FRAMES_CSV=$TRAIN_FRAMES_CSV
EVAL_VIEWS_CSV=$EVAL_VIEWS_CSV
EVAL_FRAMES_CSV=$EVAL_FRAMES_CSV
LOG_DIR=$LOG_DIR

Goal:
  v312 audits whether residual contributors correlate with explicit-binding
  geometry mismatch: current x_bar vs free_lbs / geometry target in screen space.
  It is an audit only; no checkpoint edit and no training.
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

run_audit() {
  local variant="$1"
  local out_dir="$2"
  log_event "audit_start" "$variant"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/audit_377_stageB_v312_geometry_mismatch.py \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --out-dir "$out_dir" \
    --dataset-root "$DATA_ROOT" \
    --component-csv "$COMPONENT_CSV" \
    --point-csv "$POINT_CSV" \
    --variant "$variant" \
    --train-views "$TRAIN_VIEWS_CSV" \
    --train-frames "$TRAIN_FRAMES_CSV" \
    --eval-views "$EVAL_VIEWS_CSV" \
    --eval-frames "$EVAL_FRAMES_CSV" \
    > "$LOG_DIR/audit_${variant}.log" 2>&1
  log_event "audit_done" "$variant"
}

run_audit baseline "$BASELINE_OUT"
run_audit adopted "$ADOPTED_OUT"

"$PYTHON_BIN" - "$SUMMARY" "$BASELINE_OUT/geometry_mismatch_summary.json" "$ADOPTED_OUT/geometry_mismatch_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
paths = {
    "baseline": Path(sys.argv[2]),
    "adopted": Path(sys.argv[3]),
}

def flat_metrics(data):
    outer = data.get("outer_summary", {})
    inner = data.get("inner_summary", {})
    return {
        "mean_inner": float(data.get("frame_mean_inner_missing_pixels", 0.0)),
        "mean_outer": float(data.get("frame_mean_outer_leak_pixels", 0.0)),
        "mean_hard": float(data.get("frame_mean_hard_residual_score", 0.0)),
        "outer_xfree_mean": float(outer.get("outer_xfree_delta_px_weighted_mean", 0.0)),
        "inner_xfree_mean": float(inner.get("inner_xfree_delta_px_weighted_mean", 0.0)),
        "outer_pregeom_mean": float(outer.get("outer_pre_geometry_delta_px_weighted_mean", 0.0)),
        "inner_pregeom_mean": float(inner.get("inner_pre_geometry_delta_px_weighted_mean", 0.0)),
        "outer_fidelity_mean": float(outer.get("outer_fidelity_weight_weighted_mean", 0.0)),
        "inner_fidelity_mean": float(inner.get("inner_fidelity_weight_weighted_mean", 0.0)),
        "outer_center_blend_mean": float(outer.get("outer_center_blend_weighted_mean", 0.0)),
        "inner_center_blend_mean": float(inner.get("inner_center_blend_weighted_mean", 0.0)),
        "outer_xfree_corr": float(outer.get("outer_xfree_delta_px_corr", 0.0)),
        "inner_xfree_corr": float(inner.get("inner_xfree_delta_px_corr", 0.0)),
    }

rows = []
for name, path in paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    item = flat_metrics(data)
    item["variant"] = name
    item["summary_json"] = str(path)
    rows.append(item)

fields = ["variant", "summary_json"] + [key for key in rows[0].keys() if key not in ("variant", "summary_json")]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary": str(summary_path), "rows": rows}, indent=2), flush=True)
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "BASELINE_OUT=$BASELINE_OUT"
  echo "ADOPTED_OUT=$ADOPTED_OUT"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "BASELINE_OUT=$BASELINE_OUT"
echo "ADOPTED_OUT=$ADOPTED_OUT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
