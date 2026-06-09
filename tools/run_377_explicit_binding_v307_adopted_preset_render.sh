#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v307_adopted_preset_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$ROOT/exp/stageB/logs/377_explicit_binding_v305_boundary_safe_sh_refine_v305_startup_probe_20260519_112551_bjt/refine/no_train_summary.tsv}"
RENDER_BASELINE="${RENDER_BASELINE:-0}"
EXPORT_EDITABLE="${EXPORT_EDITABLE:-0}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v307_adopted_preset_render_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v307_adopted_preset_render_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
if [ "$RENDER_BASELINE" != "1" ] && [ ! -s "$BASELINE_SUMMARY" ]; then
  echo "baseline summary missing, falling back to RENDER_BASELINE=1: $BASELINE_SUMMARY" >&2
  RENDER_BASELINE=1
fi

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-900}"
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
BASELINE_SUMMARY=$BASELINE_SUMMARY
RENDER_BASELINE=$RENDER_BASELINE
EXPORT_EDITABLE=$EXPORT_EDITABLE
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v307 verifies the formal render preset path for the adopted explicit-binding
  geometry correction. It uses no training and avoids the old support/color
  refinement branches. The intended result should match v306 strict-pass metrics.
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

render_variant() {
  local variant="$1"
  local render_exp="$2"
  local preset="$3"
  local hydra_dir="$4"

  local preset_args=()
  if [ "$preset" != "none" ]; then
    preset_args=(
      "++explicit_binding_render_preset=$preset"
      "++explicit_binding_adopted_component_csv=$COMPONENT_CSV"
      "++explicit_binding_adopted_point_csv=$POINT_CSV"
      "++explicit_binding_adopted_center_strength=0.45"
      "++explicit_binding_adopted_outer_px=0.35"
      "++explicit_binding_adopted_component_required=true"
      "++explicit_binding_adopted_improvement_guard=true"
      "++explicit_binding_adopted_max_points=96"
    )
  fi

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "${preset_args[@]}" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=$EXPORT_EDITABLE" \
    "++export_opacity_maps=$EXPORT_EDITABLE" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
}

analyze_variant() {
  local variant="$1"
  local render_exp="$2"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${variant}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${variant}.log" 2>&1
}

ADOPTED_RENDER_EXP="$EXP_ROOT/adopted_preset_guard_045_outer035_m0"
if [ "$RENDER_BASELINE" = "1" ]; then
  BASELINE_RENDER_EXP="$EXP_ROOT/baseline_no_preset"
  log_event "render_start" "baseline_no_preset"
  render_variant baseline_no_preset "$BASELINE_RENDER_EXP" none "$HYDRA_RUN_ROOT/baseline_no_preset"
  analyze_variant baseline_no_preset "$BASELINE_RENDER_EXP"
  log_event "render_done" "baseline_no_preset"
fi

log_event "render_start" "adopted_preset_guard_045_outer035_m0"
render_variant adopted_preset_guard_045_outer035_m0 "$ADOPTED_RENDER_EXP" v307_adopted_geometry "$HYDRA_RUN_ROOT/adopted_preset_guard_045_outer035_m0"
analyze_variant adopted_preset_guard_045_outer035_m0 "$ADOPTED_RENDER_EXP"
log_event "render_done" "adopted_preset_guard_045_outer035_m0"

"$PYTHON_BIN" - "$SUMMARY" "$ADOPTED_RENDER_EXP" "$BASELINE_SUMMARY" "$RENDER_BASELINE" "${BASELINE_RENDER_EXP:-}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
adopted_render = Path(sys.argv[2])
baseline_summary = Path(sys.argv[3])
render_baseline = sys.argv[4] == "1"
baseline_render = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None

def metrics_from_render(render_exp):
    contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
    }

def metrics_from_summary(summary_file):
    rows = list(csv.DictReader(summary_file.open("r", encoding="utf-8"), delimiter="\t"))
    for row in rows:
        if row.get("variant") == "baseline_v281_screen_mid":
            return {key: float(row[key]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
    raise RuntimeError(f"baseline_v281_screen_mid not found in {summary_file}")

baseline = metrics_from_render(baseline_render) if render_baseline and baseline_render else metrics_from_summary(baseline_summary)
adopted = metrics_from_render(adopted_render)
delta = {key: adopted[key] - baseline[key] for key in adopted}
strict = (
    delta["inner"] < -0.05
    and delta["outer"] <= 0.0
    and delta["fg"] <= 0.0
    and delta["boundary"] <= 0.0
    and delta["edge"] <= 0.0
    and delta["hard"] < -0.000001
)
probe = (
    delta["hard"] < -0.00001
    and delta["fg"] <= 0.000015
    and delta["boundary"] <= 0.000015
    and delta["edge"] <= 0.003
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)
status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")

header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_v281", "boundary_delta_v281", "edge_delta_v281",
    "inner_delta_v281", "outer_delta_v281", "hard_delta_v281",
    "strict_pass", "probe_pass", "status",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerow([
        "adopted_preset_guard_045_outer035_m0",
        str(adopted_render),
        f"{adopted['fg']:.8f}",
        f"{adopted['boundary']:.8f}",
        f"{adopted['edge']:.6f}",
        f"{adopted['inner']:.4f}",
        f"{adopted['outer']:.4f}",
        f"{adopted['hard']:.8f}",
        f"{delta['fg']:.8f}",
        f"{delta['boundary']:.8f}",
        f"{delta['edge']:.6f}",
        f"{delta['inner']:.4f}",
        f"{delta['outer']:.4f}",
        f"{delta['hard']:.8f}",
        "1" if strict else "0",
        "1" if probe else "0",
        status,
    ])
print(json.dumps({"status": status, "delta": delta, "render_exp": str(adopted_render)}, indent=2))
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "ADOPTED_RENDER_EXP=$ADOPTED_RENDER_EXP"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "ADOPTED_RENDER_EXP=$ADOPTED_RENDER_EXP"
echo "SUMMARY=$SUMMARY"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
