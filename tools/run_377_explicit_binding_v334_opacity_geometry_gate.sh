#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v334_opacity_geometry_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,1]}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v334_opacity_geometry_gate_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v334_opacity_geometry_gate_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"
for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/manifest.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" "$START_BJT" <<'PY' || true
import json, sys, time
from pathlib import Path
path, run_id, gpu, phase, detail, start_bjt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_bjt": start_bjt,
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
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
DATA_ROOT=$DATA_ROOT
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v334 separates RGB support from opacity/coverage support on the full v333
  frame set. This is a no-train geometry gate: RGB contour metrics remain
  secondary, opacity footprint is the primary raw geometry support signal.
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
  shift 2
  log_event "render_${variant}_start" "$render_exp"
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
    "$@" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "render_${variant}_done" "status=0"

  log_event "contours_${variant}_start" "$render_exp"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 30 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${variant}.log" 2>&1
  log_event "contours_${variant}_done" "status=0"

  log_event "rgb_residuals_${variant}_start" "$render_exp"
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 30 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${variant}.log" 2>&1
  log_event "rgb_residuals_${variant}_done" "status=0"

  log_event "opacity_footprint_${variant}_start" "$render_exp"
  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 \
    --rgb-close-kernel 5 \
    --opacity-close-kernel 3 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 30 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${variant}.log" 2>&1
  log_event "opacity_footprint_${variant}_done" "status=0"
}

BASELINE_EXP="$EXP_ROOT/baseline_no_preset"
FORMAL_EXP="$EXP_ROOT/formal_v320_v307_signed_geometry"

render_variant baseline_no_preset "$BASELINE_EXP" \
  "pipeline.compute_cov3D_python=true" \
  "++pipeline.covariance_mode=default" \
  "++pipeline.covariance_signed_dynamic_enable=false" \
  "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
  "++pipeline.covariance_signed_center_offset_enable=false" \
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

render_variant formal_v320_v307_signed_geometry "$FORMAL_EXP" \
  "++explicit_binding_render_preset=v320_v307_signed_geometry"

"$PYTHON_BIN" - "$SUMMARY" baseline_no_preset "$BASELINE_EXP" formal_v320_v307_signed_geometry "$FORMAL_EXP" <<'PY'
import csv, json, sys
from pathlib import Path

summary_path = Path(sys.argv[1])
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))

def load(render_exp):
    render_exp = Path(render_exp)
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8"))
    opacity = json.loads((render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8"))
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "rgb_inner": float(residual["mean_inner_missing_pixels"]),
        "rgb_outer": float(residual["mean_outer_leak_pixels"]),
        "rgb_hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
        "opacity_on_rgb_inner_ratio": float(opacity["mean_primary_opacity_on_rgb_inner_ratio"]),
        "both_rgb_opacity_inner_ratio": float(opacity["mean_primary_both_rgb_opacity_inner_missing_ratio"]),
        "rgb_outer_with_opacity_ratio": float(opacity["mean_primary_rgb_outer_with_opacity_ratio"]),
        "diagnosis": str(opacity.get("diagnosis", "")),
    }

metrics = {name: load(path) for name, path in pairs}
base = metrics["baseline_no_preset"]
header = [
    "variant", "render_exp", "fg", "boundary", "edge", "rgb_inner", "rgb_outer", "rgb_hard",
    "opacity_inner", "opacity_outer", "opacity_on_rgb_inner_ratio", "both_rgb_opacity_inner_ratio",
    "rgb_outer_with_opacity_ratio", "diagnosis",
    "fg_delta", "boundary_delta", "edge_delta", "rgb_inner_delta", "rgb_outer_delta", "rgb_hard_delta",
    "opacity_inner_delta", "opacity_outer_delta", "geometry_status",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in pairs:
        m = metrics[name]
        d = {key: m[key] - base[key] for key in ("fg", "boundary", "edge", "rgb_inner", "rgb_outer", "rgb_hard", "opacity_inner", "opacity_outer")}
        if name == "baseline_no_preset":
            status = "baseline"
        else:
            strict = d["opacity_inner"] <= 0.0 and d["opacity_outer"] <= 0.0 and d["rgb_hard"] <= 0.000001
            status = "geometry_strict_pass" if strict else "geometry_rejected"
        writer.writerow([
            name, render_exp,
            f'{m["fg"]:.8f}', f'{m["boundary"]:.8f}', f'{m["edge"]:.6f}',
            f'{m["rgb_inner"]:.4f}', f'{m["rgb_outer"]:.4f}', f'{m["rgb_hard"]:.8f}',
            f'{m["opacity_inner"]:.4f}', f'{m["opacity_outer"]:.4f}',
            f'{m["opacity_on_rgb_inner_ratio"]:.6f}', f'{m["both_rgb_opacity_inner_ratio"]:.6f}',
            f'{m["rgb_outer_with_opacity_ratio"]:.6f}', m["diagnosis"],
            f'{d["fg"]:.8f}', f'{d["boundary"]:.8f}', f'{d["edge"]:.6f}',
            f'{d["rgb_inner"]:.4f}', f'{d["rgb_outer"]:.4f}', f'{d["rgb_hard"]:.8f}',
            f'{d["opacity_inner"]:.4f}', f'{d["opacity_outer"]:.4f}', status,
        ])
print(summary_path)
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
