#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v273_footprint_factor_probe_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v273_footprint_factor_probe_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v273_footprint_factor_probe_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v273 no-train footprint factor probe.
  This is a render-only factor sweep, not a training run.

Factors:
  baseline: current raw explicit_binding render.
  scale_*: global Gaussian footprint shrink/grow via render_scaling_modifier.
  orth_*: explicit_binding R_bar orthogonalization and its interaction with scale.
  cov_raster_rotation: disable python covariance so rasterizer uses original Gaussian quaternion/scales.
  no_camera_geometry: disable camera geometry correction at render time.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\trender_exp\torth\tscale\tcompute_cov3D_python\tcamera_geometry\tmean_fg_l1\tmean_boundary_l1\tmean_edge_symmetric_dist_px\tmean_inner_missing_pixels\tmean_outer_leak_pixels\tmean_hard_residual_score\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\tstatus\n' > "$SUMMARY"

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

render_variant() {
  local variant="$1"
  local orth_flag="$2"
  local scale="$3"
  local cov_python="$4"
  local camera_geometry="$5"
  local render_exp="$EXP_ROOT/${variant}"

  log_event "render_start" "$variant orth=$orth_flag scale=$scale cov_python=$cov_python camera_geometry=$camera_geometry"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "pipeline.compute_cov3D_python=$cov_python" \
    "++render_scaling_modifier=$scale" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=$orth_flag" \
    "++opt.camera_geometry_enable=$camera_geometry" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/render_${variant}" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1

  log_event "contour_start" "$variant"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${variant}.log" 2>&1

  log_event "residual_start" "$variant"
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

  "$PYTHON_BIN" - "$variant" "$render_exp" "$orth_flag" "$scale" "$cov_python" "$camera_geometry" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

variant, render_exp, orth, scale, cov_python, camera_geometry, summary_path = sys.argv[1:8]
render_exp = Path(render_exp)
summary_path = Path(summary_path)
contour_path = render_exp / "diagnostics" / "contours" / "contour_summary.json"
residual_path = render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json"
contour = json.loads(contour_path.read_text(encoding="utf-8")) if contour_path.exists() else {}
residual = json.loads(residual_path.read_text(encoding="utf-8")) if residual_path.exists() else {}

def as_float(value):
    try:
        return float(value)
    except Exception:
        return None

def fmt(value, digits=8):
    value = as_float(value)
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"

baseline = None
rows = []
if summary_path.exists():
    lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
    header = lines[0] if lines else []
    for row in lines[1:]:
        if row and row[0] == "baseline":
            baseline = dict(zip(header, row))
            break

metrics = {
    "mean_fg_l1": contour.get("mean_fg_l1"),
    "mean_boundary_l1": contour.get("mean_boundary_l1"),
    "mean_edge_symmetric_dist_px": contour.get("mean_edge_symmetric_dist_px"),
    "mean_inner_missing_pixels": residual.get("mean_inner_missing_pixels"),
    "mean_outer_leak_pixels": residual.get("mean_outer_leak_pixels"),
    "mean_hard_residual_score": residual.get("mean_hard_residual_score"),
}

if baseline is None or variant == "baseline":
    deltas = {"fg": 0.0, "boundary": 0.0, "edge": 0.0, "inner": 0.0, "outer": 0.0}
else:
    deltas = {
        "fg": as_float(metrics["mean_fg_l1"]) - as_float(baseline["mean_fg_l1"]),
        "boundary": as_float(metrics["mean_boundary_l1"]) - as_float(baseline["mean_boundary_l1"]),
        "edge": as_float(metrics["mean_edge_symmetric_dist_px"]) - as_float(baseline["mean_edge_symmetric_dist_px"]),
        "inner": as_float(metrics["mean_inner_missing_pixels"]) - as_float(baseline["mean_inner_missing_pixels"]),
        "outer": as_float(metrics["mean_outer_leak_pixels"]) - as_float(baseline["mean_outer_leak_pixels"]),
    }

row = [
    variant,
    str(render_exp),
    orth,
    scale,
    cov_python,
    camera_geometry,
    fmt(metrics["mean_fg_l1"]),
    fmt(metrics["mean_boundary_l1"]),
    fmt(metrics["mean_edge_symmetric_dist_px"], 6),
    fmt(metrics["mean_inner_missing_pixels"], 4),
    fmt(metrics["mean_outer_leak_pixels"], 4),
    fmt(metrics["mean_hard_residual_score"], 8),
    fmt(deltas["fg"]),
    fmt(deltas["boundary"]),
    fmt(deltas["edge"], 6),
    fmt(deltas["inner"], 4),
    fmt(deltas["outer"], 4),
    "ok",
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY

  log_event "variant_done" "$render_exp"
}

render_variant baseline false 1.00 true true
render_variant scale_097 false 0.97 true true
render_variant scale_095 false 0.95 true true
render_variant scale_090 false 0.90 true true
render_variant scale_103 false 1.03 true true
render_variant orth_100 true 1.00 true true
render_variant orth_scale_095 true 0.95 true true
render_variant cov_raster_rotation false 1.00 false true
render_variant no_camera_geometry false 1.00 true false

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
