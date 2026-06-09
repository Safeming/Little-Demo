#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v283_opacity_footprint_audit_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v283_opacity_footprint_audit_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v283_opacity_footprint_audit_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v283 is a no-train opacity footprint audit.
  It renders v271 ckpt136410 with the v281 dynamic_screen_mid render-time actuator,
  saves raw opacity maps, and compares RGB support vs opacity support on held-out views.
  The next train should only run if opacity also misses the same inner gaps.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'label\trender_exp\tfg\tboundary\tedge\trgb_inner\trgb_outer\thard\topacity_inner\topacity_outer\topacity_on_rgb_inner_ratio\tboth_rgb_opacity_inner_ratio\trgb_outer_with_opacity_ratio\tdiagnosis\tstatus\n' > "$SUMMARY"

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

render_opacity_audit() {
  local label="$1"
  local render_exp="$2"
  local hydra_dir="$3"

  log_event "render_start" "$label"
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
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_shrink_factor=1.000" \
    "++pipeline.covariance_signed_grow_factor=1.000" \
    "++pipeline.covariance_signed_anisotropic_axis=all" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=96" \
    "++pipeline.covariance_signed_dynamic_max_under_points=96" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${label}.log" 2>&1
}

analyze_all() {
  local label="$1"
  local render_exp="$2"

  log_event "analyze_contour_start" "$label"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1

  log_event "analyze_residual_start" "$label"
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
    > "$LOG_DIR/boundary_residuals_${label}.log" 2>&1

  log_event "analyze_opacity_start" "$label"
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
    --topk 16 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${label}.log" 2>&1
}

append_summary() {
  local label="$1"
  local render_exp="$2"
  "$PYTHON_BIN" - "$SUMMARY" "$label" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

summary_path, label, render_exp = sys.argv[1:4]
summary_path = Path(summary_path)
render_exp = Path(render_exp)
contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
opacity = json.loads((render_exp / "diagnostics" / "opacity_footprint" / "opacity_footprint_summary.json").read_text(encoding="utf-8"))

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    label,
    str(render_exp),
    fmt(contour["mean_fg_l1"]),
    fmt(contour["mean_boundary_l1"]),
    fmt(contour["mean_edge_symmetric_dist_px"], 6),
    fmt(residual["mean_inner_missing_pixels"], 4),
    fmt(residual["mean_outer_leak_pixels"], 4),
    fmt(residual["mean_hard_residual_score"]),
    fmt(opacity["mean_primary_opacity_inner_missing_pixels"], 4),
    fmt(opacity["mean_primary_opacity_outer_leak_pixels"], 4),
    fmt(opacity["mean_primary_opacity_on_rgb_inner_ratio"], 6),
    fmt(opacity["mean_primary_both_rgb_opacity_inner_missing_ratio"], 6),
    fmt(opacity["mean_primary_rgb_outer_with_opacity_ratio"], 6),
    opacity["diagnosis"],
    "ok",
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

LABEL="dynamic_screen_mid_opacity"
RENDER_EXP="$EXP_ROOT/no_train_${LABEL}"
render_opacity_audit "$LABEL" "$RENDER_EXP" "$HYDRA_RUN_ROOT/render_${LABEL}"
analyze_all "$LABEL" "$RENDER_EXP"
append_summary "$LABEL" "$RENDER_EXP"

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
