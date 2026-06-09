#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v292_score_weighted_actuator_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
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

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v292_score_weighted_actuator_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v292_score_weighted_actuator_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/no_train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tscreen_shrink\tscreen_grow\tcomponent_required\tcomponent_signature\tcomponent_pad\tcomponent_scale\tmax_over\tmax_under\tscore_weight\tscore_power\tscore_min\tscore_quantile\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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
  v292 keeps correction view/component-conditioned and no-train. It tests a
  score-weighted screen covariance actuator: component scores scale the normal
  shrink/grow factor continuously instead of applying a binary full-strength
  point mask. No checkpoint parameters are trained.
EOF

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

render_raw() {
  local variant="$1"
  local render_exp="$2"
  local screen_shrink="$3"
  local screen_grow="$4"
  local component_required="$5"
  local component_signature="$6"
  local component_pad="$7"
  local component_scale="$8"
  local max_over="$9"
  local max_under="${10}"
  local score_weight="${11}"
  local score_power="${12}"
  local score_min="${13}"
  local score_quantile="${14}"
  local hydra_dir="${15}"

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
    "++pipeline.covariance_signed_dynamic_component_signature_enable=$component_signature" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=$component_pad" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=$component_scale" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=$component_required" \
    "++pipeline.covariance_signed_dynamic_score_weighting_enable=$score_weight" \
    "++pipeline.covariance_signed_dynamic_score_weight_power=$score_power" \
    "++pipeline.covariance_signed_dynamic_score_weight_min=$score_min" \
    "++pipeline.covariance_signed_dynamic_score_weight_quantile=$score_quantile" \
    "++pipeline.covariance_signed_dynamic_max_over_points=$max_over" \
    "++pipeline.covariance_signed_dynamic_max_under_points=$max_under" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=$screen_shrink" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=$screen_grow" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true"
}

analyze_raw() {
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

append_summary() {
  local variant="$1"
  local screen_shrink="$2"
  local screen_grow="$3"
  local component_required="$4"
  local component_signature="$5"
  local component_pad="$6"
  local component_scale="$7"
  local max_over="$8"
  local max_under="$9"
  local score_weight="${10}"
  local score_power="${11}"
  local score_min="${12}"
  local score_quantile="${13}"
  local render_exp="${14}"

  "$PYTHON_BIN" - "$SUMMARY" "$variant" "$screen_shrink" "$screen_grow" "$component_required" "$component_signature" "$component_pad" "$component_scale" "$max_over" "$max_under" "$score_weight" "$score_power" "$score_min" "$score_quantile" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    variant,
    screen_shrink,
    screen_grow,
    component_required,
    component_signature,
    component_pad,
    component_scale,
    max_over,
    max_under,
    score_weight,
    score_power,
    score_min,
    score_quantile,
    render_exp,
) = sys.argv[1:16]
summary_path = Path(summary_path)
render_exp = Path(render_exp)
contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
metrics = {
    "fg": float(contour["mean_fg_l1"]),
    "boundary": float(contour["mean_boundary_l1"]),
    "edge": float(contour["mean_edge_symmetric_dist_px"]),
    "inner": float(residual["mean_inner_missing_pixels"]),
    "outer": float(residual["mean_outer_leak_pixels"]),
    "hard": float(residual["mean_hard_residual_score"]),
}
lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = lines[0]
baseline = None
for row in lines[1:]:
    if row and row[0] == "baseline":
        baseline = {key: float(row[header.index(key)]) for key in metrics}
        break
if baseline is None or variant == "baseline":
    baseline = dict(metrics)
delta = {key: metrics[key] - baseline[key] for key in metrics}
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
    and delta["fg"] <= 0.00002
    and delta["boundary"] <= 0.00002
    and delta["edge"] <= 0.003
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    variant,
    screen_shrink,
    screen_grow,
    component_required,
    component_signature,
    component_pad,
    component_scale,
    max_over,
    max_under,
    score_weight,
    score_power,
    score_min,
    score_quantile,
    str(render_exp),
    fmt(metrics["fg"]),
    fmt(metrics["boundary"]),
    fmt(metrics["edge"], 6),
    fmt(metrics["inner"], 4),
    fmt(metrics["outer"], 4),
    fmt(metrics["hard"]),
    fmt(delta["fg"]),
    fmt(delta["boundary"]),
    fmt(delta["edge"], 6),
    fmt(delta["inner"], 4),
    fmt(delta["outer"], 4),
    fmt(delta["hard"]),
    "1" if strict else "0",
    "1" if probe else "0",
    "strict_pass" if strict else ("probe_pass" if probe else "rejected"),
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

run_variant() {
  local variant="$1"
  local screen_shrink="$2"
  local screen_grow="$3"
  local component_required="$4"
  local component_signature="$5"
  local component_pad="$6"
  local component_scale="$7"
  local max_over="$8"
  local max_under="$9"
  local score_weight="${10}"
  local score_power="${11}"
  local score_min="${12}"
  local score_quantile="${13}"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  log_event "render_start" "$variant"
  render_raw "$variant" "$render_exp" "$screen_shrink" "$screen_grow" "$component_required" "$component_signature" "$component_pad" "$component_scale" "$max_over" "$max_under" "$score_weight" "$score_power" "$score_min" "$score_quantile" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  append_summary "$variant" "$screen_shrink" "$screen_grow" "$component_required" "$component_signature" "$component_pad" "$component_scale" "$max_over" "$max_under" "$score_weight" "$score_power" "$score_min" "$score_quantile" "$render_exp"
  log_event "variant_done" "$variant"
}

run_variant baseline 1.000 1.000 false false 10 1.25 0 0 false 1.0 0.0 0.90
run_variant v281_screen_mid_binary 0.940 1.025 false false 10 1.25 96 96 false 1.0 0.0 0.90
run_variant score_mid_power1_min0 0.940 1.025 false false 10 1.25 96 96 true 1.0 0.0 0.90
run_variant score_mid_power1_min35 0.940 1.025 false false 10 1.25 96 96 true 1.0 0.35 0.90
run_variant score_mid_power2_min25 0.940 1.025 false false 10 1.25 96 96 true 2.0 0.25 0.90
run_variant score_component_mid_min25 0.940 1.025 true false 14 1.35 96 96 true 1.0 0.25 0.90
run_variant score_signature_mid_min25 0.940 1.025 true true 18 1.55 96 96 true 1.0 0.25 0.90
run_variant score_soft_min25 0.970 1.020 false false 10 1.25 96 96 true 1.0 0.25 0.90

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
