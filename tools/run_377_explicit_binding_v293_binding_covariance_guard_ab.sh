#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v293_binding_covariance_guard_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v293_binding_covariance_guard_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v293_binding_covariance_guard_ab_${RUN_ID}}"
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
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tmode\tstrength\tboundary_min\tlayer_ids\tregion_ids\tjoint_ids\tthin_min\tsurface_min\tsurface_max\tpower\tmax_points\taniso_clamp\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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
  v293 tests whether the explicit-binding footprint can be corrected without
  residual component CSVs or point residual training. It applies a local
  binding-derived covariance guard on high boundary/thin soft/free points only.
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
  local enable="$3"
  local mode="$4"
  local strength="$5"
  local boundary_min="$6"
  local layer_ids="$7"
  local region_ids="$8"
  local joint_ids="$9"
  local thin_min="${10}"
  local surface_min="${11}"
  local surface_max="${12}"
  local power="${13}"
  local max_points="${14}"
  local aniso_clamp="${15}"
  local hydra_dir="${16}"

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
    "++pipeline.covariance_anisotropy_clamp=0.0" \
    "++pipeline.binding_covariance_guard_enable=$enable" \
    "++pipeline.binding_covariance_guard_mode=$mode" \
    "++pipeline.binding_covariance_guard_strength=$strength" \
    "++pipeline.binding_covariance_guard_boundary_min=$boundary_min" \
    "++pipeline.binding_covariance_guard_layer_ids='$layer_ids'" \
    "++pipeline.binding_covariance_guard_region_ids='$region_ids'" \
    "++pipeline.binding_covariance_guard_joint_ids='$joint_ids'" \
    "++pipeline.binding_covariance_guard_thin_min=$thin_min" \
    "++pipeline.binding_covariance_guard_surface_min=$surface_min" \
    "++pipeline.binding_covariance_guard_surface_max=$surface_max" \
    "++pipeline.binding_covariance_guard_power=$power" \
    "++pipeline.binding_covariance_guard_max_points=$max_points" \
    "++pipeline.binding_covariance_guard_anisotropy_clamp=$aniso_clamp" \
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
  local mode="$2"
  local strength="$3"
  local boundary_min="$4"
  local layer_ids="$5"
  local region_ids="$6"
  local joint_ids="$7"
  local thin_min="$8"
  local surface_min="$9"
  local surface_max="${10}"
  local power="${11}"
  local max_points="${12}"
  local aniso_clamp="${13}"
  local render_exp="${14}"

  "$PYTHON_BIN" - "$SUMMARY" "$variant" "$mode" "$strength" "$boundary_min" "$layer_ids" "$region_ids" "$joint_ids" "$thin_min" "$surface_min" "$surface_max" "$power" "$max_points" "$aniso_clamp" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    variant,
    mode,
    strength,
    boundary_min,
    layer_ids,
    region_ids,
    joint_ids,
    thin_min,
    surface_min,
    surface_max,
    power,
    max_points,
    aniso_clamp,
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
    and delta["edge"] <= 0.004
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    variant,
    mode,
    strength,
    boundary_min,
    layer_ids,
    region_ids,
    joint_ids,
    thin_min,
    surface_min,
    surface_max,
    power,
    max_points,
    aniso_clamp,
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
  local enable="$2"
  local mode="$3"
  local strength="$4"
  local boundary_min="$5"
  local layer_ids="$6"
  local region_ids="$7"
  local joint_ids="$8"
  local thin_min="$9"
  local surface_min="${10}"
  local surface_max="${11}"
  local power="${12}"
  local max_points="${13}"
  local aniso_clamp="${14}"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  log_event "render_start" "$variant"
  render_raw "$variant" "$render_exp" "$enable" "$mode" "$strength" "$boundary_min" "$layer_ids" "$region_ids" "$joint_ids" "$thin_min" "$surface_min" "$surface_max" "$power" "$max_points" "$aniso_clamp" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  append_summary "$variant" "$mode" "$strength" "$boundary_min" "$layer_ids" "$region_ids" "$joint_ids" "$thin_min" "$surface_min" "$surface_max" "$power" "$max_points" "$aniso_clamp" "$render_exp"
  log_event "variant_done" "$variant"
}

run_variant baseline false canonical_blend 0.0 0.08 soft,free cloth,soft "" "" "" "" 1.0 0 1.25
run_variant canonical_soft_35 true canonical_blend 0.35 0.08 soft,free cloth,soft "" "" "" "" 1.0 512 1.25
run_variant canonical_soft_55 true canonical_blend 0.55 0.08 soft,free cloth,soft "" "" "" "" 1.0 512 1.25
run_variant canonical_boundary_35 true canonical_blend 0.35 0.12 soft,free cloth,soft "" "" "" "" 1.0 256 1.25
run_variant aniso_clamp_35 true aniso_clamp 0.35 0.08 soft,free cloth,soft "" "" "" "" 1.0 512 1.20
run_variant rot_iso_25 true rotation_isotropic_geom 0.25 0.08 soft,free cloth,soft "" "" "" "" 1.0 512 1.25
run_variant canonical_thin_45 true canonical_blend 0.45 0.05 soft,free cloth,soft "" 0.10 "" "" 1.0 512 1.25
run_variant canonical_armcloth_35 true canonical_blend 0.35 0.08 soft,free cloth "" "" "" "" 1.0 512 1.25

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
