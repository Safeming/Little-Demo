#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v279_covariance_footprint_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v279_covariance_footprint_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v279_covariance_footprint_ab_${RUN_ID}}"
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
  v279 no-train explicit_binding covariance / footprint A/B.
  Same checkpoint, same test-view frames, render_export_refine=false.
  This probes the root cause after v278 showed support selection has only tiny effect.

Factors:
  baseline: current binding covariance path.
  cov_orth: orthogonalize the final covariance rotation only.
  cov_canonical_rotation: ignore binding rotation_precomp for covariance.
  cov_rot_iso_geom: isotropic scale but still through binding rotation_precomp.
  cov_world_iso_geom: isotropic world covariance, ignores rotation.
  cov_aniso_clamp_*: clamp Gaussian axis anisotropy before covariance.
  *_scale_*: same covariance mode with render_scaling_modifier shrink/grow.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\trender_exp\tcov_mode\tanisotropy_clamp\tisotropic_reduce\trender_scale\torth\tcamera_geometry\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tstatus\n' > "$SUMMARY"

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

append_summary_row() {
  local variant="$1"
  local render_exp="$2"
  local cov_mode="$3"
  local anisotropy_clamp="$4"
  local isotropic_reduce="$5"
  local render_scale="$6"
  local orth="$7"
  local camera_geometry="$8"

  "$PYTHON_BIN" - "$variant" "$render_exp" "$cov_mode" "$anisotropy_clamp" "$isotropic_reduce" "$render_scale" "$orth" "$camera_geometry" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

(
    variant,
    render_exp,
    cov_mode,
    anisotropy_clamp,
    isotropic_reduce,
    render_scale,
    orth,
    camera_geometry,
    summary_path,
) = sys.argv[1:10]
render_exp = Path(render_exp)
summary_path = Path(summary_path)

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

baseline = None
if summary_path.exists():
    lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
    header = lines[0] if lines else []
    for row in lines[1:]:
        if row and row[0] == "baseline":
            baseline = {
                "fg": float(row[header.index("fg")]),
                "boundary": float(row[header.index("boundary")]),
                "edge": float(row[header.index("edge")]),
                "inner": float(row[header.index("inner")]),
                "outer": float(row[header.index("outer")]),
                "hard": float(row[header.index("hard")]),
            }
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

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    variant,
    str(render_exp),
    cov_mode,
    anisotropy_clamp,
    isotropic_reduce,
    render_scale,
    orth,
    camera_geometry,
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
    "ok",
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

render_variant() {
  local variant="$1"
  local cov_mode="$2"
  local anisotropy_clamp="$3"
  local isotropic_reduce="$4"
  local render_scale="$5"
  local orth="$6"
  local camera_geometry="$7"
  local render_exp="$EXP_ROOT/${variant}"

  log_event "render_start" "$variant cov_mode=$cov_mode clamp=$anisotropy_clamp reduce=$isotropic_reduce scale=$render_scale orth=$orth camera_geometry=$camera_geometry"
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
    "++pipeline.covariance_mode=$cov_mode" \
    "++pipeline.covariance_anisotropy_clamp=$anisotropy_clamp" \
    "++pipeline.covariance_isotropic_reduce=$isotropic_reduce" \
    "++render_scaling_modifier=$render_scale" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=$orth" \
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

  append_summary_row "$variant" "$render_exp" "$cov_mode" "$anisotropy_clamp" "$isotropic_reduce" "$render_scale" "$orth" "$camera_geometry"
  log_event "variant_done" "$variant"
}

render_variant baseline default 0.0 geom 1.00 false true
render_variant cov_orth orthogonalized 0.0 geom 1.00 false true
render_variant cov_canonical_rotation canonical_rotation 0.0 geom 1.00 false true
render_variant cov_rot_iso_geom rotation_isotropic_geom 0.0 geom 1.00 false true
render_variant cov_world_iso_geom world_isotropic_geom 0.0 geom 1.00 false true
render_variant cov_aniso_clamp_150 default 1.50 geom 1.00 false true
render_variant cov_aniso_clamp_125 default 1.25 geom 1.00 false true
render_variant cov_aniso_clamp_110 default 1.10 geom 1.00 false true
render_variant cov_orth_scale_097 orthogonalized 0.0 geom 0.97 false true
render_variant cov_world_iso_scale_103 world_isotropic_geom 0.0 geom 1.03 false true

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
