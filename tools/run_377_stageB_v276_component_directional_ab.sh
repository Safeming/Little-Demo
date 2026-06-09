#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v276_component_directional_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_stageB_v276_component_directional_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v276_component_directional_ab_${RUN_ID}}"
CKPT_DIR="$EXP_ROOT/checkpoints"
PLAN_DIR="$LOG_DIR/component_plan"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PLAN_JSON="$PLAN_DIR/component_direction_plan.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$CKPT_DIR" "$PLAN_DIR" "$HYDRA_RUN_ROOT"

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
PLAN_DIR=$PLAN_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v276 no-train component-level signed directional actuator probe.
  Build residual-component directional plan, edit local canonical xyz with
  binding_state sync, render raw explicit_binding test-view, and gate.

Gate:
  strict_pass requires:
    inner_delta < 0
    outer_delta <= 0
    fg_delta <= 0
    boundary_delta <= 0
    edge_delta <= 0
  probe_pass requires hard_delta < -0.0005 without large RGB/edge regressions.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tdirection\tdelta_scale\tmax_points\tmax_point_step\tselected_count\tckpt\trender_exp\tmean_fg_l1\tmean_boundary_l1\tmean_edge_symmetric_dist_px\tmean_inner_missing_pixels\tmean_outer_leak_pixels\tmean_hard_residual_score\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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

log_event "plan_start" "$PLAN_DIR"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/plan_377_stageB_v276_component_directions.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --out-dir "$PLAN_DIR" \
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
  --component-shift-px 1.25 \
  --component-weight-area-power 0.5 \
  --jacobian-eps 0.001 \
  --jacobian-damping 0.00001 \
  --max-component-world-step 0.003 \
  --max-point-canonical-step 0.006 \
  --min-point-weight 1.0 \
  --min-direction-consistency 0.25 \
  --max-plan-points 384 \
  --render-scaling-modifier 1.0 \
  --compute-cov3d-python \
  --camera-geometry \
  > "$LOG_DIR/plan.log" 2>&1
log_event "plan_done" "$PLAN_JSON"

render_and_score() {
  local variant="$1"
  local direction="$2"
  local delta_scale="$3"
  local max_points="$4"
  local max_point_step="$5"
  local selected_count="$6"
  local ckpt="$7"
  local render_exp="$8"

  log_event "render_start" "$variant ckpt=$ckpt"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "pipeline.compute_cov3D_python=true" \
    "++render_scaling_modifier=1.0" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++opt.camera_geometry_enable=true" \
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

  "$PYTHON_BIN" - "$variant" "$direction" "$delta_scale" "$max_points" "$max_point_step" "$selected_count" "$ckpt" "$render_exp" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

variant, direction, delta_scale, max_points, max_point_step, selected_count, ckpt, render_exp, summary_path = sys.argv[1:10]
render_exp = Path(render_exp)
summary_path = Path(summary_path)
contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))

def f(value):
    return float(value)

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = lines[0]
baseline = None
for row in lines[1:]:
    if row and row[0] == "baseline":
        baseline = dict(zip(header, row))
        break

metrics = {
    "mean_fg_l1": f(contour["mean_fg_l1"]),
    "mean_boundary_l1": f(contour["mean_boundary_l1"]),
    "mean_edge_symmetric_dist_px": f(contour["mean_edge_symmetric_dist_px"]),
    "mean_inner_missing_pixels": f(residual["mean_inner_missing_pixels"]),
    "mean_outer_leak_pixels": f(residual["mean_outer_leak_pixels"]),
    "mean_hard_residual_score": f(residual["mean_hard_residual_score"]),
}
if baseline is None or variant == "baseline":
    delta = {key: 0.0 for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
else:
    delta = {
        "fg": metrics["mean_fg_l1"] - f(baseline["mean_fg_l1"]),
        "boundary": metrics["mean_boundary_l1"] - f(baseline["mean_boundary_l1"]),
        "edge": metrics["mean_edge_symmetric_dist_px"] - f(baseline["mean_edge_symmetric_dist_px"]),
        "inner": metrics["mean_inner_missing_pixels"] - f(baseline["mean_inner_missing_pixels"]),
        "outer": metrics["mean_outer_leak_pixels"] - f(baseline["mean_outer_leak_pixels"]),
        "hard": metrics["mean_hard_residual_score"] - f(baseline["mean_hard_residual_score"]),
    }
strict_pass = (
    delta["inner"] < -1.0
    and delta["outer"] <= 0.0
    and delta["fg"] <= 0.0
    and delta["boundary"] <= 0.0
    and delta["edge"] <= 0.0
)
probe_pass = (
    delta["hard"] < -0.0005
    and delta["fg"] <= 0.0015
    and delta["boundary"] <= 0.0015
    and delta["edge"] <= 0.12
    and delta["inner"] <= 80.0
    and delta["outer"] <= 160.0
)
row = [
    variant,
    direction,
    delta_scale,
    max_points,
    max_point_step,
    selected_count,
    ckpt,
    str(render_exp),
    fmt(metrics["mean_fg_l1"]),
    fmt(metrics["mean_boundary_l1"]),
    fmt(metrics["mean_edge_symmetric_dist_px"], 6),
    fmt(metrics["mean_inner_missing_pixels"], 4),
    fmt(metrics["mean_outer_leak_pixels"], 4),
    fmt(metrics["mean_hard_residual_score"], 8),
    fmt(delta["fg"]),
    fmt(delta["boundary"]),
    fmt(delta["edge"], 6),
    fmt(delta["inner"], 4),
    fmt(delta["outer"], 4),
    fmt(delta["hard"], 8),
    "1" if strict_pass else "0",
    "1" if probe_pass else "0",
    "ok",
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY

  log_event "variant_done" "$variant"
}

make_variant() {
  local variant="$1"
  local direction="$2"
  local delta_scale="$3"
  local max_points="$4"
  local max_point_step="$5"
  local ckpt="$CKPT_DIR/${variant}.pth"
  local report="$CKPT_DIR/${variant}_edit_report.json"
  local render_exp="$EXP_ROOT/${variant}"
  local selected_count="0"

  if [ "$variant" = "baseline" ]; then
    render_and_score "$variant" "$direction" "$delta_scale" "$max_points" "$max_point_step" "$selected_count" "$BASE_CKPT" "$render_exp"
    return
  fi

  log_event "edit_start" "$variant direction=$direction delta_scale=$delta_scale max_points=$max_points max_point_step=$max_point_step"
  "$PYTHON_BIN" tools/make_377_stageB_v276_directional_checkpoint.py \
    --input-ckpt "$BASE_CKPT" \
    --plan-json "$PLAN_JSON" \
    --output-ckpt "$ckpt" \
    --report-json "$report" \
    --direction "$direction" \
    --delta-scale "$delta_scale" \
    --max-points "$max_points" \
    --max-point-step "$max_point_step" \
    --min-direction-consistency 0.25 \
    --max-conflict-ratio 0.70 \
    --sync-binding-state \
    --reset-gaussian-optimizer-state \
    > "$LOG_DIR/edit_${variant}.log" 2>&1
  selected_count="$("$PYTHON_BIN" - "$report" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('selected_count', 0))
PY
)"
  render_and_score "$variant" "$direction" "$delta_scale" "$max_points" "$max_point_step" "$selected_count" "$ckpt" "$render_exp"
}

make_variant baseline none 0.0 0 0.000
make_variant all_small all 0.50 384 0.003
make_variant all_mid all 1.00 384 0.006
make_variant all_top128 all 1.00 128 0.006
make_variant outer_mid outer 1.00 192 0.006
make_variant inner_mid inner 1.00 192 0.006
make_variant all_strong all 1.50 384 0.008

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "PLAN_JSON=$PLAN_JSON"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "PLAN_JSON=$PLAN_JSON"
echo "END_BJT=$END_BJT"
