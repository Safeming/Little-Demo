#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v277_verified_support_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
APPEND_ITER="${APPEND_ITER:-136411}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_stageB_v277_verified_support_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v277_verified_support_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
CAND_DIR="$LOG_DIR/candidates"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

BASE_RENDER_EXP="$EXP_ROOT/baseline"
VISUAL_CSV="$CAND_DIR/visual_hull_candidates.csv"
VISUAL_SUMMARY="$CAND_DIR/visual_hull_candidate_summary.json"
ACTUAL_DIR="$CAND_DIR/actual_radii"
ACCEPTED_CSV="$ACTUAL_DIR/actual_radii_accepted_candidates.csv"
ACTUAL_SUMMARY="$ACTUAL_DIR/actual_radii_candidate_summary.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$CAND_DIR" "$ACTUAL_DIR"

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
CAND_DIR=$CAND_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v277 component-level verified support placement A/B.
  Do not move old Gaussians and do not train.
  Generate visual-hull 3D support candidates from raw inner residuals, validate
  them with actual rasterizer radii across held-out views, append only accepted
  candidates, render raw test-view, then gate.

Gate:
  strict_pass requires:
    inner_delta < -1
    outer_delta <= 0
    fg_delta <= 0
    boundary_delta <= 0
    edge_delta <= 0
  probe_pass is diagnostic only and does not authorize long training.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tmax_candidates\topacity_factor\tscale_factor\tscale_max\tappended_count\tckpt\trender_exp\tmean_fg_l1\tmean_boundary_l1\tmean_edge_symmetric_dist_px\tmean_inner_missing_pixels\tmean_outer_leak_pixels\tmean_hard_residual_score\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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
  "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64"
)

render_and_score() {
  local variant="$1"
  local max_candidates="$2"
  local opacity_factor="$3"
  local scale_factor="$4"
  local scale_max="$5"
  local appended_count="$6"
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
  local render_status=$?
  if [ "$render_status" -ne 0 ]; then
    log_event "render_failed" "$variant status=$render_status"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t0\t0\tfailed_render\n' \
      "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" "$appended_count" "$ckpt" "$render_exp" >> "$SUMMARY"
    return "$render_status"
  fi

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
  local contour_status=$?

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
  local residual_status=$?
  if [ "$contour_status" -ne 0 ] || [ "$residual_status" -ne 0 ]; then
    log_event "metrics_failed" "$variant contour=$contour_status residual=$residual_status"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t0\t0\tfailed_metrics\n' \
      "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" "$appended_count" "$ckpt" "$render_exp" >> "$SUMMARY"
    return 2
  fi

  "$PYTHON_BIN" - "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" "$appended_count" "$ckpt" "$render_exp" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

variant, max_candidates, opacity_factor, scale_factor, scale_max, appended_count, ckpt, render_exp, summary_path = sys.argv[1:10]
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
    max_candidates,
    opacity_factor,
    scale_factor,
    scale_max,
    appended_count,
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

append_and_render_variant() {
  local variant="$1"
  local max_candidates="$2"
  local opacity_factor="$3"
  local scale_factor="$4"
  local scale_max="$5"
  local append_exp="$EXP_ROOT/${variant}_append"
  local render_exp="$EXP_ROOT/${variant}"
  local ckpt="$append_exp/ckpt${APPEND_ITER}.pth"

  if [ ! -s "$ACCEPTED_CSV" ]; then
    log_event "append_skip" "$variant no accepted candidate csv"
    printf '%s\t%s\t%s\t%s\t%s\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t0\t0\tblocked_no_accepted_candidates\n' \
      "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" >> "$SUMMARY"
    return 0
  fi

  log_event "append_start" "$variant max=$max_candidates opacity=$opacity_factor scale=$scale_factor scale_max=$scale_max"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/append_377_stageB_v277_verified_support.py \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --candidates-csv "$ACCEPTED_CSV" \
    --out-dir "$append_exp" \
    --dataset-root "$DATA_ROOT" \
    --max-candidates "$max_candidates" \
    --checkpoint-iteration "$APPEND_ITER" \
    --parent-screen-radius 42.0 \
    --child-opacity-factor "$opacity_factor" \
    --child-opacity-floor 0.040 \
    --child-opacity-ceiling 0.32 \
    --child-scale-factor "$scale_factor" \
    --child-scale-max "$scale_max" \
    > "$LOG_DIR/append_${variant}.log" 2>&1
  local append_status=$?
  if [ "$append_status" -ne 0 ]; then
    log_event "append_failed" "$variant status=$append_status"
    printf '%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t0\t0\tfailed_append\n' \
      "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" "$ckpt" "$render_exp" >> "$SUMMARY"
    return "$append_status"
  fi

  local appended_count
  appended_count="$("$PYTHON_BIN" - "$append_exp/v277_verified_support_append_summary.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('appended_count', 0))
PY
)"
  render_and_score "$variant" "$max_candidates" "$opacity_factor" "$scale_factor" "$scale_max" "$appended_count" "$ckpt" "$render_exp"
}

log_event "baseline_render" "$BASE_RENDER_EXP"
render_and_score "baseline" "0" "0" "0" "0" "0" "$BASE_CKPT" "$BASE_RENDER_EXP" || exit $?

log_event "visual_hull_start" "$VISUAL_CSV"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v269_visual_hull_candidates.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --base-render-exp "$BASE_RENDER_EXP" \
  --out-csv "$VISUAL_CSV" \
  --out-summary "$VISUAL_SUMMARY" \
  --dataset-root "$DATA_ROOT" \
  --eval-views "21,22,23" \
  --frames "0,60,120,180,240,300,360,420,480,540" \
  --render-support-threshold 0.025 \
  --close-kernel 5 \
  --band-width 7 \
  --search-band-width 24 \
  --min-component-area 14 \
  --max-components-per-view 6 \
  --depth-samples 128 \
  --depth-margin 0.08 \
  --disk-radius 3.5 \
  --max-output-candidates 72 \
  --min-inner-pixels 6.0 \
  --max-outer-pixels 8.0 \
  --min-mask-views 1 \
  --max-per-frame 10 \
  > "$LOG_DIR/visual_hull_candidates.log" 2>&1
visual_status=$?
if [ "$visual_status" -ne 0 ]; then
  log_event "visual_hull_blocked" "status=$visual_status"
else
  log_event "visual_hull_done" "$VISUAL_SUMMARY"
fi

if [ -s "$VISUAL_CSV" ]; then
  log_event "actual_radii_start" "$ACTUAL_DIR"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/validate_377_stageB_v264_actual_radii_candidates.py \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --candidates-csv "$VISUAL_CSV" \
    --out-dir "$ACTUAL_DIR" \
    --dataset-root "$DATA_ROOT" \
    --eval-views "21,22,23" \
    --max-input-candidates 72 \
    --max-output-candidates 24 \
    --parent-screen-radius 42.0 \
    --child-opacity-factor 0.80 \
    --child-opacity-floor 0.040 \
    --child-opacity-ceiling 0.32 \
    --child-scale-factor 0.55 \
    --radii-scale 1.0 \
    --min-radius-px 1.0 \
    --max-radius-px 12.0 \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --min-component-area 18 \
    --min-actual-inner-pixels 4.0 \
    --max-actual-outer-pixels 3.0 \
    --max-actual-outer-inner-ratio 0.25 \
    --max-per-frame 8 \
    > "$LOG_DIR/actual_radii_candidates.log" 2>&1
  actual_status=$?
  if [ "$actual_status" -ne 0 ]; then
    log_event "actual_radii_blocked" "status=$actual_status"
  else
    log_event "actual_radii_done" "$ACTUAL_SUMMARY"
  fi
else
  log_event "actual_radii_skip" "no visual candidates"
fi

append_and_render_variant "support_top8_soft" "8" "0.55" "0.35" "0.006"
append_and_render_variant "support_top16_mid" "16" "0.80" "0.55" "0.008"
append_and_render_variant "support_top24_mid" "24" "0.80" "0.55" "0.008"

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "VISUAL_SUMMARY=$VISUAL_SUMMARY"
  echo "ACTUAL_SUMMARY=$ACTUAL_SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "VISUAL_SUMMARY=$VISUAL_SUMMARY"
echo "ACTUAL_SUMMARY=$ACTUAL_SUMMARY"
echo "END_BJT=$END_BJT"
