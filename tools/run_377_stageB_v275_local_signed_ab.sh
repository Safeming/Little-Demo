#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v275_local_signed_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_JSON="${CANDIDATE_JSON:-$ROOT/exp/stageB/logs/377_stageB_v274_contributor_audit_v274_contributor_audit_final_20260517_191438_bjt/v275_candidate_point_sets.json}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_stageB_v275_local_signed_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v275_local_signed_ab_${RUN_ID}}"
CKPT_DIR="$EXP_ROOT/checkpoints"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$CKPT_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_JSON" "$DATA_ROOT"; do
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
CANDIDATE_JSON=$CANDIDATE_JSON
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v275 no-train local signed calibration A/B.
  Use v274 shrink/grow point sets.
  Generate edited checkpoint variants, render raw explicit_binding test-view,
  then run contour and boundary residual gates.

Gate:
  strict_pass requires:
    inner_delta < 0
    outer_delta <= 0
    fg_delta <= 0
    boundary_delta <= 0
    edge_delta <= 0
  probe_pass requires:
    hard_delta < 0
    and no single RGB/edge metric has a large regression.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tedit_mode\tshrink_scale\tgrow_scale\tshrink_opacity\tgrow_opacity\tmax_shrink\tmax_grow\tckpt\trender_exp\tmean_fg_l1\tmean_boundary_l1\tmean_edge_symmetric_dist_px\tmean_inner_missing_pixels\tmean_outer_leak_pixels\tmean_hard_residual_score\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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

render_and_score() {
  local variant="$1"
  local edit_mode="$2"
  local shrink_scale="$3"
  local grow_scale="$4"
  local shrink_opacity="$5"
  local grow_opacity="$6"
  local max_shrink="$7"
  local max_grow="$8"
  local ckpt="$9"
  local render_exp="${10}"

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

  "$PYTHON_BIN" - "$variant" "$edit_mode" "$shrink_scale" "$grow_scale" "$shrink_opacity" "$grow_opacity" "$max_shrink" "$max_grow" "$ckpt" "$render_exp" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

(
    variant,
    edit_mode,
    shrink_scale,
    grow_scale,
    shrink_opacity,
    grow_opacity,
    max_shrink,
    max_grow,
    ckpt,
    render_exp,
    summary_path,
) = sys.argv[1:12]
render_exp = Path(render_exp)
summary_path = Path(summary_path)
contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))

def f(value):
    return float(value)

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

baseline = None
lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = lines[0]
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
    edit_mode,
    shrink_scale,
    grow_scale,
    shrink_opacity,
    grow_opacity,
    max_shrink,
    max_grow,
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
  local edit_mode="$2"
  local shrink_scale="$3"
  local grow_scale="$4"
  local shrink_opacity="$5"
  local grow_opacity="$6"
  local max_shrink="$7"
  local max_grow="$8"

  local ckpt="$CKPT_DIR/${variant}.pth"
  local report="$CKPT_DIR/${variant}_edit_report.json"
  local render_exp="$EXP_ROOT/${variant}"

  if [ "$variant" = "baseline" ]; then
    render_and_score "$variant" "$edit_mode" "$shrink_scale" "$grow_scale" "$shrink_opacity" "$grow_opacity" "$max_shrink" "$max_grow" "$BASE_CKPT" "$render_exp"
    return
  fi

  log_event "edit_start" "$variant shrink_scale=$shrink_scale grow_scale=$grow_scale shrink_opacity=$shrink_opacity grow_opacity=$grow_opacity max_shrink=$max_shrink max_grow=$max_grow"
  "$PYTHON_BIN" tools/make_377_stageB_v275_local_signed_checkpoint.py \
    --input-ckpt "$BASE_CKPT" \
    --candidate-json "$CANDIDATE_JSON" \
    --output-ckpt "$ckpt" \
    --report-json "$report" \
    --edit-mode "$edit_mode" \
    --max-shrink-points "$max_shrink" \
    --max-grow-points "$max_grow" \
    --shrink-scale-factor "$shrink_scale" \
    --grow-scale-factor "$grow_scale" \
    --shrink-opacity-factor "$shrink_opacity" \
    --grow-opacity-factor "$grow_opacity" \
    --reset-gaussian-optimizer-state \
    > "$LOG_DIR/edit_${variant}.log" 2>&1

  render_and_score "$variant" "$edit_mode" "$shrink_scale" "$grow_scale" "$shrink_opacity" "$grow_opacity" "$max_shrink" "$max_grow" "$ckpt" "$render_exp"
}

make_variant baseline none 1.000 1.000 1.000 1.000 0 0
make_variant local_scale_soft residual 0.985 1.015 1.000 1.000 96 96
make_variant local_scale_mid residual 0.970 1.020 1.000 1.000 96 96
make_variant local_scale_strong residual 0.950 1.030 1.000 1.000 96 96
make_variant shrink_only_mid residual 0.970 1.000 1.000 1.000 96 0
make_variant grow_only_mid residual 1.000 1.020 1.000 1.000 0 96
make_variant top48_scale_mid residual 0.970 1.020 1.000 1.000 48 48

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
