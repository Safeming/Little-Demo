#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-v341_point_screen_guard_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v341_point_screen_guard_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v341_point_screen_guard_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,1]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"
POINT_SCREEN_DROP_IMAGES="${POINT_SCREEN_DROP_IMAGES:-c21_f000090,c21_f000091,c21_f000092,c21_f000093,c21_f000094,c21_f000095,c21_f000096,c21_f000097,c23_f000315,c23_f000316,c23_f000317,c23_f000318,c23_f000319,c23_f000320,c23_f000321,c23_f000322,c23_f000323,c23_f000324,c23_f000325,c23_f000411}"

SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

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

render_and_analyze() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  shift 3

  log_event "render_${variant}_start" "$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    "$@" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "render_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
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
    > "$LOG_DIR/opacity_footprint_${variant}.log" 2>&1
  log_event "analyze_${variant}_done" "status=0"
}

render_and_analyze baseline_no_preset "$BASE_CKPT" "$EXP_ROOT/baseline_no_preset" \
  "pipeline.compute_cov3D_python=true" \
  "++pipeline.covariance_mode=default" \
  "++pipeline.covariance_signed_dynamic_enable=false" \
  "++pipeline.covariance_signed_point_json=" \
  "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
  "++pipeline.covariance_signed_center_offset_enable=false" \
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

render_and_analyze formal_v338_current "$CANDIDATE_CKPT" "$EXP_ROOT/formal_v338_current" \
  "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

render_and_analyze formal_v338_point_screen_guard "$CANDIDATE_CKPT" "$EXP_ROOT/formal_v338_point_screen_guard" \
  "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
  "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'"

"$PYTHON_BIN" - "$SUMMARY" "$EXP_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
variants = ["baseline_no_preset", "formal_v338_current", "formal_v338_point_screen_guard"]
metrics = ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")


def load_metrics(render_exp):
    render_exp = Path(render_exp)
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8"))
    opacity = json.loads((render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8"))
    return {
        "samples": int(contour.get("n_samples", residual.get("n_samples", opacity.get("n_samples", 0)))),
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    }


def status_for(delta):
    strict = all(delta[k] <= 0.0 for k in metrics)
    probe = (
        delta["fg"] <= 0.00002
        and delta["boundary"] <= 0.00002
        and delta["edge"] <= 0.01
        and delta["inner"] <= 2.0
        and delta["outer"] <= 2.0
        and delta["hard"] <= 0.00025
        and delta["opacity_inner"] <= 2.0
        and delta["opacity_outer"] <= 2.0
    )
    return "strict_pass" if strict else ("probe_pass" if probe else "regresses")


loaded = {variant: load_metrics(exp_root / variant) for variant in variants}
base = loaded["baseline_no_preset"]
rows = []
for variant, data in loaded.items():
    delta = {k: data[k] - base[k] for k in metrics}
    rows.append({
        "variant": variant,
        "samples": data["samples"],
        **{k: data[k] for k in metrics},
        **{f"{k}_delta_base": delta[k] for k in metrics},
        "status": "baseline" if variant == "baseline_no_preset" else status_for(delta),
    })

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY

log_event "summary_done" "$SUMMARY"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "POINT_SCREEN_DROP_IMAGES=$POINT_SCREEN_DROP_IMAGES"
