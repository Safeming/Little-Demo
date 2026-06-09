#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-v346_component_local_asset_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v346_component_local_asset_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v346_component_local_asset_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
WINDOW_SPECS="${WINDOW_SPECS:-c23_f298_301|[23]|[298,301,1];c22_f000_002|[22]|[0,2,1];c21_f162_165|[21]|[162,165,1];c23_f060_062|[23]|[60,62,1];c21_f480_482|[21]|[480,482,1];c22_f240_242|[22]|[240,242,1]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"
V345_DENSE_EXP="${V345_DENSE_EXP:-$ROOT/exp/formal/377_v338_raw_contour_gate_formal_v345_temporal_screen_guard_dense_gate_20260523}"
V346_SOURCE_CURRENT_VARIANT="${V346_SOURCE_CURRENT_VARIANT:-candidate_v345_temporal_screen_guard}"
V345_SCREEN_GUARD_JSON="${V345_SCREEN_GUARD_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v345_temporal_screen_guard_20260523/assets/v345_combined_screen_guard.json}"
V345_SCREEN_DROP_LIST="${V345_SCREEN_DROP_LIST:-$ROOT/exp/stageB/logs/377_explicit_binding_v345_temporal_screen_guard_20260523/assets/v345_combined_screen_guard_drop_images.txt}"
V344_ROW_GUARD_JSON="${V344_ROW_GUARD_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v344_component_row_guard_probe_v344_component_row_guard_geometry_20260523/assets/v344_component_row_guard_top4.json}"

MIN_POSITIVE="${MIN_POSITIVE:-1.0}"
MIN_HARD_POSITIVE="${MIN_HARD_POSITIVE:-0.00005}"
MIN_EDGE_POSITIVE="${MIN_EDGE_POSITIVE:-0.004}"
TOP_FRAMES="${TOP_FRAMES:-80}"
COMPONENTS_PER_FRAME="${COMPONENTS_PER_FRAME:-2}"
MAX_ACTIONS="${MAX_ACTIONS:-160}"
MIN_OWNER_CONSISTENCY="${MIN_OWNER_CONSISTENCY:-0.50}"

SUMMARY="$LOG_DIR/summary.tsv"
WINDOW_SUMMARY="$LOG_DIR/window_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
ASSET_DIR="$LOG_DIR/assets"
V346_ASSET_JSON="$ASSET_DIR/v346_component_local_asset.json"
V346_ROW_GUARD_JSON="$ASSET_DIR/v346_component_local_row_guard_probe.json"
V346_CANDIDATES_TSV="$ASSET_DIR/v346_component_local_candidates.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$COMPONENT_CSV" "$POINT_CSV" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json" \
  "$V345_DENSE_EXP/baseline_no_preset/diagnostics/contours/contour_samples.csv" \
  "$V345_DENSE_EXP/$V346_SOURCE_CURRENT_VARIANT/diagnostics/contours/contour_samples.csv" \
  "$V345_SCREEN_GUARD_JSON" "$V345_SCREEN_DROP_LIST" "$V344_ROW_GUARD_JSON"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

log_event "make_asset_start" "$V346_ASSET_JSON"
"$PYTHON_BIN" tools/make_377_stageB_v346_component_local_asset.py \
  --baseline-render-exp "$V345_DENSE_EXP/baseline_no_preset" \
  --current-render-exp "$V345_DENSE_EXP/$V346_SOURCE_CURRENT_VARIANT" \
  --component-csv "$COMPONENT_CSV" \
  --point-csv "$POINT_CSV" \
  --exclude-drop-json "$V345_SCREEN_GUARD_JSON" \
  --min-positive "$MIN_POSITIVE" \
  --min-hard-positive "$MIN_HARD_POSITIVE" \
  --min-edge-positive "$MIN_EDGE_POSITIVE" \
  --top-frames "$TOP_FRAMES" \
  --components-per-frame "$COMPONENTS_PER_FRAME" \
  --max-actions "$MAX_ACTIONS" \
  --min-owner-consistency "$MIN_OWNER_CONSISTENCY" \
  --out-json "$V346_ASSET_JSON" \
  --out-row-guard-json "$V346_ROW_GUARD_JSON" \
  --out-candidates-tsv "$V346_CANDIDATES_TSV" \
  > "$LOG_DIR/make_v346_asset.log" 2>&1
log_event "make_asset_done" "$V346_ASSET_JSON"

POINT_SCREEN_DROP_IMAGES="$(tr '\n' ',' < "$V345_SCREEN_DROP_LIST" | sed 's/,$//')"

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
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local variant="$4"
  local ckpt="$5"
  local render_exp="$6"
  shift 6

  log_event "render_${window}_${variant}_start" "$render_exp"
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
    "dataset.test_views.view=$views_spec" \
    "dataset.test_frames.view=$frames_spec" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/${window}_${variant}" \
    "wandb_disable=true" \
    "$@" \
    > "$LOG_DIR/render_${window}_${variant}.log" 2>&1
  log_event "render_${window}_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${window}_${variant}.log" 2>&1

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
    > "$LOG_DIR/boundary_residuals_${window}_${variant}.log" 2>&1

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
    > "$LOG_DIR/opacity_footprint_${window}_${variant}.log" 2>&1
  log_event "analyze_${window}_${variant}_done" "status=0"
}

run_window() {
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local window_root="$EXP_ROOT/$window"

  render_and_analyze "$window" "$views_spec" "$frames_spec" baseline_no_preset "$BASE_CKPT" "$window_root/baseline_no_preset" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=false" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
    "++pipeline.covariance_signed_center_offset_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

  render_and_analyze "$window" "$views_spec" "$frames_spec" formal_v338_current "$CANDIDATE_CKPT" "$window_root/formal_v338_current" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v345_screen_guard "$CANDIDATE_CKPT" "$window_root/v345_screen_guard" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v344_plus_v345 "$CANDIDATE_CKPT" "$window_root/v344_plus_v345" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
    "++explicit_binding_adopted_signed_dynamic_component_row_guard_json=$V344_ROW_GUARD_JSON"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v346_component_local "$CANDIDATE_CKPT" "$window_root/v346_component_local" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_dynamic_component_local_asset_json=$V346_ASSET_JSON"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v346_component_local_plus_v345 "$CANDIDATE_CKPT" "$window_root/v346_component_local_plus_v345" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
    "++explicit_binding_adopted_signed_dynamic_component_local_asset_json=$V346_ASSET_JSON"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v346_row_guard_upper_bound_plus_v345 "$CANDIDATE_CKPT" "$window_root/v346_row_guard_upper_bound_plus_v345" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
    "++explicit_binding_adopted_signed_dynamic_component_row_guard_json=$V346_ROW_GUARD_JSON"
}

IFS=';' read -r -a WINDOWS <<< "$WINDOW_SPECS"
for item in "${WINDOWS[@]}"; do
  IFS='|' read -r window views frames <<< "$item"
  run_window "$window" "$views" "$frames"
done

"$PYTHON_BIN" - "$SUMMARY" "$WINDOW_SUMMARY" "$EXP_ROOT" "${WINDOWS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
window_summary_path = Path(sys.argv[2])
exp_root = Path(sys.argv[3])
windows = [item.split("|")[0] for item in sys.argv[4:]]
variants = [
    "baseline_no_preset",
    "formal_v338_current",
    "v345_screen_guard",
    "v344_plus_v345",
    "v346_component_local",
    "v346_component_local_plus_v345",
    "v346_row_guard_upper_bound_plus_v345",
]
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


window_rows = []
totals = {variant: {"samples": 0, **{k: 0.0 for k in metrics}} for variant in variants}
for window in windows:
    loaded = {variant: load_metrics(exp_root / window / variant) for variant in variants}
    base = loaded["baseline_no_preset"]
    for variant, data in loaded.items():
        delta = {k: data[k] - base[k] for k in metrics}
        row = {
            "window": window,
            "variant": variant,
            "samples": data["samples"],
            **{k: data[k] for k in metrics},
            **{f"{k}_delta_base": delta[k] for k in metrics},
            "status": "baseline" if variant == "baseline_no_preset" else status_for(delta),
        }
        window_rows.append(row)
        totals[variant]["samples"] += data["samples"]
        for key in metrics:
            totals[variant][key] += data[key] * data["samples"]

base_total = totals["baseline_no_preset"]
base_samples = max(1, base_total["samples"])
base_mean = {k: base_total[k] / base_samples for k in metrics}
rows = []
for variant in variants:
    samples = max(1, totals[variant]["samples"])
    data = {"samples": totals[variant]["samples"], **{k: totals[variant][k] / samples for k in metrics}}
    delta = {k: data[k] - base_mean[k] for k in metrics}
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
with window_summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(window_rows)
PY

log_event "summary_done" "$SUMMARY"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "WINDOW_SUMMARY=$WINDOW_SUMMARY"
echo "V346_ASSET_JSON=$V346_ASSET_JSON"
echo "V346_ROW_GUARD_JSON=$V346_ROW_GUARD_JSON"
echo "V346_CANDIDATES_TSV=$V346_CANDIDATES_TSV"
echo
echo "Dense gate command after local probe passes:"
echo "CANDIDATE_CKPT=$CANDIDATE_CKPT \\"
echo "CANDIDATE_VARIANT_NAME=candidate_v346_component_local_plus_v345 \\"
echo "CANDIDATE_SIGNED_POINT_SCREEN_ACTUATOR_DROP_IMAGES='$POINT_SCREEN_DROP_IMAGES' \\"
echo "CANDIDATE_SIGNED_DYNAMIC_COMPONENT_LOCAL_ASSET_JSON=$V346_ASSET_JSON \\"
echo "TEST_VIEWS_SPEC='[21,22,23]' TEST_FRAMES_SPEC='[0,570,1]' \\"
echo "RUN_ID='formal_v346_component_local_dense_gate_$(TZ=Asia/Shanghai date '+%Y%m%d')' \\"
echo "tools/formal/run_377_v338_raw_contour_gate.sh"
