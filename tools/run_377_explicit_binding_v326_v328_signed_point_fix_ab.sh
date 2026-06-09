#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v326_v328_signed_point_fix_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"
EXACT_COMPONENT_CSV="${EXACT_COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v320_paired_signed_selector_v320_paired_signed_selector_20260519_212529_bjt/selected_components.csv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
SOURCE_VIEWS_CSV="${SOURCE_VIEWS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TRAIN_VIEWS_CSV="${TRAIN_VIEWS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TARGET_VIEWS_CSV="${TARGET_VIEWS_CSV:-21,22,23}"
TRAIN_FRAMES_CSV="${TRAIN_FRAMES_CSV:-0,570,60}"
TARGET_FRAMES_CSV="${TARGET_FRAMES_CSV:-0,570,60}"

NEAREST_VIEWS="${NEAREST_VIEWS:-4}"
VIEW_POWER="${VIEW_POWER:-3.0}"
MIN_ABS_SCORE="${MIN_ABS_SCORE:-0.0}"
MAX_SHRINK="${MAX_SHRINK:-96}"
MAX_GROW="${MAX_GROW:-96}"

INNER_GROW_MIN_BOUNDARY="${INNER_GROW_MIN_BOUNDARY:-0.18}"
INNER_GROW_MIN_THIN="${INNER_GROW_MIN_THIN:-0.05}"
INNER_GROW_MIN_VISIBLE_HITS="${INNER_GROW_MIN_VISIBLE_HITS:-40}"
INNER_GROW_SURFACE_MAX="${INNER_GROW_SURFACE_MAX:-0.080}"
INNER_GROW_ALLOWED_REGIONS="${INNER_GROW_ALLOWED_REGIONS:-1,2}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v326_v328_signed_point_fix_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v326_v328_signed_point_fix_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
PRIOR_DIR="$LOG_DIR/prior"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$PRIOR_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$EXACT_COMPONENT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXACT_COMPONENT_CSV=$EXACT_COMPONENT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v326: outer-only view-conditioned signed point prior.
  v327: inner-only conservative center-fidelity gate sweep.
  v328: combined outer shrink plus conservative inner gate.
  Acceptance is raw contour gate only. Color/SH training remains out of the
  adoption path unless a no-train variant first passes raw gate.
EOF

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

make_prior() {
  local label="$1"
  local direction_mode="$2"
  local max_shrink="$3"
  local max_grow="$4"
  shift 4
  local out_json="$PRIOR_DIR/${label}.json"
  log_event "prior_start" "$label"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v325_view_signed_point_prior.py \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --dataset-root "$DATA_ROOT" \
    --component-csv "$COMPONENT_CSV" \
    --point-csv "$POINT_CSV" \
    --out-dir "$PRIOR_DIR/${label}_scene" \
    --out-json "$out_json" \
    --source-views "$SOURCE_VIEWS_CSV" \
    --train-views "$TRAIN_VIEWS_CSV" \
    --target-views "$TARGET_VIEWS_CSV" \
    --train-frames "$TRAIN_FRAMES_CSV" \
    --target-frames "$TARGET_FRAMES_CSV" \
    --nearest-views "$NEAREST_VIEWS" \
    --direction-mode "$direction_mode" \
    --view-power "$VIEW_POWER" \
    --max-shrink "$max_shrink" \
    --max-grow "$max_grow" \
    --min-abs-score "$MIN_ABS_SCORE" \
    "$@" \
    > "$LOG_DIR/prior_${label}.log" 2>&1
  log_event "prior_done" "$out_json"
}

render_variant() {
  local variant="$1"
  local render_exp="$2"
  shift 2
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "$@" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
}

analyze_variant() {
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

run_variant() {
  local variant="$1"
  local render_exp="$2"
  shift 2
  log_event "render_start" "$variant"
  render_variant "$variant" "$render_exp" "$@"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
}

OUTER_PRIOR="$PRIOR_DIR/v326_outer_only.json"
INNER_PRIOR="$PRIOR_DIR/v327_inner_conservative.json"
COMBINED_PRIOR="$PRIOR_DIR/v328_combined.json"

make_prior v326_outer_only outer "$MAX_SHRINK" 0
make_prior v327_inner_conservative inner 0 "$MAX_GROW" \
  --grow-min-boundary "$INNER_GROW_MIN_BOUNDARY" \
  --grow-min-thin "$INNER_GROW_MIN_THIN" \
  --grow-min-visible-hits "$INNER_GROW_MIN_VISIBLE_HITS" \
  --grow-surface-max "$INNER_GROW_SURFACE_MAX" \
  --grow-allowed-regions "$INNER_GROW_ALLOWED_REGIONS"
make_prior v328_combined both "$MAX_SHRINK" "$MAX_GROW" \
  --grow-min-boundary "$INNER_GROW_MIN_BOUNDARY" \
  --grow-min-thin "$INNER_GROW_MIN_THIN" \
  --grow-min-visible-hits "$INNER_GROW_MIN_VISIBLE_HITS" \
  --grow-surface-max "$INNER_GROW_SURFACE_MAX" \
  --grow-allowed-regions "$INNER_GROW_ALLOWED_REGIONS"

BASELINE_EXP="$EXP_ROOT/baseline_no_preset"
EXACT_EXP="$EXP_ROOT/v320_selected_exact_component"
V326_EXP="$EXP_ROOT/v326_outer_only"
V327_24_EXP="$EXP_ROOT/v327_inner_g24_s025"
V327_48_EXP="$EXP_ROOT/v327_inner_g48_s035"
V327_96_EXP="$EXP_ROOT/v327_inner_g96_s045"
V328_24_EXP="$EXP_ROOT/v328_combined_g24_s025"
V328_48_EXP="$EXP_ROOT/v328_combined_g48_s035"
V328_96_EXP="$EXP_ROOT/v328_combined_g96_s045"

COMMON_NO_PRESET_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_point_screen_actuator_enable=false"
  "++pipeline.covariance_signed_center_offset_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
)

OUTER_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_point_json=$OUTER_PRIOR"
  "++pipeline.covariance_signed_point_screen_actuator_enable=true"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_screen_actuator_enable=true"
  "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940"
  "++pipeline.covariance_signed_screen_normal_grow_factor=1.000"
  "++pipeline.covariance_signed_screen_tangent_factor=1.000"
  "++pipeline.covariance_signed_center_offset_enable=true"
  "++pipeline.covariance_signed_center_offset_outer_px=0.35"
  "++pipeline.covariance_signed_center_offset_inner_px=0.0"
  "++pipeline.covariance_signed_center_offset_outer_direction=view_center"
  "++pipeline.covariance_signed_center_offset_inner_direction=component_center"
  "++pipeline.covariance_signed_center_offset_max_world_step=0.0020"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
)

inner_args() {
  local prior="$1"
  local max_points="$2"
  local strength="$3"
  printf '%s\n' \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=false" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
    "++pipeline.covariance_signed_center_offset_enable=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=true" \
    "++model.deformer.rigid.geometry_fidelity_target=free_lbs" \
    "++model.deformer.rigid.geometry_fidelity_center_strength=$strength" \
    "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0" \
    "++model.deformer.rigid.geometry_fidelity_boundary_min=0.0" \
    "++model.deformer.rigid.geometry_fidelity_layer_ids='soft,rigid,free'" \
    "++model.deformer.rigid.geometry_fidelity_region_ids='cloth,body,soft'" \
    "++model.deformer.rigid.geometry_fidelity_joint_ids=" \
    "++model.deformer.rigid.geometry_fidelity_power=1.2" \
    "++model.deformer.rigid.geometry_fidelity_max_points=1024" \
    "++model.deformer.rigid.geometry_fidelity_component_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_signed_point_json=$prior" \
    "++model.deformer.rigid.geometry_fidelity_signed_point_max=$max_points"
}

combined_args() {
  local max_points="$1"
  local strength="$2"
  local -a args
  mapfile -t args < <(inner_args "$COMBINED_PRIOR" "$max_points" "$strength")
  printf '%s\n' \
    "${args[@]}" \
    "++pipeline.covariance_signed_point_json=$COMBINED_PRIOR" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.000" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=true" \
    "++pipeline.covariance_signed_center_offset_outer_px=0.35" \
    "++pipeline.covariance_signed_center_offset_inner_px=0.0" \
    "++pipeline.covariance_signed_center_offset_outer_direction=view_center" \
    "++pipeline.covariance_signed_center_offset_inner_direction=component_center" \
    "++pipeline.covariance_signed_center_offset_max_world_step=0.0020"
}

run_variant baseline_no_preset "$BASELINE_EXP" "${COMMON_NO_PRESET_ARGS[@]}"

run_variant v320_selected_exact_component "$EXACT_EXP" \
  "++explicit_binding_render_preset=v307_adopted_geometry" \
  "++explicit_binding_adopted_component_csv=$EXACT_COMPONENT_CSV" \
  "++explicit_binding_adopted_point_csv=$POINT_CSV" \
  "++explicit_binding_adopted_center_strength=0.45" \
  "++explicit_binding_adopted_outer_px=0.35" \
  "++explicit_binding_adopted_component_required=true" \
  "++explicit_binding_adopted_improvement_guard=true" \
  "++explicit_binding_adopted_max_points=96"

run_variant v326_outer_only "$V326_EXP" "${OUTER_ARGS[@]}"

mapfile -t V327_24_ARGS < <(inner_args "$INNER_PRIOR" 24 0.25)
mapfile -t V327_48_ARGS < <(inner_args "$INNER_PRIOR" 48 0.35)
mapfile -t V327_96_ARGS < <(inner_args "$INNER_PRIOR" 96 0.45)
run_variant v327_inner_g24_s025 "$V327_24_EXP" "${V327_24_ARGS[@]}"
run_variant v327_inner_g48_s035 "$V327_48_EXP" "${V327_48_ARGS[@]}"
run_variant v327_inner_g96_s045 "$V327_96_EXP" "${V327_96_ARGS[@]}"

mapfile -t V328_24_ARGS < <(combined_args 24 0.25)
mapfile -t V328_48_ARGS < <(combined_args 48 0.35)
mapfile -t V328_96_ARGS < <(combined_args 96 0.45)
run_variant v328_combined_g24_s025 "$V328_24_EXP" "${V328_24_ARGS[@]}"
run_variant v328_combined_g48_s035 "$V328_48_EXP" "${V328_48_ARGS[@]}"
run_variant v328_combined_g96_s045 "$V328_96_EXP" "${V328_96_ARGS[@]}"

"$PYTHON_BIN" - "$SUMMARY" \
  baseline_no_preset "$BASELINE_EXP" \
  v320_selected_exact_component "$EXACT_EXP" \
  v326_outer_only "$V326_EXP" \
  v327_inner_g24_s025 "$V327_24_EXP" \
  v327_inner_g48_s035 "$V327_48_EXP" \
  v327_inner_g96_s045 "$V327_96_EXP" \
  v328_combined_g24_s025 "$V328_24_EXP" \
  v328_combined_g48_s035 "$V328_48_EXP" \
  v328_combined_g96_s045 "$V328_96_EXP" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))

def load_metrics(render_exp):
    render_exp = Path(render_exp)
    contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
    }

def delta(metrics, base):
    return {key: float(metrics[key]) - float(base[key]) for key in metrics}

def status_for(d):
    strict = (
        d["fg"] <= 0.0
        and d["boundary"] <= 0.0
        and d["edge"] <= 0.0
        and d["inner"] <= 0.0
        and d["outer"] <= 0.0
        and d["hard"] <= -0.000001
    )
    probe = (
        d["hard"] < -0.00001
        and d["fg"] <= 0.000015
        and d["boundary"] <= 0.000015
        and d["edge"] <= 0.003
        and d["inner"] <= 0.5
        and d["outer"] <= 0.5
    )
    return int(strict), int(probe), "strict_pass" if strict else ("probe_pass" if probe else "rejected")

metrics = {name: load_metrics(path) for name, path in pairs}
base = metrics["baseline_no_preset"]
exact = metrics["v320_selected_exact_component"]
header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base", "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "fg_delta_exact", "boundary_delta_exact", "edge_delta_exact", "inner_delta_exact", "outer_delta_exact", "hard_delta_exact",
    "strict_pass", "probe_pass", "status",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in pairs:
        m = metrics[name]
        db = delta(m, base)
        de = delta(m, exact)
        strict, probe, status = status_for(db)
        writer.writerow([
            name, str(render_exp),
            f'{m["fg"]:.8f}', f'{m["boundary"]:.8f}', f'{m["edge"]:.6f}', f'{m["inner"]:.4f}', f'{m["outer"]:.4f}', f'{m["hard"]:.8f}',
            f'{db["fg"]:.8f}', f'{db["boundary"]:.8f}', f'{db["edge"]:.6f}', f'{db["inner"]:.4f}', f'{db["outer"]:.4f}', f'{db["hard"]:.8f}',
            f'{de["fg"]:.8f}', f'{de["boundary"]:.8f}', f'{de["edge"]:.6f}', f'{de["inner"]:.4f}', f'{de["outer"]:.4f}', f'{de["hard"]:.8f}',
            strict, probe, status,
        ])
print(summary_path)
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event "done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
