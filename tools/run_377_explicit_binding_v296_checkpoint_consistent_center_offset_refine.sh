#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v296_checkpoint_consistent_center_offset_refine_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
DO_TRAIN="${DO_TRAIN:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-200}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-100,200}"
BASE_ITER="${BASE_ITER:-136410}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v296_checkpoint_consistent_center_offset_refine_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v296_checkpoint_consistent_center_offset_refine_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/no_train_summary.tsv"
TRAIN_SUMMARY="$LOG_DIR/train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

ACTUAL_BASE_ITER="$("$PYTHON_BIN" - "$BASE_CKPT" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt[-1]))
PY
)"
if [ "$ACTUAL_BASE_ITER" != "$BASE_ITER" ]; then
  echo "BASE_ITER mismatch: BASE_ITER=$BASE_ITER but checkpoint iteration=$ACTUAL_BASE_ITER ($BASE_CKPT)" >&2
  exit 2
fi

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tscreen_shrink\tscreen_grow\tcenter_enable\touter_px\tinner_px\touter_dir\tinner_dir\tmax_world_step\tcomponent_required\tcomponent_signature\tcomponent_pad\tcomponent_scale\tmax_over\tmax_under\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"
printf 'label\tvariant\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$TRAIN_SUMMARY"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
DO_TRAIN=$DO_TRAIN
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS

Goal:
  v296 repeats the useful v295 no-train center-offset probe with deterministic
  checkpoint semantics. The previous v295 short train passed load_ckpt but
  train.py used the inherited start_checkpoint, so training silently started
  from v270 instead of the no-train v271 ckpt. This script makes
  start_checkpoint=load_ckpt=BASE_CKPT and keeps BASE_ITER aligned with the
  checkpoint iteration.

Gate:
  Run raw no-train A/B first. Only a strict/probe pass gets a short color/SH/
  texture refine from the same checkpoint with xyz/opacity/scale/rotation
  frozen and render_export_refine=false.
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
  local ckpt="$2"
  local render_exp="$3"
  local screen_shrink="$4"
  local screen_grow="$5"
  local center_enable="$6"
  local outer_px="$7"
  local inner_px="$8"
  local outer_dir="$9"
  local inner_dir="${10}"
  local max_world_step="${11}"
  local component_required="${12}"
  local component_signature="${13}"
  local component_pad="${14}"
  local component_scale="${15}"
  local max_over="${16}"
  local max_under="${17}"
  local hydra_dir="${18}"

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
    "++pipeline.covariance_mode=default" \
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
    "++pipeline.covariance_signed_dynamic_component_top_ids_enable=false" \
    "++pipeline.covariance_signed_dynamic_component_top_ids_only=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=$max_over" \
    "++pipeline.covariance_signed_dynamic_max_under_points=$max_under" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=$screen_shrink" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=$screen_grow" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=$center_enable" \
    "++pipeline.covariance_signed_center_offset_outer_px=$outer_px" \
    "++pipeline.covariance_signed_center_offset_inner_px=$inner_px" \
    "++pipeline.covariance_signed_center_offset_outer_direction=$outer_dir" \
    "++pipeline.covariance_signed_center_offset_inner_direction=$inner_dir" \
    "++pipeline.covariance_signed_center_offset_score_weight_power=1.0" \
    "++pipeline.covariance_signed_center_offset_score_weight_min=0.15" \
    "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90" \
    "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001" \
    "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001" \
    "++pipeline.covariance_signed_center_offset_max_world_step=$max_world_step" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++pipeline.binding_covariance_guard_enable=false" \
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
  local summary_path="$1"
  local baseline_label="$2"
  local row_prefix="$3"
  local render_exp="$4"
  "$PYTHON_BIN" - "$summary_path" "$baseline_label" "$row_prefix" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

summary_path, baseline_label, row_prefix, render_exp = sys.argv[1:5]
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
    if row and row[0] == baseline_label:
        baseline = {key: float(row[header.index(key)]) for key in metrics}
        break
if baseline is None or row_prefix.split("\t", 1)[0] == baseline_label:
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
    and delta["fg"] <= 0.000025
    and delta["boundary"] <= 0.000025
    and delta["edge"] <= 0.004
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)
def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"
suffix = [
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
    handle.write(row_prefix + "\t" + "\t".join(suffix) + "\n")
PY
}

run_variant() {
  local variant="$1"
  local screen_shrink="$2"
  local screen_grow="$3"
  local center_enable="$4"
  local outer_px="$5"
  local inner_px="$6"
  local outer_dir="$7"
  local inner_dir="$8"
  local max_world_step="$9"
  local component_required="${10}"
  local component_signature="${11}"
  local component_pad="${12}"
  local component_scale="${13}"
  local max_over="${14}"
  local max_under="${15}"
  local render_exp="$EXP_ROOT/no_train_${variant}"
  local row_prefix
  printf -v row_prefix '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "$variant" "$screen_shrink" "$screen_grow" "$center_enable" "$outer_px" "$inner_px" \
    "$outer_dir" "$inner_dir" "$max_world_step" "$component_required" "$component_signature" \
    "$component_pad" "$component_scale" "$max_over" "$max_under" "$render_exp"

  log_event "no_train_render_start" "$variant"
  render_raw "$variant" "$BASE_CKPT" "$render_exp" "$screen_shrink" "$screen_grow" "$center_enable" "$outer_px" "$inner_px" "$outer_dir" "$inner_dir" "$max_world_step" "$component_required" "$component_signature" "$component_pad" "$component_scale" "$max_over" "$max_under" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "no_train_analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  append_summary "$SUMMARY" "baseline_v281_screen_mid" "$row_prefix" "$render_exp"
  log_event "no_train_variant_done" "$variant"
}

select_variant() {
  "$PYTHON_BIN" - "$SUMMARY" "$SELECTED_ENV" <<'PY'
import csv
import sys
from pathlib import Path

summary = Path(sys.argv[1])
selected_env = Path(sys.argv[2])
rows = list(csv.DictReader(summary.open(encoding="utf-8"), delimiter="\t"))
candidates = [row for row in rows if row["variant"] != "baseline_v281_screen_mid" and row["strict_pass"] == "1"]
reason = "strict_pass"
if not candidates:
    candidates = [row for row in rows if row["variant"] != "baseline_v281_screen_mid" and row["probe_pass"] == "1"]
    reason = "probe_pass"
if not candidates:
    selected_env.write_text("SELECTED_VARIANT=\nSELECT_REASON=no_gate_pass\n", encoding="utf-8")
    print("no_gate_pass")
    raise SystemExit(0)
candidates.sort(key=lambda row: (
    int(row["strict_pass"]),
    -float(row["hard_delta"]),
    -float(row["outer_delta"]),
    -float(row["inner_delta"]),
    -float(row["fg_delta"]),
), reverse=True)
row = candidates[0]
lines = [
    f"SELECTED_VARIANT={row['variant']}",
    f"SELECT_REASON={reason}",
]
for key, value in row.items():
    env_key = "SELECTED_" + key.upper()
    safe_value = str(value).replace("'", "'\\''")
    lines.append(f"{env_key}='{safe_value}'")
selected_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"selected={row['variant']} reason={reason}")
PY
}

append_train_summary() {
  local label="$1"
  local variant="$2"
  local ckpt="$3"
  local render_exp="$4"
  "$PYTHON_BIN" - "$TRAIN_SUMMARY" "$SUMMARY" "$label" "$variant" "$ckpt" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

train_summary, no_train_summary, label, variant, ckpt, render_exp = sys.argv[1:7]
train_summary = Path(train_summary)
no_train_summary = Path(no_train_summary)
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
lines = [line.rstrip("\n").split("\t") for line in no_train_summary.read_text(encoding="utf-8").splitlines()]
header = lines[0]
baseline = None
for row in lines[1:]:
    if row and row[0] == "baseline_v281_screen_mid":
        baseline = {key: float(row[header.index(key)]) for key in metrics}
        break
if baseline is None:
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
    and delta["fg"] <= 0.000025
    and delta["boundary"] <= 0.000025
    and delta["edge"] <= 0.004
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)
def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"
row = [
    label,
    variant,
    ckpt,
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
with train_summary.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

train_selected_variant() {
  source "$SELECTED_ENV"
  if [ -z "${SELECTED_VARIANT:-}" ]; then
    log_event "train_skip" "${SELECT_REASON:-no_gate_pass}"
    return
  fi
  if [ "$DO_TRAIN" != "1" ]; then
    log_event "train_skip" "DO_TRAIN=$DO_TRAIN selected=$SELECTED_VARIANT"
    return
  fi
  local train_exp="$EXP_ROOT/train_${SELECTED_VARIANT}"
  local checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"
  log_event "train_start" "$SELECTED_VARIANT reason=$SELECT_REASON exp=$train_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "start_checkpoint=$BASE_CKPT" \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$train_exp" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_${SELECTED_VARIANT}" \
    "mode=train" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=$SELECTED_COMPONENT_SIGNATURE" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=$SELECTED_COMPONENT_PAD" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=$SELECTED_COMPONENT_SCALE" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=$SELECTED_COMPONENT_REQUIRED" \
    "++pipeline.covariance_signed_dynamic_max_over_points=$SELECTED_MAX_OVER" \
    "++pipeline.covariance_signed_dynamic_max_under_points=$SELECTED_MAX_UNDER" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=$SELECTED_SCREEN_SHRINK" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=$SELECTED_SCREEN_GROW" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=$SELECTED_CENTER_ENABLE" \
    "++pipeline.covariance_signed_center_offset_outer_px=$SELECTED_OUTER_PX" \
    "++pipeline.covariance_signed_center_offset_inner_px=$SELECTED_INNER_PX" \
    "++pipeline.covariance_signed_center_offset_outer_direction=$SELECTED_OUTER_DIR" \
    "++pipeline.covariance_signed_center_offset_inner_direction=$SELECTED_INNER_DIR" \
    "++pipeline.covariance_signed_center_offset_score_weight_power=1.0" \
    "++pipeline.covariance_signed_center_offset_score_weight_min=0.15" \
    "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90" \
    "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001" \
    "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001" \
    "++pipeline.covariance_signed_center_offset_max_world_step=$SELECTED_MAX_WORLD_STEP" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++pipeline.binding_covariance_guard_enable=false" \
    "++resume.allow_start_load_ckpt_mismatch=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=false" \
    "model.pose_correction.delay=1" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "opt.iterations=$TRAIN_ITERS" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=0.00014" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.texture_lr=0.00000035" \
    "opt.tex_latent_lr=0.0" \
    "++opt.texture_trainable_name_patterns=[*]" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.stageB_semantic_loss_enable=false" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0" \
    "++opt.semantic_region_logits_lr=0.0" \
    "++opt.semantic_compact_logits_lr=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.018" \
    "++opt.train_sample_camera_max_prob=0.125" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=1" \
    "opt.lambda_l1=0.060" \
    "opt.lambda_l1_fg=0.140" \
    "opt.lambda_l1_boundary=0.080" \
    "opt.lambda_perceptual=0.020" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_silhouette_outer=0.0" \
    "++opt.lambda_silhouette_inner=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "opt.percent_dense=0.0" \
    "opt.densify_until_iter=0" \
    "opt.densify_from_iter=1000000" \
    "opt.opacity_reset_interval=1000000" \
    "best_eval_split=test" \
    "best_metric=l1_fg" \
    "best_metric_mode=min" \
    "best_metric_source=best_eval" \
    "test_interval=0" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0020" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "wandb_disable=true" \
    > "$LOG_DIR/train_${SELECTED_VARIANT}.log" 2>&1
  log_event "train_done" "$train_exp"

  IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    local global_iter=$((BASE_ITER + step))
    local ckpt="$train_exp/ckpt${global_iter}.pth"
    local label="ckpt${global_iter}"
    local render_exp="${train_exp}_raw_render_${label}"
    if [ ! -f "$ckpt" ]; then
      log_event "train_render_skip" "missing=$ckpt"
      continue
    fi
    log_event "train_render_start" "$label"
    render_raw "${SELECTED_VARIANT}_${label}" "$ckpt" "$render_exp" "$SELECTED_SCREEN_SHRINK" "$SELECTED_SCREEN_GROW" "$SELECTED_CENTER_ENABLE" "$SELECTED_OUTER_PX" "$SELECTED_INNER_PX" "$SELECTED_OUTER_DIR" "$SELECTED_INNER_DIR" "$SELECTED_MAX_WORLD_STEP" "$SELECTED_COMPONENT_REQUIRED" "$SELECTED_COMPONENT_SIGNATURE" "$SELECTED_COMPONENT_PAD" "$SELECTED_COMPONENT_SCALE" "$SELECTED_MAX_OVER" "$SELECTED_MAX_UNDER" "$HYDRA_RUN_ROOT/render_${SELECTED_VARIANT}_${label}" \
      > "$LOG_DIR/render_${SELECTED_VARIANT}_${label}.log" 2>&1
    analyze_raw "${SELECTED_VARIANT}_${label}" "$render_exp"
    append_train_summary "$label" "$SELECTED_VARIANT" "$ckpt" "$render_exp"
    log_event "train_render_done" "$label"
  done
}

run_variant baseline_v281_screen_mid 0.940 1.025 false 0.00 0.00 view_center component_center 0.000 false false 10 1.25 96 96
run_variant outer_center_025 0.940 1.025 true 0.25 0.00 view_center component_center 0.0015 false false 10 1.25 96 96
run_variant outer_center_050 0.940 1.025 true 0.50 0.00 view_center component_center 0.0025 false false 10 1.25 96 96
run_variant inner_gap_015 0.940 1.025 true 0.00 0.15 view_center component_center 0.0010 false false 10 1.25 96 96
run_variant inner_gap_025 0.940 1.025 true 0.00 0.25 view_center component_center 0.0015 false false 10 1.25 96 96
run_variant signed_offset_soft 0.940 1.025 true 0.25 0.12 view_center component_center 0.0015 false false 10 1.25 96 96
run_variant signed_offset_outer_only_component 0.940 1.025 true 0.35 0.00 component_center component_center 0.0020 true false 14 1.35 96 96
run_variant signed_offset_signature 0.940 1.025 true 0.25 0.10 view_center component_center 0.0015 true true 18 1.55 96 96

select_variant
train_selected_variant

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
  echo "SELECTED_ENV=$SELECTED_ENV"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
echo "SELECTED_ENV=$SELECTED_ENV"
echo "END_BJT=$END_BJT"
