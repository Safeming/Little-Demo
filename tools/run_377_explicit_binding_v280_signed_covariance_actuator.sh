#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v280_signed_covariance_actuator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
DO_TRAIN="${DO_TRAIN:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-200}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-100,200}"
BASE_ITER="${BASE_ITER:-136410}"
SKIP_NO_TRAIN="${SKIP_NO_TRAIN:-0}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_JSON="${CANDIDATE_JSON:-$ROOT/exp/stageB/logs/377_stageB_v274_contributor_audit_v274_contributor_audit_final_20260517_191438_bjt/v275_candidate_point_sets.json}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v280_signed_covariance_actuator_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v280_signed_covariance_actuator_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/no_train_summary.tsv"
TRAIN_SUMMARY="$LOG_DIR/train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_JSON" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"

if [ "$SKIP_NO_TRAIN" = "1" ] && [ -f "$LOG_DIR/run_info.txt" ]; then
  {
    echo "RESUME_START_BJT=$START_BJT"
    echo "RESUME_SKIP_NO_TRAIN=$SKIP_NO_TRAIN"
  } >> "$LOG_DIR/run_info.txt"
else
  cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
CANDIDATE_JSON=$CANDIDATE_JSON
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
DO_TRAIN=$DO_TRAIN
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS

Goal:
  v280 signed covariance footprint actuator.
  Do not edit checkpoint xyz/opacity/scale/rotation.
  Use v274 contributor point sets only as render-time covariance masks.
  First run no-train raw gate, then short color/texture-only train for the best probe/strict variant.
EOF
fi

if [ "$SKIP_NO_TRAIN" != "1" ] || [ ! -f "$EVENTS" ]; then
  printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
fi
if [ "$SKIP_NO_TRAIN" != "1" ] || [ ! -f "$SUMMARY" ]; then
  printf 'variant\tcov_mode\taxis\tshrink_factor\tgrow_factor\tmax_shrink\tmax_grow\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"
fi
if [ "$SKIP_NO_TRAIN" != "1" ] || [ ! -f "$TRAIN_SUMMARY" ]; then
  printf 'label\tvariant\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tstatus\n' > "$TRAIN_SUMMARY"
fi

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
  local cov_mode="$4"
  local axis="$5"
  local shrink_factor="$6"
  local grow_factor="$7"
  local max_shrink="$8"
  local max_grow="$9"
  local hydra_dir="${10}"

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
    "++pipeline.covariance_mode=$cov_mode" \
    "++pipeline.covariance_signed_point_json=$CANDIDATE_JSON" \
    "++pipeline.covariance_signed_shrink_factor=$shrink_factor" \
    "++pipeline.covariance_signed_grow_factor=$grow_factor" \
    "++pipeline.covariance_signed_max_shrink_points=$max_shrink" \
    "++pipeline.covariance_signed_max_grow_points=$max_grow" \
    "++pipeline.covariance_signed_anisotropic_axis=$axis" \
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

append_no_train_summary() {
  local variant="$1"
  local cov_mode="$2"
  local axis="$3"
  local shrink_factor="$4"
  local grow_factor="$5"
  local max_shrink="$6"
  local max_grow="$7"
  local render_exp="$8"

  "$PYTHON_BIN" - "$SUMMARY" "$variant" "$cov_mode" "$axis" "$shrink_factor" "$grow_factor" "$max_shrink" "$max_grow" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

summary_path, variant, cov_mode, axis, shrink_factor, grow_factor, max_shrink, max_grow, render_exp = sys.argv[1:10]
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
baseline = None
lines = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = lines[0]
for row in lines[1:]:
    if row and row[0] == "baseline":
        baseline = {key: float(row[header.index(key)]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
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
    and delta["fg"] <= 0.0001
    and delta["boundary"] <= 0.0001
    and delta["edge"] <= 0.005
    and delta["inner"] <= 1.0
    and delta["outer"] <= 1.0
)

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    variant,
    cov_mode,
    axis,
    shrink_factor,
    grow_factor,
    max_shrink,
    max_grow,
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
    "ok",
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

render_variant() {
  local variant="$1"
  local cov_mode="$2"
  local axis="$3"
  local shrink_factor="$4"
  local grow_factor="$5"
  local max_shrink="$6"
  local max_grow="$7"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  log_event "no_train_render_start" "$variant axis=$axis shrink=$shrink_factor grow=$grow_factor max=($max_shrink,$max_grow)"
  render_raw "$variant" "$BASE_CKPT" "$render_exp" "$cov_mode" "$axis" "$shrink_factor" "$grow_factor" "$max_shrink" "$max_grow" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "no_train_analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  append_no_train_summary "$variant" "$cov_mode" "$axis" "$shrink_factor" "$grow_factor" "$max_shrink" "$max_grow" "$render_exp"
  log_event "no_train_variant_done" "$variant"
}

select_variant() {
  "$PYTHON_BIN" - "$SUMMARY" "$SELECTED_ENV" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
selected_env = Path(sys.argv[2])
rows = list(csv.DictReader(summary_path.open(encoding="utf-8"), delimiter="\t"))
candidates = [row for row in rows if row["variant"] != "baseline" and row["strict_pass"] == "1"]
source = "strict"
if not candidates:
    candidates = [row for row in rows if row["variant"] != "baseline" and row["probe_pass"] == "1"]
    source = "probe"
if not candidates:
    selected_env.write_text("TRAIN_SELECTED=0\nSELECT_REASON=no_gate_pass\n", encoding="utf-8")
    sys.exit(0)

def score(row):
    return (
        float(row["hard_delta"]),
        float(row["edge_delta"]),
        float(row["outer_delta"]),
        float(row["inner_delta"]),
    )

best = sorted(candidates, key=score)[0]
lines = {
    "TRAIN_SELECTED": "1",
    "SELECT_REASON": source,
    "SELECTED_VARIANT": best["variant"],
    "SELECTED_COV_MODE": best["cov_mode"],
    "SELECTED_AXIS": best["axis"],
    "SELECTED_SHRINK_FACTOR": best["shrink_factor"],
    "SELECTED_GROW_FACTOR": best["grow_factor"],
    "SELECTED_MAX_SHRINK": best["max_shrink"],
    "SELECTED_MAX_GROW": best["max_grow"],
}
selected_env.write_text(
    "".join(f"{key}={shlex.quote(value)}\n" for key, value in lines.items()),
    encoding="utf-8",
)
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
rows = [line.rstrip("\n").split("\t") for line in no_train_summary.read_text(encoding="utf-8").splitlines()]
header = rows[0]
baseline = None
for row in rows[1:]:
    if row and row[0] == "baseline":
        baseline = {key: float(row[header.index(key)]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
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
    "ok",
]
with train_summary.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

train_selected_variant() {
  # shellcheck disable=SC1090
  source "$SELECTED_ENV"
  if [ "${TRAIN_SELECTED:-0}" != "1" ]; then
    log_event "train_skip" "${SELECT_REASON:-no_gate_pass}"
    return 0
  fi
  if [ "$DO_TRAIN" != "1" ]; then
    log_event "train_skip" "DO_TRAIN=$DO_TRAIN selected=$SELECTED_VARIANT"
    return 0
  fi

  local train_exp="$EXP_ROOT/train_${SELECTED_VARIANT}"
  local checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"
  log_event "train_start" "$SELECTED_VARIANT reason=$SELECT_REASON exp=$train_exp"

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,60]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "dataset.parsing_prior.compact_mapping_file=" \
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$train_exp" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_${SELECTED_VARIANT}" \
    "seed=-1" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=true" \
    "++resume.clear_binding_state_on_resume=false" \
    "pipeline.pose_noise=0.0" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=$SELECTED_COV_MODE" \
    "++pipeline.covariance_signed_point_json=$CANDIDATE_JSON" \
    "++pipeline.covariance_signed_shrink_factor=$SELECTED_SHRINK_FACTOR" \
    "++pipeline.covariance_signed_grow_factor=$SELECTED_GROW_FACTOR" \
    "++pipeline.covariance_signed_max_shrink_points=$SELECTED_MAX_SHRINK" \
    "++pipeline.covariance_signed_max_grow_points=$SELECTED_MAX_GROW" \
    "++pipeline.covariance_signed_anisotropic_axis=$SELECTED_AXIS" \
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
    "++opt.stageB_semantic_body_cloth_weight=0.0" \
    "++opt.stageB_semantic_compact_weight=0.0" \
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
    "opt.lambda_l1_face=0.020" \
    "opt.lambda_l1_shoulder_arm=0.016" \
    "opt.lambda_l1_waist=0.012" \
    "opt.lambda_edge_face=0.003" \
    "opt.lambda_edge_shoulder_arm=0.003" \
    "opt.lambda_edge_waist=0.0015" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.002" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.002" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.002" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.002" \
    "++opt.lambda_detail_waist_luma_dog=0.001" \
    "++opt.lambda_perceptual_face=0.006" \
    "++opt.lambda_perceptual_shoulder_arm=0.004" \
    "++opt.lambda_perceptual_waist=0.002" \
    "++opt.lambda_perceptual_face_patch=0.003" \
    "++opt.lambda_perceptual_shoulder_arm_patch=0.003" \
    "++opt.lambda_perceptual_upper_torso_patch=0.003" \
    "++opt.lambda_perceptual_upper_torso_core_patch=0.002" \
    "++opt.lambda_perceptual_waist_patch=0.0015" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_mask_boundary_hard=0.0" \
    "++opt.lambda_silhouette_outer=0.0" \
    "++opt.lambda_silhouette_inner=0.0" \
    "++opt.lambda_silhouette_shoulder_arm_outer_shell=0.0" \
    "++opt.lambda_silhouette_upper_torso_outer_shell=0.0" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0" \
    "++opt.lambda_boundary_scaling_residual_reg=0.0" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0" \
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
    render_raw "${SELECTED_VARIANT}_${label}" "$ckpt" "$render_exp" "$SELECTED_COV_MODE" "$SELECTED_AXIS" "$SELECTED_SHRINK_FACTOR" "$SELECTED_GROW_FACTOR" "$SELECTED_MAX_SHRINK" "$SELECTED_MAX_GROW" "$HYDRA_RUN_ROOT/render_${SELECTED_VARIANT}_${label}" \
      > "$LOG_DIR/render_${SELECTED_VARIANT}_${label}.log" 2>&1
    analyze_raw "${SELECTED_VARIANT}_${label}" "$render_exp"
    append_train_summary "$label" "$SELECTED_VARIANT" "$ckpt" "$render_exp"
    log_event "train_render_done" "$label"
  done
}

if [ "$SKIP_NO_TRAIN" != "1" ]; then
  render_variant baseline default all 1.000 1.000 0 0
  render_variant signed_all_tiny default all 0.985 1.015 96 96
  render_variant signed_all_mid default all 0.970 1.020 96 96
  render_variant signed_major_soft default major 0.970 1.020 96 96
  render_variant signed_major_mid default major 0.940 1.025 96 96
  render_variant over_major_only default major 0.940 1.000 96 0
  render_variant under_major_only default major 1.000 1.025 0 96
  render_variant top48_major_mid default major 0.940 1.025 48 48
  select_variant
elif [ ! -f "$SELECTED_ENV" ]; then
  select_variant
else
  log_event "select_reuse" "$SELECTED_ENV"
fi
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
