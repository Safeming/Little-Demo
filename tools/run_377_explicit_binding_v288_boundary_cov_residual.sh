#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v288_boundary_cov_residual_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
TRAIN_ITERS="${TRAIN_ITERS:-80}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-40,80}"
BASE_ITER="${BASE_ITER:-136410}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"
TRAIN_COMPONENT_CSV="${TRAIN_COMPONENT_CSV:-$COMPONENT_CSV}"
TRAIN_POINT_CSV="${TRAIN_POINT_CSV:-$POINT_CSV}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v288_boundary_cov_residual_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v288_boundary_cov_residual_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
INIT_CKPT="$EXP_ROOT/init_boundary_cov_residual/ckpt${BASE_ITER}.pth"
INIT_REPORT="$LOG_DIR/init_boundary_cov_residual.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$(dirname "$INIT_CKPT")"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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
BASE_ITER=$BASE_ITER
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
TRAIN_COMPONENT_CSV=$TRAIN_COMPONENT_CSV
TRAIN_POINT_CSV=$TRAIN_POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
INIT_CKPT=$INIT_CKPT
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS

Goal:
  v288 converts the v281 render-time dynamic screen actuator into a checkpoint-internal
  local boundary_cov_residual. Only this residual is trainable. Color, xyz, opacity,
  scale, rotation, pose, and texture remain frozen.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'label\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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

DYNAMIC_ACTUATOR_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_point_json="
  "++pipeline.covariance_signed_shrink_factor=1.000"
  "++pipeline.covariance_signed_grow_factor=1.000"
  "++pipeline.covariance_signed_anisotropic_axis=all"
  "++pipeline.covariance_signed_dynamic_enable=true"
  "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV"
  "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV"
  "++pipeline.covariance_signed_dynamic_component_signature_enable=${COMPONENT_SIGNATURE_ENABLE:-false}"
  "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'"
  "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'"
  "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'"
  "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'"
  "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'"
  "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'"
  "++pipeline.covariance_signed_dynamic_boundary_min=0.0"
  "++pipeline.covariance_signed_dynamic_component_pad_px=10"
  "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25"
  "++pipeline.covariance_signed_dynamic_component_max_over=16"
  "++pipeline.covariance_signed_dynamic_component_max_under=16"
  "++pipeline.covariance_signed_dynamic_component_min_area=20"
  "++pipeline.covariance_signed_dynamic_component_required=${COMPONENT_REQUIRED:-false}"
  "++pipeline.covariance_signed_dynamic_component_top_ids_enable=${COMPONENT_TOP_IDS_ENABLE:-false}"
  "++pipeline.covariance_signed_dynamic_component_top_ids_only=${COMPONENT_TOP_IDS_ONLY:-false}"
  "++pipeline.covariance_signed_dynamic_max_over_points=96"
  "++pipeline.covariance_signed_dynamic_max_under_points=96"
  "++pipeline.covariance_signed_screen_actuator_enable=true"
  "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940"
  "++pipeline.covariance_signed_screen_normal_grow_factor=1.025"
  "++pipeline.covariance_signed_screen_tangent_factor=1.000"
  "++pipeline.boundary_cov_residual_enable=true"
  "++pipeline.boundary_cov_residual_max_abs=0.080"
)

TRAIN_DYNAMIC_ACTUATOR_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_point_json="
  "++pipeline.covariance_signed_shrink_factor=1.000"
  "++pipeline.covariance_signed_grow_factor=1.000"
  "++pipeline.covariance_signed_anisotropic_axis=all"
  "++pipeline.covariance_signed_dynamic_enable=true"
  "++pipeline.covariance_signed_dynamic_component_csv=$TRAIN_COMPONENT_CSV"
  "++pipeline.covariance_signed_dynamic_point_csv=$TRAIN_POINT_CSV"
  "++pipeline.covariance_signed_dynamic_component_signature_enable=${TRAIN_COMPONENT_SIGNATURE_ENABLE:-${COMPONENT_SIGNATURE_ENABLE:-false}}"
  "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'"
  "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'"
  "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'"
  "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'"
  "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'"
  "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'"
  "++pipeline.covariance_signed_dynamic_boundary_min=0.0"
  "++pipeline.covariance_signed_dynamic_component_pad_px=10"
  "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25"
  "++pipeline.covariance_signed_dynamic_component_max_over=16"
  "++pipeline.covariance_signed_dynamic_component_max_under=16"
  "++pipeline.covariance_signed_dynamic_component_min_area=20"
  "++pipeline.covariance_signed_dynamic_component_required=${TRAIN_COMPONENT_REQUIRED:-${COMPONENT_REQUIRED:-false}}"
  "++pipeline.covariance_signed_dynamic_component_top_ids_enable=${TRAIN_COMPONENT_TOP_IDS_ENABLE:-${COMPONENT_TOP_IDS_ENABLE:-false}}"
  "++pipeline.covariance_signed_dynamic_component_top_ids_only=${TRAIN_COMPONENT_TOP_IDS_ONLY:-${COMPONENT_TOP_IDS_ONLY:-false}}"
  "++pipeline.covariance_signed_dynamic_max_over_points=96"
  "++pipeline.covariance_signed_dynamic_max_under_points=96"
  "++pipeline.covariance_signed_screen_actuator_enable=true"
  "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940"
  "++pipeline.covariance_signed_screen_normal_grow_factor=1.025"
  "++pipeline.covariance_signed_screen_tangent_factor=1.000"
  "++pipeline.boundary_cov_residual_enable=true"
  "++pipeline.boundary_cov_residual_max_abs=0.080"
)

render_raw() {
  local ckpt="$1"
  local render_exp="$2"
  local hydra_dir="$3"

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
    "${DYNAMIC_ACTUATOR_ARGS[@]}" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++model.gaussian.boundary_cov_residual_channels=${BOUNDARY_COV_RESIDUAL_CHANNELS:-1}" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true"
}

analyze_raw() {
  local label="$1"
  local render_exp="$2"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1

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
    > "$LOG_DIR/boundary_residuals_${label}.log" 2>&1
}

append_summary() {
  local label="$1"
  local ckpt="$2"
  local render_exp="$3"

  "$PYTHON_BIN" - "$SUMMARY" "$label" "$ckpt" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

summary_path, label, ckpt, render_exp = sys.argv[1:5]
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
rows = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = rows[0]
baseline = None
for row in rows[1:]:
    if row and row[0] == "baseline_dynamic_screen_mid":
        baseline = {key: float(row[header.index(key)]) for key in metrics}
        break
if baseline is None or label == "baseline_dynamic_screen_mid":
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
    delta["inner"] < -0.20
    and delta["outer"] <= 1.25
    and delta["fg"] <= 0.000050
    and delta["boundary"] <= 0.000050
    and delta["edge"] <= 0.006000
    and delta["hard"] <= 0.000020
)
status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

row = [
    label,
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
    status,
]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

render_and_gate() {
  local label="$1"
  local ckpt="$2"
  local render_exp="$3"
  local hydra_dir="$4"
  log_event "render_start" "$label"
  render_raw "$ckpt" "$render_exp" "$hydra_dir" > "$LOG_DIR/render_${label}.log" 2>&1
  log_event "analyze_start" "$label"
  analyze_raw "$label" "$render_exp"
  append_summary "$label" "$ckpt" "$render_exp"
  log_event "gate_done" "$label"
}

log_event "init_ckpt_start" "$INIT_CKPT"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v288_boundary_cov_checkpoint.py \
  --input-ckpt "$BASE_CKPT" \
  --point-csv "$POINT_CSV" \
  --output-ckpt "$INIT_CKPT" \
  --report-json "$INIT_REPORT" \
  --max-over-points 96 \
  --max-under-points 96 \
  --min-visible-hits 12 \
  --min-signed-margin 0.5 \
  --over-residual -0.010 \
  --under-residual 0.010 \
  --cov-residual-channels "${BOUNDARY_COV_RESIDUAL_CHANNELS:-1}" \
  ${BOUNDARY_COV_SIGNED_TWO_CHANNEL:+--signed-two-channel} \
  --mark-boundary-tags \
  --reset-gaussian-optimizer-state \
  --save-gaussian-device cuda \
  > "$LOG_DIR/init_ckpt.log" 2>&1
log_event "init_ckpt_done" "$INIT_CKPT"

render_and_gate "baseline_dynamic_screen_mid" "$BASE_CKPT" "$EXP_ROOT/no_train_baseline_dynamic_screen_mid" "$HYDRA_RUN_ROOT/render_baseline_dynamic_screen_mid"
render_and_gate "init_boundary_cov_residual" "$INIT_CKPT" "$EXP_ROOT/no_train_init_boundary_cov_residual" "$HYDRA_RUN_ROOT/render_init_boundary_cov_residual"

TRAIN_EXP="$EXP_ROOT/train_boundary_cov_residual"
CHECKPOINT_LIST="[$TRAIN_CHECKPOINT_STEPS]"
log_event "train_start" "$TRAIN_EXP"
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
  "start_checkpoint=$INIT_CKPT" \
  "exp_dir=$TRAIN_EXP" \
  "hydra.run.dir=$HYDRA_RUN_ROOT/train_boundary_cov_residual" \
  "seed=-1" \
  "wandb_disable=true" \
  "++resume.restore_gaussian_optimizer_state=false" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=false" \
  "++resume.clear_binding_state_on_resume=false" \
  "pipeline.pose_noise=0.0" \
  "${TRAIN_DYNAMIC_ACTUATOR_ARGS[@]}" \
  "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
  "++model.gaussian.boundary_cov_residual_channels=${BOUNDARY_COV_RESIDUAL_CHANNELS:-1}" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "opt.iterations=$TRAIN_ITERS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.boundary_cov_residual_lr=0.0008" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "++opt.foreground_mask_source=hard" \
  "++opt.global_mask_source=hard" \
  "++opt.boundary_target_mask_source=hard" \
  "++opt.boundary_region_source=binary" \
  "++opt.boundary_band_width=8" \
  "opt.lambda_l1=0.0" \
  "opt.lambda_l1_fg=0.0" \
  "opt.lambda_l1_boundary=0.0" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_silhouette_outer=0.0" \
  "++opt.lambda_silhouette_outer_shell=0.008" \
  "++opt.silhouette_outer_shell_start_width=1" \
  "++opt.silhouette_outer_shell_end_width=18" \
  "++opt.lambda_silhouette_inner=0.008" \
  "++opt.silhouette_inner_ring_width=9" \
  "++opt.lambda_raw_support_hinge_inner=0.006" \
  "++opt.lambda_raw_support_hinge_outer=0.006" \
  "++opt.raw_support_hinge_inner_band_width=9" \
  "++opt.raw_support_hinge_outer_start_width=1" \
  "++opt.raw_support_hinge_outer_end_width=24" \
  "++opt.raw_support_hinge_min_pixels=8" \
  "++opt.lambda_boundary_cov_residual_reg=0.020" \
  "++opt.lambda_boundary_cov_residual_smooth=0.010" \
  "++opt.boundary_residual_smooth_k=8" \
  "++opt.boundary_residual_smooth_distance_quantile=0.5" \
  "++opt.lambda_boundary_opacity_residual_reg=0.0" \
  "++opt.lambda_boundary_scaling_residual_reg=0.0" \
  "++opt.stageB_semantic_loss_enable=false" \
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
  "test_iterations=$CHECKPOINT_LIST" \
  "save_iterations=$CHECKPOINT_LIST" \
  "checkpoint_iterations=$CHECKPOINT_LIST" \
  "++validation_image_log_limit=0" \
  "opt.grad_clip=0.0010" \
  > "$LOG_DIR/train_boundary_cov_residual.log" 2>&1
log_event "train_done" "$TRAIN_EXP"

IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
for step in "${steps[@]}"; do
  global_iter=$((BASE_ITER + step))
  ckpt="$TRAIN_EXP/ckpt${global_iter}.pth"
  label="boundary_cov_residual_ckpt${global_iter}"
  if [ ! -f "$ckpt" ]; then
    log_event "train_render_skip" "missing=$ckpt"
    continue
  fi
  render_and_gate "$label" "$ckpt" "$EXP_ROOT/raw_render_${label}" "$HYDRA_RUN_ROOT/render_${label}"
done

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "INIT_REPORT=$INIT_REPORT"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
