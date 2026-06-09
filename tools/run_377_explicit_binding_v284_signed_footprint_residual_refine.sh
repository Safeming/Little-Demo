#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v284_signed_footprint_residual_refine_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
TRAIN_ITERS="${TRAIN_ITERS:-160}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-80,160}"
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

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v284_signed_footprint_residual_refine_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v284_signed_footprint_residual_refine_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

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
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS

Goal:
  v284 is a short signed footprint residual refine.
  It keeps the v281 dynamic_screen_mid render-time actuator as the baseline geometry,
  freezes old xyz/base opacity/base scale/rotation/color/texture/pose, and trains only
  boundary_opacity_residual + boundary_scaling_residual through signed opacity footprint losses.
  Raw contour gate decides whether a checkpoint is usable.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'label\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\topacity_inner\topacity_outer\topacity_on_rgb_inner_ratio\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\topacity_inner_delta\topacity_outer_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"

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
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_shrink_factor=1.000" \
    "++pipeline.covariance_signed_grow_factor=1.000" \
    "++pipeline.covariance_signed_anisotropic_axis=all" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=96" \
    "++pipeline.covariance_signed_dynamic_max_under_points=96" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
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
    > "$LOG_DIR/opacity_footprint_${label}.log" 2>&1
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
opacity = json.loads((render_exp / "diagnostics" / "opacity_footprint" / "opacity_footprint_summary.json").read_text(encoding="utf-8"))
metrics = {
    "fg": float(contour["mean_fg_l1"]),
    "boundary": float(contour["mean_boundary_l1"]),
    "edge": float(contour["mean_edge_symmetric_dist_px"]),
    "inner": float(residual["mean_inner_missing_pixels"]),
    "outer": float(residual["mean_outer_leak_pixels"]),
    "hard": float(residual["mean_hard_residual_score"]),
    "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
    "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    "opacity_on_rgb_inner_ratio": float(opacity["mean_primary_opacity_on_rgb_inner_ratio"]),
}
rows = [line.rstrip("\n").split("\t") for line in summary_path.read_text(encoding="utf-8").splitlines()]
header = rows[0]
baseline = None
for row in rows[1:]:
    if row and row[0] == "baseline_dynamic_screen_mid":
        baseline = {key: float(row[header.index(key)]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")}
        break
if baseline is None or label == "baseline_dynamic_screen_mid":
    baseline = {key: metrics[key] for key in ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")}
delta = {key: metrics[key] - baseline[key] for key in baseline}
strict = (
    delta["inner"] < -0.05
    and delta["opacity_inner"] < -0.05
    and delta["outer"] <= 0.0
    and delta["opacity_outer"] <= 0.0
    and delta["fg"] <= 0.0
    and delta["boundary"] <= 0.0
    and delta["edge"] <= 0.0
    and delta["hard"] < -0.000001
)
probe = (
    (delta["inner"] < -0.05 or delta["opacity_inner"] < -0.05)
    and delta["outer"] <= 1.0
    and delta["opacity_outer"] <= 2.0
    and delta["fg"] <= 0.00008
    and delta["boundary"] <= 0.00008
    and delta["edge"] <= 0.008
    and delta["hard"] <= 0.00001
)

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
    fmt(metrics["opacity_inner"], 4),
    fmt(metrics["opacity_outer"], 4),
    fmt(metrics["opacity_on_rgb_inner_ratio"], 6),
    fmt(delta["fg"]),
    fmt(delta["boundary"]),
    fmt(delta["edge"], 6),
    fmt(delta["inner"], 4),
    fmt(delta["outer"], 4),
    fmt(delta["hard"]),
    fmt(delta["opacity_inner"], 4),
    fmt(delta["opacity_outer"], 4),
    "1" if strict else "0",
    "1" if probe else "0",
    "ok",
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

  log_event "render_start" "$label ckpt=$ckpt"
  render_raw "$ckpt" "$render_exp" "$hydra_dir" > "$LOG_DIR/render_${label}.log" 2>&1
  log_event "analyze_start" "$label"
  analyze_raw "$label" "$render_exp"
  append_summary "$label" "$ckpt" "$render_exp"
  log_event "gate_done" "$label"
}

train_v284() {
  local train_exp="$EXP_ROOT/train_signed_footprint_residual"
  local checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"
  log_event "train_start" "exp=$train_exp"

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
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_signed_footprint_residual" \
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
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_shrink_factor=1.000" \
    "++pipeline.covariance_signed_grow_factor=1.000" \
    "++pipeline.covariance_signed_anisotropic_axis=all" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=96" \
    "++pipeline.covariance_signed_dynamic_max_under_points=96" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
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
    "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.000018" \
    "++opt.boundary_scaling_residual_lr=0.000006" \
    "++opt.stageB_semantic_loss_enable=false" \
    "++opt.stageB_semantic_body_cloth_weight=0.0" \
    "++opt.stageB_semantic_compact_weight=0.0" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0" \
    "++opt.semantic_region_logits_lr=0.0" \
    "++opt.semantic_compact_logits_lr=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.018" \
    "++opt.train_sample_camera_max_prob=0.125" \
    "++opt.train_sample_log_interval=80" \
    "++opt.train_sample_accumulation_steps=1" \
    "++opt.foreground_mask_source=hard" \
    "++opt.global_mask_source=hard" \
    "++opt.boundary_target_mask_source=hard" \
    "++opt.boundary_region_source=binary" \
    "++opt.boundary_band_width=8" \
    "++opt.silhouette_outer_ring_width=7" \
    "++opt.silhouette_outer_shell_start_width=1" \
    "++opt.silhouette_outer_shell_end_width=20" \
    "++opt.silhouette_inner_ring_width=7" \
    "++opt.mask_loss_type=bce" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_mask_boundary_hard=0.0" \
    "++opt.lambda_mask_shoulder_arm_boundary_hard=0.0" \
    "++opt.lambda_mask_upper_torso_boundary_hard=0.0" \
    "++opt.lambda_mask_shoulder_arm_region_hard=0.0" \
    "++opt.lambda_mask_shoulder_arm_disagreement_hard=0.0" \
    "++opt.lambda_mask_shoulder_focus_small_fp_hard=0.0" \
    "++opt.lambda_mask_shoulder_focus_small_fn_hard=0.0" \
    "++opt.lambda_silhouette_outer=0.0024" \
    "++opt.lambda_silhouette_outer_shell=0.0038" \
    "++opt.lambda_silhouette_inner=0.0034" \
    "++opt.lambda_silhouette_head_outer_shell=0.0" \
    "++opt.lambda_silhouette_shoulder_arm_outer_shell=0.0" \
    "++opt.lambda_silhouette_upper_torso_outer_shell=0.0" \
    "++opt.lambda_silhouette_outer_spike=0.0" \
    "++opt.lambda_silhouette_outer_fragment=0.0" \
    "++opt.lambda_silhouette_outer_bead=0.0" \
    "++opt.lambda_silhouette_outer_chain=0.0" \
    "++opt.lambda_silhouette_arm_stipple=0.0" \
    "++opt.lambda_silhouette_arm_tail=0.0" \
    "++opt.lambda_silhouette_arm_fringe=0.0" \
    "++opt.lambda_silhouette_arm_attached_fragment=0.0" \
    "++opt.lambda_silhouette_shoulder_attached_fragment=0.0" \
    "++opt.lambda_silhouette_arm_notch=0.0" \
    "++opt.lambda_silhouette_arm_hole=0.0" \
    "++opt.lambda_silhouette_arm_gap=0.0" \
    "++opt.lambda_silhouette_shoulder_bead=0.0" \
    "++opt.lambda_silhouette_shoulder_chain=0.0" \
    "++opt.lambda_silhouette_shoulder_hole=0.0" \
    "++opt.lambda_silhouette_shoulder_gap=0.0" \
    "++opt.lambda_silhouette_shoulder_pinhole=0.0" \
    "++opt.boundary_image_error_score_enable=true" \
    "++opt.boundary_image_error_score_signed_enable=true" \
    "++opt.boundary_image_error_score_mix=1.0" \
    "++opt.boundary_image_error_score_gain=1.08" \
    "++opt.boundary_image_error_score_power=1.0" \
    "++opt.boundary_image_error_score_min=0.010" \
    "++opt.boundary_image_error_score_band_width=8" \
    "++opt.boundary_image_error_pred_threshold=0.34" \
    "++opt.boundary_image_error_target_threshold=0.50" \
    "++opt.boundary_image_error_score_focus_dilate=4" \
    "++opt.boundary_image_error_score_neighborhood_radius=4" \
    "++opt.boundary_image_error_score_neighborhood_max_radius=6" \
    "++opt.boundary_image_error_score_use_radii_radius=true" \
    "++opt.boundary_image_error_score_radii_quantile=0.50" \
    "++opt.boundary_image_error_score_radii_scale=0.25" \
    "++opt.boundary_image_error_score_contributor_enable=true" \
    "++opt.boundary_image_error_score_contributor_mix=0.75" \
    "++opt.boundary_image_error_score_contributor_interval=20" \
    "++opt.boundary_image_error_contributor_residual_threshold=0.10" \
    "++opt.boundary_image_error_contributor_max_pixels=2048" \
    "++opt.boundary_image_error_contributor_pixel_chunk=384" \
    "++opt.boundary_image_error_contributor_point_chunk=8192" \
    "++opt.boundary_image_error_contributor_topk=5" \
    "++opt.boundary_image_error_contributor_radius_scale=1.25" \
    "++opt.boundary_image_error_contributor_min_radius=2.0" \
    "++opt.boundary_image_error_contributor_max_radius=14.0" \
    "++opt.boundary_image_error_contributor_sigma_scale=0.55" \
    "++opt.boundary_image_error_contributor_opacity_power=0.45" \
    "++opt.boundary_image_error_contributor_score_power=1.0" \
    "++opt.boundary_image_error_contributor_verbose=false" \
    "++opt.boundary_image_error_score_smooth_k=8" \
    "++opt.boundary_image_error_score_smooth_blend=0.14" \
    "++opt.boundary_image_error_score_prior_floor=0.04" \
    "++opt.boundary_aware_enable=true" \
    "++opt.boundary_aware_gate_l1_boundary=false" \
    "++opt.boundary_aware_gate_mask_boundary=false" \
    "++opt.boundary_aware_gate_mask_boundary_hard=false" \
    "++opt.boundary_aware_gate_mask_shoulder_arm_boundary_hard=false" \
    "++opt.boundary_aware_gate_mask_upper_torso_boundary_hard=false" \
    "++opt.boundary_aware_gate_mask_shoulder_arm_region_hard=false" \
    "++opt.boundary_aware_gate_mask_shoulder_arm_disagreement_hard=false" \
    "++opt.boundary_aware_gate_mask_shoulder_focus_small_fp_hard=false" \
    "++opt.boundary_aware_gate_mask_shoulder_focus_small_fn_hard=false" \
    "++opt.boundary_aware_gate_silhouette_outer=true" \
    "++opt.boundary_aware_gate_silhouette_outer_shell=true" \
    "++opt.boundary_aware_gate_silhouette_inner=true" \
    "++opt.boundary_aware_gate_silhouette_head_outer_shell=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_arm_outer_shell=false" \
    "++opt.boundary_aware_gate_silhouette_upper_torso_outer_shell=false" \
    "++opt.boundary_aware_gate_silhouette_outer_spike=false" \
    "++opt.boundary_aware_gate_silhouette_outer_fragment=false" \
    "++opt.boundary_aware_gate_silhouette_outer_bead=false" \
    "++opt.boundary_aware_gate_silhouette_outer_chain=false" \
    "++opt.boundary_aware_gate_silhouette_arm_stipple=false" \
    "++opt.boundary_aware_gate_silhouette_arm_tail=false" \
    "++opt.boundary_aware_gate_silhouette_arm_fringe=false" \
    "++opt.boundary_aware_gate_silhouette_arm_attached_fragment=false" \
    "++opt.boundary_aware_gate_silhouette_arm_notch=false" \
    "++opt.boundary_aware_gate_silhouette_arm_hole=false" \
    "++opt.boundary_aware_gate_silhouette_arm_gap=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_attached_fragment=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_bead=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_chain=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_hole=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_gap=false" \
    "++opt.boundary_aware_gate_silhouette_shoulder_pinhole=false" \
    "++opt.boundary_aware_threshold=0.010" \
    "++opt.boundary_aware_score_power=0.90" \
    "++opt.boundary_tag_schedule_use_local_iteration=true" \
    "++opt.boundary_tag_enable=true" \
    "++opt.boundary_tag_init_iter=1" \
    "++opt.boundary_tag_update_interval=20" \
    "++opt.boundary_tag_update_until_iter=120" \
    "++opt.boundary_tag_mode=topk_ratio" \
    "++opt.boundary_tag_topk_ratio=0.08" \
    "++opt.boundary_tag_min_ratio=0.04" \
    "++opt.boundary_tag_threshold=0.12" \
    "++opt.boundary_tag_binary=true" \
    "++opt.boundary_tag_use_score_within_subset=true" \
    "++opt.boundary_tag_score_smooth_blend=0.12" \
    "++opt.boundary_tag_score_smooth_k=8" \
    "++opt.boundary_tag_support_k=6" \
    "++opt.boundary_tag_support_threshold=0.12" \
    "++opt.boundary_signed_routing_enable=true" \
    "++opt.boundary_signed_mixed_loss_scale=0.0" \
    "++opt.boundary_signed_shrink_loss_scale=0.42" \
    "++opt.boundary_signed_grow_loss_scale=0.42" \
    "++opt.boundary_signed_share_gain=1.00" \
    "++opt.boundary_signed_share_power=1.0" \
    "++opt.boundary_signed_shrink_share_gain=1.00" \
    "++opt.boundary_signed_shrink_share_power=1.0" \
    "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
    "++opt.boundary_aware_feature_dc_scale=0.0" \
    "++opt.boundary_aware_feature_rest_scale=0.0" \
    "++opt.boundary_aware_xyz_scale=0.0" \
    "++opt.boundary_aware_opacity_scale=0.0" \
    "++opt.boundary_aware_scaling_scale=0.0" \
    "++opt.boundary_aware_rotation_scale=0.0" \
    "++opt.boundary_aware_boundary_opacity_residual_scale=0.85" \
    "++opt.boundary_aware_boundary_scaling_residual_scale=0.55" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0040" \
    "++opt.lambda_boundary_scaling_residual_reg=0.0060" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0030" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0040" \
    "opt.lambda_l1=0.0" \
    "opt.lambda_l1_fg=0.0" \
    "opt.lambda_l1_boundary=0.0" \
    "opt.lambda_perceptual=0.0" \
    "opt.lambda_l1_face=0.0" \
    "opt.lambda_l1_shoulder_arm=0.0" \
    "opt.lambda_l1_waist=0.0" \
    "opt.lambda_edge_face=0.0" \
    "opt.lambda_edge_shoulder_arm=0.0" \
    "opt.lambda_edge_waist=0.0" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.0" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
    "++opt.lambda_detail_waist_luma_dog=0.0" \
    "++opt.lambda_perceptual_face=0.0" \
    "++opt.lambda_perceptual_shoulder_arm=0.0" \
    "++opt.lambda_perceptual_waist=0.0" \
    "++opt.lambda_perceptual_face_patch=0.0" \
    "++opt.lambda_perceptual_shoulder_arm_patch=0.0" \
    "++opt.lambda_perceptual_upper_torso_patch=0.0" \
    "++opt.lambda_perceptual_upper_torso_core_patch=0.0" \
    "++opt.lambda_perceptual_waist_patch=0.0" \
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
    "opt.grad_clip=0.0010" \
    > "$LOG_DIR/train_signed_footprint_residual.log" 2>&1

  log_event "train_done" "$train_exp"
}

render_and_gate \
  baseline_dynamic_screen_mid \
  "$BASE_CKPT" \
  "$EXP_ROOT/no_train_baseline_dynamic_screen_mid" \
  "$HYDRA_RUN_ROOT/render_baseline_dynamic_screen_mid"

TRAIN_EXP="$EXP_ROOT/train_signed_footprint_residual"
train_v284

IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
for step in "${steps[@]}"; do
  global_iter=$((BASE_ITER + step))
  ckpt="$TRAIN_EXP/ckpt${global_iter}.pth"
  label="ckpt${global_iter}"
  render_exp="${TRAIN_EXP}_raw_render_${label}"
  if [ ! -f "$ckpt" ]; then
    log_event "train_render_skip" "missing=$ckpt"
    continue
  fi
  render_and_gate "$label" "$ckpt" "$render_exp" "$HYDRA_RUN_ROOT/render_${label}"
done

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
