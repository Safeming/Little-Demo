#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v321_teacher_locked_paired_signed_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"
SELECTED_CSV="${SELECTED_CSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v320_paired_signed_selector_v320_paired_signed_selector_20260519_212529_bjt/selected_components.csv}"
REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-$ROOT/exp/stageB/logs/377_explicit_binding_v320_paired_signed_selector_v320_paired_signed_selector_20260519_212529_bjt/selection_summary.tsv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,1]}"
EVAL_TRAIN_FRAMES_SPEC="${EVAL_TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

TRAIN_ITERS="${TRAIN_ITERS:-200}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-100,200}"
FEATURE_LR="${FEATURE_LR:-0.00002}"
TEXTURE_LR="${TEXTURE_LR:-0.0}"
FG_L1_REGION_MODE="${FG_L1_REGION_MODE:-foreground}"
FG_L1_INTERIOR_ERODE_KERNEL_SIZE="${FG_L1_INTERIOR_ERODE_KERNEL_SIZE:-0}"
LAMBDA_L1="${LAMBDA_L1:-0.025}"
LAMBDA_L1_FG="${LAMBDA_L1_FG:-0.070}"
LAMBDA_L1_BOUNDARY="${LAMBDA_L1_BOUNDARY:-0.010}"
TEACHER_RENDER_DISTILL_ENABLE="${TEACHER_RENDER_DISTILL_ENABLE:-${TEACHER_DISTILL_ENABLE:-true}}"
LAMBDA_TEACHER_RENDER_DISTILL_BOUNDARY="${LAMBDA_TEACHER_RENDER_DISTILL_BOUNDARY:-1.00}"
LAMBDA_TEACHER_RENDER_DISTILL_OUTER="${LAMBDA_TEACHER_RENDER_DISTILL_OUTER:-1.60}"
LAMBDA_TEACHER_RENDER_DISTILL_INNER="${LAMBDA_TEACHER_RENDER_DISTILL_INNER:-0.08}"
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_BOUNDARY="${LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_BOUNDARY:-2.00}"
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_OUTER="${LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_OUTER:-4.00}"
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_INNER="${LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_INNER:-0.25}"
BOUNDARY_COLOR_PROTECT_ENABLE="${BOUNDARY_COLOR_PROTECT_ENABLE:-false}"
BOUNDARY_COLOR_PROTECT_VERBOSE="${BOUNDARY_COLOR_PROTECT_VERBOSE:-false}"
SCREEN_COLOR_PROTECT_ENABLE="${SCREEN_COLOR_PROTECT_ENABLE:-false}"
SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH="${SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH:-17}"
SCREEN_COLOR_PROTECT_OUTER_START="${SCREEN_COLOR_PROTECT_OUTER_START:-1}"
SCREEN_COLOR_PROTECT_OUTER_END="${SCREEN_COLOR_PROTECT_OUTER_END:-38}"
SCREEN_COLOR_PROTECT_RADIUS_PAD="${SCREEN_COLOR_PROTECT_RADIUS_PAD:-5}"
SCREEN_COLOR_PROTECT_MIN_RADIUS="${SCREEN_COLOR_PROTECT_MIN_RADIUS:-0.0}"
SCREEN_COLOR_PROTECT_MAX_POINTS="${SCREEN_COLOR_PROTECT_MAX_POINTS:-0}"
STATIC_COLOR_PROTECT_ENABLE="${STATIC_COLOR_PROTECT_ENABLE:-false}"
STATIC_COLOR_PROTECT_COMPONENT_CSV="${STATIC_COLOR_PROTECT_COMPONENT_CSV:-$SELECTED_CSV}"
STATIC_COLOR_PROTECT_POINT_CSV="${STATIC_COLOR_PROTECT_POINT_CSV:-$POINT_CSV}"
STATIC_COLOR_PROTECT_MAX_POINT_CSV_POINTS="${STATIC_COLOR_PROTECT_MAX_POINT_CSV_POINTS:-512}"
STATIC_COLOR_PROTECT_MIN_ABS_SIGNED_SCORE="${STATIC_COLOR_PROTECT_MIN_ABS_SIGNED_SCORE:-75.0}"
STATIC_COLOR_PROTECT_MIN_VISIBLE_HITS="${STATIC_COLOR_PROTECT_MIN_VISIBLE_HITS:-80}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v321_teacher_locked_paired_signed_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v321_teacher_locked_paired_signed_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
TRAIN_EXP="$EXP_ROOT/teacher_locked_color_train"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/train_summary.tsv"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$POINT_CSV" "$SELECTED_CSV" "$REFERENCE_SUMMARY" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

BASE_ITER="$("$PYTHON_BIN" - "$BASE_CKPT" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt[-1]))
PY
)"
CHECKPOINT_LIST="[$TRAIN_CHECKPOINT_STEPS]"

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-1800}"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$((START_EPOCH + EST_SECONDS))" '+%F %T BJT')"

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'label\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta_baseline\tboundary_delta_baseline\tedge_delta_baseline\tinner_delta_baseline\touter_delta_baseline\thard_delta_baseline\tfg_delta_selected\tboundary_delta_selected\tedge_delta_selected\tinner_delta_selected\touter_delta_selected\thard_delta_selected\tstrict_vs_baseline\tdo_no_harm_vs_selected\tstatus\n' > "$SUMMARY"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
SELECTED_CSV=$SELECTED_CSV
POINT_CSV=$POINT_CSV
REFERENCE_SUMMARY=$REFERENCE_SUMMARY
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS
FEATURE_LR=$FEATURE_LR
TEXTURE_LR=$TEXTURE_LR
FG_L1_REGION_MODE=$FG_L1_REGION_MODE
FG_L1_INTERIOR_ERODE_KERNEL_SIZE=$FG_L1_INTERIOR_ERODE_KERNEL_SIZE
LAMBDA_L1=$LAMBDA_L1
LAMBDA_L1_FG=$LAMBDA_L1_FG
LAMBDA_L1_BOUNDARY=$LAMBDA_L1_BOUNDARY
TEACHER_RENDER_DISTILL_ENABLE=$TEACHER_RENDER_DISTILL_ENABLE
LAMBDA_TEACHER_RENDER_DISTILL_BOUNDARY=$LAMBDA_TEACHER_RENDER_DISTILL_BOUNDARY
LAMBDA_TEACHER_RENDER_DISTILL_OUTER=$LAMBDA_TEACHER_RENDER_DISTILL_OUTER
LAMBDA_TEACHER_RENDER_DISTILL_INNER=$LAMBDA_TEACHER_RENDER_DISTILL_INNER
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_BOUNDARY=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_BOUNDARY
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_OUTER=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_OUTER
LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_INNER=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_INNER
BOUNDARY_COLOR_PROTECT_ENABLE=$BOUNDARY_COLOR_PROTECT_ENABLE
SCREEN_COLOR_PROTECT_ENABLE=$SCREEN_COLOR_PROTECT_ENABLE
SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH=$SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH
SCREEN_COLOR_PROTECT_OUTER_START=$SCREEN_COLOR_PROTECT_OUTER_START
SCREEN_COLOR_PROTECT_OUTER_END=$SCREEN_COLOR_PROTECT_OUTER_END
SCREEN_COLOR_PROTECT_RADIUS_PAD=$SCREEN_COLOR_PROTECT_RADIUS_PAD
STATIC_COLOR_PROTECT_ENABLE=$STATIC_COLOR_PROTECT_ENABLE
STATIC_COLOR_PROTECT_COMPONENT_CSV=$STATIC_COLOR_PROTECT_COMPONENT_CSV
STATIC_COLOR_PROTECT_POINT_CSV=$STATIC_COLOR_PROTECT_POINT_CSV
STATIC_COLOR_PROTECT_MAX_POINT_CSV_POINTS=$STATIC_COLOR_PROTECT_MAX_POINT_CSV_POINTS
STATIC_COLOR_PROTECT_MIN_ABS_SIGNED_SCORE=$STATIC_COLOR_PROTECT_MIN_ABS_SIGNED_SCORE
STATIC_COLOR_PROTECT_MIN_VISIBLE_HITS=$STATIC_COLOR_PROTECT_MIN_VISIBLE_HITS
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v321 tests the current root cause directly: v320 no-train paired signed
  geometry passes the raw gate, but color/SH training can move RGB support
  across the silhouette threshold. Keep xyz/opacity/scale/rotation/pose/camera
  frozen, train only color/SH lightly, and distill the teacher/base raw support
  in boundary and outer shells so training cannot erase the no-train contour
  gain.
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

SELECTED_GEOM_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=true"
  "++pipeline.covariance_signed_dynamic_component_csv=$SELECTED_CSV"
  "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV"
  "++pipeline.covariance_signed_dynamic_component_signature_enable=false"
  "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'"
  "++pipeline.covariance_signed_dynamic_over_region_ids=cloth"
  "++pipeline.covariance_signed_dynamic_over_joint_ids='6,9,12,13,14,15'"
  "++pipeline.covariance_signed_dynamic_under_layer_ids='soft,rigid,free'"
  "++pipeline.covariance_signed_dynamic_under_region_ids='cloth,body,soft'"
  "++pipeline.covariance_signed_dynamic_under_joint_ids='0,1,2,4,7,8,10'"
  "++pipeline.covariance_signed_dynamic_boundary_min=0.0"
  "++pipeline.covariance_signed_dynamic_component_pad_px=10"
  "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25"
  "++pipeline.covariance_signed_dynamic_component_max_over=16"
  "++pipeline.covariance_signed_dynamic_component_max_under=16"
  "++pipeline.covariance_signed_dynamic_component_min_area=20"
  "++pipeline.covariance_signed_dynamic_component_required=true"
  "++pipeline.covariance_signed_dynamic_max_over_points=96"
  "++pipeline.covariance_signed_dynamic_max_under_points=96"
  "++pipeline.covariance_signed_screen_actuator_enable=true"
  "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940"
  "++pipeline.covariance_signed_screen_normal_grow_factor=1.025"
  "++pipeline.covariance_signed_screen_tangent_factor=1.000"
  "++pipeline.covariance_signed_center_offset_enable=true"
  "++pipeline.covariance_signed_center_offset_outer_px=0.35"
  "++pipeline.covariance_signed_center_offset_inner_px=0.0"
  "++pipeline.covariance_signed_center_offset_outer_direction=view_center"
  "++pipeline.covariance_signed_center_offset_inner_direction=component_center"
  "++pipeline.covariance_signed_center_offset_score_weight_power=1.0"
  "++pipeline.covariance_signed_center_offset_score_weight_min=0.15"
  "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90"
  "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001"
  "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001"
  "++pipeline.covariance_signed_center_offset_max_world_step=0.0020"
  "++pipeline.boundary_cov_residual_enable=false"
  "++pipeline.binding_covariance_guard_enable=false"
  "++model.deformer.rigid.rotation_orthogonalize_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=true"
  "++model.deformer.rigid.geometry_fidelity_target=free_lbs"
  "++model.deformer.rigid.geometry_fidelity_center_strength=0.45"
  "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_boundary_min=0.12"
  "++model.deformer.rigid.geometry_fidelity_layer_ids='soft,free'"
  "++model.deformer.rigid.geometry_fidelity_region_ids='cloth,soft'"
  "++model.deformer.rigid.geometry_fidelity_joint_ids="
  "++model.deformer.rigid.geometry_fidelity_non_rigid_min=0.0"
  "++model.deformer.rigid.geometry_fidelity_power=1.2"
  "++model.deformer.rigid.geometry_fidelity_max_points=1024"
  "++model.deformer.rigid.geometry_fidelity_component_enable=true"
  "++model.deformer.rigid.geometry_fidelity_component_csv=$SELECTED_CSV"
  "++model.deformer.rigid.geometry_fidelity_component_direction=inner"
  "++model.deformer.rigid.geometry_fidelity_component_pad_px=2"
  "++model.deformer.rigid.geometry_fidelity_component_ellipse_scale=1.05"
  "++model.deformer.rigid.geometry_fidelity_component_max=12"
  "++model.deformer.rigid.geometry_fidelity_component_min_area=40"
  "++model.deformer.rigid.geometry_fidelity_component_required=true"
  "++model.deformer.rigid.geometry_fidelity_component_improvement_enable=true"
)

log_event "train_start" "$TRAIN_EXP"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.val_views=$TEST_VIEWS_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.val_frames=$TEST_FRAMES_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.compact_mapping_file=" \
  "start_checkpoint=$BASE_CKPT" \
  "exp_dir=$TRAIN_EXP" \
  "hydra.run.dir=$HYDRA_RUN_ROOT/train" \
  "seed=-1" \
  "wandb_disable=true" \
  "++resume.allow_partial_converter_load=true" \
  "++resume.restore_gaussian_optimizer_state=false" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=true" \
  "++resume.clear_binding_state_on_resume=false" \
  "pipeline.pose_noise=0.0" \
  "${SELECTED_GEOM_ARGS[@]}" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "opt.iterations=$TRAIN_ITERS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=$FEATURE_LR" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=$TEXTURE_LR" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "++opt.binding_layer_logits_lr=0.0" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.stageB_semantic_body_cloth_weight=0.0" \
  "++opt.stageB_semantic_compact_weight=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "opt.lambda_l1=$LAMBDA_L1" \
  "opt.lambda_l1_fg=$LAMBDA_L1_FG" \
  "++opt.fg_l1_region_mode=$FG_L1_REGION_MODE" \
  "++opt.fg_l1_interior_erode_kernel_size=$FG_L1_INTERIOR_ERODE_KERNEL_SIZE" \
  "opt.lambda_l1_boundary=$LAMBDA_L1_BOUNDARY" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_silhouette_outer=0.0" \
  "++opt.lambda_silhouette_inner=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "++opt.teacher_render_distill_enable=$TEACHER_RENDER_DISTILL_ENABLE" \
  "++opt.teacher_render_distill_ckpt=$BASE_CKPT" \
  "++opt.teacher_render_distill_iteration_mode=teacher" \
  "++opt.lambda_teacher_render_distill_boundary=$LAMBDA_TEACHER_RENDER_DISTILL_BOUNDARY" \
  "++opt.lambda_teacher_render_distill_outer=$LAMBDA_TEACHER_RENDER_DISTILL_OUTER" \
  "++opt.lambda_teacher_render_distill_inner=$LAMBDA_TEACHER_RENDER_DISTILL_INNER" \
  "++opt.teacher_render_distill_boundary_width=11" \
  "++opt.teacher_render_distill_outer_start_width=1" \
  "++opt.teacher_render_distill_outer_end_width=28" \
  "++opt.teacher_render_distill_inner_erode_width=15" \
  "++opt.lambda_teacher_render_distill_raw_support_boundary=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_BOUNDARY" \
  "++opt.lambda_teacher_render_distill_raw_support_outer=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_OUTER" \
  "++opt.lambda_teacher_render_distill_raw_support_inner=$LAMBDA_TEACHER_RENDER_DISTILL_RAW_SUPPORT_INNER" \
  "++opt.teacher_render_distill_raw_support_threshold=0.025" \
  "++opt.teacher_render_distill_raw_support_chroma_factor=0.75" \
  "++opt.teacher_render_distill_raw_support_margin=0.004" \
  "++opt.teacher_render_distill_raw_support_close_kernel=5" \
  "++opt.teacher_render_distill_raw_support_boundary_width=13" \
  "++opt.teacher_render_distill_raw_support_outer_start_width=1" \
  "++opt.teacher_render_distill_raw_support_outer_end_width=30" \
  "++opt.teacher_render_distill_raw_support_inner_erode_width=13" \
  "++opt.teacher_render_distill_raw_support_boundary_mode=both" \
  "++opt.teacher_render_distill_raw_support_outer_mode=suppress" \
  "++opt.teacher_render_distill_raw_support_inner_mode=keep" \
  "++opt.teacher_render_distill_raw_support_min_pixels=8" \
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
  "++opt.boundary_color_grad_protect_enable=$BOUNDARY_COLOR_PROTECT_ENABLE" \
  "++opt.boundary_color_grad_protect_verbose=$BOUNDARY_COLOR_PROTECT_VERBOSE" \
  "++opt.screen_space_color_grad_protect_enable=$SCREEN_COLOR_PROTECT_ENABLE" \
  "++opt.screen_space_color_grad_protect_boundary_width=$SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH" \
  "++opt.screen_space_color_grad_protect_outer_start_width=$SCREEN_COLOR_PROTECT_OUTER_START" \
  "++opt.screen_space_color_grad_protect_outer_end_width=$SCREEN_COLOR_PROTECT_OUTER_END" \
  "++opt.screen_space_color_grad_protect_radius_pad_px=$SCREEN_COLOR_PROTECT_RADIUS_PAD" \
  "++opt.screen_space_color_grad_protect_min_radius_px=$SCREEN_COLOR_PROTECT_MIN_RADIUS" \
  "++opt.screen_space_color_grad_protect_max_points=$SCREEN_COLOR_PROTECT_MAX_POINTS" \
  "++opt.static_color_grad_protect_enable=$STATIC_COLOR_PROTECT_ENABLE" \
  "++opt.static_color_grad_protect_component_csv=$STATIC_COLOR_PROTECT_COMPONENT_CSV" \
  "++opt.static_color_grad_protect_point_csv=$STATIC_COLOR_PROTECT_POINT_CSV" \
  "++opt.static_color_grad_protect_max_point_csv_points=$STATIC_COLOR_PROTECT_MAX_POINT_CSV_POINTS" \
  "++opt.static_color_grad_protect_min_abs_signed_score=$STATIC_COLOR_PROTECT_MIN_ABS_SIGNED_SCORE" \
  "++opt.static_color_grad_protect_min_visible_hits=$STATIC_COLOR_PROTECT_MIN_VISIBLE_HITS" \
  > "$LOG_DIR/train.log" 2>&1
log_event "train_done" "status=0"

render_eval() {
  local label="$1"
  local ckpt="$2"
  local config_dir="$3"
  local render_exp="$EXP_ROOT/$label"
  log_event "render_start" "$label"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$config_dir" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$EVAL_TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=false" \
    "++render_export_refine=false" \
    "++explicit_binding_render_preset=v307_adopted_geometry" \
    "++explicit_binding_adopted_component_csv=$SELECTED_CSV" \
    "++explicit_binding_adopted_point_csv=$POINT_CSV" \
    "++explicit_binding_adopted_center_strength=0.45" \
    "++explicit_binding_adopted_outer_px=0.35" \
    "++explicit_binding_adopted_component_required=true" \
    "++explicit_binding_adopted_improvement_guard=true" \
    "++explicit_binding_adopted_max_points=96" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/render_$label" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${label}.log" 2>&1
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
  "$PYTHON_BIN" - "$SUMMARY" "$REFERENCE_SUMMARY" "$label" "$ckpt" "$render_exp" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
reference_path = Path(sys.argv[2])
label = sys.argv[3]
ckpt = sys.argv[4]
render_exp = Path(sys.argv[5])

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

refs = {}
with reference_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        refs[row["label"]] = {k: float(row[k]) for k in metrics}
baseline = refs["baseline"]
selected = refs["selected"]
db = {k: metrics[k] - baseline[k] for k in metrics}
ds = {k: metrics[k] - selected[k] for k in metrics}
strict = (
    db["inner"] < -0.05
    and db["outer"] <= 0.0
    and db["fg"] <= 0.0
    and db["boundary"] <= 0.0
    and db["edge"] <= 0.0
    and db["hard"] < -0.000001
)
do_no_harm = (
    ds["fg"] <= 0.0
    and ds["boundary"] <= 0.0
    and ds["edge"] <= 0.0
    and ds["inner"] <= 0.0
    and ds["outer"] <= 0.0
    and ds["hard"] <= 0.0
)
status = "strict_and_teacher_safe" if strict and do_no_harm else ("strict_vs_baseline_only" if strict else "rejected")
with summary_path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow([
        label,
        ckpt,
        str(render_exp),
        f'{metrics["fg"]:.8f}',
        f'{metrics["boundary"]:.8f}',
        f'{metrics["edge"]:.6f}',
        f'{metrics["inner"]:.4f}',
        f'{metrics["outer"]:.4f}',
        f'{metrics["hard"]:.8f}',
        f'{db["fg"]:.8f}',
        f'{db["boundary"]:.8f}',
        f'{db["edge"]:.6f}',
        f'{db["inner"]:.4f}',
        f'{db["outer"]:.4f}',
        f'{db["hard"]:.8f}',
        f'{ds["fg"]:.8f}',
        f'{ds["boundary"]:.8f}',
        f'{ds["edge"]:.6f}',
        f'{ds["inner"]:.4f}',
        f'{ds["outer"]:.4f}',
        f'{ds["hard"]:.8f}',
        int(strict),
        int(do_no_harm),
        status,
    ])
print(status, flush=True)
PY
  log_event "render_done" "$label"
}

for step in ${TRAIN_CHECKPOINT_STEPS//,/ }; do
  ckpt="$TRAIN_EXP/ckpt$((BASE_ITER + step)).pth"
  if [ -f "$ckpt" ]; then
    render_eval "ckpt$((BASE_ITER + step))" "$ckpt" "$TRAIN_EXP/.hydra"
  else
    log_event "ckpt_missing" "$ckpt"
  fi
done

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event "done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
