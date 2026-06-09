#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
ASSET_JSON="${ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v387_runtime_bounded_marginal_selector_v387_runtime_bounded_marginal_selector_20260530_094841_bjt/v387_selector/assets/v387_final_runtime_bounded_marginal_selector_asset.json}"

RUN_ID="${RUN_ID:-v388_raw_config_boundary_residual_tune_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v388_raw_config_boundary_residual_tune_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v388_raw_config_boundary_residual_tune_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
TEST_INTERVAL="${TEST_INTERVAL:-500}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[1000,1500,2000]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[500,1000,1500,2000]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$ASSET_JSON" "$DATA_ROOT" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_DIR" "$LOG_DIR" "$HYDRA_RUN_DIR"

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+85 minutes' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<INFO
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
ASSET_JSON=$ASSET_JSON
EXP_DIR=$EXP_DIR
LOG_DIR=$LOG_DIR
TRAIN_STEPS=$TRAIN_STEPS
CONFIG_CONTRACT=raw_gate_base_config_consistent
INFO

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

EXTRA_TRAIN_ARGS_ARRAY=()
if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
  read -r -a EXTRA_TRAIN_ARGS_ARRAY <<< "$EXTRA_TRAIN_ARGS"
fi

env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "start_checkpoint=$BASE_CKPT" \
  "load_ckpt=$BASE_CKPT" \
  "exp_dir=$EXP_DIR" \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.subject=CoreView_377" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
  "++pipeline.split_child_component_enable=true" \
  "++pipeline.split_child_component_asset_json=$ASSET_JSON" \
  "++pipeline.split_child_component_action_required=false" \
  "++pipeline.split_child_component_opacity=0.045" \
  "++pipeline.split_child_component_radius_scale=1.0" \
  "++pipeline.split_child_component_max_children=-1" \
  "++model.gaussian.semantic_logits_adapter_enable=true" \
  "++model.gaussian.semantic_logits_adapter_max_delta=2.25" \
  "++model.gaussian.semantic_asset_logits_adapter_enable=false" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.stageB_semantic_adapter_only_train=false" \
  "opt.iterations=$TRAIN_STEPS" \
  "test_interval=$TEST_INTERVAL" \
  "test_iterations=$CHECKPOINT_ITERATIONS" \
  "save_iterations=$SAVE_ITERATIONS" \
  "checkpoint_iterations=$CHECKPOINT_ITERATIONS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.000090" \
  "++opt.boundary_scaling_residual_lr=0.000030" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "++opt.binding_layer_logits_lr=0.0" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.semantic_asset_region_logits_lr=0.0" \
  "++opt.semantic_asset_compact_logits_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "++opt.nr_latent_lr=0.0" \
  "++opt.texture_lr=0.0" \
  "++opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.percent_dense=0.0" \
  "opt.densify_from_iter=999999999" \
  "opt.densify_until_iter=0" \
  "opt.lambda_l1=0.0" \
  "opt.lambda_l1_fg=0.0" \
  "opt.lambda_l1_boundary=0.0" \
  "opt.lambda_l1_face=0.0" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_perceptual_face=0.0" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "opt.lambda_opacity=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_pose=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "++opt.lambda_binding_rigid=0.0" \
  "++opt.lambda_binding_soft=0.0" \
  "++opt.lambda_binding_canonical=0.0" \
  "++opt.lambda_binding_surface=0.0" \
  "++opt.lambda_binding_entropy=0.0" \
  "++opt.lambda_binding_temporal=0.0" \
  "++opt.lambda_binding_semantic=0.0" \
  "++opt.lambda_binding_body=0.0" \
  "++opt.lambda_binding_cloth=0.0" \
  "++opt.lambda_nr_xyz=0.0" \
  "++opt.lambda_nr_scale=0.0" \
  "++opt.lambda_nr_rot=0.0" \
  "++opt.boundary_region_source=soft_alpha" \
  "++opt.boundary_target_mask_source=soft" \
  "++opt.boundary_aware_enable=true" \
  "++opt.boundary_aware_threshold=0.035" \
  "++opt.boundary_aware_score_power=1.0" \
  "++opt.boundary_aware_freeze_converter_for_boundary_loss=true" \
  "++opt.boundary_image_error_score_enable=true" \
  "++opt.boundary_image_error_score_signed_enable=true" \
  "++opt.boundary_signed_routing_enable=true" \
  "++opt.boundary_tag_schedule_use_local_iteration=true" \
  "++opt.boundary_tag_enable=true" \
  "++opt.boundary_tag_mode=topk_ratio" \
  "++opt.boundary_tag_topk_ratio=0.09" \
  "++opt.boundary_tag_min_ratio=0.07" \
  "++opt.boundary_tag_update_interval=40" \
  "++opt.boundary_tag_update_until_iter=1600" \
  "++opt.boundary_tag_binary=true" \
  "++opt.boundary_tag_use_score_within_subset=true" \
  "++opt.boundary_tag_score_smooth_blend=0.14" \
  "++opt.boundary_tag_score_smooth_k=12" \
  "++opt.boundary_tag_support_k=8" \
  "++opt.boundary_tag_support_threshold=0.24" \
  "++opt.boundary_band_width=5" \
  "++opt.boundary_image_error_score_band_width=5" \
  "++opt.boundary_image_error_score_focus_dilate=2" \
  "++opt.boundary_image_error_score_smooth_blend=0.10" \
  "++opt.boundary_image_error_score_prior_floor=0.10" \
  "++opt.boundary_signed_mixed_loss_scale=0.08" \
  "++opt.boundary_signed_shrink_loss_scale=0.95" \
  "++opt.boundary_signed_grow_loss_scale=0.28" \
  "++opt.boundary_signed_share_gain=1.02" \
  "++opt.boundary_signed_share_power=1.12" \
  "++opt.boundary_signed_shrink_share_gain=1.10" \
  "++opt.boundary_signed_shrink_share_power=1.24" \
  "++opt.boundary_aware_opacity_scale=0.0" \
  "++opt.boundary_aware_scaling_scale=0.0" \
  "++opt.boundary_aware_xyz_scale=0.0" \
  "++opt.boundary_aware_rotation_scale=0.0" \
  "++opt.boundary_aware_feature_dc_scale=0.0" \
  "++opt.boundary_aware_feature_rest_scale=0.0" \
  "++opt.boundary_aware_boundary_opacity_residual_scale=0.45" \
  "++opt.boundary_aware_boundary_scaling_residual_scale=0.28" \
  "++opt.lambda_boundary_opacity_residual_reg=0.00075" \
  "++opt.lambda_boundary_scaling_residual_reg=0.00018" \
  "++opt.lambda_boundary_opacity_residual_smooth=0.0010" \
  "++opt.lambda_boundary_scaling_residual_smooth=0.00070" \
  "++opt.boundary_residual_smooth_k=10" \
  "++opt.boundary_residual_smooth_distance_quantile=0.50" \
  "++opt.silhouette_outer_ring_width=4" \
  "++opt.silhouette_outer_shell_start_width=1" \
  "++opt.silhouette_outer_shell_end_width=10" \
  "++opt.silhouette_outer_shell_soft_weights=true" \
  "++opt.silhouette_outer_shell_weight_min=0.22" \
  "++opt.lambda_silhouette_outer=[0.002,1,0.004,800,0.005]" \
  "++opt.lambda_silhouette_outer_shell=[0.008,1,0.014,800,0.018,1600,0.020]" \
  "++opt.lambda_silhouette_upper_torso_outer_shell=[0.008,1,0.018,800,0.022,1600,0.024]" \
  "++opt.lambda_silhouette_shoulder_arm_outer_shell=[0.006,1,0.014,800,0.018,1600,0.020]" \
  "++opt.lambda_silhouette_outer_spike=[0.004,1,0.010,1000,0.014]" \
  "++opt.silhouette_outer_spike_opacity_threshold=0.055" \
  "++opt.silhouette_outer_spike_support_threshold=0.30" \
  "++opt.silhouette_outer_spike_power=1.08" \
  "++opt.silhouette_inner_ring_width=4" \
  "++opt.lambda_silhouette_inner=[0.004,1,0.008,800,0.010,1600,0.012]" \
  "resume.disable_densify_on_resume=true" \
  "resume.disable_opacity_reset_on_resume=true" \
  "resume.use_checkpoint_iteration_as_offset=true" \
  "resume.clear_boundary_tags_on_resume=false" \
  "resume.clear_binding_state_on_resume=false" \
  "++resume.allow_start_load_ckpt_mismatch=false" \
  "export_interpretability=false" \
  "export_semantic_editable_assets=false" \
  "++export_opacity_maps=true" \
  "++render_export_refine=false" \
  "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
  "hydra.run.dir=$HYDRA_RUN_DIR" \
  "wandb_disable=true" \
  "${EXTRA_TRAIN_ARGS_ARRAY[@]}" \
  > "$LOG_DIR/train.log" 2>&1

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
echo "EXP_DIR=$EXP_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
