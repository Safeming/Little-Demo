#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
RUN_ID="${RUN_ID:-formal_377_v338_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v338_semantic_train_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
TRAIN_STEPS="${TRAIN_STEPS:-${ITERATIONS:-2000}}"
TEST_INTERVAL="${TEST_INTERVAL:-1000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[$TRAIN_STEPS]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[$TRAIN_STEPS]}"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" "$COMPACT_MAPPING_FILE" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_DIR" "$HYDRA_RUN_DIR"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
)

echo "EXP_DIR=$EXP_DIR"
echo "BASE_EXP=$BASE_EXP"
echo "BASE_CKPT=$BASE_CKPT"
echo "formal_preset=v338_temporal_selector_grow_only_guard"
echo "PARSER_ROOT=$PARSER_ROOT"
echo "TRAIN_STEPS=$TRAIN_STEPS"

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
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING_FILE" \
  "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
  "++model.gaussian.semantic_logits_adapter_enable=true" \
  "++model.gaussian.semantic_logits_adapter_max_delta=1.0" \
  "++model.gaussian.semantic_asset_logits_adapter_enable=true" \
  "++model.gaussian.semantic_asset_logits_adapter_max_delta=1.0" \
  "++opt.stageB_semantic_loss_enable=true" \
  "++opt.stageB_semantic_adapter_only_train=true" \
  "++opt.stageB_semantic_use_opacity_support=true" \
  "++opt.stageB_semantic_min_valid_pixels=64" \
  "++opt.stageB_semantic_body_cloth_weight=1.0" \
  "++opt.stageB_semantic_compact_weight=1.0" \
  "++opt.stageB_semantic_parent_consistency_weight=0.36" \
  "++opt.stageB_semantic_exclusive_weight=0.12" \
  "++opt.stageB_semantic_adapter_smooth_weight=0.05" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.lambda_binding_semantic_asset_adapter_reg=0.001" \
  "++opt.lambda_binding_layer_logits_adapter_reg=0.0" \
  "opt.iterations=$TRAIN_STEPS" \
  "test_interval=$TEST_INTERVAL" \
  "save_iterations=$SAVE_ITERATIONS" \
  "checkpoint_iterations=$CHECKPOINT_ITERATIONS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "++opt.binding_layer_logits_lr=0.0" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.semantic_asset_region_logits_lr=0.0005" \
  "++opt.semantic_asset_compact_logits_lr=0.0005" \
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
  "resume.disable_densify_on_resume=true" \
  "resume.disable_opacity_reset_on_resume=true" \
  "resume.use_checkpoint_iteration_as_offset=true" \
  "resume.clear_boundary_tags_on_resume=false" \
  "resume.clear_binding_state_on_resume=false" \
  "++resume.allow_start_load_ckpt_mismatch=false" \
  "export_interpretability=false" \
  "export_semantic_editable_assets=false" \
  "++render_export_refine=false" \
  "hydra.run.dir=$HYDRA_RUN_DIR" \
  "wandb_disable=true" \
  "$@"

echo "TRAIN_EXP_DIR=$EXP_DIR"
