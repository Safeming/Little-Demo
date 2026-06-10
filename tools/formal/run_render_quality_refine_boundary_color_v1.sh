#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
SUBJECT="${SUBJECT:-CoreView_377}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/formal/377_v395_dense_canary_semantic_train_formal_377_v395_dense_canary_semantic_train_v395_dense_canary_selector_batch_20260531_232920_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt139910.pth}"
RUN_ID="${RUN_ID:-render_quality_refine_boundary_color_v1_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/${SUBJECT}_render_quality_refine_boundary_color_v1_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
TRAIN_STEPS="${TRAIN_STEPS:-800}"
TEST_INTERVAL="${TEST_INTERVAL:-400}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[$TRAIN_STEPS]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[$TRAIN_STEPS]}"

FEATURE_LR="${FEATURE_LR:-0.00008}"
OPACITY_LR="${OPACITY_LR:-0.00006}"
SCALING_LR="${SCALING_LR:-0.0}"
BOUNDARY_OPACITY_RESIDUAL_LR="${BOUNDARY_OPACITY_RESIDUAL_LR:-0.000015}"
BOUNDARY_SCALING_RESIDUAL_LR="${BOUNDARY_SCALING_RESIDUAL_LR:-0.000004}"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
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
echo "SUBJECT=$SUBJECT"
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
  "dataset.subject=$SUBJECT" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.stageB_semantic_adapter_only_train=false" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.semantic_asset_region_logits_lr=0.0" \
  "++opt.semantic_asset_compact_logits_lr=0.0" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.lambda_binding_semantic_asset_adapter_reg=0.0" \
  "++opt.binding_layer_logits_lr=0.0" \
  "opt.iterations=$TRAIN_STEPS" \
  "test_interval=$TEST_INTERVAL" \
  "save_iterations=$SAVE_ITERATIONS" \
  "checkpoint_iterations=$CHECKPOINT_ITERATIONS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=$FEATURE_LR" \
  "opt.opacity_lr=$OPACITY_LR" \
  "opt.scaling_lr=$SCALING_LR" \
  "opt.rotation_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=$BOUNDARY_OPACITY_RESIDUAL_LR" \
  "++opt.boundary_scaling_residual_lr=$BOUNDARY_SCALING_RESIDUAL_LR" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "++opt.nr_latent_lr=0.0" \
  "++opt.texture_lr=0.0" \
  "++opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "opt.lambda_l1=0.035" \
  "opt.lambda_l1_fg=0.105" \
  "opt.lambda_l1_boundary=0.145" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.020" \
  "opt.lambda_perceptual_face=0.006" \
  "opt.lambda_l1_face=0.020" \
  "opt.lambda_l1_shoulder_arm=0.016" \
  "opt.lambda_l1_waist=0.012" \
  "opt.lambda_edge_face=0.004" \
  "opt.lambda_edge_shoulder_arm=0.004" \
  "opt.lambda_edge_waist=0.002" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_boundary_opacity_residual_reg=0.0005" \
  "++opt.lambda_boundary_scaling_residual_reg=0.0005" \
  "opt.lambda_opacity=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_pose=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "opt.percent_dense=0.0" \
  "opt.densify_from_iter=1000000" \
  "opt.densify_until_iter=0" \
  "opt.opacity_reset_interval=1000000" \
  "resume.disable_densify_on_resume=true" \
  "resume.disable_opacity_reset_on_resume=true" \
  "resume.require_no_densify_on_resume=true" \
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
