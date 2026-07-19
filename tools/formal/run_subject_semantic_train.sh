#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
SUBJECT="${SUBJECT:?set SUBJECT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:?set BASE_EXP}"
BASE_CKPT="${BASE_CKPT:?set BASE_CKPT}"
EXP_DIR="${EXP_DIR:?set EXP_DIR}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
ALLOWED_CAMERA_IDS="${ALLOWED_CAMERA_IDS:-$TRAIN_VIEWS_SPEC}"
ALLOWED_FRAME_IDS="${ALLOWED_FRAME_IDS:-[0,120,240,360,480]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[17,18,19,20]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[60,540,240]}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
TEST_INTERVAL="${TEST_INTERVAL:-1000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[$TRAIN_STEPS]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[$TRAIN_STEPS]}"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" \
  "$DATA_ROOT/$SUBJECT" "$PARSER_ROOT/$SUBJECT/mask_cihp" "$COMPACT_MAPPING_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_DIR" "$HYDRA_RUN_DIR"

env \
  "CUDA_VISIBLE_DEVICES=$GPU" \
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB" \
  "PYTHONUNBUFFERED=1" \
  "$PYTHON_BIN" train.py \
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
  "dataset.parsing_prior.enable=true" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
  "dataset.parsing_prior.parser_layout=cihp_subject" \
  "dataset.parsing_prior.use_direct_parser_labels=true" \
  "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING_FILE" \
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
  "++opt.stageB_semantic_allowed_camera_ids=$ALLOWED_CAMERA_IDS" \
  "++opt.stageB_semantic_allowed_frame_ids=$ALLOWED_FRAME_IDS" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.lambda_binding_semantic_asset_adapter_reg=0.001" \
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
  "opt.pose_correction_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "++opt.texture_lr=0.0" \
  "++opt.tex_latent_lr=0.0" \
  "++opt.semantic_asset_region_logits_lr=0.0005" \
  "++opt.semantic_asset_compact_logits_lr=0.0005" \
  "++opt.percent_dense=0.0" \
  "opt.densify_from_iter=999999999" \
  "opt.densify_until_iter=0" \
  "opt.lambda_l1=0.0" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_mask=0.0" \
  "opt.lambda_opacity=0.0" \
  "++resume.reset_semantic_state_on_resume=true" \
  "resume.disable_densify_on_resume=true" \
  "resume.disable_opacity_reset_on_resume=true" \
  "resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,camera_geometry.,texture.shadow_handoff_approved]" \
  "export_interpretability=false" \
  "export_semantic_editable_assets=false" \
  "hydra.run.dir=$HYDRA_RUN_DIR" \
  "wandb_disable=true" \
  "$@"

echo "TRAIN_EXP_DIR=$EXP_DIR"
