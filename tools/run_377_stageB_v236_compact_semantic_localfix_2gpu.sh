#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageB_v236_compact_localfix_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPUS="${GPUS:-2,3}"
SEED="${SEED:--1}"
ITERATIONS="${ITERATIONS:-6000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2000,4000,6000}"
DO_RENDER="${DO_RENDER:-1}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-5000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-25}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
QUEUE_LAUNCH_STAGGER_SECONDS="${QUEUE_LAUNCH_STAGGER_SECONDS:-20}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-7}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"
BASE_ITER="${BASE_ITER:-135710}"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v236_compact_semantic_localfix_${RUN_ID}}"
SUMMARY="$LOG_DIR/summary.tsv"
EVAL_SUMMARY="$LOG_DIR/eval_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
mkdir -p "$LOG_DIR"

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CHECKPOINT_LIST="[$CHECKPOINT_STEPS]"
SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tkind\texp_dir\trender_exp\teval_dir\tstagec_demo\tstatus\tdetail\n' > "$SUMMARY"
printf 'name\tkind\trender_exp\tregion\tiou\tprecision\trecall\tpred_pixels\tparser_pixels\tstatus\n' > "$EVAL_SUMMARY"
printf 'name\tgpu\tpid\n' > "$PIDS"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPUS=$GPUS"
  echo "ITERATIONS=$ITERATIONS"
  echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "BASE_ITER=$BASE_ITER"
  echo "PURPOSE=v236 compact semantic local fix only; opacity/geometry/texture frozen"
  echo "DEBUG=StageBSemanticDbg every 250 train iterations with per-class IoU/precision/recall/pred/target/prob"
} | tee "$LOG_DIR/run_info.txt"

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

summary_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$SUMMARY"
}

gpu_stats() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local gpu="$1"
  local name="$2"
  if [ "$WAIT_FOR_FREE_GPUS" != "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats "$gpu")
    used="${used:-999999}"
    util="${util:-100}"
    if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
      log_event "$gpu" "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$gpu" "$name" "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

append_eval_summary() {
  local name="$1"
  local kind="$2"
  local render_exp="$3"
  local eval_dir="$4"
  local status="$5"
  "$PYTHON_BIN" - "$name" "$kind" "$render_exp" "$eval_dir" "$status" "$EVAL_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
name, kind, render_exp, eval_dir, status, out = sys.argv[1:]
summary = Path(eval_dir) / "summary.json"
if not summary.exists():
    raise SystemExit(0)
data = json.loads(summary.read_text())
with open(out, "a", encoding="utf-8") as f:
    for region, metrics in data.items():
        row = [
            name,
            kind,
            render_exp,
            region,
            f"{float(metrics.get('iou', 0.0)):.6f}",
            f"{float(metrics.get('precision', 0.0)):.6f}",
            f"{float(metrics.get('recall', 0.0)):.6f}",
            f"{float(metrics.get('pred_pixels', 0.0)):.1f}",
            f"{float(metrics.get('parser_pixels', 0.0)):.1f}",
            status,
        ]
        f.write("\t".join(row) + "\n")
PY
}

render_export_eval() {
  local name="$1"
  local gpu="$2"
  local config_exp="$3"
  local ckpt="$4"
  local out_exp="$5"
  local kind="$6"
  local eval_dir="$out_exp/diagnostics/compact_vs_hulk_parser"
  local stagec_dir="$out_exp/test-view/stageC_compact_demo_selected"

  if [ ! -f "$ckpt" ]; then
    summary_row "$name" "$kind" "$config_exp" "$out_exp" "$eval_dir" "$stagec_dir" "render_skipped" "missing_ckpt=$ckpt"
    log_event "$gpu" "$name" "render_skipped" "missing_ckpt=$ckpt"
    return 0
  fi

  wait_for_gpu "$gpu" "${name}_${kind}_render"
  log_event "$gpu" "$name" "render_start" "$kind:$out_exp"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  "$PYTHON_BIN" "$ROOT/render.py" \
    --config-path "$config_exp/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$out_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "export_interpretability=true" \
    "++binding_map_names=$BINDING_MAPS" \
    "++binding_map_use_opacity_mask=true" \
    "++binding_map_hard_fg_opacity_threshold=0.030" \
    "++binding_map_opacity_threshold=0.030" \
    "++binding_map_compact_semantic_opacity_threshold=0.025" \
    "++binding_map_hard_fg_close_kernel=5" \
    "++binding_map_hard_fg_erode_kernel=1" \
    "++binding_map_support_close_kernel=5" \
    "++binding_map_support_erode_kernel=1" \
    "++render_export_opacity_threshold=0.025" \
    "++render_export_close_kernel=5" \
    "++render_export_erode_kernel=1" \
    "export_semantic_editable_assets=true" \
    "semantic_editable_parser_root=$PARSER_ROOT" \
    "semantic_editable_parser_layout=cihp_subject" \
    "semantic_editable_direct_parser_mode=false" \
    "semantic_editable_export_compact_head=true" \
    "semantic_editable_include_binding_summary=true" \
    "++semantic_editable_compact_opacity_threshold=0.025" \
    "++semantic_editable_compact_confidence_threshold=0.0" \
    "+semantic_editable_preview_min_area=18" \
    "hydra.run.dir=$LOG_DIR/hydra_${name}_${kind}_render" \
    "wandb_disable=true" \
    > "$LOG_DIR/${name}_render_${kind}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "$kind" "$config_exp" "$out_exp" "$eval_dir" "$stagec_dir" "render_failed" "status=$status"
    log_event "$gpu" "$name" "render_failed" "status=$status"
    return "$status"
  fi

  "$PYTHON_BIN" "$ROOT/tools/eval_377_stageB_compact_vs_hulk_parser.py" \
    --exp-dir "$out_exp" \
    --split test-view \
    --parser-root "$PARSER_ROOT" \
    --dataset-root "$DATA_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --out-dir "$eval_dir" \
    --select "${SELECT[@]}" \
    --panel-width 190 \
    --header-height 30 \
    --gap 6 > "$LOG_DIR/${name}_eval_${kind}.log" 2>&1 || true

  "$PYTHON_BIN" "$ROOT/tools/make_377_stageC_editable_demo.py" \
    --exp-dir "$out_exp" \
    --split test-view \
    --output-dir "$stagec_dir" \
    --select "${SELECT[@]}" \
    --mask-source compact \
    --parser-root "$PARSER_ROOT" \
    --dataset-root "$DATA_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --panel-width 220 \
    --header-height 34 \
    --gap 8 > "$LOG_DIR/${name}_stagec_${kind}.log" 2>&1 || true

  "$PYTHON_BIN" "$ROOT/tools/make_binding_paper_montage.py" \
    --exp-dir "$out_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
    --select "${SELECT[@]}" \
    --output-dir "$out_exp/test-view/paper_montages_selected" > "$LOG_DIR/${name}_montage_${kind}.log" 2>&1 || true

  append_eval_summary "$name" "$kind" "$out_exp" "$eval_dir" "ok"
  summary_row "$name" "$kind" "$config_exp" "$out_exp" "$eval_dir" "$stagec_dir" "ok" "$ckpt"
  log_event "$gpu" "$name" "render_done" "$kind:$out_exp"
}

train_branch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local overrides=("$@")
  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local hydra_run_dir="$LOG_DIR/hydra_${name}_train"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + ITERATIONS)).pth"
  mkdir -p "$exp_dir"

  wait_for_gpu "$gpu" "$name"
  log_event "$gpu" "$name" "train_start" "iterations=$ITERATIONS checkpoints=$CHECKPOINT_LIST"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  "$PYTHON_BIN" "$ROOT/train.py" \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$ALL20" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,60]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "dataset.parsing_prior.skip_empty_samples=false" \
    "dataset.parsing_prior.skip_empty_min_pixels=96" \
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$exp_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "hydra.run.dir=$hydra_run_dir" \
    "++resume.allow_partial_converter_load=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=true" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.gaussian.semantic_logits_adapter_enable=true" \
    "++model.gaussian.semantic_logits_adapter_compact_classes=6" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$ITERATIONS" \
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
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_debug_interval=250" \
    "++opt.stageB_semantic_debug_regions=[hair,face,skin,upper,lower,shoes]" \
    "++opt.stageB_semantic_debug_pred_threshold=0.5" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_use_opacity_support=true" \
    "++opt.stageB_semantic_min_valid_pixels=64" \
    "++opt.stageB_semantic_body_cloth_bce_weight=1.0" \
    "++opt.stageB_semantic_body_cloth_dice_weight=0.75" \
    "++opt.stageB_semantic_compact_bce_weight=0.45" \
    "++opt.stageB_semantic_compact_dice_weight=0.70" \
    "++opt.stageB_semantic_compact_ce_weight=0.70" \
    "++opt.stageB_semantic_parent_target_use_compact_groups=true" \
    "++opt.stageB_semantic_parent_body_indices=[0,1,2]" \
    "++opt.stageB_semantic_parent_cloth_indices=[3,4,5]" \
    "++opt.stageB_semantic_parent_consistency_enable=true" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.018" \
    "++opt.train_sample_camera_max_prob=0.130" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=2" \
    "opt.lambda_l1=0.0" \
    "opt.lambda_l1_fg=0.0" \
    "opt.lambda_l1_boundary=0.0" \
    "opt.lambda_l1_face=0.0" \
    "opt.lambda_l1_shoulder_arm=0.0" \
    "opt.lambda_l1_waist=0.0" \
    "opt.lambda_perceptual=0.0" \
    "opt.lambda_edge_face=0.0" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_mask_boundary_hard=0.0" \
    "++opt.lambda_silhouette_outer=0.0" \
    "++opt.lambda_silhouette_inner=0.0" \
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
    "best_metric=lpips_fg" \
    "best_metric_mode=min" \
    "best_metric_source=best_eval" \
    "test_interval=2000" \
    "test_iterations=$CHECKPOINT_LIST" \
    "save_iterations=$CHECKPOINT_LIST" \
    "checkpoint_iterations=$CHECKPOINT_LIST" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0055" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "train" "$exp_dir" "" "" "" "train_failed" "status=$status"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi

  summary_row "$name" "train" "$exp_dir" "" "" "" "ok" "final=$final_ckpt"
  log_event "$gpu" "$name" "train_done" "$final_ckpt"
  if [ "$DO_RENDER" = "1" ]; then
    render_export_eval "$name" "$gpu" "$exp_dir" "$final_ckpt" "${exp_dir}_render_compact_final" "final"
  fi
}

run_branch() {
  local gpu="$1"
  local name="$2"
  case "$name" in
    v236a_lower_upper_cleanup)
      train_branch "$name" "$gpu" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.78" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.22" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.026" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.97" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.72" \
        "++model.deformer.rigid.compact_semantic_lower_head_suppress=0.62" \
        "++model.deformer.rigid.compact_semantic_lower_torso_support=0.03" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.07" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.10" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=0.96" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.18" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.34" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.14" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.014" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.052" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.082" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.27" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.04" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.48" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.80" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.42" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.70" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.20" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.03" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.64" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.52" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.70" \
        "++opt.semantic_region_logits_lr=0.00070" \
        "++opt.semantic_compact_logits_lr=0.00100" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0012" \
        "++opt.stageB_semantic_ignore_boundary_width=4" \
        "++opt.stageB_semantic_opacity_threshold=0.030" \
        "++opt.stageB_semantic_body_cloth_weight=0.55" \
        "++opt.stageB_semantic_compact_weight=1.08" \
        "++opt.stageB_semantic_parent_consistency_weight=0.62" \
        "++opt.stageB_semantic_exclusive_weight=0.24" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.022" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.30,2.45,1.08,1.08,6.10]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,10.50,5.30,1.50,1.45,15.00]"
      ;;
    v236b_shoes_recall_local)
      train_branch "$name" "$gpu" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.88" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.30" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.034" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.99" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.80" \
        "++model.deformer.rigid.compact_semantic_lower_head_suppress=0.52" \
        "++model.deformer.rigid.compact_semantic_lower_torso_support=0.04" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.07" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.04" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=0.91" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.08" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.30" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.10" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.010" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.036" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.056" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.24" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.06" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.01" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.36" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.72" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.30" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.74" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.26" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.03" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.52" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.40" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.95" \
        "++opt.semantic_region_logits_lr=0.00085" \
        "++opt.semantic_compact_logits_lr=0.00125" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0010" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.45" \
        "++opt.stageB_semantic_compact_weight=1.22" \
        "++opt.stageB_semantic_parent_consistency_weight=0.48" \
        "++opt.stageB_semantic_exclusive_weight=0.18" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.016" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.30,2.15,1.00,1.00,8.20]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,10.50,4.40,1.18,1.16,22.00]"
      ;;
    *)
      echo "unknown branch: $name" >&2
      return 2
      ;;
  esac
}

IFS=',' read -ra GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt 2 ]; then
  echo "GPUS must contain at least two ids, got: $GPUS" >&2
  exit 2
fi

(
  run_branch "${GPU_LIST[0]}" v236a_lower_upper_cleanup
) &
pid=$!
printf 'v236a_lower_upper_cleanup\t%s\t%s\n' "${GPU_LIST[0]}" "$pid" | tee -a "$PIDS"

sleep "$QUEUE_LAUNCH_STAGGER_SECONDS"

(
  run_branch "${GPU_LIST[1]}" v236b_shoes_recall_local
) &
pid=$!
printf 'v236b_shoes_recall_local\t%s\t%s\n' "${GPU_LIST[1]}" "$pid" | tee -a "$PIDS"

wait_status=0
wait || wait_status=$?

{
  echo "END_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "WAIT_STATUS=$wait_status"
  echo "SUMMARY=$SUMMARY"
  echo "EVAL_SUMMARY=$EVAL_SUMMARY"
} | tee -a "$LOG_DIR/run_info.txt"

exit "$wait_status"
