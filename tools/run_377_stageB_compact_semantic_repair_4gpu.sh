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
RUN_ID="${RUN_ID:-stageB_compact_semantic_v233_skincloth_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
GPUS="${GPUS:-0,1,2,3}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
QUEUE_LAUNCH_STAGGER_SECONDS="${QUEUE_LAUNCH_STAGGER_SECONDS:-25}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"

export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS_PER_JOB"
export BLIS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENCV_FOR_THREADS_NUM="$CPU_THREADS_PER_JOB"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v224c_head_reliable_views_preserve_stageB_headfix_fixed_20260511_103654_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
BASE_ITER="${BASE_ITER:-111710}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_compact_semantic_repair_4gpu_$RUN_ID}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVAL_SUMMARY="$LOG_DIR/eval_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
for cam in $(seq 1 23); do
  parser_dir="$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B${cam}"
  if [ ! -d "$parser_dir" ]; then
    echo "missing parser dir: $parser_dir" >&2
    exit 3
  fi
done

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
ALL23="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]"
HELDOUT_DIAG="[18,19,20,21,22,23]"
SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"
START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"
printf 'name\tkind\texp_dir\trender_exp\teval_dir\tstagec_demo\tstatus\tdetail\n' > "$SUMMARY"
printf 'name\tkind\trender_exp\tregion\tiou\tprecision\trecall\tpred_pixels\tparser_pixels\tstatus\n' > "$EVAL_SUMMARY"
printf 'RUN_ID=%s\nSTART_BJT=%s\nGPUS=%s\nBASE_EXP=%s\nBASE_CKPT=%s\nBASE_ITER=%s\nDATA_ROOT=%s\nPARSER_ROOT=%s\nCOMPACT_MAPPING=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$GPUS" "$BASE_EXP" "$BASE_CKPT" "$BASE_ITER" \
  "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

Purpose:
  Repair StageB compact semantic for true StageC. This v233 line keeps the v232
  shoes prior, but adds explicit skin/cloth competition guards for bare arms and
  lower legs. The branches vary skin floors, upper/lower limb suppressors, and
  parent body/cloth consistency, then train on B1-B20 and evaluate on held-out
  B21/B22/B23 against Hulk parser compact masks.
EOF

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$1" "$2" "$3" <<'PY'
import json, sys, time
from pathlib import Path
path, run_id, gpu, phase, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

summary_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$SUMMARY"
}

split_csv() {
  local value="$1"
  local IFS=','
  read -ra _items <<< "$value"
  printf '%s\n' "${_items[@]}"
}

gpu_stats() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local gpu="$1"
  local name="$2"
  if [ "$WAIT_FOR_FREE_GPUS" != "1" ] || [ "$SMOKE" = "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats "$gpu")
    used="${used:-0}"
    util="${util:-0}"
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
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_${kind}"

  if [ ! -f "$ckpt" ]; then
    summary_row "$name" "$kind" "$config_exp" "$out_exp" "$eval_dir" "$stagec_dir" "render_skipped" "missing_ckpt=$ckpt"
    log_event "$gpu" "$name" "render_skipped" "missing_ckpt=$ckpt"
    return 0
  fi

  wait_for_gpu "$gpu" "${name}_${kind}_render"
  log_event "$gpu" "$name" "render_start" "$kind:$out_exp"
  write_status "$gpu" "render_start" "$name:$kind"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" render.py \
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
    "++binding_map_hard_fg_opacity_threshold=0.080" \
    "++binding_map_opacity_threshold=0.080" \
    "++binding_map_compact_semantic_opacity_threshold=0.050" \
    "++binding_map_hard_fg_close_kernel=1" \
    "++binding_map_hard_fg_erode_kernel=1" \
    "++binding_map_support_close_kernel=1" \
    "++binding_map_support_erode_kernel=1" \
    "++render_export_opacity_threshold=0.050" \
    "++render_export_close_kernel=1" \
    "++render_export_erode_kernel=1" \
    "export_semantic_editable_assets=true" \
    "semantic_editable_parser_root=$PARSER_ROOT" \
    "semantic_editable_parser_layout=cihp_subject" \
    "semantic_editable_direct_parser_mode=false" \
    "semantic_editable_export_compact_head=true" \
    "semantic_editable_include_binding_summary=true" \
    "++semantic_editable_compact_opacity_threshold=0.050" \
    "++semantic_editable_compact_confidence_threshold=0.0" \
    "+semantic_editable_preview_min_area=18" \
    "hydra.run.dir=$hydra_run_dir" \
    wandb_disable=true > "$LOG_DIR/${name}_render_${kind}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "$kind" "$config_exp" "$out_exp" "$eval_dir" "$stagec_dir" "render_failed" "status=$status"
    log_event "$gpu" "$name" "render_failed" "status=$status"
    return "$status"
  fi

  "$PYTHON_BIN" tools/eval_377_stageB_compact_vs_hulk_parser.py \
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

  "$PYTHON_BIN" tools/make_377_stageC_editable_demo.py \
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
    --gap 8 > "$LOG_DIR/${name}_stagec_compact_${kind}.log" 2>&1 || true

  "$PYTHON_BIN" tools/make_binding_paper_montage.py \
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

train_compact_repair() {
  local name="$1"
  local gpu="$2"
  local train_views="$3"
  local iterations="$4"
  local checkpoint_list="$5"
  shift 5
  local overrides=("$@")
  if [ "$SMOKE" = "1" ]; then
    iterations=4
    checkpoint_list="[2,4]"
  fi

  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  local final_ckpt="$exp_dir/ckpt$((BASE_ITER + iterations)).pth"
  mkdir -p "$exp_dir"

  wait_for_gpu "$gpu" "$name"
  log_event "$gpu" "$name" "train_start" "iterations=$iterations views=$train_views"
  write_status "$gpu" "train_start" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$train_views" \
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
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
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
    "opt.iterations=$iterations" \
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
    "++opt.stageB_semantic_exclusive_weight=0.10" \
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
    "test_interval=600" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0060" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    summary_row "$name" "train" "$exp_dir" "" "" "" "train_failed" "status=$status"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi

  summary_row "$name" "train" "$exp_dir" "" "" "" "ok" "best=$exp_dir/best_ckpt.pth final=$final_ckpt"
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    render_export_eval "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_compact_best" "best"
    render_export_eval "$name" "$gpu" "$exp_dir" "$final_ckpt" "${exp_dir}_render_compact_final" "final"
  fi
}

run_named_job() {
  local gpu="$1"
  local job="$2"
  case "$job" in
    v233a_skincloth_balanced)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.74" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.20" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.024" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.96" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.62" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.08" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.06" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=0.92" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.12" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.40" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.11" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.010" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.040" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.060" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.24" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.08" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.02" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.38" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.74" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.30" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.72" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.28" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.05" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.46" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.34" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.15" \
        "++opt.semantic_region_logits_lr=0.00100" \
        "++opt.semantic_compact_logits_lr=0.00145" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0008" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.42" \
        "++opt.stageB_semantic_compact_weight=1.22" \
        "++opt.stageB_semantic_parent_consistency_weight=0.42" \
        "++opt.stageB_semantic_exclusive_weight=0.14" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.012" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.60,2.05,0.95,0.95,6.20]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,11.00,4.20,1.05,1.05,16.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0070"
      ;;
    v233b_skin_guard_strong)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.76" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.20" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.024" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.97" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.68" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.06" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.16" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=1.00" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.22" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.35" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.14" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.016" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.060" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.085" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.20" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.04" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.58" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.84" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.46" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.64" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.18" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.03" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.66" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.52" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.05" \
        "++opt.semantic_region_logits_lr=0.00095" \
        "++opt.semantic_compact_logits_lr=0.00135" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0009" \
        "++opt.stageB_semantic_ignore_boundary_width=4" \
        "++opt.stageB_semantic_opacity_threshold=0.030" \
        "++opt.stageB_semantic_body_cloth_weight=0.48" \
        "++opt.stageB_semantic_compact_weight=1.12" \
        "++opt.stageB_semantic_parent_consistency_weight=0.50" \
        "++opt.stageB_semantic_exclusive_weight=0.18" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.016" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.40,2.45,0.90,0.90,6.00]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,10.50,5.20,1.00,1.00,15.50]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0065"
      ;;
    v233c_parent_boundary_strong)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.72" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.18" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.022" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.95" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.58" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.08" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.024" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.24" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.10" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=0.96" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.16" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.36" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.12" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.012" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.050" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.070" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.22" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.06" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.01" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.48" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.80" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.40" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.68" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.22" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.04" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.58" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.46" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.95" \
        "++opt.semantic_region_logits_lr=0.00085" \
        "++opt.semantic_compact_logits_lr=0.00125" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0011" \
        "++opt.stageB_semantic_ignore_boundary_width=5" \
        "++opt.stageB_semantic_opacity_threshold=0.035" \
        "++opt.stageB_semantic_body_cloth_weight=0.58" \
        "++opt.stageB_semantic_compact_weight=1.02" \
        "++opt.stageB_semantic_parent_consistency_weight=0.62" \
        "++opt.stageB_semantic_exclusive_weight=0.22" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.022" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.20,2.20,1.00,1.00,5.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,10.00,4.80,1.15,1.15,14.50]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0060"
      ;;
    v233d_shoes_preserve_control)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.78" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.22" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.026" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.96" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.70" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.08" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=1.02" \
        "++model.deformer.rigid.compact_semantic_skin_thigh_support=0.90" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_support=1.06" \
        "++model.deformer.rigid.compact_semantic_skin_foot_support=0.35" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.10" \
        "++model.deformer.rigid.compact_semantic_skin_floor=0.008" \
        "++model.deformer.rigid.compact_semantic_skin_arm_floor=0.032" \
        "++model.deformer.rigid.compact_semantic_skin_lower_leg_floor=0.050" \
        "++model.deformer.rigid.compact_semantic_upper_core_support=0.24" \
        "++model.deformer.rigid.compact_semantic_upper_arm_body_support=0.08" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_body_support=0.00" \
        "++model.deformer.rigid.compact_semantic_upper_arm_support=0.02" \
        "++model.deformer.rigid.compact_semantic_upper_arm_suppress=0.30" \
        "++model.deformer.rigid.compact_semantic_upper_forearm_suppress=0.66" \
        "++model.deformer.rigid.compact_semantic_upper_skin_suppress=0.24" \
        "++model.deformer.rigid.compact_semantic_lower_thigh_support=0.76" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_support=0.32" \
        "++model.deformer.rigid.compact_semantic_lower_foot_support=0.05" \
        "++model.deformer.rigid.compact_semantic_lower_lower_leg_suppress=0.38" \
        "++model.deformer.rigid.compact_semantic_lower_skin_suppress=0.28" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.25" \
        "++opt.semantic_region_logits_lr=0.00105" \
        "++opt.semantic_compact_logits_lr=0.00150" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0007" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.36" \
        "++opt.stageB_semantic_compact_weight=1.26" \
        "++opt.stageB_semantic_parent_consistency_weight=0.36" \
        "++opt.stageB_semantic_exclusive_weight=0.12" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.010" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,4.50,1.85,0.92,0.92,6.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.25,10.80,3.80,1.00,1.00,18.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0075"
      ;;
    v232a_prior_balanced)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.55" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.14" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.018" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.92" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.45" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.04" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.12" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.025" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.25" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.10" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.05" \
        "++opt.semantic_region_logits_lr=0.00100" \
        "++opt.semantic_compact_logits_lr=0.00145" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0008" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.28" \
        "++opt.stageB_semantic_compact_weight=1.25" \
        "++opt.stageB_semantic_parent_consistency_weight=0.25" \
        "++opt.stageB_semantic_exclusive_weight=0.10" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.010" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,5.00,1.25,0.95,0.95,6.00]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,12.00,1.55,1.05,1.05,16.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0075"
      ;;
    v232b_shoes_aggressive)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.78" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.22" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.026" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.96" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.70" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.00" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.08" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.020" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.20" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.09" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.40" \
        "++opt.semantic_region_logits_lr=0.00110" \
        "++opt.semantic_compact_logits_lr=0.00170" \
        "++opt.lambda_binding_semantic_adapter_reg=0.00055" \
        "++opt.stageB_semantic_ignore_boundary_width=2" \
        "++opt.stageB_semantic_opacity_threshold=0.020" \
        "++opt.stageB_semantic_body_cloth_weight=0.22" \
        "++opt.stageB_semantic_compact_weight=1.38" \
        "++opt.stageB_semantic_parent_consistency_weight=0.20" \
        "++opt.stageB_semantic_exclusive_weight=0.08" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.008" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,4.60,1.15,0.90,0.90,7.20]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.20,11.00,1.40,1.00,1.00,20.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0080"
      ;;
    v232c_face_skin_balance)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.48" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.12" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.016" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.90" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.38" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.05" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.12" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.045" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.45" \
        "++model.deformer.rigid.compact_semantic_hair_face_suppress=0.60" \
        "++model.deformer.rigid.compact_semantic_skin_arm_support=0.96" \
        "++model.deformer.rigid.compact_semantic_skin_leg_support=0.92" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.16" \
        "++model.gaussian.semantic_logits_adapter_max_delta=2.10" \
        "++opt.semantic_region_logits_lr=0.00095" \
        "++opt.semantic_compact_logits_lr=0.00145" \
        "++opt.lambda_binding_semantic_adapter_reg=0.00075" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.30" \
        "++opt.stageB_semantic_compact_weight=1.30" \
        "++opt.stageB_semantic_parent_consistency_weight=0.28" \
        "++opt.stageB_semantic_exclusive_weight=0.10" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.010" \
        "++opt.stageB_semantic_compact_class_weights=[1.35,5.80,1.80,0.98,0.98,5.30]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.80,15.00,3.20,1.10,1.10,14.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0075"
      ;;
    v232d_prior_conservative)
      train_compact_repair "$job" "$gpu" "$ALL20" 24000 "[6000,12000,18000,24000]" \
        "++model.deformer.rigid.compact_semantic_shoes_ankle_support=0.35" \
        "++model.deformer.rigid.compact_semantic_shoes_thin_support=0.10" \
        "++model.deformer.rigid.compact_semantic_shoes_floor=0.014" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_suppress=0.88" \
        "++model.deformer.rigid.compact_semantic_lower_shoe_post_suppress=0.25" \
        "++model.deformer.rigid.compact_semantic_lower_leg_boost=0.08" \
        "++model.deformer.rigid.compact_semantic_lower_pelvis_boost=0.14" \
        "++model.deformer.rigid.compact_semantic_face_floor=0.015" \
        "++model.deformer.rigid.compact_semantic_face_body_boost=0.15" \
        "++model.deformer.rigid.compact_semantic_skin_torso_support=0.08" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.75" \
        "++opt.semantic_region_logits_lr=0.00085" \
        "++opt.semantic_compact_logits_lr=0.00115" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0010" \
        "++opt.stageB_semantic_ignore_boundary_width=4" \
        "++opt.stageB_semantic_opacity_threshold=0.030" \
        "++opt.stageB_semantic_body_cloth_weight=0.34" \
        "++opt.stageB_semantic_compact_weight=1.15" \
        "++opt.stageB_semantic_parent_consistency_weight=0.34" \
        "++opt.stageB_semantic_exclusive_weight=0.12" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.016" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,4.60,1.35,1.00,1.00,4.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,11.00,1.80,1.10,1.10,12.00]" \
        "test_interval=6000" \
        "opt.grad_clip=0.0065"
      ;;
    v231a_majority_stable_9h)
      train_compact_repair "$job" "$gpu" "$ALL20" 47000 "[11750,23500,35250,47000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.35" \
        "++opt.semantic_region_logits_lr=0.00055" \
        "++opt.semantic_compact_logits_lr=0.00070" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0018" \
        "++opt.stageB_semantic_ignore_boundary_width=7" \
        "++opt.stageB_semantic_opacity_threshold=0.040" \
        "++opt.stageB_semantic_body_cloth_weight=0.46" \
        "++opt.stageB_semantic_compact_weight=0.95" \
        "++opt.stageB_semantic_parent_consistency_weight=0.45" \
        "++opt.stageB_semantic_exclusive_weight=0.14" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.032" \
        "++opt.stageB_semantic_compact_class_weights=[1.25,3.10,1.55,1.10,1.10,2.00]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.50,6.00,2.20,1.25,1.25,3.00]" \
        "test_interval=11750" \
        "opt.grad_clip=0.0050"
      ;;
    v231b_skin_cloth_boundary_9h)
      train_compact_repair "$job" "$gpu" "$ALL20" 47000 "[11750,23500,35250,47000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.45" \
        "++opt.semantic_region_logits_lr=0.00065" \
        "++opt.semantic_compact_logits_lr=0.00085" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0015" \
        "++opt.stageB_semantic_ignore_boundary_width=9" \
        "++opt.stageB_semantic_opacity_threshold=0.045" \
        "++opt.stageB_semantic_body_cloth_weight=0.55" \
        "++opt.stageB_semantic_compact_weight=0.88" \
        "++opt.stageB_semantic_parent_consistency_weight=0.55" \
        "++opt.stageB_semantic_exclusive_weight=0.18" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.038" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,2.80,1.90,1.25,1.25,1.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,5.50,3.20,1.50,1.50,2.50]" \
        "test_interval=11750" \
        "opt.grad_clip=0.0055"
      ;;
    v231c_hair_face_preserve_9h)
      train_compact_repair "$job" "$gpu" "$ALL20" 47000 "[11750,23500,35250,47000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.55" \
        "++opt.semantic_region_logits_lr=0.00075" \
        "++opt.semantic_compact_logits_lr=0.00095" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0014" \
        "++opt.stageB_semantic_ignore_boundary_width=5" \
        "++opt.stageB_semantic_opacity_threshold=0.035" \
        "++opt.stageB_semantic_body_cloth_weight=0.38" \
        "++opt.stageB_semantic_compact_weight=1.05" \
        "++opt.stageB_semantic_parent_consistency_weight=0.36" \
        "++opt.stageB_semantic_exclusive_weight=0.12" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.024" \
        "++opt.stageB_semantic_compact_class_weights=[1.45,4.20,1.35,1.00,1.00,1.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[2.00,8.50,1.80,1.20,1.20,2.60]" \
        "test_interval=11750" \
        "opt.grad_clip=0.0060"
      ;;
    v231d_conservative_smooth_9h)
      train_compact_repair "$job" "$gpu" "$ALL20" 47000 "[11750,23500,35250,47000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.10" \
        "++opt.semantic_region_logits_lr=0.00045" \
        "++opt.semantic_compact_logits_lr=0.00060" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0024" \
        "++opt.stageB_semantic_ignore_boundary_width=11" \
        "++opt.stageB_semantic_opacity_threshold=0.050" \
        "++opt.stageB_semantic_body_cloth_weight=0.62" \
        "++opt.stageB_semantic_compact_weight=0.75" \
        "++opt.stageB_semantic_parent_consistency_weight=0.68" \
        "++opt.stageB_semantic_exclusive_weight=0.22" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.050" \
        "++opt.stageB_semantic_compact_class_weights=[1.20,2.80,1.60,1.20,1.20,1.60]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.40,5.50,2.60,1.50,1.50,2.20]" \
        "test_interval=11750" \
        "opt.grad_clip=0.0045"
      ;;
    v228a_compact_live_all20_balanced|v230a_parentnorm_all20_balanced)
      train_compact_repair "$job" "$gpu" "$ALL20" 3600 "[900,1800,2700,3600]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.55" \
        "++opt.semantic_region_logits_lr=0.00085" \
        "++opt.semantic_compact_logits_lr=0.00105" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0012" \
        "++opt.stageB_semantic_ignore_boundary_width=5" \
        "++opt.stageB_semantic_opacity_threshold=0.035" \
        "++opt.stageB_semantic_body_cloth_weight=0.34" \
        "++opt.stageB_semantic_compact_weight=1.00" \
        "++opt.stageB_semantic_parent_consistency_weight=0.34" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.016" \
        "++opt.stageB_semantic_compact_class_weights=[1.15,3.20,1.25,1.00,1.00,3.60]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.30,7.50,1.50,1.10,1.10,8.50]" \
        "opt.grad_clip=0.0055"
      ;;
    v228b_compact_live_all20_face_shoes_recall|v230b_parentnorm_face_shoes_recall)
      train_compact_repair "$job" "$gpu" "$ALL20" 4200 "[1050,2100,3150,4200]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.80" \
        "++opt.semantic_region_logits_lr=0.00100" \
        "++opt.semantic_compact_logits_lr=0.00135" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0009" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.26" \
        "++opt.stageB_semantic_compact_weight=1.18" \
        "++opt.stageB_semantic_parent_consistency_weight=0.26" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.010" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,4.80,1.15,0.95,0.95,5.20]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.20,12.00,1.35,1.05,1.05,13.00]" \
        "opt.grad_clip=0.0070"
      ;;
    v228c_compact_live_all20_lowdelta_smooth|v230c_parentnorm_lowdelta_smooth)
      train_compact_repair "$job" "$gpu" "$ALL20" 3600 "[900,1800,2700,3600]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.20" \
        "++opt.semantic_region_logits_lr=0.00062" \
        "++opt.semantic_compact_logits_lr=0.00078" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0020" \
        "++opt.stageB_semantic_ignore_boundary_width=7" \
        "++opt.stageB_semantic_opacity_threshold=0.040" \
        "++opt.stageB_semantic_body_cloth_weight=0.42" \
        "++opt.stageB_semantic_compact_weight=0.92" \
        "++opt.stageB_semantic_parent_consistency_weight=0.42" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.030" \
        "++opt.stageB_semantic_compact_class_weights=[1.05,2.80,1.30,1.05,1.05,3.20]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.20,6.50,1.60,1.10,1.10,7.50]" \
        "opt.grad_clip=0.0045"
      ;;
    v228d_compact_live_testview_oracle_diag|v230d_parentnorm_testview_oracle_diag)
      train_compact_repair "$job" "$gpu" "$ALL23" 1800 "[450,900,1350,1800]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.65" \
        "++opt.semantic_region_logits_lr=0.00090" \
        "++opt.semantic_compact_logits_lr=0.00120" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0010" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.26" \
        "++opt.stageB_semantic_compact_weight=1.10" \
        "++opt.stageB_semantic_parent_consistency_weight=0.26" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.012" \
        "++opt.train_sample_camera_weights={21:2.20,22:2.20,23:2.20}" \
        "++opt.train_sample_camera_min_prob=0.035" \
        "++opt.train_sample_camera_max_prob=0.220" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,4.20,1.20,0.95,0.95,4.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.20,10.00,1.40,1.05,1.05,11.00]" \
        "opt.grad_clip=0.0065"
      ;;
    v228e_compact_live_heldout_diag|v230e_parentnorm_heldout_diag)
      train_compact_repair "$job" "$gpu" "$HELDOUT_DIAG" 1800 "[450,900,1350,1800]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.65" \
        "++opt.semantic_region_logits_lr=0.00090" \
        "++opt.semantic_compact_logits_lr=0.00120" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0010" \
        "++opt.stageB_semantic_ignore_boundary_width=3" \
        "++opt.stageB_semantic_opacity_threshold=0.025" \
        "++opt.stageB_semantic_body_cloth_weight=0.26" \
        "++opt.stageB_semantic_compact_weight=1.10" \
        "++opt.stageB_semantic_parent_consistency_weight=0.26" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.012" \
        "++opt.train_sample_camera_weights={21:2.40,22:2.40,23:2.40}" \
        "++opt.train_sample_camera_min_prob=0.060" \
        "++opt.train_sample_camera_max_prob=0.260" \
        "++opt.stageB_semantic_compact_class_weights=[1.10,4.20,1.20,0.95,0.95,4.80]" \
        "++opt.stageB_semantic_compact_positive_weights=[1.20,10.00,1.40,1.05,1.05,11.00]" \
        "opt.grad_clip=0.0065"
      ;;
    *)
      log_event "$gpu" "$job" "unknown_job" "skipped"
      return 2
      ;;
  esac
}

queue_gpu() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local launched=0
  for job in "${jobs[@]}"; do
    run_named_job "$gpu" "$job"
    local status=$?
    launched=$((launched + 1))
    if [ "$status" -eq 0 ]; then
      log_event "$gpu" "$job" "job_done" "ok"
    else
      log_event "$gpu" "$job" "job_failed" "status=$status"
    fi
  done
  write_status "$gpu" "queue_done" "launched=$launched"
  log_event "$gpu" "queue" "done" "launched=$launched"
}

launch_queue() {
  local gpu="$1"
  local delay="${2:-0}"
  shift 2
  local jobs=("$@")
  (
    if [ "$delay" -gt 0 ]; then
      log_event "$gpu" "queue" "launch_delay" "${delay}s"
      sleep "$delay"
    fi
    queue_gpu "$gpu" "${jobs[@]}"
  ) &
  local pid=$!
  printf 'gpu%s_queue\t%s\t%s\n' "$gpu" "$gpu" "$pid" >> "$PIDS"
}

mapfile -t SELECTED_GPUS < <(split_csv "$GPUS")
if [ "${#SELECTED_GPUS[@]}" -lt 4 ]; then
  echo "expected four GPUs in GPUS, got: $GPUS" >&2
  exit 4
fi

GPU0="${SELECTED_GPUS[0]}"
GPU1="${SELECTED_GPUS[1]}"
GPU2="${SELECTED_GPUS[2]}"
GPU3="${SELECTED_GPUS[3]}"

launch_queue "$GPU0" 0 v233a_skincloth_balanced
launch_queue "$GPU1" "$QUEUE_LAUNCH_STAGGER_SECONDS" v233b_skin_guard_strong
launch_queue "$GPU2" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 2))" v233c_parent_boundary_strong
launch_queue "$GPU3" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 3))" v233d_shoes_preserve_control

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "EVAL_SUMMARY=$EVAL_SUMMARY"
cat "$PIDS"

wait
log_event "all" "queue" "all_done" "summary=$SUMMARY eval_summary=$EVAL_SUMMARY"
echo "SUMMARY=$SUMMARY"
echo "EVAL_SUMMARY=$EVAL_SUMMARY"
