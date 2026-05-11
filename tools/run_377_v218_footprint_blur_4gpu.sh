#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-v218_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-7200}"
START_EPOCH="${START_EPOCH:-$(date +%s)}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$((START_EPOCH + TIME_BUDGET_SECONDS))}"
MIN_START_SECONDS="${MIN_START_SECONDS:-900}"
SMOKE="${SMOKE:-0}"
SMOKE_MAX_JOBS_PER_GPU="${SMOKE_MAX_JOBS_PER_GPU:-1}"
DO_RENDER="${DO_RENDER:-1}"

V215C_EXP="${V215C_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt}"
V215C410_CKPT="${V215C410_CKPT:-$V215C_EXP/ckpt109410.pth}"
V215C410_RENDER="${V215C410_RENDER:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt_render_quick_ckpt109410}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
V211_ANALYSIS_DIR="${V211_ANALYSIS_DIR:-$ROOT/exp/stageA2/logs/v211_region_view_conflict_20260509_211000_bjt/v211a_region_view_conflict_map}"
V211_PLAN="$V211_ANALYSIS_DIR/region_training_plan.json"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v218_footprint_blur_4gpu_$RUN_ID}"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"
cd "$ROOT" || exit 1

for required in "$V215C_EXP/.hydra/config.yaml" "$V215C410_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$V211_PLAN"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

build_v211_cloth_camera_weights() {
  "$PYTHON_BIN" - "$V211_PLAN" <<'PY'
import json
import sys
from collections import defaultdict

plan = json.loads(open(sys.argv[1], encoding="utf-8").read())
scores = defaultdict(lambda: 0.42)
for item in plan.get("recommended_regions", []):
    region = item.get("region")
    if region == "upper_cloth":
        strength = 1.05
    elif region == "lower_cloth":
        strength = 0.95
    else:
        continue
    for rank, cam in enumerate(item.get("top_cameras", [])[:8]):
        scores[int(cam)] += strength * max(0.32, 1.0 - 0.09 * rank)

values = {cam: scores[cam] for cam in range(1, 21)}
mean = sum(values.values()) / len(values)
weights = {cam: max(0.35, min(2.0, value / max(mean, 1.0e-6))) for cam, value in values.items()}
print("{" + ",".join(f"{cam}:{weights[cam]:.4f}" for cam in range(1, 21)) + "}")
PY
}

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CAM_WEIGHTS="${CAM_WEIGHTS:-$(build_v211_cloth_camera_weights)}"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
DEADLINE_BJT="$(TZ=Asia/Shanghai date -d "@$DEADLINE_EPOCH" '+%F %T BJT')"
HP_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
HP_CROP=(150 35 650 430)

printf 'RUN_ID=%s\nSTART_BJT=%s\nDEADLINE_BJT=%s\nTIME_BUDGET_SECONDS=%s\nMIN_START_SECONDS=%s\nV215C410_CKPT=%s\nV215C410_RENDER=%s\nV211_ANALYSIS_DIR=%s\nALL20=%s\nCAM_WEIGHTS=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$DEADLINE_BJT" "$TIME_BUDGET_SECONDS" "$MIN_START_SECONDS" \
  "$V215C410_CKPT" "$V215C410_RENDER" "$V211_ANALYSIS_DIR" "$ALL20" "$CAM_WEIGHTS" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

v218 purpose:
  Test whether the current clarity ceiling is caused by render footprint /
  Gaussian scale / slight alignment blur rather than missing texture HF loss.

candidates:
  v218a: texture frozen, boundary opacity/scale residual only.
  v218b: texture frozen, tiny global opacity/scale update.
  v218c: texture frozen, tiny upper-body pose update only.
  v218d: mild texture HF polish plus boundary residual.

acceptance:
  Compare against v215c@109410. A useful result should raise heldout
  high-pass ratio and visible cloth/waist detail without increasing
  boundary_l1 or edge distance beyond the v215c stable band.
EOF

printf 'name\tlabel\texp_dir\trender_exp\ttrain_lpips_fg\ttrain_l1_fg\ttrain_psnr_fg\trender_lpips\trender_psnr\trender_ssim\tfg_l1\tboundary_l1\tedge_px\tfg_hp_ratio\tcrop_hp_ratio\tfg_hp_l1\tcrop_hp_l1\tstatus\n' > "$SUMMARY"
printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $((DEADLINE_EPOCH - now))
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$START_EPOCH" "$DEADLINE_EPOCH" "$1" "$2" "$3" <<'PY'
import json
import sys
import time
from pathlib import Path

path, run_id, start_epoch, deadline_epoch, gpu, phase, detail = sys.argv[1:]
start_epoch = int(start_epoch)
deadline_epoch = int(deadline_epoch)
now = int(time.time())
data = {
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_epoch": start_epoch,
    "deadline_epoch": deadline_epoch,
    "now_epoch": now,
    "elapsed_seconds": now - start_epoch,
    "remaining_seconds": max(0, deadline_epoch - now),
}
Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
}

csv_to_hydra_list() {
  local csv="$1"
  if [ -z "$csv" ]; then
    echo "[]"
  else
    echo "[$csv]"
  fi
}

write_summary_row() {
  local name="$1"
  local label="$2"
  local exp_dir="$3"
  local render_exp="$4"
  local status="$5"
  "$PYTHON_BIN" - "$name" "$label" "$exp_dir" "$render_exp" "$status" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

name, label, exp_dir, render_exp, status, summary = sys.argv[1:7]
exp = Path(exp_dir)
render = Path(render_exp) if render_exp else None

def fmt(value, digits=8):
    if value is None:
        return "nan"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"

train = {}
train_path = exp / "best_test_metrics.json"
if train_path.exists():
    train = json.loads(train_path.read_text())

render_metrics = {}
if render is not None:
    render_path = render / "test-view" / "results.npz"
    if render_path.exists():
        data = np.load(render_path)
        for key in ("lpips", "psnr", "ssim"):
            if key in data.files:
                render_metrics[key] = float(np.asarray(data[key]).mean())

contour = {}
highpass = {}
if render is not None:
    contour_path = render / "diagnostics" / "contour_summary.json"
    if contour_path.exists():
        contour = json.loads(contour_path.read_text())
    highpass_path = render / "diagnostics" / "highpass_summary.json"
    if highpass_path.exists():
        highpass = json.loads(highpass_path.read_text())

row = [
    name,
    label,
    str(exp),
    str(render) if render is not None else "",
    fmt(train.get("lpips_fg")),
    fmt(train.get("l1_fg")),
    fmt(train.get("psnr_fg"), 6),
    fmt(render_metrics.get("lpips")),
    fmt(render_metrics.get("psnr"), 6),
    fmt(render_metrics.get("ssim")),
    fmt(contour.get("mean_fg_l1"), 6),
    fmt(contour.get("mean_boundary_l1"), 6),
    fmt(contour.get("mean_edge_symmetric_dist_px"), 4),
    fmt(highpass.get("fg_hp_ratio_mean"), 4),
    fmt(highpass.get("crop_hp_ratio_mean"), 4),
    fmt(highpass.get("fg_hp_l1_mean"), 5),
    fmt(highpass.get("crop_hp_l1_mean"), 5),
    status,
]
with open(summary, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

run_highpass_diag() {
  local name="$1"
  local gpu="$2"
  local render_exp="$3"
  local label="$4"
  local highpass_log="$LOG_DIR/${name}_highpass_${label}.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_highpass_energy.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --select "${HP_SELECT[@]}" \
    --crop "${HP_CROP[@]}" > "$highpass_log" 2>&1
}

render_and_diag() {
  local name="$1"
  local gpu="$2"
  local exp_dir="$3"
  local ckpt_path="$4"
  local label="$5"

  if [ "$DO_RENDER" != "1" ]; then
    return 0
  fi

  local render_exp="${exp_dir}_render_quick_${label}"
  local render_log="$LOG_DIR/${name}_render_${label}.log"
  local contour_log="$LOG_DIR/${name}_contour_${label}.log"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_render_${label}"

  if [ ! -f "$ckpt_path" ]; then
    write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "missing_ckpt"
    return 0
  fi

  log_event "$gpu" "$name" "render_start" "$label"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" render.py \
    --config-path "$exp_dir/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt_path" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "hydra.run.dir=$hydra_run_dir" \
    wandb_disable=true > "$render_log" 2>&1
  local render_status=$?
  if [ "$render_status" -ne 0 ]; then
    write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "render_failed"
    log_event "$gpu" "$name" "render_failed" "$label status=$render_status"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 > "$contour_log" 2>&1
  local contour_status=$?
  if [ "$contour_status" -ne 0 ]; then
    write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "contour_failed"
    log_event "$gpu" "$name" "contour_failed" "$label status=$contour_status"
    return 0
  fi

  run_highpass_diag "$name" "$gpu" "$render_exp" "$label"
  local highpass_status=$?
  if [ "$highpass_status" -ne 0 ]; then
    write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "highpass_failed"
    log_event "$gpu" "$name" "highpass_failed" "$label status=$highpass_status"
    return 0
  fi

  write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "ok"
  log_event "$gpu" "$name" "render_done" "$label"
}

run_one() {
  local name="$1"
  local gpu="$2"
  local iterations="$3"
  local texture_lr="$4"
  local start_ckpt="$5"
  local base_iter="$6"
  local checkpoint_csv="$7"
  local train_views="$8"
  shift 8

  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_csv=""
  fi

  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${name}_${RUN_ID}"
  local train_log="$LOG_DIR/${name}.log"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  local checkpoint_hydra
  checkpoint_hydra="$(csv_to_hydra_list "$checkpoint_csv")"

  mkdir -p "$exp_dir"
  log_event "$gpu" "$name" "train_start" "iterations=$iterations lr=$texture_lr start=$(basename "$start_ckpt")"
  write_status "$gpu" "train_start" "$name"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
    --config-path "$V215C_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$train_views" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,30]" \
    "dataset.test_frames.view=[0,570,30]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=" \
    "dataset.parsing_prior.skip_empty_samples=false" \
    "start_checkpoint=$start_ckpt" \
    "exp_dir=$exp_dir" \
    "hydra.run.dir=$hydra_run_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.clear_boundary_tags_on_resume=false" \
    "opt.iterations=$iterations" \
    "pipeline.pose_noise=0.0" \
    "model.pose_correction.delay=1" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
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
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.latent_weight_decay=0.0" \
    "opt.tex_latent_lr=0.0" \
    "opt.texture_lr=$texture_lr" \
    "++opt.lambda_binding_parsing=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_weights=$CAM_WEIGHTS" \
    "++opt.train_sample_camera_min_prob=0.015" \
    "++opt.train_sample_camera_max_prob=0.100" \
    "++opt.train_sample_log_interval=100" \
    "++opt.upper_torso_region_source=parser_only" \
    "++opt.upper_torso_region_parser_dilate=1" \
    "++opt.waist_region_source=parser_only" \
    "++opt.waist_region_mode=lower_cloth" \
    "++opt.waist_region_parser_dilate=1" \
    "++opt.face_region_source=parser_only" \
    "++opt.shoulder_arm_region_source=parser_only" \
    "++opt.clarity_debug_enable=true" \
    "++opt.clarity_debug_interval=100" \
    "++opt.clarity_debug_warmup_iters=0" \
    "++opt.photometric_correction_enable=false" \
    "++opt.reliable_view_supervision_enable=true" \
    "++opt.reliable_view_camera_quality_weights=$CAM_WEIGHTS" \
    "++opt.reliable_view_default_highfreq_weight=0.84" \
    "++opt.reliable_view_unknown_highfreq_weight=0.48" \
    "++opt.reliable_view_highfreq_power=1.25" \
    "++opt.reliable_view_highfreq_min_weight=0.22" \
    "++opt.reliable_view_highfreq_max_weight=1.55" \
    "++opt.reliable_view_apply_edge=false" \
    "++opt.reliable_view_apply_detail=false" \
    "++opt.reliable_view_apply_luma_dog=false" \
    "++opt.reliable_view_apply_patch_perceptual=false" \
    "++opt.reliable_view_apply_region_perceptual=false" \
    "++opt.perceptual_exclude_boundary_width=28" \
    "++opt.upper_torso_perceptual_exclude_boundary_width=28" \
    "++opt.upper_torso_core_perceptual_exclude_boundary_width=18" \
    "++opt.waist_perceptual_exclude_boundary_width=28" \
    "++opt.perceptual_adaptive_edge_protect=0.98" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.0" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
    "++opt.lambda_detail_waist_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
    "opt.lambda_edge_face=0.0" \
    "opt.lambda_edge_shoulder_arm=0.0" \
    "opt.lambda_edge_waist=0.0" \
    "opt.lambda_l1_fg=0.118" \
    "opt.lambda_l1_boundary=0.214" \
    "opt.lambda_perceptual=0.082" \
    "opt.grad_clip=0.0025" \
    "test_interval=0" \
    "test_iterations=$checkpoint_hydra" \
    "save_iterations=$checkpoint_hydra" \
    "checkpoint_iterations=$checkpoint_hydra" \
    "++validation_image_log_limit=0" \
    "$@" > "$train_log" 2>&1

  local train_status=$?
  if [ "$train_status" -ne 0 ]; then
    write_summary_row "$name" "train" "$exp_dir" "" "train_failed"
    log_event "$gpu" "$name" "train_failed" "status=$train_status log=$train_log"
    return "$train_status"
  fi

  log_event "$gpu" "$name" "train_done" "$train_log"

  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    if [ -n "$checkpoint_csv" ]; then
      IFS=',' read -ra checkpoints <<< "$checkpoint_csv"
      for local_ckpt in "${checkpoints[@]}"; do
        local global_ckpt=$((base_iter + local_ckpt))
        render_and_diag "$name" "$gpu" "$exp_dir" "$exp_dir/ckpt${global_ckpt}.pth" "ckpt${global_ckpt}"
      done
    fi
    render_and_diag "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "best"
  fi
}

run_named_job() {
  local gpu="$1"
  local job="$2"
  local fast300="60,120,220,300"
  local fast360="80,160,260,360"
  local freeze_texture="++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]"
  local owner_hf_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_structure_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*]"

  case "$job" in
    v218a_v215c410_boundary_residual_probe)
      run_one "$job" "$gpu" 300 0.0 "$V215C410_CKPT" 109410 "$fast300" "$ALL20" \
        "$freeze_texture" \
        "opt.lambda_l1_fg=0.116" \
        "opt.lambda_l1_boundary=0.226" \
        "opt.lambda_perceptual=0.072" \
        "++opt.boundary_aware_enable=true" \
        "++opt.boundary_aware_gate_l1_boundary=true" \
        "++opt.boundary_aware_gate_mask_boundary=true" \
        "++opt.boundary_aware_threshold=0.030" \
        "++opt.boundary_aware_score_power=1.0" \
        "++opt.boundary_aware_opacity_scale=0.0" \
        "++opt.boundary_aware_scaling_scale=0.0" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=0.58" \
        "++opt.boundary_aware_boundary_scaling_residual_scale=0.40" \
        "++opt.boundary_opacity_residual_lr=2.2e-05" \
        "++opt.boundary_scaling_residual_lr=7.0e-06" \
        "++opt.lambda_boundary_opacity_residual_reg=0.0018" \
        "++opt.lambda_boundary_scaling_residual_reg=0.00055" \
        "++opt.lambda_boundary_opacity_residual_smooth=0.0018" \
        "++opt.lambda_boundary_scaling_residual_smooth=0.0013" \
        "opt.grad_clip=0.0024"
      ;;
    v218b_v215c410_tiny_scale_opacity_probe)
      run_one "$job" "$gpu" 300 0.0 "$V215C410_CKPT" 109410 "$fast300" "$ALL20" \
        "$freeze_texture" \
        "opt.opacity_lr=7.0e-07" \
        "opt.scaling_lr=2.2e-07" \
        "opt.lambda_l1_fg=0.124" \
        "opt.lambda_l1_boundary=0.232" \
        "opt.lambda_perceptual=0.060" \
        "++opt.boundary_aware_enable=false" \
        "++opt.lambda_mask_boundary=0.010" \
        "++opt.lambda_mask_boundary_hard=0.018" \
        "++opt.lambda_silhouette_outer=0.004" \
        "++opt.lambda_silhouette_outer_shell=0.006" \
        "++opt.lambda_silhouette_inner=0.002" \
        "opt.grad_clip=0.0016"
      ;;
    v218c_v215c410_micro_pose_probe)
      run_one "$job" "$gpu" 300 0.0 "$V215C410_CKPT" 109410 "$fast300" "$ALL20" \
        "$freeze_texture" \
        "++model.pose_correction.train_pose_body=true" \
        "++model.pose_correction.pose_body_train_joint_ids=[12,13,14,15,16,17]" \
        "opt.pose_correction_lr=1.2e-07" \
        "opt.lambda_pose=0.38" \
        "opt.lambda_l1_fg=0.120" \
        "opt.lambda_l1_boundary=0.218" \
        "opt.lambda_perceptual=0.098" \
        "++opt.boundary_aware_enable=false" \
        "opt.grad_clip=0.0028"
      ;;
    v218d_v215c410_boundary_texture_combo)
      run_one "$job" "$gpu" 360 7.0e-06 "$V215C410_CKPT" 109410 "$fast360" "$ALL20" \
        "$owner_hf_patterns" \
        "opt.lambda_l1_fg=0.120" \
        "opt.lambda_l1_boundary=0.224" \
        "opt.lambda_perceptual=0.096" \
        "++opt.reliable_view_apply_luma_dog=true" \
        "++opt.reliable_view_apply_patch_perceptual=true" \
        "++opt.lambda_detail_waist_luma_dog=0.005" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.007" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.006" \
        "++opt.lambda_perceptual_upper_torso_patch=0.004" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.004" \
        "++opt.lambda_perceptual_waist_patch=0.004" \
        "++opt.owner_local_detail_boost_enable=true" \
        "++opt.owner_local_detail_boost_warmup_iters=0" \
        "++opt.owner_local_detail_boost_takeover_floor=0.26" \
        "++opt.owner_local_detail_boost_takeover_gain=1.12" \
        "++opt.owner_local_detail_boost_takeover_power=0.90" \
        "++opt.owner_local_detail_boost_luma_max_extra=[0.30,1,0.58,140,0.78]" \
        "++opt.owner_local_detail_boost_patch_max_extra=[0.16,1,0.32,140,0.46]" \
        "++opt.owner_local_detail_boost_boundary_max_extra=[0.10,1,0.20,140,0.30]" \
        "++opt.owner_local_detail_boost_face_strength=0.25" \
        "++opt.owner_local_detail_boost_shoulder_strength=0.80" \
        "++opt.owner_local_detail_boost_upper_torso_strength=1.00" \
        "++opt.owner_local_detail_boost_upper_torso_core_strength=1.06" \
        "++opt.boundary_aware_enable=true" \
        "++opt.boundary_aware_gate_l1_boundary=true" \
        "++opt.boundary_aware_threshold=0.032" \
        "++opt.boundary_aware_opacity_scale=0.0" \
        "++opt.boundary_aware_scaling_scale=0.0" \
        "++opt.boundary_aware_boundary_opacity_residual_scale=0.42" \
        "++opt.boundary_aware_boundary_scaling_residual_scale=0.28" \
        "++opt.boundary_opacity_residual_lr=1.4e-05" \
        "++opt.boundary_scaling_residual_lr=4.5e-06" \
        "++opt.lambda_boundary_opacity_residual_reg=0.0022" \
        "++opt.lambda_boundary_scaling_residual_reg=0.00075" \
        "++opt.lambda_boundary_opacity_residual_smooth=0.0020" \
        "++opt.lambda_boundary_scaling_residual_smooth=0.0016" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.22,1,0.40,160,0.54]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.upper_torso=0.28" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.lower=0.24" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost_max=1.70" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=0.0" \
        "opt.grad_clip=0.0030"
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
  local remain
  for job in "${jobs[@]}"; do
    if [ "$SMOKE" = "1" ] && [ "$launched" -ge "$SMOKE_MAX_JOBS_PER_GPU" ]; then
      log_event "$gpu" "$job" "smoke_skip" "max_jobs=$SMOKE_MAX_JOBS_PER_GPU"
      continue
    fi
    remain="$(remaining_seconds)"
    if [ "$SMOKE" != "1" ] && [ "$remain" -le "$MIN_START_SECONDS" ]; then
      log_event "$gpu" "$job" "deadline_skip" "remaining=${remain}s"
      continue
    fi
    run_named_job "$gpu" "$job"
    local status=$?
    launched=$((launched + 1))
    if [ "$status" -eq 0 ]; then
      log_event "$gpu" "$job" "job_done" "remaining=$(remaining_seconds)s"
    else
      log_event "$gpu" "$job" "job_failed" "status=$status remaining=$(remaining_seconds)s"
    fi
  done
  write_status "$gpu" "queue_done" "launched=$launched"
  log_event "$gpu" "queue" "done" "launched=$launched remaining=$(remaining_seconds)s"
}

launch_queue() {
  local gpu="$1"
  shift
  (
    queue_gpu "$gpu" "$@"
  ) &
  local pid=$!
  printf 'gpu%s_queue\t%s\t%s\n' "$gpu" "$gpu" "$pid" >> "$PIDS"
}

build_compare_panels() {
  if [ "$DO_RENDER" != "1" ] || [ "$SMOKE" = "1" ] || [ ! -d "$V215C410_RENDER/test-view/renders" ]; then
    return 0
  fi
  local out_main="$LOG_DIR/compare_panels/main_upper_crop"
  local out_full="$LOG_DIR/compare_panels/main_full_body"
  local render_exps=(
    "$V215C410_RENDER"
    "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v218a_v215c410_boundary_residual_probe_${RUN_ID}_render_quick_best"
    "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v218b_v215c410_tiny_scale_opacity_probe_${RUN_ID}_render_quick_best"
    "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v218c_v215c410_micro_pose_probe_${RUN_ID}_render_quick_best"
    "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v218d_v215c410_boundary_texture_combo_${RUN_ID}_render_quick_best"
  )
  local labels=(v215c v218a v218b v218c v218d)
  local existing_exps=()
  local existing_labels=()
  local idx
  for idx in "${!render_exps[@]}"; do
    if [ -d "${render_exps[$idx]}/test-view/renders" ]; then
      existing_exps+=("${render_exps[$idx]}")
      existing_labels+=("${labels[$idx]}")
    fi
  done
  if [ "${#existing_exps[@]}" -lt 2 ]; then
    return 0
  fi
  "$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
    --render-exp "${existing_exps[@]}" \
    --labels "${existing_labels[@]}" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --output-dir "$out_main" \
    --select "${HP_SELECT[@]}" \
    --crop 150 35 650 430 \
    --panel-width 250 \
    --stack > "$LOG_DIR/compare_main_upper_crop.log" 2>&1 || true
  "$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
    --render-exp "${existing_exps[@]}" \
    --labels "${existing_labels[@]}" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --output-dir "$out_full" \
    --select "${HP_SELECT[@]}" \
    --panel-width 210 \
    --stack > "$LOG_DIR/compare_main_full_body.log" 2>&1 || true
}

launch_queue 0 \
  v218a_v215c410_boundary_residual_probe

launch_queue 1 \
  v218b_v215c410_tiny_scale_opacity_probe

launch_queue 2 \
  v218c_v215c410_micro_pose_probe

launch_queue 3 \
  v218d_v215c410_boundary_texture_combo

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "DEADLINE_BJT=$DEADLINE_BJT"
cat "$PIDS"

wait

build_compare_panels
log_event "all" "queue" "all_done" "summary=$SUMMARY"
echo "SUMMARY=$SUMMARY"
