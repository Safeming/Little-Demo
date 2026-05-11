#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-7200}"
START_EPOCH="${START_EPOCH:-$(date +%s)}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$((START_EPOCH + TIME_BUDGET_SECONDS))}"
MIN_START_SECONDS="${MIN_START_SECONDS:-900}"
SMOKE="${SMOKE:-0}"
SMOKE_MAX_JOBS_PER_GPU="${SMOKE_MAX_JOBS_PER_GPU:-1}"
DO_RENDER="${DO_RENDER:-1}"

BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
V210B_EXP="${V210B_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v210b_v198a_ownerlocal_cloth_teacher_20260509_191634_bjt}"
V210B_CKPT="${V210B_CKPT:-$V210B_EXP/ckpt109150.pth}"
V213A_EXP="${V213A_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v213a_v198a_cloth_cache_open_20260509_224650_bjt}"
V213A100_CKPT="${V213A100_CKPT:-$V213A_EXP/ckpt109100.pth}"
V214J_EXP="${V214J_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v214j_v198a_cap085_probe_20260510_001225_bjt}"
V214J300_CKPT="${V214J300_CKPT:-$V214J_EXP/ckpt109300.pth}"
V214V_EXP="${V214V_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v214v_v210b150_boundary_20260510_001225_bjt}"
V214V350_CKPT="${V214V350_CKPT:-$V214V_EXP/ckpt109350.pth}"
V215C_EXP="${V215C_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt}"
V215C410_CKPT="${V215C410_CKPT:-$V215C_EXP/ckpt109410.pth}"
V216E_EXP="${V216E_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v216e_v210b150_owner_hf_rebuild_v216_20260510_111055_bjt}"
V216E470_CKPT="${V216E470_CKPT:-$V216E_EXP/ckpt109470.pth}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
V211_ANALYSIS_DIR="${V211_ANALYSIS_DIR:-$ROOT/exp/stageA2/logs/v211_region_view_conflict_20260509_211000_bjt/v211a_region_view_conflict_map}"
V211_PLAN="$V211_ANALYSIS_DIR/region_training_plan.json"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v217_reliable_region_hf_4gpu_$RUN_ID}"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"
cd "$ROOT" || exit 1

for required in "$BASE_CKPT" "$V210B_CKPT" "$V213A100_CKPT" "$V214J300_CKPT" "$V214V350_CKPT" "$V215C410_CKPT" "$V216E470_CKPT" "$PARSER_ROOT" "$V211_PLAN"; do
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
CLOTH12="[2,3,6,7,8,9,11,13,15,18,19,20]"
CAM_WEIGHTS="${CAM_WEIGHTS:-$(build_v211_cloth_camera_weights)}"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
DEADLINE_BJT="$(TZ=Asia/Shanghai date -d "@$DEADLINE_EPOCH" '+%F %T BJT')"

printf 'RUN_ID=%s\nSTART_BJT=%s\nDEADLINE_BJT=%s\nTIME_BUDGET_SECONDS=%s\nMIN_START_SECONDS=%s\nBASE_CKPT=%s\nV210B_CKPT=%s\nV213A100_CKPT=%s\nV214J300_CKPT=%s\nV214V350_CKPT=%s\nV215C410_CKPT=%s\nV216E470_CKPT=%s\nV211_ANALYSIS_DIR=%s\nALL20=%s\nCLOTH12=%s\nCAM_WEIGHTS=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$DEADLINE_BJT" "$TIME_BUDGET_SECONDS" "$MIN_START_SECONDS" \
  "$BASE_CKPT" "$V210B_CKPT" "$V213A100_CKPT" "$V214J300_CKPT" "$V214V350_CKPT" "$V215C410_CKPT" "$V216E470_CKPT" \
  "$V211_ANALYSIS_DIR" "$ALL20" "$CLOTH12" "$CAM_WEIGHTS" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

v217 purpose:
  Probe the current clarity root-cause directly: conflicting multi-view
  high-frequency supervision is being averaged inside a shared canonical
  appearance path. This queue keeps geometry/pose/opacity/scaling frozen and
  tests reliable-view high-frequency supervision, trusted cloth camera subsets,
  and the explicit cloth high-frequency cache path.

acceptance:
  - v198a remains strict anchor
  - v210b@109150 remains auxiliary edge anchor
  - v215c@109410 is current stable-edge v215 candidate
  - v216e@109470 is the best low-frequency/color tradeoff but not the anchor
  - A useful result must improve cloth/waist high-frequency visibility without
    pushing boundary_l1 above the v215/v216 stable band or edge_px into the
    v216e-best soft-edge range.
EOF

printf 'name\tlabel\texp_dir\trender_exp\ttrain_lpips_fg\ttrain_l1_fg\ttrain_psnr_fg\trender_lpips\trender_psnr\trender_ssim\tfg_l1\tboundary_l1\tedge_px\tstatus\n' > "$SUMMARY"
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
if render is not None:
    contour_path = render / "diagnostics" / "contour_summary.json"
    if contour_path.exists():
        contour = json.loads(contour_path.read_text())

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
    status,
]
with open(summary, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
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
    --config-path "$BASE_EXP/.hydra" \
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
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.]" \
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
    "++opt.camera_affine_train_camera_ids=$ALL20" \
    "++opt.camera_affine_max_camera_id=23" \
    "++opt.camera_affine_strength=[0.25,1,0.55,160,0.80]" \
    "++opt.camera_affine_scale_max_delta=0.050" \
    "++opt.camera_affine_shift_max_abs=0.026" \
    "++opt.camera_affine_clamp_colors=true" \
    "++opt.camera_affine_apply_unknown=false" \
    "++opt.camera_affine_scale_reg_weight=1.0" \
    "++opt.camera_affine_shift_reg_weight=1.0" \
    "++opt.lambda_camera_affine_reg=0.0030" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.latent_weight_decay=0.0" \
    "opt.tex_latent_lr=0.0" \
    "opt.texture_lr=$texture_lr" \
    "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_structure_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*]" \
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
    "++opt.lambda_detail_face=0.002" \
    "++opt.lambda_detail_shoulder_arm=0.003" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.002" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.005" \
    "++opt.lambda_detail_waist_luma_dog=0.007" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.010" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.008" \
    "++opt.lambda_perceptual_upper_torso_patch=0.006" \
    "++opt.lambda_perceptual_upper_torso_core_patch=0.005" \
    "++opt.lambda_perceptual_waist_patch=0.006" \
    "++opt.owner_local_detail_boost_enable=true" \
    "++opt.owner_local_detail_boost_warmup_iters=0" \
    "++opt.owner_local_detail_boost_takeover_floor=0.24" \
    "++opt.owner_local_detail_boost_takeover_gain=1.25" \
    "++opt.owner_local_detail_boost_takeover_power=0.85" \
    "++opt.owner_local_detail_boost_min_signal=0.01" \
    "++opt.owner_local_detail_boost_detail_max_extra=[0.35,1,0.72,160,0.96]" \
    "++opt.owner_local_detail_boost_luma_max_extra=[0.50,1,0.95,160,1.26]" \
    "++opt.owner_local_detail_boost_patch_max_extra=[0.26,1,0.52,160,0.72]" \
    "++opt.owner_local_detail_boost_edge_max_extra=[0.06,1,0.14,160,0.20]" \
    "++opt.owner_local_detail_boost_boundary_max_extra=[0.06,1,0.12,160,0.18]" \
    "++opt.owner_local_detail_boost_face_strength=0.55" \
    "++opt.owner_local_detail_boost_shoulder_strength=1.05" \
    "++opt.owner_local_detail_boost_upper_torso_strength=1.16" \
    "++opt.owner_local_detail_boost_upper_torso_core_strength=1.22" \
    "++opt.reliable_view_supervision_enable=true" \
    "++opt.reliable_view_camera_quality_weights=$CAM_WEIGHTS" \
    "++opt.reliable_view_default_highfreq_weight=0.95" \
    "++opt.reliable_view_unknown_highfreq_weight=0.62" \
    "++opt.reliable_view_highfreq_power=1.10" \
    "++opt.reliable_view_highfreq_min_weight=0.35" \
    "++opt.reliable_view_highfreq_max_weight=1.45" \
    "++opt.reliable_view_apply_edge=false" \
    "++opt.reliable_view_apply_detail=false" \
    "++opt.reliable_view_apply_luma_dog=true" \
    "++opt.reliable_view_apply_patch_perceptual=true" \
    "++opt.reliable_view_apply_region_perceptual=false" \
    "opt.lambda_l1_fg=0.112" \
    "opt.lambda_l1_boundary=0.204" \
    "opt.lambda_perceptual=0.116" \
    "++opt.photometric_correction_enable=false" \
    "++opt.photometric_correction_strength=0.30" \
    "++opt.photometric_correction_erode_kernel_size=11" \
    "++opt.photometric_correction_min_pixels=512" \
    "++opt.photometric_correction_min_scale=0.88" \
    "++opt.photometric_correction_max_scale=1.13" \
    "++opt.photometric_correction_max_shift=0.050" \
    "++opt.photometric_contour_debug_interval=100" \
    "++opt.perceptual_exclude_boundary_width=28" \
    "++opt.upper_torso_perceptual_exclude_boundary_width=28" \
    "++opt.upper_torso_core_perceptual_exclude_boundary_width=18" \
    "++opt.waist_perceptual_exclude_boundary_width=28" \
    "++opt.perceptual_adaptive_edge_protect=0.94" \
    "opt.lambda_edge_face=0.0" \
    "opt.lambda_edge_shoulder_arm=0.0015" \
    "opt.lambda_edge_waist=0.0" \
    "++model.texture.structured_trunk.output_head.local_color.owner.scale=[0.54,1,0.70,160,0.82]" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.scale=[0.54,1,0.72,160,0.86]" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_gain=1.24" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_bias=0.02" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.min_gate=0.10" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.local_color_output_scale=0.18" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=[0.42,1,0.72,160,0.94]" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.max=1.0" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.region_strength.upper_torso=0.96" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.scale=[0.46,1,0.66,160,0.78]" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.gate_gain=1.18" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.min_gate=0.12" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.takeover.scale=[0.26,1,0.42,160,0.54]" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.50,1,0.68,160,0.82]" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.use_local_color=true" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.local_color_scale=[0.82,1,1.00,160,1.12]" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.gate_gain=1.24" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.gate_bias=0.00" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.min_gate=0.08" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.face=0.30" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.shoulder_upper=0.38" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.upper_torso=0.34" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost_max=2.10" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.enable=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=0.50" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.tiny_repair_scale=1.0" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.018" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.input_detach=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.chroma_center=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_bias=-0.50" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.min_gate=0.04" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.init_from=shared_chroma" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_init_from=gate_mlp" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.init_missing_only=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.init_output_scale=0.28" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_init_output_scale=1.0" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.inherit_point_gate=false" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate_combine_mode=mul" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.enable=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.combine_mode=max" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.fallback_to_full=false" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.semantic_id_weights=[[3,1.00],[4,0.90]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.semantic_name_weights=[[upper,1.00],[lower,0.90]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.exclude_semantic_id_weights=[[0,1.00],[1,1.00],[2,0.55],[5,1.00]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.exclude_semantic_name_weights=[[hair,1.00],[face,1.00],[skin,0.55],[shoes,1.00]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.joint_id_weights=[[0,0.55],[1,0.90],[2,0.90],[3,0.95],[6,0.85],[9,0.90],[12,0.55]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.exclude_joint_id_weights=[[7,1.00],[8,1.00],[10,1.00],[11,1.00],[15,1.00],[16,0.90],[17,0.90],[18,0.50],[19,0.50],[20,1.00],[21,1.00],[22,1.00],[23,1.00]]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.min_gate=0.10" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.enable=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.strength=[0.0,1,0.36,60,0.52]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.threshold=0.18" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.power=1.35" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.min_scale=0.38" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.detach=true" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_neurons=72" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_hidden_layers=2" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.skip_in=[]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.cond_in=[]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.multires=0" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.last_layer_init=false" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_neurons=48" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_hidden_layers=1" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.skip_in=[]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.cond_in=[]" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.multires=0" \
    "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.last_layer_init=false" \
    "opt.grad_clip=0.0018" \
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
  local cache360="80,180,280,360"
  local owner_only_patterns="++opt.texture_trainable_name_patterns=[structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*]"
  local hf_owner_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_structure_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*]"
  local cache_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*]"

  case "$job" in
    v217a_v215c410_reliable_hf_strict)
      run_one "$job" "$gpu" 300 1.1e-05 "$V215C410_CKPT" 109410 "$fast300" "$ALL20" \
        "$owner_only_patterns" \
        "opt.lambda_l1_fg=0.120" \
        "opt.lambda_l1_boundary=0.218" \
        "opt.lambda_perceptual=0.106" \
        "++opt.lambda_detail_face=0.0" \
        "++opt.lambda_detail_shoulder_arm=0.0" \
        "++opt.lambda_detail_face_luma_dog=0.0" \
        "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
        "++opt.lambda_detail_waist_luma_dog=0.009" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.014" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.011" \
        "++opt.lambda_perceptual_upper_torso_patch=0.008" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.006" \
        "++opt.lambda_perceptual_waist_patch=0.008" \
        "++opt.reliable_view_default_highfreq_weight=0.82" \
        "++opt.reliable_view_unknown_highfreq_weight=0.42" \
        "++opt.reliable_view_highfreq_power=1.45" \
        "++opt.reliable_view_highfreq_min_weight=0.16" \
        "++opt.reliable_view_highfreq_max_weight=1.72" \
        "++opt.owner_local_detail_boost_luma_max_extra=[0.48,1,0.88,140,1.10]" \
        "++opt.owner_local_detail_boost_patch_max_extra=[0.24,1,0.46,140,0.64]" \
        "++opt.owner_local_detail_boost_upper_torso_strength=1.24" \
        "++opt.owner_local_detail_boost_upper_torso_core_strength=1.30" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=0.0" \
        "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=[0.44,1,0.68,140,0.86]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.46,1,0.64,140,0.78]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.face=0.00" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.shoulder_upper=0.20" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.upper_torso=0.46" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.lower=0.42" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost_max=2.40" \
        "opt.grad_clip=0.0042"
      ;;
    v217b_v215c410_cloth12_subset_oracle)
      run_one "$job" "$gpu" 300 8.0e-06 "$V215C410_CKPT" 109410 "$fast300" "$CLOTH12" \
        "$owner_only_patterns" \
        "opt.lambda_l1_fg=0.118" \
        "opt.lambda_l1_boundary=0.214" \
        "opt.lambda_perceptual=0.108" \
        "++opt.train_sample_camera_weights={2:1.80,3:1.10,6:1.15,7:1.20,8:1.25,9:0.95,11:1.80,13:0.92,15:0.95,18:1.10,19:1.35,20:1.45}" \
        "++opt.train_sample_camera_min_prob=0.030" \
        "++opt.train_sample_camera_max_prob=0.160" \
        "++opt.lambda_detail_face=0.0" \
        "++opt.lambda_detail_shoulder_arm=0.0" \
        "++opt.lambda_detail_face_luma_dog=0.0" \
        "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
        "++opt.lambda_detail_waist_luma_dog=0.010" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.015" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.012" \
        "++opt.lambda_perceptual_upper_torso_patch=0.010" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.008" \
        "++opt.lambda_perceptual_waist_patch=0.009" \
        "++opt.reliable_view_default_highfreq_weight=1.00" \
        "++opt.reliable_view_unknown_highfreq_weight=0.40" \
        "++opt.reliable_view_highfreq_power=1.25" \
        "++opt.reliable_view_highfreq_min_weight=0.22" \
        "++opt.reliable_view_highfreq_max_weight=1.60" \
        "++opt.owner_local_detail_boost_luma_max_extra=[0.56,1,1.00,150,1.26]" \
        "++opt.owner_local_detail_boost_patch_max_extra=[0.30,1,0.56,150,0.74]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=0.0" \
        "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=[0.46,1,0.72,150,0.92]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.50,1,0.70,150,0.84]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.upper_torso=0.54" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost.lower=0.48" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.region_boost_max=2.60" \
        "opt.grad_clip=0.0038"
      ;;
    v217c_v213a100_cache_reliable_asset)
      run_one "$job" "$gpu" 360 6.0e-06 "$V213A100_CKPT" 109100 "$cache360" "$ALL20" \
        "$cache_patterns" \
        "opt.lambda_l1_fg=0.116" \
        "opt.lambda_l1_boundary=0.196" \
        "opt.lambda_perceptual=0.116" \
        "++opt.lambda_detail_face=0.0" \
        "++opt.lambda_detail_shoulder_arm=0.0" \
        "++opt.lambda_detail_face_luma_dog=0.0" \
        "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
        "++opt.lambda_detail_waist_luma_dog=0.010" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.013" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.010" \
        "++opt.lambda_perceptual_upper_torso_patch=0.010" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.008" \
        "++opt.lambda_perceptual_waist_patch=0.009" \
        "++opt.reliable_view_default_highfreq_weight=0.92" \
        "++opt.reliable_view_unknown_highfreq_weight=0.50" \
        "++opt.reliable_view_highfreq_power=1.30" \
        "++opt.reliable_view_highfreq_min_weight=0.24" \
        "++opt.reliable_view_highfreq_max_weight=1.68" \
        "++opt.owner_local_detail_boost_luma_max_extra=[0.48,1,0.88,160,1.12]" \
        "++opt.owner_local_detail_boost_patch_max_extra=[0.26,1,0.50,160,0.68]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=[0.0,1,0.24,90,0.46,180,0.62]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.022" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_bias=-0.56" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.min_gate=0.05" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.init_output_scale=0.24" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.min_gate=0.14" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.strength=[0.0,1,0.52,90,0.70]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.min_scale=0.30" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.38,1,0.54,160,0.68]" \
        "opt.grad_clip=0.0036"
      ;;
    v217d_v216e470_edge_safe_polish)
      run_one "$job" "$gpu" 300 8.5e-06 "$V216E470_CKPT" 109470 "$fast300" "$ALL20" \
        "$hf_owner_patterns" \
        "opt.lambda_l1_fg=0.124" \
        "opt.lambda_l1_boundary=0.224" \
        "opt.lambda_perceptual=0.106" \
        "++opt.lambda_detail_face=0.0" \
        "++opt.lambda_detail_shoulder_arm=0.0" \
        "++opt.lambda_detail_face_luma_dog=0.0" \
        "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
        "++opt.lambda_detail_waist_luma_dog=0.007" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.010" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.008" \
        "++opt.lambda_perceptual_upper_torso_patch=0.006" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.005" \
        "++opt.lambda_perceptual_waist_patch=0.006" \
        "++opt.reliable_view_default_highfreq_weight=0.84" \
        "++opt.reliable_view_unknown_highfreq_weight=0.44" \
        "++opt.reliable_view_highfreq_power=1.40" \
        "++opt.reliable_view_highfreq_min_weight=0.18" \
        "++opt.reliable_view_highfreq_max_weight=1.60" \
        "++opt.perceptual_adaptive_edge_protect=1.00" \
        "++opt.owner_local_detail_boost_luma_max_extra=[0.34,1,0.66,140,0.86]" \
        "++opt.owner_local_detail_boost_patch_max_extra=[0.18,1,0.36,140,0.50]" \
        "++opt.owner_local_detail_boost_boundary_max_extra=[0.12,1,0.24,160,0.34]" \
        "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=[0.34,1,0.52,140,0.68]" \
        "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.scale=[0.54,1,0.72,140,0.86]" \
        "++model.texture.structured_trunk.output_head.local_color.owner.head.boundary.takeover.scale=[0.40,1,0.58,140,0.70]" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=[0.32,1,0.48,140,0.62]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=0.18" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.010" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.strength=[0.0,1,0.62,100,0.82]" \
        "++model.texture.detail_residual.high_frequency.view_conflict_residual.boundary_suppress.min_scale=0.24" \
        "opt.grad_clip=0.0032"
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

launch_queue 0 \
  v217a_v215c410_reliable_hf_strict

launch_queue 1 \
  v217b_v215c410_cloth12_subset_oracle

launch_queue 2 \
  v217c_v213a100_cache_reliable_asset

launch_queue 3 \
  v217d_v216e470_edge_safe_polish

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "DEADLINE_BJT=$DEADLINE_BJT"
cat "$PIDS"

wait

log_event "all" "queue" "all_done" "summary=$SUMMARY"
echo "SUMMARY=$SUMMARY"
