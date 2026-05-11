#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageB_v221_rootcause_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-7200}"
START_EPOCH="${START_EPOCH:-$(date +%s)}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$((START_EPOCH + TIME_BUDGET_SECONDS))}"
MIN_START_SECONDS="${MIN_START_SECONDS:-900}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"
JOBS="${JOBS:-}"
GPUS="${GPUS:-}"

V215C_EXP="${V215C_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt}"
V215C410_CKPT="${V215C410_CKPT:-$V215C_EXP/ckpt109410.pth}"
V215C410_RENDER="${V215C410_RENDER:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt_render_quick_ckpt109410}"
V198A_EXP="${V198A_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue}"
V198A_CKPT="${V198A_CKPT:-$V198A_EXP/best_ckpt.pth}"
V198A_RENDER="${V198A_RENDER:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue_render_quick}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
V211_PLAN="${V211_PLAN:-$ROOT/exp/stageA2/logs/v211_region_view_conflict_20260509_211000_bjt/v211a_region_view_conflict_map/region_training_plan.json}"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_v221_rootcause_4gpu_$RUN_ID}"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"
cd "$ROOT" || exit 1

for required in "$V215C_EXP/.hydra/config.yaml" "$V215C410_CKPT" "$V198A_EXP/.hydra/config.yaml" "$V198A_CKPT" "$DATA_ROOT" "$PARSER_ROOT" "$V211_PLAN"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
for cam in $(seq 1 20); do
  parser_dir="$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B${cam}"
  if [ ! -d "$parser_dir" ]; then
    echo "missing parser dir: $parser_dir" >&2
    exit 3
  fi
done

build_region_camera_weights() {
  "$PYTHON_BIN" - "$V211_PLAN" <<'PY'
import json
import sys
from collections import defaultdict

plan = json.loads(open(sys.argv[1], encoding="utf-8").read())
scores = defaultdict(lambda: 0.44)
for item in plan.get("recommended_regions", []):
    region = str(item.get("region", ""))
    if region in {"upper_cloth", "lower_cloth"}:
        strength = 1.08
    elif region in {"skin", "face", "arm"}:
        strength = 0.82
    else:
        strength = 0.55
    for rank, cam in enumerate(item.get("top_cameras", [])[:8]):
        scores[int(cam)] += strength * max(0.30, 1.0 - 0.09 * rank)
values = {cam: scores[cam] for cam in range(1, 21)}
mean = sum(values.values()) / len(values)
weights = {cam: max(0.35, min(2.0, value / max(mean, 1.0e-6))) for cam, value in values.items()}
print("{" + ",".join(f"{cam}:{weights[cam]:.4f}" for cam in range(1, 21)) + "}")
PY
}

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CORE12="[2,3,6,7,8,9,11,13,15,18,19,20]"
RELIABLE5="[2,8,11,19,20]"
MINI_VIEWS="[2,11]"
MINI_FRAMES="[240,421,60]"
CAM_WEIGHTS="${CAM_WEIGHTS:-$(build_region_camera_weights)}"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
DEADLINE_BJT="$(TZ=Asia/Shanghai date -d "@$DEADLINE_EPOCH" '+%F %T BJT')"
HP_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
HP_SELECT_MINI=(render_c2_f000240.png render_c2_f000300.png render_c2_f000360.png render_c2_f000420.png render_c11_f000240.png render_c11_f000300.png render_c11_f000360.png render_c11_f000420.png)
HP_CROP=(150 35 650 430)

printf 'RUN_ID=%s\nSTART_BJT=%s\nDEADLINE_BJT=%s\nTIME_BUDGET_SECONDS=%s\nV215C410_CKPT=%s\nV198A_CKPT=%s\nV215C410_RENDER=%s\nV198A_RENDER=%s\nALL20=%s\nCORE12=%s\nRELIABLE5=%s\nMINI_VIEWS=%s\nMINI_FRAMES=%s\nCAM_WEIGHTS=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$DEADLINE_BJT" "$TIME_BUDGET_SECONDS" \
  "$V215C410_CKPT" "$V198A_CKPT" "$V215C410_RENDER" "$V198A_RENDER" "$ALL20" "$CORE12" "$RELIABLE5" "$MINI_VIEWS" "$MINI_FRAMES" "$CAM_WEIGHTS" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

v221 StageB clarity root-cause probes:
  This run is not another long StageA continuation. It asks where the blur is
  coming from: texture/region asset capacity, multi-view averaging, or splat
  footprint low-pass filtering. Most variants keep geometry, pose, gaussian
  primitive parameters, latent codes, camera affine, and camera geometry frozen.

variants:
  v221a_stageB_mini_overfit_texture_capacity:
    v215c anchor + two reliable training cameras with selected-frame train
    probe. If this cannot recover local high-frequency detail on train views,
    the texture/region asset path is the bottleneck.
  v221b_stageB_reliable5_region_capacity:
    v215c anchor + reliable camera subset. If this improves over all20, blur is
    dominated by multi-view conflict or low-quality views.
  v221c_stageB_all20_highcap_region_asset:
    v215c anchor + all training cameras with the same high-capacity region asset.
    This isolates subset-vs-all-view averaging.
  v221d_stageB_footprint_scaling_probe:
    v215c anchor + high-capacity texture plus very small scaling/opacity learning.
    If only this branch raises true highpass, splat footprint/compositing is likely
    suppressing details.

acceptance:
  - LPIPS alone is not enough.
  - v221a train-probe highpass should approach GT on selected train views.
  - v221b should beat v221c on crop highpass if multi-view conflict is the issue.
  - v221d should improve highpass only with stable boundary/edge metrics.
  - Visual crops must show face/arm/cloth details, not boundary drift.
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
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_epoch": start_epoch,
    "deadline_epoch": deadline_epoch,
    "now_epoch": now,
    "elapsed_seconds": now - start_epoch,
    "remaining_seconds": max(0, deadline_epoch - now),
}, indent=2), encoding="utf-8")
PY
}

split_csv() {
  local value="$1"
  if [ -z "$value" ]; then
    return 0
  fi
  local IFS=','
  read -ra _split_items <<< "$value"
  printf '%s\n' "${_split_items[@]}"
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
if (exp / "best_test_metrics.json").exists():
    train = json.loads((exp / "best_test_metrics.json").read_text())

render_metrics = {}
if render is not None and (render / "test-view" / "results.npz").exists():
    data = np.load(render / "test-view" / "results.npz")
    for key in ("lpips", "psnr", "ssim"):
        if key in data.files:
            render_metrics[key] = float(np.asarray(data[key]).mean())

contour = {}
highpass = {}
if render is not None:
    contour_path = render / "diagnostics" / "contour_summary.json"
    highpass_path = render / "diagnostics" / "highpass_summary.json"
    if contour_path.exists():
        contour = json.loads(contour_path.read_text())
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
  shift 4
  local select_files=("$@")
  if [ "${#select_files[@]}" -eq 0 ]; then
    select_files=("${HP_SELECT[@]}")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_highpass_energy.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --select "${select_files[@]}" \
    --crop "${HP_CROP[@]}" > "$LOG_DIR/${name}_highpass_${label}.log" 2>&1
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
    wandb_disable=true > "$LOG_DIR/${name}_render_${label}.log" 2>&1
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
    --topk 12 > "$LOG_DIR/${name}_contour_${label}.log" 2>&1
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

render_train_probe_and_diag() {
  local name="$1"
  local gpu="$2"
  local exp_dir="$3"
  local ckpt_path="$4"
  local label="$5"
  local test_views="$6"
  local test_frames="$7"
  shift 7
  local select_files=("$@")

  if [ "$DO_RENDER" != "1" ]; then
    return 0
  fi

  local render_exp="${exp_dir}_render_trainprobe_${label}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_render_trainprobe_${label}"
  if [ ! -f "$ckpt_path" ]; then
    write_summary_row "$name" "trainprobe_${label}" "$exp_dir" "$render_exp" "missing_ckpt"
    return 0
  fi

  log_event "$gpu" "$name" "render_trainprobe_start" "$label views=$test_views frames=$test_frames"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" render.py \
    --config-path "$exp_dir/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt_path" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=$test_views" \
    "dataset.test_frames.view=$test_frames" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "hydra.run.dir=$hydra_run_dir" \
    wandb_disable=true > "$LOG_DIR/${name}_render_trainprobe_${label}.log" 2>&1
  local render_status=$?
  if [ "$render_status" -ne 0 ]; then
    write_summary_row "$name" "trainprobe_${label}" "$exp_dir" "$render_exp" "render_trainprobe_failed"
    log_event "$gpu" "$name" "render_trainprobe_failed" "$label status=$render_status"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 > "$LOG_DIR/${name}_contour_trainprobe_${label}.log" 2>&1
  local contour_status=$?
  if [ "$contour_status" -ne 0 ]; then
    write_summary_row "$name" "trainprobe_${label}" "$exp_dir" "$render_exp" "contour_trainprobe_failed"
    log_event "$gpu" "$name" "contour_trainprobe_failed" "$label status=$contour_status"
    return 0
  fi

  run_highpass_diag "$name" "$gpu" "$render_exp" "trainprobe_${label}" "${select_files[@]}"
  local highpass_status=$?
  if [ "$highpass_status" -ne 0 ]; then
    write_summary_row "$name" "trainprobe_${label}" "$exp_dir" "$render_exp" "highpass_trainprobe_failed"
    log_event "$gpu" "$name" "highpass_trainprobe_failed" "$label status=$highpass_status"
    return 0
  fi

  write_summary_row "$name" "trainprobe_${label}" "$exp_dir" "$render_exp" "ok"
  log_event "$gpu" "$name" "render_trainprobe_done" "$label"
}

run_one() {
  local name="$1"
  local gpu="$2"
  local base_exp="$3"
  local start_ckpt="$4"
  local base_iter="$5"
  local iterations="$6"
  local texture_lr="$7"
  local checkpoint_csv="$8"
  local train_views="$9"
  shift 9

  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_csv=""
  fi

  local exp_dir="$ROOT/exp/stageB/377_region_asset_clarity_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  local checkpoint_hydra
  checkpoint_hydra="$(csv_to_hydra_list "$checkpoint_csv")"
  mkdir -p "$exp_dir"

  log_event "$gpu" "$name" "train_start" "iterations=$iterations lr=$texture_lr start=$start_ckpt"
  write_status "$gpu" "train_start" "$name"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
    --config-path "$base_exp/.hydra" \
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
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
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
    "++opt.train_sample_camera_max_prob=0.105" \
    "++opt.train_sample_log_interval=100" \
    "++opt.clarity_debug_enable=true" \
    "++opt.clarity_debug_interval=100" \
    "++opt.clarity_debug_warmup_iters=0" \
    "++opt.face_region_source=parser_only" \
    "++opt.face_region_parser_dilate=1" \
    "++opt.face_region_source_aware_validity_enable=true" \
    "++opt.face_region_min_pixels_parser=18" \
    "++opt.shoulder_arm_region_source=parser_only" \
    "++opt.shoulder_arm_region_parser_dilate=1" \
    "++opt.shoulder_arm_region_source_aware_validity_enable=true" \
    "++opt.shoulder_arm_region_min_pixels_parser=32" \
    "++opt.upper_torso_region_source=parser_only" \
    "++opt.upper_torso_region_parser_dilate=1" \
    "++opt.upper_torso_region_min_pixels=32" \
    "++opt.waist_region_source=parser_only" \
    "++opt.waist_region_mode=lower_cloth" \
    "++opt.waist_region_parser_dilate=1" \
    "++opt.waist_region_min_pixels=32" \
    "++opt.detail_interior_erode_kernel_size=1" \
    "++opt.detail_interior_exclude_boundary_width=18" \
    "++opt.face_detail_interior_exclude_boundary_width=10" \
    "++opt.shoulder_arm_detail_interior_exclude_boundary_width=18" \
    "++opt.upper_torso_detail_interior_exclude_boundary_width=18" \
    "++opt.upper_torso_core_erode_kernel_size=3" \
    "++opt.upper_torso_core_center_width_ratio=0.74" \
    "++opt.upper_torso_core_top_trim_ratio=0.04" \
    "++opt.upper_torso_core_bottom_trim_ratio=0.10" \
    "++opt.waist_detail_interior_exclude_boundary_width=18" \
    "++opt.perceptual_exclude_boundary_width=24" \
    "++opt.face_perceptual_exclude_boundary_width=12" \
    "++opt.shoulder_arm_perceptual_exclude_boundary_width=24" \
    "++opt.upper_torso_perceptual_exclude_boundary_width=24" \
    "++opt.upper_torso_core_perceptual_exclude_boundary_width=16" \
    "++opt.waist_perceptual_exclude_boundary_width=24" \
    "++opt.perceptual_adaptive_edge_protect=0.92" \
    "++opt.reliable_view_supervision_enable=false" \
    "++opt.reliable_view_camera_quality_weights=$CAM_WEIGHTS" \
    "++opt.reliable_view_default_highfreq_weight=0.80" \
    "++opt.reliable_view_unknown_highfreq_weight=0.48" \
    "++opt.reliable_view_highfreq_power=1.20" \
    "++opt.reliable_view_highfreq_min_weight=0.28" \
    "++opt.reliable_view_highfreq_max_weight=1.46" \
    "++opt.reliable_view_apply_edge=false" \
    "++opt.reliable_view_apply_detail=false" \
    "++opt.reliable_view_apply_luma_dog=false" \
    "++opt.reliable_view_apply_patch_perceptual=false" \
    "++opt.reliable_view_apply_region_perceptual=false" \
    "++opt.owner_local_detail_boost_enable=true" \
    "++opt.owner_local_detail_boost_warmup_iters=0" \
    "++opt.owner_local_detail_boost_takeover_floor=0.18" \
    "++opt.owner_local_detail_boost_takeover_gain=1.10" \
    "++opt.owner_local_detail_boost_takeover_power=1.10" \
    "++opt.owner_local_detail_boost_min_signal=0.002" \
    "++opt.owner_local_detail_boost_detail_max_extra=0.16" \
    "++opt.owner_local_detail_boost_luma_max_extra=0.22" \
    "++opt.owner_local_detail_boost_patch_max_extra=0.20" \
    "++opt.owner_local_detail_boost_edge_max_extra=0.08" \
    "++opt.lambda_l1=0.018" \
    "opt.lambda_l1_fg=0.110" \
    "opt.lambda_l1_boundary=0.214" \
    "opt.lambda_perceptual=0.068" \
    "opt.lambda_l1_face=0.050" \
    "opt.lambda_l1_shoulder_arm=0.040" \
    "opt.lambda_l1_waist=0.036" \
    "opt.lambda_edge_face=0.010" \
    "opt.lambda_edge_shoulder_arm=0.012" \
    "opt.lambda_edge_waist=0.006" \
    "++opt.lambda_detail_face=0.000" \
    "++opt.lambda_detail_shoulder_arm=0.000" \
    "++opt.lambda_detail_waist=0.000" \
    "++opt.lambda_detail_face_luma_dog=0.016" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.014" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.014" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.018" \
    "++opt.lambda_detail_waist_luma_dog=0.010" \
    "++opt.lambda_perceptual_face_patch=0.014" \
    "++opt.lambda_perceptual_shoulder_arm_patch=0.010" \
    "++opt.lambda_perceptual_upper_torso_patch=0.010" \
    "++opt.lambda_perceptual_upper_torso_core_patch=0.012" \
    "++opt.lambda_perceptual_waist_patch=0.006" \
    "++opt.lambda_perceptual_shoulder_arm=0.012" \
    "++opt.lambda_perceptual_waist=0.006" \
    "++opt.lambda_mask_shoulder_arm_boundary_hard=0.0" \
    "++opt.lambda_mask_upper_torso_boundary_hard=0.0" \
    "++opt.lambda_mask_shoulder_arm_region_hard=0.0" \
    "++opt.lambda_silhouette_shoulder_arm_outer_shell=0.0" \
    "++opt.lambda_silhouette_upper_torso_outer_shell=0.0" \
    "opt.grad_clip=0.0022" \
    "test_interval=0" \
    "test_iterations=$checkpoint_hydra" \
    "save_iterations=$checkpoint_hydra" \
    "checkpoint_iterations=$checkpoint_hydra" \
    "++validation_image_log_limit=0" \
    "$@" > "$LOG_DIR/${name}.log" 2>&1

  local train_status=$?
  if [ "$train_status" -ne 0 ]; then
    write_summary_row "$name" "train" "$exp_dir" "" "train_failed"
    log_event "$gpu" "$name" "train_failed" "status=$train_status log=$LOG_DIR/${name}.log"
    return "$train_status"
  fi

  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"

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
  local ckpts360="90,180,270,360"
  local ckpts420="100,220,320,420"
  local region_asset_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_structure_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*,structured_trunk_output_head_local_fusion_projs.*,structured_trunk_output_head_local_geometry_fusion_projs.*]"
  local highcap_region_overrides=(
    "++model.texture.detail_residual.high_frequency.enable=true"
    "++model.texture.detail_residual.high_frequency.tiny_repair_scale=0.030"
    "++model.texture.detail_residual.high_frequency.scale=0.070"
    "++model.texture.detail_residual.high_frequency.gate_bias=-0.02"
    "++model.texture.detail_residual.high_frequency.min_gate=0.04"
    "++model.texture.detail_residual.high_frequency.face_branch.enable=true"
    "++model.texture.detail_residual.high_frequency.face_branch.tiny_repair_scale=0.016"
    "++model.texture.detail_residual.high_frequency.face_branch.scale=0.240"
    "++model.texture.detail_residual.high_frequency.face_branch.gate_bias=-0.08"
    "++model.texture.detail_residual.high_frequency.face_branch.min_gate=0.08"
    "++model.texture.structured_trunk.output_head.local_color.owner.enable=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.use_support=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.use_region_gate=false"
    "++model.texture.structured_trunk.output_head.local_color.owner.scale=0.82"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.enable=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.scale=0.88"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_color_input=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_color_output=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_geometry_raw=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_support_feature=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.init_from_local_color=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.init_scale=0.70"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_gain=1.30"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_bias=-0.01"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.min_gate=0.10"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.enable=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.scale=1.0"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=0.96"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.base_mix=0.0"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.max=1.0"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.use_support=true"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_offset=0.18"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_gain=7.0"
    "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_power=1.18"
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=0.60"
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.gate_gain=1.24"
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.min_gate=0.08"
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.max_residual=0.22"
  )
  local aggressive_detail_losses=(
    "++opt.owner_local_detail_boost_enable=true"
    "++opt.owner_local_detail_boost_warmup_iters=0"
    "++opt.owner_local_detail_boost_takeover_floor=0.12"
    "++opt.owner_local_detail_boost_takeover_gain=1.42"
    "++opt.owner_local_detail_boost_takeover_power=1.18"
    "++opt.owner_local_detail_boost_min_signal=0.0"
    "++opt.owner_local_detail_boost_detail_max_extra=0.90"
    "++opt.owner_local_detail_boost_luma_max_extra=1.25"
    "++opt.owner_local_detail_boost_patch_max_extra=0.92"
    "++opt.owner_local_detail_boost_edge_max_extra=0.42"
    "++opt.lambda_detail_face=0.010"
    "++opt.lambda_detail_shoulder_arm=0.010"
    "++opt.lambda_detail_waist=0.006"
    "++opt.lambda_detail_face_luma_dog=0.038"
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.034"
    "++opt.lambda_detail_upper_torso_luma_dog=0.038"
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.052"
    "++opt.lambda_detail_waist_luma_dog=0.024"
    "++opt.lambda_perceptual_face_patch=0.024"
    "++opt.lambda_perceptual_shoulder_arm_patch=0.018"
    "++opt.lambda_perceptual_upper_torso_patch=0.018"
    "++opt.lambda_perceptual_upper_torso_core_patch=0.024"
    "++opt.lambda_perceptual_waist_patch=0.012"
    "opt.lambda_l1_fg=0.128"
    "opt.lambda_l1_boundary=0.230"
    "opt.lambda_perceptual=0.052"
    "opt.lambda_edge_face=0.020"
    "opt.lambda_edge_shoulder_arm=0.024"
    "opt.lambda_edge_waist=0.010"
    "opt.grad_clip=0.0060"
  )
  local reliable_loss_overrides=(
    "++opt.reliable_view_supervision_enable=true"
    "++opt.reliable_view_apply_luma_dog=true"
    "++opt.reliable_view_apply_patch_perceptual=true"
    "++opt.reliable_view_apply_region_perceptual=false"
    "++opt.reliable_view_apply_global_perceptual=false"
    "++opt.reliable_view_default_highfreq_weight=0.88"
    "++opt.reliable_view_unknown_highfreq_weight=0.42"
    "++opt.reliable_view_highfreq_power=1.35"
    "++opt.reliable_view_highfreq_min_weight=0.22"
    "++opt.reliable_view_highfreq_max_weight=1.68"
  )

  case "$job" in
    v221a_stageB_mini_overfit_texture_capacity)
      run_one "$job" "$gpu" "$V215C_EXP" "$V215C410_CKPT" 109410 420 2.4e-06 "$ckpts420" "$MINI_VIEWS" \
        "$region_asset_patterns" \
        "++opt.train_sample_mode=random" \
        "++opt.train_sample_log_interval=40" \
        "${highcap_region_overrides[@]}" \
        "${aggressive_detail_losses[@]}" \
        "++opt.reliable_view_supervision_enable=false" \
        "opt.lambda_l1_boundary=0.245" \
        "opt.lambda_perceptual=0.045" \
        "opt.grad_clip=0.0100"
      local run_status=$?
      if [ "$run_status" -ne 0 ]; then
        return "$run_status"
      fi
      if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
        local exp_dir="$ROOT/exp/stageB/377_region_asset_clarity_${job}_${RUN_ID}"
        render_train_probe_and_diag "$job" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "best" "$MINI_VIEWS" "$MINI_FRAMES" "${HP_SELECT_MINI[@]}"
      fi
      ;;
    v221b_stageB_reliable5_region_capacity)
      run_one "$job" "$gpu" "$V215C_EXP" "$V215C410_CKPT" 109410 420 1.25e-06 "$ckpts420" "$RELIABLE5" \
        "$region_asset_patterns" \
        "${highcap_region_overrides[@]}" \
        "${aggressive_detail_losses[@]}" \
        "${reliable_loss_overrides[@]}" \
        "opt.lambda_l1_boundary=0.232" \
        "opt.lambda_perceptual=0.050" \
        "opt.grad_clip=0.0060"
      ;;
    v221c_stageB_all20_highcap_region_asset)
      run_one "$job" "$gpu" "$V215C_EXP" "$V215C410_CKPT" 109410 420 1.15e-06 "$ckpts420" "$ALL20" \
        "$region_asset_patterns" \
        "${highcap_region_overrides[@]}" \
        "${aggressive_detail_losses[@]}" \
        "${reliable_loss_overrides[@]}" \
        "opt.lambda_l1_boundary=0.236" \
        "opt.lambda_perceptual=0.050" \
        "opt.grad_clip=0.0055"
      ;;
    v221d_stageB_footprint_scaling_probe)
      run_one "$job" "$gpu" "$V215C_EXP" "$V215C410_CKPT" 109410 360 1.05e-06 "$ckpts360" "$RELIABLE5" \
        "$region_asset_patterns" \
        "${highcap_region_overrides[@]}" \
        "${aggressive_detail_losses[@]}" \
        "${reliable_loss_overrides[@]}" \
        "opt.opacity_lr=0.00030" \
        "opt.scaling_lr=0.000035" \
        "opt.rotation_lr=0.0" \
        "++resume.require_no_densify_on_resume=true" \
        "++resume.disable_densify_on_resume=true" \
        "++opt.boundary_opacity_residual_lr=0.000025" \
        "++opt.boundary_scaling_residual_lr=0.000008" \
        "opt.lambda_l1_boundary=0.250" \
        "opt.lambda_perceptual=0.046" \
        "opt.grad_clip=0.0040"
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
  if [ "$DO_RENDER" != "1" ] || [ "$SMOKE" = "1" ]; then
    return 0
  fi
  local render_exps=()
  local labels=()
  if [ -d "$V215C410_RENDER/test-view/renders" ]; then
    render_exps+=("$V215C410_RENDER")
    labels+=(v215c)
  fi
  if [ -d "$V198A_RENDER/test-view/renders" ]; then
    render_exps+=("$V198A_RENDER")
    labels+=(v198a)
  fi
  local names=(
    v221a_stageB_mini_overfit_texture_capacity
    v221b_stageB_reliable5_region_capacity
    v221c_stageB_all20_highcap_region_asset
    v221d_stageB_footprint_scaling_probe
  )
  local short_labels=(v221a v221b v221c v221d)
  local idx
  for idx in "${!names[@]}"; do
    local path="$ROOT/exp/stageB/377_region_asset_clarity_${names[$idx]}_${RUN_ID}_render_quick_best"
    if [ -d "$path/test-view/renders" ]; then
      render_exps+=("$path")
      labels+=("${short_labels[$idx]}")
    fi
  done
  if [ "${#render_exps[@]}" -lt 2 ]; then
    return 0
  fi
  "$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
    --render-exp "${render_exps[@]}" \
    --labels "${labels[@]}" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --output-dir "$LOG_DIR/compare_panels/main_upper_crop" \
    --select "${HP_SELECT[@]}" \
    --crop 150 35 650 430 \
    --panel-width 230 \
    --stack > "$LOG_DIR/compare_main_upper_crop.log" 2>&1 || true
  "$PYTHON_BIN" tools/make_377_render_comparison_montage.py \
    --render-exp "${render_exps[@]}" \
    --labels "${labels[@]}" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --output-dir "$LOG_DIR/compare_panels/main_full_body" \
    --select "${HP_SELECT[@]}" \
    --panel-width 190 \
    --stack > "$LOG_DIR/compare_main_full_body.log" 2>&1 || true
}

DEFAULT_JOBS=(
  v221a_stageB_mini_overfit_texture_capacity
  v221b_stageB_reliable5_region_capacity
  v221c_stageB_all20_highcap_region_asset
  v221d_stageB_footprint_scaling_probe
)
DEFAULT_GPUS=(0 1 2 3)

if [ -n "$JOBS" ]; then
  mapfile -t SELECTED_JOBS < <(split_csv "$JOBS")
else
  SELECTED_JOBS=("${DEFAULT_JOBS[@]}")
fi
if [ -n "$GPUS" ]; then
  mapfile -t SELECTED_GPUS < <(split_csv "$GPUS")
else
  SELECTED_GPUS=("${DEFAULT_GPUS[@]}")
fi
if [ "${#SELECTED_GPUS[@]}" -eq 0 ]; then
  echo "no GPUs selected" >&2
  exit 4
fi

declare -A JOBS_BY_GPU=()
QUEUE_GPUS=()
for idx in "${!SELECTED_JOBS[@]}"; do
  gpu="${SELECTED_GPUS[$((idx % ${#SELECTED_GPUS[@]}))]}"
  if [[ -z "${JOBS_BY_GPU[$gpu]+set}" ]]; then
    QUEUE_GPUS+=("$gpu")
    JOBS_BY_GPU[$gpu]=""
  fi
  JOBS_BY_GPU[$gpu]+=" ${SELECTED_JOBS[$idx]}"
done

for gpu in "${QUEUE_GPUS[@]}"; do
  read -ra gpu_jobs <<< "${JOBS_BY_GPU[$gpu]}"
  launch_queue "$gpu" "${gpu_jobs[@]}"
done

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "DEADLINE_BJT=$DEADLINE_BJT"
cat "$PIDS"

wait

build_compare_panels
log_event "all" "queue" "all_done" "summary=$SUMMARY"
echo "SUMMARY=$SUMMARY"
