#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
ITERATIONS="${ITERATIONS:-800}"
OFFSET_ITER="${OFFSET_ITER:-109000}"
SEED="${SEED:--1}"

BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
DATA_ROOT="$ROOT/data/ZJUMoCap"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
ANALYSIS_DIR="${ANALYSIS_DIR:?ANALYSIS_DIR is required}"
ANALYSIS_JSON="$ANALYSIS_DIR/reliable_teacher_summary.json"

LOG_DIR="$ROOT/exp/stageA2/logs/v205_reliable_view_subset_$RUN_ID"
SUMMARY="$LOG_DIR/summary.tsv"
PIDS="$LOG_DIR/pids.tsv"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"
cd "$ROOT"

if [ ! -f "$BASE_CKPT" ]; then
  echo "missing checkpoint: $BASE_CKPT" >&2
  exit 2
fi
if [ ! -f "$ANALYSIS_JSON" ]; then
  echo "missing analysis json: $ANALYSIS_JSON" >&2
  exit 3
fi

read_analysis_value() {
  local expr="$1"
  "$PYTHON_BIN" - "$ANALYSIS_JSON" "$expr" <<'PY'
import json
import sys

path, expr = sys.argv[1:3]
data = json.loads(open(path, encoding="utf-8").read())
value = eval(expr, {"__builtins__": {}}, {"data": data, "sorted": sorted, "str": str, "int": int})
if isinstance(value, (list, tuple)):
    print("[" + ",".join(str(int(x)) for x in value) + "]")
else:
    print(str(value))
PY
}

TOP8="$(read_analysis_value 'sorted(data["view_sets"]["top8"])')"
TOP12="$(read_analysis_value 'sorted(data["view_sets"]["top12"])')"
DROP_BAD4="$(read_analysis_value 'sorted(data["view_sets"]["drop_bad4"])')"
ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CAM_WEIGHTS="$(read_analysis_value 'data["camera_weights_omega"]')"

printf 'RUN_ID=%s\nANALYSIS_DIR=%s\nTOP8=%s\nTOP12=%s\nDROP_BAD4=%s\nCAM_WEIGHTS=%s\n' \
  "$RUN_ID" "$ANALYSIS_DIR" "$TOP8" "$TOP12" "$DROP_BAD4" "$CAM_WEIGHTS" | tee "$LOG_DIR/run_info.txt"

printf 'name\tlabel\texp_dir\trender_exp\ttrain_lpips_fg\ttrain_l1_fg\ttrain_psnr_fg\trender_lpips\trender_psnr\trender_ssim\tfg_l1\tboundary_l1\tedge_px\tstatus\n' > "$SUMMARY"
printf 'name\tgpu\tpid\n' > "$PIDS"

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
render = Path(render_exp)

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
render_path = render / "test-view" / "results.npz"
if render_path.exists():
    data = np.load(render_path)
    render_metrics = {key: float(data[key]) for key in data.files if key in ("lpips", "psnr", "ssim")}

contour = {}
contour_path = render / "diagnostics" / "contour_summary.json"
if contour_path.exists():
    contour = json.loads(contour_path.read_text())

row = [
    name,
    label,
    str(exp),
    str(render),
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
  local render_exp="${exp_dir}_render_quick_${label}"
  local render_log="$LOG_DIR/${name}_render_${label}.log"
  local contour_log="$LOG_DIR/${name}_contour_${label}.log"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_render_${label}"

  if [ ! -f "$ckpt_path" ]; then
    write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "missing_ckpt"
    return 0
  fi

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

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 > "$contour_log" 2>&1

  write_summary_row "$name" "$label" "$exp_dir" "$render_exp" "ok"
}

run_one() {
  local name="$1"
  local gpu="$2"
  local views="$3"
  local texture_lr="$4"
  shift 4

  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${name}_${RUN_ID}"
  local train_log="$LOG_DIR/${name}.log"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"

  mkdir -p "$exp_dir"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$views" \
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
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$exp_dir" \
    "hydra.run.dir=$hydra_run_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.clear_boundary_tags_on_resume=false" \
    "opt.iterations=$ITERATIONS" \
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
    "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*]" \
    "++opt.lambda_binding_parsing=0.0" \
    "++opt.face_region_source=parser_prefer" \
    "++opt.face_region_parser_dilate=1" \
    "++opt.face_region_source_aware_validity_enable=true" \
    "++opt.face_region_min_pixels_parser=24" \
    "++opt.shoulder_arm_region_source=parser_prefer" \
    "++opt.shoulder_arm_region_parser_dilate=3" \
    "++opt.shoulder_arm_region_source_aware_validity_enable=true" \
    "++opt.shoulder_arm_region_min_pixels_parser=40" \
    "++opt.upper_torso_region_source=parser_prefer" \
    "++opt.upper_torso_region_parser_dilate=3" \
    "++opt.waist_region_source=parser_prefer" \
    "++opt.waist_region_parser_dilate=1" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.0" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
    "++opt.lambda_detail_waist_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
    "++opt.reliable_view_apply_detail=false" \
    "++opt.reliable_view_apply_luma_dog=false" \
    "opt.lambda_l1_fg=0.11" \
    "opt.lambda_l1_boundary=0.16" \
    "opt.lambda_perceptual=0.145" \
    "++opt.perceptual_exclude_boundary_width=24" \
    "++opt.face_perceptual_exclude_boundary_width=10" \
    "++opt.shoulder_arm_perceptual_exclude_boundary_width=24" \
    "++opt.upper_torso_perceptual_exclude_boundary_width=24" \
    "++opt.upper_torso_core_perceptual_exclude_boundary_width=14" \
    "++opt.waist_perceptual_exclude_boundary_width=24" \
    "++opt.perceptual_adaptive_edge_protect=0.72" \
    "opt.lambda_edge_face=0.010" \
    "opt.lambda_edge_shoulder_arm=0.020" \
    "opt.lambda_edge_waist=0.008" \
    "opt.grad_clip=0.0045" \
    "test_interval=400" \
    "test_iterations=[400,800]" \
    "save_iterations=[400,800]" \
    "checkpoint_iterations=[400,800]" \
    "++validation_image_log_limit=0" \
    "$@" > "$train_log" 2>&1

  local ckpt400="$exp_dir/ckpt$((OFFSET_ITER + 400)).pth"
  local ckpt800="$exp_dir/ckpt$((OFFSET_ITER + 800)).pth"
  render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt400" "ckpt$((OFFSET_ITER + 400))"
  if [ "$ITERATIONS" -ge 800 ]; then
    render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt800" "ckpt$((OFFSET_ITER + 800))"
  fi
  render_and_diag "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "best"
}

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  (
    echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] start $name gpu=$gpu" > "$LOG_DIR/${name}.status"
    if run_one "$name" "$gpu" "$@"; then
      echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] done $name" >> "$LOG_DIR/${name}.status"
    else
      echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] failed $name" >> "$LOG_DIR/${name}.status"
      write_summary_row "$name" "train" "$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${name}_${RUN_ID}" "" "train_failed"
    fi
  ) &
  local pid=$!
  printf '%s\t%s\t%s\n' "$name" "$gpu" "$pid" >> "$PIDS"
}

launch "v205a_top8_reliable" 0 "$TOP8" 1.5e-06
launch "v205b_top12_reliable" 1 "$TOP12" 1.7e-06
launch "v205c_weighted_all20" 2 "$ALL20" 1.8e-06 \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_weights=$CAM_WEIGHTS" \
  "++opt.train_sample_camera_min_prob=0.015" \
  "++opt.train_sample_camera_max_prob=0.100" \
  "++opt.train_sample_log_interval=200"
launch "v205d_drop_bad4" 3 "$DROP_BAD4" 1.8e-06

cat "$PIDS"
wait

echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] all done"
echo "SUMMARY=$SUMMARY"
