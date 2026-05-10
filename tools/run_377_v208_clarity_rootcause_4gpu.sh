#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
OFFSET_ITER="${OFFSET_ITER:-109000}"
SEED="${SEED:--1}"

BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
DATA_ROOT="$ROOT/data/ZJUMoCap"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT/exp/stageA2/logs/v205teacher_20260509_130130_bjt/reliable_teacher_analysis}"
ANALYSIS_JSON="$ANALYSIS_DIR/reliable_teacher_summary.json"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v208_clarity_rootcause_$RUN_ID}"
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

ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CAM_WEIGHTS="$(read_analysis_value 'data["camera_weights_omega"]')"
UNIFORM_WEIGHTS="{1:1.0,2:1.0,3:1.0,4:1.0,5:1.0,6:1.0,7:1.0,8:1.0,9:1.0,10:1.0,11:1.0,12:1.0,13:1.0,14:1.0,15:1.0,16:1.0,17:1.0,18:1.0,19:1.0,20:1.0}"

printf 'RUN_ID=%s\nBASE_CKPT=%s\nANALYSIS_DIR=%s\nALL20=%s\nCAM_WEIGHTS=%s\nUNIFORM_WEIGHTS=%s\n' \
  "$RUN_ID" "$BASE_CKPT" "$ANALYSIS_DIR" "$ALL20" "$CAM_WEIGHTS" "$UNIFORM_WEIGHTS" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

variants:
  v208a_unweighted_all20_edge_guard:
    purpose: isolate whether v206 gain comes from camera weighting or common edge/loss setup
    trainable: high-frequency texture heads only
    sampler: frame-balanced uniform camera weights
  v208b_owner_local_trainable:
    purpose: test whether clarity is capped because v206 only trained high-frequency heads
    trainable: high-frequency heads + local_color / owner / owner-boundary heads
  v208c_owner_takeover_boost:
    purpose: test whether stronger region-owner takeover helps reduce shared texture averaging
    trainable: same as v208b, with stronger owner takeover and owner-local detail boost
  v208d_view_conflict_residual_safe:
    purpose: test whether a gated view-conflict residual can absorb multi-view high-frequency conflict
    trainable: newly added view_conflict residual MLP/gate only

checks:
  ckpt109300 / ckpt109450 / ckpt109600 / ckpt109800 / best quick render + contour diagnostics
EOF

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
  local iterations="$3"
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
    "dataset.train_views=$ALL20" \
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
    "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*]" \
    "++opt.lambda_binding_parsing=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_weights=$CAM_WEIGHTS" \
    "++opt.train_sample_camera_min_prob=0.015" \
    "++opt.train_sample_camera_max_prob=0.100" \
    "++opt.train_sample_log_interval=200" \
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
    "++opt.clarity_debug_enable=true" \
    "++opt.clarity_debug_interval=200" \
    "++opt.clarity_debug_warmup_iters=0" \
    "++opt.lambda_detail_face=0.0" \
    "++opt.lambda_detail_shoulder_arm=0.0" \
    "++opt.lambda_detail_waist=0.0" \
    "++opt.lambda_detail_face_luma_dog=0.0" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
    "++opt.lambda_detail_waist_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
    "++opt.reliable_view_supervision_enable=false" \
    "++opt.reliable_view_apply_edge=false" \
    "++opt.reliable_view_apply_detail=false" \
    "++opt.reliable_view_apply_luma_dog=false" \
    "++opt.reliable_view_apply_patch_perceptual=false" \
    "++opt.reliable_view_apply_region_perceptual=false" \
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
    "test_interval=150" \
    "test_iterations=[300,450,600,800]" \
    "save_iterations=[300,450,600,800]" \
    "checkpoint_iterations=[300,450,600,800]" \
    "++validation_image_log_limit=0" \
    "$@" > "$train_log" 2>&1

  local ckpt300="$exp_dir/ckpt$((OFFSET_ITER + 300)).pth"
  local ckpt450="$exp_dir/ckpt$((OFFSET_ITER + 450)).pth"
  local ckpt600="$exp_dir/ckpt$((OFFSET_ITER + 600)).pth"
  local ckpt800="$exp_dir/ckpt$((OFFSET_ITER + 800)).pth"
  render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt300" "ckpt$((OFFSET_ITER + 300))"
  render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt450" "ckpt$((OFFSET_ITER + 450))"
  render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt600" "ckpt$((OFFSET_ITER + 600))"
  render_and_diag "$name" "$gpu" "$exp_dir" "$ckpt800" "ckpt$((OFFSET_ITER + 800))"
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

owner_local_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*,structured_trunk_output_head_local_fusion_projs.*,structured_trunk_output_head_local_geometry_fusion_projs.*]"

view_conflict_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*]"

view_conflict_overrides=(
  "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_]"
  "$view_conflict_patterns"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.enable=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=[0.00,1,0.25,300,0.55,600,0.75]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.020"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.tiny_repair_scale=0.80"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.input_detach=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.chroma_center=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_bias=-0.65"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.min_gate=0.025"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.inherit_point_gate=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate_combine_mode=mul"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.enable=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.combine_mode=max"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.fallback_to_full=false"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.semantic_id_weights=[[1,1.00],[2,0.92],[3,0.78]]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.joint_id_weights=[[12,0.95],[13,1.00],[14,1.00],[15,1.00],[16,0.85],[17,0.85],[18,0.48],[19,0.48]]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.min_gate=0.16"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_neurons=64"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_hidden_layers=2"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.skip_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.cond_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.multires=0"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.last_layer_init=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_neurons=40"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_hidden_layers=1"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.skip_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.cond_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.multires=0"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.last_layer_init=true"
)

launch "v208a_unweighted_all20_edge_guard" 0 800 1.8e-06 \
  "++opt.train_sample_camera_weights=$UNIFORM_WEIGHTS" \
  "++opt.train_sample_camera_min_prob=0.0" \
  "++opt.train_sample_camera_max_prob=0.0"

launch "v208b_owner_local_trainable" 1 800 8.0e-07 \
  "$owner_local_patterns"

launch "v208c_owner_takeover_boost" 2 800 8.0e-07 \
  "$owner_local_patterns" \
  "++model.texture.structured_trunk.output_head.local_color.owner.scale=[0.28,1,0.48,300,0.70,600,0.85]" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.scale=[0.28,1,0.48,300,0.70,600,0.85]" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.strength=[0.25,1,0.55,300,0.80,600,1.00]" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_offset=0.18" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_gain=9.0" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.support_power=1.15" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.region_strength.face=1.05" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.region_strength.shoulder_upper=1.00" \
  "++model.texture.structured_trunk.output_head.local_color.owner.head.takeover.region_strength.upper_torso=0.95" \
  "++opt.owner_local_detail_boost_warmup_iters=80" \
  "++opt.owner_local_detail_boost_takeover_floor=0.22" \
  "++opt.owner_local_detail_boost_takeover_gain=1.35" \
  "++opt.owner_local_detail_boost_takeover_power=1.20" \
  "++opt.owner_local_detail_boost_min_signal=0.025" \
  "++opt.owner_local_detail_boost_luma_max_extra=[0.40,1,0.95,300,1.45,600,1.80]" \
  "++opt.owner_local_detail_boost_patch_max_extra=[0.28,1,0.70,300,1.10,600,1.35]" \
  "++opt.owner_local_detail_boost_edge_max_extra=[0.05,1,0.12,300,0.20,600,0.25]" \
  "++opt.owner_local_detail_boost_boundary_max_extra=[0.04,1,0.10,300,0.16,600,0.20]"

launch "v208d_view_conflict_residual_safe" 3 800 3.0e-06 \
  "${view_conflict_overrides[@]}" \
  "opt.lambda_perceptual=0.130" \
  "opt.lambda_l1_boundary=0.17" \
  "++opt.perceptual_adaptive_edge_protect=0.78" \
  "opt.grad_clip=0.0040"

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
cat "$PIDS"

wait

echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] all done"
echo "SUMMARY=$SUMMARY"
