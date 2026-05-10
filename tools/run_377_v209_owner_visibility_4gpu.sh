#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"

BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/best_ckpt.pth}"
V208B_EXP="${V208B_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v208b_owner_local_trainable_20260509_150110_bjt}"
V208B_CKPT="${V208B_CKPT:-$V208B_EXP/ckpt109600.pth}"
DATA_ROOT="$ROOT/data/ZJUMoCap"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT/exp/stageA2/logs/v205teacher_20260509_130130_bjt/reliable_teacher_analysis}"
ANALYSIS_JSON="$ANALYSIS_DIR/reliable_teacher_summary.json"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v209_owner_visibility_$RUN_ID}"
SUMMARY="$LOG_DIR/summary.tsv"
PIDS="$LOG_DIR/pids.tsv"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"

mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"
cd "$ROOT"

for required in "$BASE_CKPT" "$V208B_CKPT" "$ANALYSIS_JSON"; do
  if [ ! -f "$required" ]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

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

printf 'RUN_ID=%s\nBASE_CKPT=%s\nV208B_CKPT=%s\nANALYSIS_DIR=%s\nALL20=%s\nCAM_WEIGHTS=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$BASE_CKPT" "$V208B_CKPT" "$ANALYSIS_DIR" "$ALL20" "$CAM_WEIGHTS" "$SMOKE" "$DO_RENDER" \
  | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

variants:
  v209a_v208b600_owner_local_vc_lowgate:
    base: v208b ckpt109600
    purpose: combine the v208b owner/local carrier signal with a low-strength view-conflict gate
  v209b_v198a_owner_local_short:
    base: v198a
    purpose: confirm the owner/local effect in a shorter early-stop run without view-conflict residual
  v209c_v198a_owner_local_vc_lowgate:
    base: v198a
    purpose: test owner/local + low-strength view-conflict directly from the main anchor
  v209d_v198a_owner_local_reliable_view:
    base: v198a
    purpose: test owner/local carrier plus conservative reliable-view detail/patch supervision

acceptance:
  - v198a remains the strict main anchor
  - LPIPS alone is not enough; fg_l1, boundary_l1, edge_px, and crop montage must stay stable
  - best candidate should beat v198a LPIPS without the v206-style soft-edge regression
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
        render_metrics = {key: float(data[key]) for key in data.files if key in ("lpips", "psnr", "ssim")}

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

csv_to_hydra_list() {
  local csv="$1"
  echo "[$csv]"
}

run_one() {
  local name="$1"
  local gpu="$2"
  local iterations="$3"
  local texture_lr="$4"
  local start_ckpt="$5"
  local base_iter="$6"
  local checkpoint_csv="$7"
  shift 7

  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_csv="2"
  fi

  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${name}_${RUN_ID}"
  local train_log="$LOG_DIR/${name}.log"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  local checkpoint_hydra
  checkpoint_hydra="$(csv_to_hydra_list "$checkpoint_csv")"
  local test_hydra="$checkpoint_hydra"
  local save_hydra="$checkpoint_hydra"

  if [ "$SMOKE" = "1" ]; then
    test_hydra="[]"
    save_hydra="[]"
    checkpoint_hydra="[]"
  fi

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
    "start_checkpoint=$start_ckpt" \
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
    "test_interval=0" \
    "test_iterations=$test_hydra" \
    "save_iterations=$save_hydra" \
    "checkpoint_iterations=$checkpoint_hydra" \
    "++validation_image_log_limit=0" \
    "$@" > "$train_log" 2>&1

  if [ "$DO_RENDER" = "1" ]; then
    IFS=',' read -ra checkpoints <<< "$checkpoint_csv"
    for local_ckpt in "${checkpoints[@]}"; do
      local global_ckpt=$((base_iter + local_ckpt))
      render_and_diag "$name" "$gpu" "$exp_dir" "$exp_dir/ckpt${global_ckpt}.pth" "ckpt${global_ckpt}"
    done
    render_and_diag "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "best"
  fi
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

owner_local_view_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*,structured_trunk_output_head_local_fusion_projs.*,structured_trunk_output_head_local_geometry_fusion_projs.*,detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*]"

view_conflict_common=(
  "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.enable=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.014"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.tiny_repair_scale=0.55"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.input_detach=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.chroma_center=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_bias=-0.85"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.min_gate=0.015"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.inherit_point_gate=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate_combine_mode=mul"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.enable=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.combine_mode=max"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.fallback_to_full=false"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.semantic_id_weights=[[1,1.00],[2,0.90],[3,0.75]]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.joint_id_weights=[[12,0.92],[13,1.00],[14,1.00],[15,1.00],[16,0.82],[17,0.82],[18,0.42],[19,0.42]]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.point_gate.min_gate=0.18"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_neurons=48"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.n_hidden_layers=2"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.skip_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.cond_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.multires=0"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.mlp.last_layer_init=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_neurons=32"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.n_hidden_layers=1"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.skip_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.cond_in=[]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.multires=0"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_mlp.last_layer_init=true"
)

launch "v209a_v208b600_owner_local_vc_lowgate" 0 600 6.0e-07 "$V208B_CKPT" 109600 "200,400,600" \
  "$owner_local_view_patterns" \
  "${view_conflict_common[@]}" \
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=[0.00,109601,0.10,109800,0.20,110000,0.28,110200,0.32]" \
  "opt.lambda_l1_boundary=0.165" \
  "opt.lambda_perceptual=0.138" \
  "++opt.perceptual_adaptive_edge_protect=0.78" \
  "opt.grad_clip=0.0038"

launch "v209b_v198a_owner_local_short" 1 450 8.0e-07 "$BASE_CKPT" 109000 "200,300,450" \
  "$owner_local_patterns" \
  "opt.lambda_l1_boundary=0.160" \
  "opt.lambda_perceptual=0.145" \
  "++opt.perceptual_adaptive_edge_protect=0.76" \
  "opt.grad_clip=0.0042"

launch "v209c_v198a_owner_local_vc_lowgate" 2 600 8.0e-07 "$BASE_CKPT" 109000 "200,400,600" \
  "$owner_local_view_patterns" \
  "${view_conflict_common[@]}" \
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=[0.00,109001,0.10,109200,0.20,109400,0.28,109600,0.32]" \
  "opt.lambda_l1_boundary=0.165" \
  "opt.lambda_perceptual=0.138" \
  "++opt.perceptual_adaptive_edge_protect=0.78" \
  "opt.grad_clip=0.0038"

launch "v209d_v198a_owner_local_reliable_view" 3 600 7.0e-07 "$BASE_CKPT" 109000 "200,400,600" \
  "$owner_local_patterns" \
  "opt.lambda_l1_boundary=0.165" \
  "opt.lambda_perceptual=0.132" \
  "++opt.perceptual_adaptive_edge_protect=0.80" \
  "++opt.reliable_view_supervision_enable=true" \
  "++opt.reliable_view_camera_quality_weights=$CAM_WEIGHTS" \
  "++opt.reliable_view_default_highfreq_weight=0.90" \
  "++opt.reliable_view_unknown_highfreq_weight=0.70" \
  "++opt.reliable_view_highfreq_power=1.05" \
  "++opt.reliable_view_highfreq_min_weight=0.48" \
  "++opt.reliable_view_highfreq_max_weight=1.38" \
  "++opt.reliable_view_apply_luma_dog=true" \
  "++opt.reliable_view_apply_patch_perceptual=true" \
  "++opt.reliable_view_apply_region_perceptual=false" \
  "opt.grad_clip=0.0038"

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
cat "$PIDS"

wait

echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] all done"
echo "SUMMARY=$SUMMARY"
