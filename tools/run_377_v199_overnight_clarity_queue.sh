#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
DEADLINE_EPOCH="${1:-$(( $(date +%s) + 9 * 60 * 60 ))}"

BASE_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue"
BASE_CKPT="$BASE_EXP/best_ckpt.pth"
BASE_RENDER="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v198a_v197a_boundary_substrate_continue_20260508_224752_bjt_v198a_boundary_substrate_continue_render_quick"
DATA_ROOT="$ROOT/data/ZJUMoCap"
LOG_DIR="$ROOT/exp/stageA2/logs/v199_overnight_clarity_$RUN_ID"
SUMMARY="$LOG_DIR/summary.tsv"
CANDIDATES="$LOG_DIR/candidates.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

log_msg() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$*" | tee -a "$LOG_DIR/queue.log"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $(( DEADLINE_EPOCH - now ))
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU_ID" "$DEADLINE_EPOCH" "$1" "$2" <<'PY'
import json
import sys
import time
from pathlib import Path

path, run_id, gpu_id, deadline_epoch, phase, detail = sys.argv[1:]
data = {
    "run_id": run_id,
    "gpu_id": gpu_id,
    "deadline_epoch": int(deadline_epoch),
    "phase": phase,
    "detail": detail,
    "now_epoch": int(time.time()),
}
Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
}

append_summary() {
  local name="$1"
  local status="$2"
  local exp_dir="$3"
  local log_path="$4"
  local extra="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$exp_dir" "$log_path" "$extra" >> "$SUMMARY"
}

make_iter_list() {
  local total="$1"
  local step="${2:-1500}"
  local vals=()
  local i
  for (( i=step; i<total; i+=step )); do
    vals+=("$i")
  done
  vals+=("$total")
  local IFS=,
  printf '[%s]' "${vals[*]}"
}

run_step() {
  local name="$1"
  local log_path="$2"
  shift 2
  log_msg "START $name"
  write_status "$name" "running"
  local start_ts end_ts status
  start_ts="$(date +%s)"
  "$@" >"$log_path" 2>&1
  status=$?
  end_ts="$(date +%s)"
  if [ "$status" -eq 0 ]; then
    log_msg "DONE $name elapsed=$((end_ts - start_ts))s log=$log_path"
  else
    log_msg "FAILED $name status=$status elapsed=$((end_ts - start_ts))s log=$log_path"
  fi
  return "$status"
}

extract_train_metrics() {
  local exp_dir="$1"
  "$PYTHON_BIN" - "$exp_dir" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "best_test_metrics.json"
if not path.exists():
    print("")
    raise SystemExit(0)
data = json.loads(path.read_text())
keys = ["iteration", "psnr", "ssim", "lpips", "psnr_fg", "ssim_fg", "lpips_fg", "l1_fg", "selection_source", "selection_metric"]
print(",".join(f"{key}={data.get(key)}" for key in keys if key in data))
PY
}

extract_render_metrics() {
  local render_exp="$1"
  "$PYTHON_BIN" - "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

exp = Path(sys.argv[1])
result = exp / "test-view" / "results.npz"
parts = []
if result.exists():
    data = np.load(result)
    for key in ("lpips", "psnr", "ssim", "l1"):
        if key in data.files:
            parts.append(f"{key}={float(data[key].mean()):.8f}")
contour = exp / "diagnostics" / "contour_summary.json"
if contour.exists():
    c = json.loads(contour.read_text())
    for key in ("mean_fg_l1", "mean_boundary_l1", "mean_interior_l1", "mean_edge_symmetric_dist_px"):
        if key in c:
            parts.append(f"{key}={float(c[key]):.8f}")
print(",".join(parts))
PY
}

add_candidate() {
  local name="$1"
  local exp_dir="$2"
  local render_exp="$3"
  if [ -f "$exp_dir/best_ckpt.pth" ]; then
    printf '%s\t%s\t%s\n' "$name" "$exp_dir" "$render_exp" >> "$CANDIDATES"
  fi
}

pick_best_candidate() {
  "$PYTHON_BIN" - "$CANDIDATES" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

candidate_path = Path(sys.argv[1])
best = None
if not candidate_path.exists():
    raise SystemExit(2)

for raw in candidate_path.read_text().splitlines():
    if not raw.strip() or raw.startswith("name\t"):
        continue
    fields = raw.split("\t")
    if len(fields) < 3:
        continue
    name, exp_dir, render_exp = fields[:3]
    exp = Path(exp_dir)
    render = Path(render_exp)
    ckpt = exp / "best_ckpt.pth"
    if not ckpt.exists():
        continue
    score = None
    source = None
    result = render / "test-view" / "results.npz"
    if result.exists():
        try:
            data = np.load(result)
            if "lpips" in data.files:
                score = float(data["lpips"].mean())
                source = "render_lpips"
        except Exception:
            pass
    if score is None:
        metrics = exp / "best_test_metrics.json"
        if metrics.exists():
            data = json.loads(metrics.read_text())
            if "lpips_fg" in data:
                score = float(data["lpips_fg"])
                source = "train_lpips_fg"
    if score is None:
        continue
    record = (score, name, exp.as_posix(), ckpt.as_posix(), render.as_posix(), source)
    if best is None or record[0] < best[0]:
        best = record

if best is None:
    raise SystemExit(3)

score, name, exp, ckpt, render, source = best
print("\t".join([name, exp, ckpt, render, f"{score:.8f}", source]))
PY
}

run_train() {
  local name="$1"
  local exp_dir="$2"
  local start_ckpt="$3"
  local iterations="$4"
  local log_path="$5"
  local allow_missing_patterns="$6"
  shift 6

  local test_list save_list
  test_list="$(make_iter_list "$iterations" 1500)"
  save_list="$(make_iter_list "$iterations" 3000)"

  local -a args=(
    "$PYTHON_BIN" train.py
    --config-path "$BASE_EXP/.hydra"
    --config-name config
    mode=train
    "dataset.root_dir=$DATA_ROOT"
    "dataset.preload=false"
    "dataset.train_views=[21,22,23]"
    "dataset.val_views=[21,22,23]"
    "dataset.test_views.view=[21,22,23]"
    "dataset.train_frames=[0,570,1]"
    "dataset.val_frames=[0,570,30]"
    "dataset.test_frames.view=[0,570,30]"
    "start_checkpoint=$start_ckpt"
    "exp_dir=$exp_dir"
    "wandb_disable=true"
    "++resume.restore_converter_optimizer_state=false"
    "++resume.restore_converter_scheduler_state=false"
    "++resume.partial_converter_missing_keys_allow_patterns=$allow_missing_patterns"
    "++resume.disable_densify_on_resume=true"
    "++resume.disable_opacity_reset_on_resume=true"
    "++resume.require_no_densify_on_resume=true"
    "++resume.clear_boundary_tags_on_resume=false"
    "opt.iterations=$iterations"
    "test_interval=1500"
    "test_iterations=$test_list"
    "save_iterations=$save_list"
    "checkpoint_iterations=$save_list"
    "++validation_image_log_limit=0"
  )
  args+=("$@")

  mkdir -p "$exp_dir"
  run_step "$name" "$log_path" "${args[@]}"
}

render_eval() {
  local name="$1"
  local config_exp="$2"
  local ckpt="$3"
  local out_exp="$4"
  local log_path="$5"
  run_step "$name" "$log_path" \
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
      wandb_disable=true
}

run_contour_diag() {
  local name="$1"
  local render_exp="$2"
  local log_path="$3"
  run_step "$name" "$log_path" \
    "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
      --render-exp "$render_exp" \
      --dataset-root "$DATA_ROOT" \
      --subject CoreView_377 \
      --band-width 7 \
      --topk 12
}

run_candidate() {
  local name="$1"
  local exp_dir="$2"
  local start_ckpt="$3"
  local iterations="$4"
  local allow_missing_patterns="$5"
  shift 5

  local train_log="$LOG_DIR/${name}.log"
  local render_exp="${exp_dir}_render_quick"
  local render_log="$LOG_DIR/${name}_render.log"
  local contour_log="$LOG_DIR/${name}_contour.log"

  if run_train "$name" "$exp_dir" "$start_ckpt" "$iterations" "$train_log" "$allow_missing_patterns" "$@"; then
    append_summary "$name" "ok" "$exp_dir" "$train_log" "$(extract_train_metrics "$exp_dir")"
  else
    append_summary "$name" "failed" "$exp_dir" "$train_log" "train_failed"
  fi

  if [ -f "$exp_dir/best_ckpt.pth" ]; then
    if render_eval "${name}_render" "$exp_dir" "$exp_dir/best_ckpt.pth" "$render_exp" "$render_log"; then
      append_summary "${name}_render" "ok" "$render_exp" "$render_log" "$(extract_render_metrics "$render_exp")"
      if run_contour_diag "${name}_contour" "$render_exp" "$contour_log"; then
        append_summary "${name}_contour" "ok" "$render_exp" "$contour_log" "$(extract_render_metrics "$render_exp")"
      else
        append_summary "${name}_contour" "failed" "$render_exp" "$contour_log" "contour_failed"
      fi
      add_candidate "$name" "$exp_dir" "$render_exp"
    else
      append_summary "${name}_render" "failed" "$render_exp" "$render_log" "render_failed"
      add_candidate "$name" "$exp_dir" "$render_exp"
    fi
  fi
}

run_if_enough_time() {
  local min_seconds="$1"
  local name="$2"
  shift 2
  local remain
  remain="$(remaining_seconds)"
  if [ "$remain" -lt "$min_seconds" ]; then
    log_msg "SKIP $name remaining=${remain}s min_required=${min_seconds}s"
    append_summary "$name" "skipped" "" "$LOG_DIR/queue.log" "remaining=${remain}s"
    return 0
  fi
  "$@"
}

common_freeze_geometry_overrides=(
  "pipeline.pose_noise=0.0"
  "model.pose_correction.delay=1"
  "++model.pose_correction.train_root_orient=false"
  "++model.pose_correction.train_pose_body=false"
  "++model.pose_correction.train_pose_hand=false"
  "++model.pose_correction.train_trans=false"
  "++model.pose_correction.train_betas=false"
  "opt.position_lr_init=0.0"
  "opt.position_lr_final=0.0"
  "opt.feature_lr=0.0"
  "opt.opacity_lr=0.0"
  "opt.scaling_lr=0.0"
  "opt.rotation_lr=0.0"
  "opt.rigid_lr=0.0"
  "opt.non_rigid_lr=0.0"
  "opt.nr_latent_lr=0.0"
  "opt.pose_correction_lr=0.0"
  "++opt.camera_geometry_enable=true"
  "++opt.camera_geometry_lr=0.0"
  "++opt.camera_affine_enable=false"
  "++opt.camera_affine_lr=0.0"
  "++opt.boundary_opacity_residual_lr=0.0"
  "++opt.boundary_scaling_residual_lr=0.0"
  "++opt.latent_weight_decay=0.0"
  "opt.tex_latent_lr=0.0"
  "++opt.photometric_contour_debug_interval=300"
)

detail_loss_overrides=(
  "opt.lambda_l1=0.018"
  "++opt.lambda_l1_fg=0.060"
  "++opt.lambda_mask_boundary=0.0010"
  "++opt.lambda_mask_boundary_hard=0.0016"
  "++opt.lambda_silhouette_outer=0.0010"
  "++opt.lambda_silhouette_outer_shell=0.0016"
  "++opt.lambda_silhouette_inner=0.0008"
  "++opt.lambda_perceptual_face=0.130"
  "++opt.lambda_perceptual_shoulder_arm=0.054"
  "++opt.lambda_perceptual_waist=0.028"
  "++opt.lambda_perceptual_face_patch=0.092"
  "++opt.lambda_perceptual_shoulder_arm_patch=0.050"
  "++opt.lambda_perceptual_waist_patch=0.030"
  "++opt.lambda_perceptual_upper_torso_patch=0.056"
  "++opt.lambda_perceptual_upper_torso_core_patch=0.066"
  "++opt.lambda_detail_waist=0.004"
  "++opt.lambda_detail_waist_luma_dog=0.003"
  "++opt.detail_multiscale_scales=[1,2]"
  "++opt.detail_highpass_kernel=5"
  "++opt.detail_scale_decay=0.55"
  "++opt.detail_gradient_mix=0.18"
  "++opt.reliable_view_apply_detail=true"
  "++opt.reliable_view_apply_luma_dog=true"
  "++opt.reliable_view_apply_edge=false"
  "++opt.reliable_view_apply_patch_perceptual=true"
  "++opt.reliable_view_apply_region_perceptual=true"
  "++opt.contour_uncertainty_min_weight=0.50"
)

texture_hf_patterns="++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*]"

view_conflict_overrides=(
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.enable=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.scale=[0.00,1,0.28,1800,0.62,5200,1.00]"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.max_residual=0.020"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.tiny_repair_scale=0.85"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.input_detach=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.chroma_center=true"
  "++model.texture.detail_residual.high_frequency.view_conflict_residual.gate_bias=-0.70"
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

micro_geometry_overrides=(
  "pipeline.pose_noise=0.0"
  "model.pose_correction.delay=1"
  "++model.pose_correction.train_root_orient=false"
  "++model.pose_correction.train_pose_body=true"
  "++model.pose_correction.train_pose_hand=false"
  "++model.pose_correction.train_trans=false"
  "++model.pose_correction.train_betas=false"
  "++model.pose_correction.pose_body_train_joint_ids=[12,13,14,15,16,17]"
  "opt.position_lr_init=0.0"
  "opt.position_lr_final=0.0"
  "opt.feature_lr=0.0"
  "opt.opacity_lr=0.0"
  "opt.scaling_lr=0.0"
  "opt.rotation_lr=0.0"
  "opt.rigid_lr=0.0"
  "opt.non_rigid_lr=0.0"
  "opt.nr_latent_lr=0.0"
  "opt.pose_correction_lr=1.6e-07"
  "++opt.camera_geometry_enable=true"
  "++opt.camera_geometry_train_camera_ids=[21,22,23]"
  "++opt.camera_geometry_max_camera_id=23"
  "++opt.camera_geometry_strength=0.48"
  "++opt.camera_geometry_rot_max_deg=0.18"
  "++opt.camera_geometry_trans_max=0.004"
  "++opt.camera_geometry_lr=1.5e-05"
  "++opt.lambda_camera_geometry_reg=0.045"
  "++opt.boundary_aware_boundary_opacity_residual_scale=0.42"
  "++opt.boundary_aware_boundary_scaling_residual_scale=0.26"
  "++opt.boundary_opacity_residual_lr=2.6e-05"
  "++opt.boundary_scaling_residual_lr=8.0e-06"
  "++opt.lambda_boundary_opacity_residual_reg=0.0018"
  "++opt.lambda_boundary_scaling_residual_reg=0.00055"
  "++opt.lambda_boundary_opacity_residual_smooth=0.0018"
  "++opt.lambda_boundary_scaling_residual_smooth=0.0013"
  "++opt.camera_affine_enable=false"
  "++opt.camera_affine_lr=0.0"
  "++opt.latent_weight_decay=0.0"
  "opt.tex_latent_lr=0.0"
  "++opt.photometric_contour_debug_interval=300"
)

printf 'name\tstatus\texp_dir\tlog\textra\n' > "$SUMMARY"
printf 'name\texp_dir\trender_exp\n' > "$CANDIDATES"

log_msg "v199 overnight clarity queue start run_id=$RUN_ID gpu=$GPU_ID deadline_epoch=$DEADLINE_EPOCH"
log_msg "base_exp=$BASE_EXP"
log_msg "base_ckpt=$BASE_CKPT"
write_status "queue" "started"

if [ ! -f "$BASE_CKPT" ]; then
  log_msg "missing base checkpoint: $BASE_CKPT"
  write_status "queue" "missing_base_ckpt"
  exit 2
fi

add_candidate "v198a_baseline" "$BASE_EXP" "$BASE_RENDER"

V199A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v199a_v198a_stable_detaildog_texture_$RUN_ID"
run_if_enough_time 5400 "v199a_stable_detaildog_texture" run_candidate \
  "v199a_stable_detaildog_texture" \
  "$V199A_EXP" \
  "$BASE_CKPT" \
  "${V199A_ITERATIONS:-9000}" \
  "[]" \
  "${common_freeze_geometry_overrides[@]}" \
  "opt.texture_lr=4.0e-06" \
  "$texture_hf_patterns" \
  "${detail_loss_overrides[@]}" \
  "opt.lambda_perceptual=0.24" \
  "++opt.lambda_l1_boundary=0.052" \
  "++opt.lambda_detail_face=0.010" \
  "++opt.lambda_detail_shoulder_arm=0.008" \
  "++opt.lambda_detail_face_luma_dog=0.006" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.005" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.005" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.006" \
  "opt.grad_clip=0.0045"

V199B_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v199b_v198a_view_conflict_after_geometry_$RUN_ID"
run_if_enough_time 5400 "v199b_view_conflict_after_geometry" run_candidate \
  "v199b_view_conflict_after_geometry" \
  "$V199B_EXP" \
  "$BASE_CKPT" \
  "${V199B_ITERATIONS:-9000}" \
  "[texture.detail_high_freq_view_conflict_]" \
  "${common_freeze_geometry_overrides[@]}" \
  "opt.texture_lr=8.0e-06" \
  "++opt.texture_trainable_name_patterns=[detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*]" \
  "${detail_loss_overrides[@]}" \
  "${view_conflict_overrides[@]}" \
  "opt.lambda_perceptual=0.24" \
  "++opt.lambda_l1_boundary=0.052" \
  "++opt.lambda_detail_face=0.010" \
  "++opt.lambda_detail_shoulder_arm=0.008" \
  "++opt.lambda_detail_face_luma_dog=0.006" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.005" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.005" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.006" \
  "opt.grad_clip=0.0045"

V199C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v199c_v198a_microgeom_detail_joint_$RUN_ID"
run_if_enough_time 5400 "v199c_microgeom_detail_joint" run_candidate \
  "v199c_microgeom_detail_joint" \
  "$V199C_EXP" \
  "$BASE_CKPT" \
  "${V199C_ITERATIONS:-9000}" \
  "[]" \
  "${micro_geometry_overrides[@]}" \
  "opt.texture_lr=2.8e-06" \
  "$texture_hf_patterns" \
  "${detail_loss_overrides[@]}" \
  "opt.lambda_perceptual=0.22" \
  "++opt.lambda_l1_boundary=0.082" \
  "++opt.lambda_detail_face=0.007" \
  "++opt.lambda_detail_shoulder_arm=0.006" \
  "++opt.lambda_detail_face_luma_dog=0.004" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.0035" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.004" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.005" \
  "opt.grad_clip=0.0040"

best_line="$(pick_best_candidate || true)"
if [ -n "$best_line" ]; then
  IFS=$'\t' read -r BEST_NAME BEST_EXP BEST_CKPT BEST_RENDER BEST_SCORE BEST_SOURCE <<< "$best_line"
  log_msg "CURRENT_BEST name=$BEST_NAME score=$BEST_SCORE source=$BEST_SOURCE ckpt=$BEST_CKPT"
  append_summary "current_best_after_abc" "ok" "$BEST_EXP" "$LOG_DIR/queue.log" "name=$BEST_NAME,score=$BEST_SCORE,source=$BEST_SOURCE"
else
  BEST_NAME="v198a_baseline"
  BEST_EXP="$BASE_EXP"
  BEST_CKPT="$BASE_CKPT"
  BEST_RENDER="$BASE_RENDER"
  BEST_SCORE="na"
  BEST_SOURCE="fallback"
  log_msg "CURRENT_BEST fallback base checkpoint"
fi

V199D_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v199d_best_combined_detail_conflict_$RUN_ID"
run_if_enough_time 6600 "v199d_best_combined_detail_conflict" run_candidate \
  "v199d_best_combined_detail_conflict" \
  "$V199D_EXP" \
  "$BEST_CKPT" \
  "${V199D_ITERATIONS:-12000}" \
  "[texture.detail_high_freq_view_conflict_]" \
  "${micro_geometry_overrides[@]}" \
  "opt.texture_lr=3.2e-06" \
  "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*]" \
  "${detail_loss_overrides[@]}" \
  "${view_conflict_overrides[@]}" \
  "opt.lambda_perceptual=0.23" \
  "++opt.lambda_l1_boundary=0.074" \
  "++opt.lambda_detail_face=0.008" \
  "++opt.lambda_detail_shoulder_arm=0.006" \
  "++opt.lambda_detail_face_luma_dog=0.005" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.004" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.0045" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.0055" \
  "opt.grad_clip=0.0038"

best_line="$(pick_best_candidate || true)"
if [ -n "$best_line" ]; then
  IFS=$'\t' read -r BEST_NAME BEST_EXP BEST_CKPT BEST_RENDER BEST_SCORE BEST_SOURCE <<< "$best_line"
  log_msg "CURRENT_BEST name=$BEST_NAME score=$BEST_SCORE source=$BEST_SOURCE ckpt=$BEST_CKPT"
  append_summary "current_best_after_d" "ok" "$BEST_EXP" "$LOG_DIR/queue.log" "name=$BEST_NAME,score=$BEST_SCORE,source=$BEST_SOURCE"
fi

chunk=1
while true; do
  remain="$(remaining_seconds)"
  if [ "$remain" -le 2700 ]; then
    log_msg "stop fill loop remaining=${remain}s"
    break
  fi

  if [ "$remain" -gt 7200 ]; then
    fill_iterations=9000
  elif [ "$remain" -gt 4800 ]; then
    fill_iterations=6000
  else
    fill_iterations=3000
  fi

  best_line="$(pick_best_candidate || true)"
  if [ -n "$best_line" ]; then
    IFS=$'\t' read -r BEST_NAME BEST_EXP BEST_CKPT BEST_RENDER BEST_SCORE BEST_SOURCE <<< "$best_line"
  fi
  log_msg "FILL chunk=$chunk start_from=$BEST_NAME score=$BEST_SCORE remaining=${remain}s iterations=$fill_iterations"

  FILL_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v199e_fill${chunk}_${fill_iterations}_best_${RUN_ID}"
  run_candidate \
    "v199e_fill${chunk}_${fill_iterations}_best" \
    "$FILL_EXP" \
    "$BEST_CKPT" \
    "$fill_iterations" \
    "[texture.detail_high_freq_view_conflict_]" \
    "${micro_geometry_overrides[@]}" \
    "opt.texture_lr=2.4e-06" \
    "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,detail_high_freq_structure_proj.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,detail_high_freq_view_conflict_mlp.*,detail_high_freq_view_conflict_gate_mlp.*]" \
    "${detail_loss_overrides[@]}" \
    "${view_conflict_overrides[@]}" \
    "opt.lambda_perceptual=0.22" \
    "++opt.lambda_l1_boundary=0.070" \
    "++opt.lambda_detail_face=0.007" \
    "++opt.lambda_detail_shoulder_arm=0.005" \
    "++opt.lambda_detail_face_luma_dog=0.004" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.0035" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.004" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.005" \
    "opt.grad_clip=0.0038"

  chunk=$((chunk + 1))
done

best_line="$(pick_best_candidate || true)"
if [ -n "$best_line" ]; then
  IFS=$'\t' read -r BEST_NAME BEST_EXP BEST_CKPT BEST_RENDER BEST_SCORE BEST_SOURCE <<< "$best_line"
  log_msg "FINAL_BEST name=$BEST_NAME score=$BEST_SCORE source=$BEST_SOURCE exp=$BEST_EXP render=$BEST_RENDER"
  append_summary "final_best" "ok" "$BEST_EXP" "$LOG_DIR/queue.log" "name=$BEST_NAME,score=$BEST_SCORE,source=$BEST_SOURCE,render=$BEST_RENDER"
fi

write_status "queue" "completed"
log_msg "v199 overnight clarity queue complete summary=$SUMMARY candidates=$CANDIDATES"
exit 0
