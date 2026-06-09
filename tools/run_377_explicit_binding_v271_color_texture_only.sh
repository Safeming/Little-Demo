#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v271_explicit_color_texture_only_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
ITERATIONS="${ITERATIONS:-400}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100,200,300,400}"
DO_RENDER="${DO_RENDER:-1}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_raw_rgb_refine_v270_explicit_raw_rgb_20260517_141109_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136110.pth}"
BASE_ITER="${BASE_ITER:-136110}"

EXP_DIR="${EXP_DIR:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v271_color_texture_only_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"

BASELINE_FG_L1="${BASELINE_FG_L1:-0.04530626}"
BASELINE_BOUNDARY_L1="${BASELINE_BOUNDARY_L1:-0.06608825}"
BASELINE_EDGE_DIST="${BASELINE_EDGE_DIST:-2.707462}"

mkdir -p "$EXP_DIR" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-1500}"
EST_END_EPOCH="$((START_EPOCH + EST_SECONDS))"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$EST_END_EPOCH" '+%F %T BJT')"

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" "$START_BJT" "$EST_END_BJT" <<'PY' || true
import json
import sys
import time
from pathlib import Path

path, run_id, gpu, phase, detail, start_bjt, est_end_bjt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_bjt": start_bjt,
    "est_end_bjt": est_end_bjt,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
EXP_DIR=$EXP_DIR
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
ITERATIONS=$ITERATIONS
CHECKPOINT_STEPS=$CHECKPOINT_STEPS
BASELINE_FG_L1=$BASELINE_FG_L1
BASELINE_BOUNDARY_L1=$BASELINE_BOUNDARY_L1
BASELINE_EDGE_DIST=$BASELINE_EDGE_DIST

Goal:
  v271 color/texture-only raw explicit_binding refine.

Hard rules:
  Start from v270 ckpt136110 by default.
  Freeze xyz / rotation / opacity / scaling / boundary residuals.
  Train only SH color features and texture.
  Disable StageB semantic loss, mask-only silhouette loss, densify, opacity reset, and render_export_refine.
  Select by external raw contour gate, not by train.py best_ckpt alone.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'label\tckpt\trender_exp\tmean_fg_l1\tmean_boundary_l1\tmean_edge_symmetric_dist_px\tfg_delta\tboundary_delta\tedge_delta\tgate\tstatus\n' > "$SUMMARY"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

append_summary() {
  local label="$1"
  local ckpt="$2"
  local render_exp="$3"
  local status="$4"
  "$PYTHON_BIN" - \
    "$label" "$ckpt" "$render_exp" "$status" "$SUMMARY" \
    "$BASELINE_FG_L1" "$BASELINE_BOUNDARY_L1" "$BASELINE_EDGE_DIST" <<'PY'
import json
import sys
from pathlib import Path

label, ckpt, render_exp, status, summary_path = sys.argv[1:6]
base_fg, base_bd, base_edge = map(float, sys.argv[6:9])
contour = Path(render_exp) / "diagnostics" / "contours" / "contour_summary.json"
metrics = {}
if contour.exists():
    metrics = json.loads(contour.read_text(encoding="utf-8"))

def as_float(key):
    try:
        return float(metrics[key])
    except Exception:
        return None

def fmt(value, digits=8):
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"

fg = as_float("mean_fg_l1")
bd = as_float("mean_boundary_l1")
edge = as_float("mean_edge_symmetric_dist_px")
fg_delta = None if fg is None else fg - base_fg
bd_delta = None if bd is None else bd - base_bd
edge_delta = None if edge is None else edge - base_edge
gate = "missing_metrics"
if fg is not None and bd is not None and edge is not None:
    gate = "pass" if fg <= base_fg and bd <= base_bd and edge <= base_edge else "blocked"

row = [
    label,
    ckpt,
    render_exp,
    fmt(fg),
    fmt(bd),
    fmt(edge, 6),
    fmt(fg_delta),
    fmt(bd_delta),
    fmt(edge_delta, 6),
    gate,
    status,
]
with open(summary_path, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
)

checkpoint_list="[$CHECKPOINT_STEPS]"

write_status "train_start" "$EXP_DIR"
log_event "train_start" "$EXP_DIR"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
  "dataset.val_views=[21,22,23]" \
  "dataset.test_views.view=[21,22,23]" \
  "dataset.train_frames=[0,570,1]" \
  "dataset.val_frames=[0,570,60]" \
  "dataset.test_frames.view=[0,570,60]" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.compact_mapping_file=" \
  "start_checkpoint=$BASE_CKPT" \
  "exp_dir=$EXP_DIR" \
  "hydra.run.dir=$HYDRA_RUN_ROOT/train" \
  "seed=-1" \
  "wandb_disable=true" \
  "++resume.allow_partial_converter_load=true" \
  "++resume.restore_gaussian_optimizer_state=false" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.]" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=true" \
  "++resume.clear_binding_state_on_resume=false" \
  "pipeline.pose_noise=0.0" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "opt.iterations=$ITERATIONS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.00018" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=0.00000045" \
  "opt.tex_latent_lr=0.0" \
  "++opt.texture_trainable_name_patterns=[*]" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.stageB_semantic_body_cloth_weight=0.0" \
  "++opt.stageB_semantic_compact_weight=0.0" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "++opt.train_sample_log_interval=100" \
  "++opt.train_sample_accumulation_steps=1" \
  "opt.lambda_l1=0.060" \
  "opt.lambda_l1_fg=0.140" \
  "opt.lambda_l1_boundary=0.080" \
  "opt.lambda_perceptual=0.025" \
  "opt.lambda_l1_face=0.020" \
  "opt.lambda_l1_shoulder_arm=0.016" \
  "opt.lambda_l1_waist=0.012" \
  "opt.lambda_edge_face=0.003" \
  "opt.lambda_edge_shoulder_arm=0.003" \
  "opt.lambda_edge_waist=0.0015" \
  "++opt.lambda_detail_face=0.0" \
  "++opt.lambda_detail_shoulder_arm=0.0" \
  "++opt.lambda_detail_waist=0.0" \
  "++opt.lambda_detail_face_luma_dog=0.003" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.003" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.003" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.003" \
  "++opt.lambda_detail_waist_luma_dog=0.0015" \
  "++opt.lambda_perceptual_face=0.008" \
  "++opt.lambda_perceptual_shoulder_arm=0.006" \
  "++opt.lambda_perceptual_waist=0.003" \
  "++opt.lambda_perceptual_face_patch=0.004" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.004" \
  "++opt.lambda_perceptual_upper_torso_patch=0.004" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.003" \
  "++opt.lambda_perceptual_waist_patch=0.002" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_silhouette_outer=0.0" \
  "++opt.lambda_silhouette_inner=0.0" \
  "++opt.lambda_silhouette_shoulder_arm_outer_shell=0.0" \
  "++opt.lambda_silhouette_upper_torso_outer_shell=0.0" \
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
  "best_metric=l1_fg" \
  "best_metric_mode=min" \
  "best_metric_source=best_eval" \
  "test_interval=0" \
  "test_iterations=$checkpoint_list" \
  "save_iterations=$checkpoint_list" \
  "checkpoint_iterations=$checkpoint_list" \
  "++validation_image_log_limit=0" \
  "opt.grad_clip=0.0020" \
  > "$LOG_DIR/train.log" 2>&1
write_status "train_done" "$EXP_DIR"
log_event "train_done" "$EXP_DIR"

render_checkpoint() {
  local step="$1"
  local global_iter=$((BASE_ITER + step))
  local ckpt="$EXP_DIR/ckpt${global_iter}.pth"
  local label="ckpt${global_iter}"
  local render_exp="${EXP_DIR}_raw_render_${label}"
  if [ ! -f "$ckpt" ]; then
    log_event "render_skip" "missing=$ckpt"
    append_summary "$label" "$ckpt" "$render_exp" "missing_ckpt"
    return 0
  fi
  write_status "raw_render_start" "$label"
  log_event "raw_render_start" "$label"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$EXP_DIR/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/render_${label}" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${label}.log" 2>&1

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1 || true
  append_summary "$label" "$ckpt" "$render_exp" "ok"
  log_event "raw_render_done" "$render_exp"
}

if [ "$DO_RENDER" = "1" ]; then
  IFS=',' read -ra steps <<< "$CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    render_checkpoint "$step"
  done
fi

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

write_status "all_done" "$END_BJT"
log_event "all_done" "$END_BJT"
echo "EXP_DIR=$EXP_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
