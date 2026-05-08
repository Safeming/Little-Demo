#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU_ID="${GPU_ID:-0}"
PRIMARY_RUN_ID="${1:?usage: run_377_v182_fill_remaining.sh PRIMARY_RUN_ID DEADLINE_EPOCH}"
DEADLINE_EPOCH="${2:?usage: run_377_v182_fill_remaining.sh PRIMARY_RUN_ID DEADLINE_EPOCH}"

BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
BASE_CKPT="$BASE_EXP/best_ckpt.pth"
DATA_ROOT="$ROOT/data/ZJUMoCap"
PRIMARY_LOG_ROOT="$ROOT/exp/stageA2/logs/v182_overnight_$PRIMARY_RUN_ID"
LOG_DIR="$PRIMARY_LOG_ROOT/fill_remaining"
SUMMARY="$LOG_DIR/fill_summary.tsv"
mkdir -p "$LOG_DIR"

cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

log_msg() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$*" | tee -a "$LOG_DIR/fill.log"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $(( DEADLINE_EPOCH - now ))
}

append_summary() {
  local name="$1"
  local status="$2"
  local exp_dir="$3"
  local log_path="$4"
  local extra="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$exp_dir" "$log_path" "$extra" >> "$SUMMARY"
}

extract_metrics() {
  local exp_dir="$1"
  "$PYTHON_BIN" - "$exp_dir" <<'PY'
import json
import sys
from pathlib import Path

exp = Path(sys.argv[1])
p = exp / "best_test_metrics.json"
if not p.exists():
    print("")
    raise SystemExit(0)
data = json.loads(p.read_text())
keys = ["iteration", "psnr", "ssim", "lpips", "psnr_fg", "ssim_fg", "lpips_fg", "l1_fg"]
print(",".join(f"{k}={data.get(k)}" for k in keys if k in data))
PY
}

run_train_chunk() {
  local name="$1"
  local exp_dir="$2"
  local start_ckpt="$3"
  local iterations="$4"
  local log_path="$5"
  log_msg "START $name iterations=$iterations start_ckpt=$start_ckpt"
  "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "start_checkpoint=$start_ckpt" \
    "exp_dir=$exp_dir" \
    "opt.iterations=$iterations" \
    "test_interval=1000" \
    "test_iterations=[1000,2000,3000,4000,5000,6000]" \
    "save_iterations=[$iterations]" \
    "checkpoint_iterations=[$iterations]" \
    "++validation_image_log_limit=0" \
    "opt.texture_lr=0.000030" \
    "opt.tex_latent_lr=0.000030" \
    "opt.pose_correction_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.feature_lr=0.0" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    >"$log_path" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    log_msg "DONE $name"
  else
    log_msg "FAILED $name status=$status"
  fi
  return "$status"
}

render_eval() {
  local name="$1"
  local config_exp="$2"
  local ckpt="$3"
  local out_exp="$4"
  local log_path="$5"
  log_msg "START $name"
  "$PYTHON_BIN" render.py \
    --config-path "$config_exp/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$out_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_frames.view=[0,570,60]" \
    wandb_disable=true \
    >"$log_path" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    log_msg "DONE $name"
  else
    log_msg "FAILED $name status=$status"
  fi
  return "$status"
}

run_diagnostic() {
  local name="$1"
  local render_exp="$2"
  local log_path="$3"
  log_msg "START $name"
  "$PYTHON_BIN" "$ROOT/tools/analyze_377_render_contours.py" \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 \
    >"$log_path" 2>&1
  local status=$?
  if [ "$status" -eq 0 ]; then
    log_msg "DONE $name"
  else
    log_msg "FAILED $name status=$status"
  fi
  return "$status"
}

printf 'name\tstatus\texp_dir\tlog\textra\n' > "$SUMMARY"
log_msg "fill watcher start primary=$PRIMARY_RUN_ID deadline_epoch=$DEADLINE_EPOCH"

PRIMARY_PID=""
if [ -f "$PRIMARY_LOG_ROOT/pid.txt" ]; then
  PRIMARY_PID="$(cat "$PRIMARY_LOG_ROOT/pid.txt")"
fi

if [ -n "$PRIMARY_PID" ]; then
  while ps -p "$PRIMARY_PID" >/dev/null 2>&1; do
    log_msg "waiting primary pid=$PRIMARY_PID remaining=$(remaining_seconds)s"
    sleep 300
  done
fi

source_ckpt="$BASE_CKPT"
source_exp="$BASE_EXP"
primary_v182c="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v182c_texture_after_anchor_v179c_faithful_$PRIMARY_RUN_ID"
if [ -f "$primary_v182c/best_ckpt.pth" ]; then
  source_ckpt="$primary_v182c/best_ckpt.pth"
  source_exp="$primary_v182c"
elif [ -f "$primary_v182c/ckpt106400.pth" ]; then
  source_ckpt="$primary_v182c/ckpt106400.pth"
  source_exp="$primary_v182c"
fi

chunk=1
while true; do
  remain="$(remaining_seconds)"
  if [ "$remain" -le 2700 ]; then
    log_msg "stop fill loop remaining=${remain}s"
    break
  fi

  if [ "$remain" -gt 5400 ]; then
    iterations=6000
  else
    iterations=3000
  fi

  chunk_id="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
  exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v182c_fill_chunk${chunk}_${iterations}_${chunk_id}"
  log_path="$LOG_DIR/v182c_fill_chunk${chunk}_${iterations}.log"
  if run_train_chunk "v182c_fill_chunk${chunk}_${iterations}" "$exp_dir" "$source_ckpt" "$iterations" "$log_path"; then
    metrics="$(extract_metrics "$exp_dir")"
    append_summary "v182c_fill_chunk${chunk}_${iterations}" "ok" "$exp_dir" "$log_path" "$metrics"
    if [ -f "$exp_dir/best_ckpt.pth" ]; then
      source_ckpt="$exp_dir/best_ckpt.pth"
      source_exp="$exp_dir"
    elif [ -f "$exp_dir/ckpt$iterations.pth" ]; then
      source_ckpt="$exp_dir/ckpt$iterations.pth"
      source_exp="$exp_dir"
    fi
  else
    append_summary "v182c_fill_chunk${chunk}_${iterations}" "failed" "$exp_dir" "$log_path" "stop_fill"
    break
  fi
  chunk=$((chunk + 1))
done

if [ "$source_ckpt" != "$BASE_CKPT" ] && [ "$(remaining_seconds)" -gt 600 ]; then
  render_exp="${source_exp}_render_quick"
  if render_eval "v182c_fill_final_render" "$source_exp" "$source_ckpt" "$render_exp" "$LOG_DIR/v182c_fill_final_render.log"; then
    append_summary "v182c_fill_final_render" "ok" "$render_exp" "$LOG_DIR/v182c_fill_final_render.log" "quick_render"
    run_diagnostic "v182c_fill_final_contour_diagnostic" "$render_exp" "$LOG_DIR/v182c_fill_final_contour_diagnostic.log"
    append_summary "v182c_fill_final_contour_diagnostic" "$?" "$render_exp" "$LOG_DIR/v182c_fill_final_contour_diagnostic.log" "contour_rank"
  else
    append_summary "v182c_fill_final_render" "failed" "$render_exp" "$LOG_DIR/v182c_fill_final_render.log" "render_failed"
  fi
fi

log_msg "fill watcher complete summary=$SUMMARY"
