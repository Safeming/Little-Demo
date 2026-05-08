#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU_ID="${GPU_ID:-0}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-32400}"
START_EPOCH="$(date +%s)"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"

BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
BASE_CKPT="$BASE_EXP/best_ckpt.pth"
DATA_ROOT="$ROOT/data/ZJUMoCap"
LOG_DIR="$ROOT/exp/stageA2/logs/v182_overnight_$RUN_ID"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

log_msg() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$*" | tee -a "$LOG_DIR/queue.log"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $(( TIME_BUDGET_SECONDS - (now - START_EPOCH) ))
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$START_EPOCH" "$TIME_BUDGET_SECONDS" "$GPU_ID" "$1" "$2" <<'PY'
import json
import sys
import time
from pathlib import Path

path, run_id, start_epoch, budget, gpu_id, phase, detail = sys.argv[1:]
start_epoch = int(start_epoch)
budget = int(budget)
now = int(time.time())
data = {
    "run_id": run_id,
    "phase": phase,
    "detail": detail,
    "gpu_id": gpu_id,
    "start_epoch": start_epoch,
    "now_epoch": now,
    "elapsed_seconds": now - start_epoch,
    "budget_seconds": budget,
    "remaining_seconds": max(0, budget - (now - start_epoch)),
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

render_eval() {
  local name="$1"
  local config_exp="$2"
  local ckpt="$3"
  local out_exp="$4"
  local frame_step="$5"
  local log_path="$6"
  run_step "$name" "$log_path" \
    "$PYTHON_BIN" render.py \
      --config-path "$config_exp/.hydra" \
      --config-name config \
      mode=test \
      "load_ckpt=$ckpt" \
      "exp_dir=$out_exp" \
      "dataset.root_dir=$DATA_ROOT" \
      "dataset.preload=false" \
      "dataset.test_frames.view=[0,570,$frame_step]" \
      wandb_disable=true
}

train_from_v179c_config() {
  local name="$1"
  local exp_dir="$2"
  local iterations="$3"
  local log_path="$4"
  shift 4
  run_step "$name" "$log_path" \
    "$PYTHON_BIN" train.py \
      --config-path "$BASE_EXP/.hydra" \
      --config-name config \
      mode=train \
      "dataset.root_dir=$DATA_ROOT" \
      "dataset.preload=false" \
      "start_checkpoint=$BASE_CKPT" \
      "exp_dir=$exp_dir" \
      "opt.iterations=$iterations" \
      "test_interval=500" \
      "test_iterations=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]" \
      "save_iterations=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]" \
      "checkpoint_iterations=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]" \
      "++validation_image_log_limit=0" \
      "$@"
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

compare_metric_gate() {
  local candidate="$1"
  "$PYTHON_BIN" - "$BASE_EXP/best_test_metrics.json" "$candidate/best_test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
if not candidate_path.exists():
    print("missing_candidate_metrics")
    raise SystemExit(2)
base = json.loads(base_path.read_text())
candidate = json.loads(candidate_path.read_text())
base_lpips = float(base.get("lpips_fg", 1e9))
candidate_lpips = float(candidate.get("lpips_fg", 1e9))
delta = candidate_lpips - base_lpips
print(f"base_lpips_fg={base_lpips:.6f} candidate_lpips_fg={candidate_lpips:.6f} delta={delta:.6f}")
raise SystemExit(0 if delta <= 0.015 else 3)
PY
}

run_diagnostic() {
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

printf 'name\tstatus\texp_dir\tlog\textra\n' > "$SUMMARY"
log_msg "v182 overnight queue start run_id=$RUN_ID gpu=$GPU_ID budget=${TIME_BUDGET_SECONDS}s"
log_msg "base_exp=$BASE_EXP"
write_status "queue" "started"

V182A_RENDER_EXP="$ROOT/exp/stageA2/v182a_faithful_replay_v179c_currentroot_$RUN_ID"
if render_eval \
  "v182a_faithful_replay_render" \
  "$BASE_EXP" \
  "$BASE_CKPT" \
  "$V182A_RENDER_EXP" \
  60 \
  "$LOG_DIR/v182a_faithful_replay_render.log"; then
  append_summary "v182a_faithful_replay_render" "ok" "$V182A_RENDER_EXP" "$LOG_DIR/v182a_faithful_replay_render.log" "quick_render"
  run_diagnostic "v182a_contour_diagnostic" "$V182A_RENDER_EXP" "$LOG_DIR/v182a_contour_diagnostic.log"
  append_summary "v182a_contour_diagnostic" "$?" "$V182A_RENDER_EXP" "$LOG_DIR/v182a_contour_diagnostic.log" "contour_rank"
else
  append_summary "v182a_faithful_replay_render" "failed" "$V182A_RENDER_EXP" "$LOG_DIR/v182a_faithful_replay_render.log" "stop_high_risk"
  write_status "queue" "failed_v182a_render"
  exit 1
fi

V182A_NOOP_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v182a_noop_resume_v179c_exact_$RUN_ID"
if train_from_v179c_config \
  "v182a_noop_resume_500" \
  "$V182A_NOOP_EXP" \
  500 \
  "$LOG_DIR/v182a_noop_resume_500.log" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.rigid_lr=0.0" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "+opt.camera_affine_lr=0.0" \
  "opt.lr_ratio=1.0"; then
  metrics="$(extract_metrics "$V182A_NOOP_EXP")"
  append_summary "v182a_noop_resume_500" "ok" "$V182A_NOOP_EXP" "$LOG_DIR/v182a_noop_resume_500.log" "$metrics"
else
  append_summary "v182a_noop_resume_500" "failed" "$V182A_NOOP_EXP" "$LOG_DIR/v182a_noop_resume_500.log" "stop_high_risk"
  write_status "queue" "failed_v182a_noop"
  exit 1
fi

V182A_NOOP_RENDER_EXP="${V182A_NOOP_EXP}_render_quick"
if [ -f "$V182A_NOOP_EXP/best_ckpt.pth" ]; then
  render_eval \
    "v182a_noop_resume_render" \
    "$V182A_NOOP_EXP" \
    "$V182A_NOOP_EXP/best_ckpt.pth" \
    "$V182A_NOOP_RENDER_EXP" \
    60 \
    "$LOG_DIR/v182a_noop_resume_render.log"
  append_summary "v182a_noop_resume_render" "$?" "$V182A_NOOP_RENDER_EXP" "$LOG_DIR/v182a_noop_resume_render.log" "quick_render"
  run_diagnostic "v182a_noop_contour_diagnostic" "$V182A_NOOP_RENDER_EXP" "$LOG_DIR/v182a_noop_contour_diagnostic.log"
  append_summary "v182a_noop_contour_diagnostic" "$?" "$V182A_NOOP_RENDER_EXP" "$LOG_DIR/v182a_noop_contour_diagnostic.log" "contour_rank"
fi

if compare_out="$(compare_metric_gate "$V182A_NOOP_EXP")"; then
  log_msg "GATE PASS v182a_noop $compare_out"
  append_summary "gate_v182a_noop" "pass" "$V182A_NOOP_EXP" "$LOG_DIR/queue.log" "$compare_out"
else
  gate_status=$?
  log_msg "GATE WARN v182a_noop status=$gate_status $compare_out"
  append_summary "gate_v182a_noop" "warn" "$V182A_NOOP_EXP" "$LOG_DIR/queue.log" "$compare_out"
fi

V182C_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v182c_texture_after_anchor_v179c_faithful_$RUN_ID"
V182C_ITER="${V182C_ITER:-6000}"
remain="$(remaining_seconds)"
if [ "$remain" -gt 2400 ]; then
  if train_from_v179c_config \
    "v182c_texture_after_anchor_${V182C_ITER}" \
    "$V182C_EXP" \
    "$V182C_ITER" \
    "$LOG_DIR/v182c_texture_after_anchor.log" \
    "opt.texture_lr=0.000035" \
    "opt.tex_latent_lr=0.000035" \
    "opt.pose_correction_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.feature_lr=0.0" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0"; then
    metrics="$(extract_metrics "$V182C_EXP")"
    append_summary "v182c_texture_after_anchor_${V182C_ITER}" "ok" "$V182C_EXP" "$LOG_DIR/v182c_texture_after_anchor.log" "$metrics"
  else
    append_summary "v182c_texture_after_anchor_${V182C_ITER}" "failed" "$V182C_EXP" "$LOG_DIR/v182c_texture_after_anchor.log" "continue_to_render_existing_if_any"
  fi
else
  append_summary "v182c_texture_after_anchor_${V182C_ITER}" "skipped" "$V182C_EXP" "$LOG_DIR/queue.log" "remaining_seconds=$remain"
fi

if [ -f "$V182C_EXP/best_ckpt.pth" ]; then
  V182C_RENDER_EXP="${V182C_EXP}_render_quick"
  render_eval \
    "v182c_texture_after_anchor_render" \
    "$V182C_EXP" \
    "$V182C_EXP/best_ckpt.pth" \
    "$V182C_RENDER_EXP" \
    60 \
    "$LOG_DIR/v182c_texture_after_anchor_render.log"
  append_summary "v182c_texture_after_anchor_render" "$?" "$V182C_RENDER_EXP" "$LOG_DIR/v182c_texture_after_anchor_render.log" "quick_render"
  run_diagnostic "v182c_texture_after_anchor_contour_diagnostic" "$V182C_RENDER_EXP" "$LOG_DIR/v182c_texture_after_anchor_contour_diagnostic.log"
  append_summary "v182c_texture_after_anchor_contour_diagnostic" "$?" "$V182C_RENDER_EXP" "$LOG_DIR/v182c_texture_after_anchor_contour_diagnostic.log" "contour_rank"
fi

write_status "queue" "completed"
log_msg "v182 overnight queue complete summary=$SUMMARY"
exit 0
