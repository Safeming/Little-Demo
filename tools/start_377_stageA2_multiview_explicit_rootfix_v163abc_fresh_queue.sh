#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
QUEUE_TAG="$(date +%Y%m%d_%H%M%S)"

V163A_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v163a_supportmain_clean_screen4k}"
V163B_EXP="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v163b_v163a_ownerhf_midhandoff_screen3k}"
V163C_EXP="${3:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v163c_v163b_clean_mixedboundary_cfinish_screen2500}"
V163A_ITERS="${4:-4000}"
V163B_ITERS="${5:-3000}"
V163C_ITERS="${6:-2500}"
if [ "$#" -ge 6 ]; then
  shift 6
else
  shift "$#"
fi

mkdir -p "$LOG_DIR" "$(dirname "$V163A_EXP")" "$(dirname "$V163B_EXP")" "$(dirname "$V163C_EXP")"

QUEUE_LOG="$LOG_DIR/v163abc_${QUEUE_TAG}_queue.log"
STATUS_FILE="$LOG_DIR/v163abc_${QUEUE_TAG}_status.log"
STAGE1_LOG="$LOG_DIR/v163a_${QUEUE_TAG}_stage1.log"
STAGE2_LOG="$LOG_DIR/v163b_${QUEUE_TAG}_stage2.log"
STAGE3_LOG="$LOG_DIR/v163c_${QUEUE_TAG}_stage3.log"

log_msg() {
  local msg="$1"
  echo "[$(date '+%F %T %Z')] $msg" | tee -a "$QUEUE_LOG" "$STATUS_FILE"
}

dump_stage_summary() {
  local stage_name="$1"
  local metrics_path="$2"
  local log_path="$3"
  "$PYTHON_BIN" - "$stage_name" "$metrics_path" "$log_path" <<'PY'
import json
import re
import sys

stage_name, metrics_path, log_path = sys.argv[1:4]
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

support = None
owner_takeover = None
boundary_takeover = None
boundary_focus = None
layer_mean = None

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "[ClarityTexTrunk]" in line:
            match = re.search(r"(?<!owner_)support=([0-9.]+)", line)
            if match:
                support = float(match.group(1))
            match = re.search(r"owner_takeover=([0-9.]+)", line)
            if match:
                owner_takeover = float(match.group(1))
            match = re.search(r"boundary_takeover=([0-9.]+)", line)
            if match:
                boundary_takeover = float(match.group(1))
            match = re.search(r"boundary_focus=([0-9.]+)", line)
            if match:
                boundary_focus = float(match.group(1))
        if "layer_mean=(" in line:
            match = re.search(r"layer_mean=\(([^,]+),([^,]+),([^)]+)\)", line)
            if match:
                layer_mean = tuple(float(x) for x in match.groups())

summary = {
    "stage": stage_name,
    "iteration": metrics.get("iteration"),
    "psnr": metrics.get("psnr"),
    "lpips": metrics.get("lpips"),
    "psnr_fg": metrics.get("psnr_fg"),
    "lpips_fg": metrics.get("lpips_fg"),
    "l1_fg": metrics.get("l1_fg"),
    "ssim_fg": metrics.get("ssim_fg"),
    "support": support,
    "owner_takeover": owner_takeover,
    "boundary_takeover": boundary_takeover,
    "boundary_focus": boundary_focus,
    "layer_mean": layer_mean,
}
print(json.dumps(summary, ensure_ascii=True))
PY
}

run_stage() {
  local stage_name="$1"
  local script_path="$2"
  local stage_log="$3"
  shift 3
  log_msg "$stage_name start"
  bash "$script_path" "$@" > "$stage_log" 2>&1
  log_msg "$stage_name done log=$stage_log"
}

log_msg "queue start"
log_msg "stageA_exp=$V163A_EXP iters=$V163A_ITERS"
log_msg "stageB_exp=$V163B_EXP iters=$V163B_ITERS"
log_msg "stageC_exp=$V163C_EXP iters=$V163C_ITERS"

run_stage \
  "v163a_fresh_support" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v157a_supportmain.sh" \
  "$STAGE1_LOG" \
  "$V163A_EXP" \
  "$V163A_ITERS" \
  "$@"

V163A_METRICS="$V163A_EXP/best_test_metrics.json"
V163A_CKPT="$V163A_EXP/best_ckpt.pth"
if [ ! -f "$V163A_METRICS" ] || [ ! -f "$V163A_CKPT" ]; then
  log_msg "v163a missing outputs"
  exit 1
fi
log_msg "v163a summary=$(dump_stage_summary v163a "$V163A_METRICS" "$STAGE1_LOG")"

run_stage \
  "v163b_ownerhf_midhandoff" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157b3_midhandoff.sh" \
  "$STAGE2_LOG" \
  "$V163B_EXP" \
  "$V163A_CKPT" \
  "$V163B_ITERS" \
  "$@"

V163B_METRICS="$V163B_EXP/best_test_metrics.json"
V163B_CKPT="$V163B_EXP/best_ckpt.pth"
if [ ! -f "$V163B_METRICS" ] || [ ! -f "$V163B_CKPT" ]; then
  log_msg "v163b missing outputs"
  exit 1
fi
log_msg "v163b summary=$(dump_stage_summary v163b "$V163B_METRICS" "$STAGE2_LOG")"

run_stage \
  "v163c_clean_mixedboundary_cfinish" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v163c_v163b_clean_cfinish.sh" \
  "$STAGE3_LOG" \
  "$V163C_EXP" \
  "$V163B_CKPT" \
  "$V163C_ITERS" \
  "$@"

V163C_METRICS="$V163C_EXP/best_test_metrics.json"
V163C_CKPT="$V163C_EXP/best_ckpt.pth"
if [ ! -f "$V163C_METRICS" ] || [ ! -f "$V163C_CKPT" ]; then
  log_msg "v163c missing outputs"
  exit 1
fi
log_msg "v163c summary=$(dump_stage_summary v163c "$V163C_METRICS" "$STAGE3_LOG")"

log_msg "queue done"
log_msg "stageA_log=$STAGE1_LOG"
log_msg "stageB_log=$STAGE2_LOG"
log_msg "stageC_log=$STAGE3_LOG"
log_msg "stageA_ckpt=$V163A_CKPT"
log_msg "stageB_ckpt=$V163B_CKPT"
log_msg "stageC_ckpt=$V163C_CKPT"
