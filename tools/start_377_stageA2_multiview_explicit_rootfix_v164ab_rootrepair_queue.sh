#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
QUEUE_TAG="$(date +%Y%m%d_%H%M%S)"

V164A_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v164a_v160a_contouralpha_rootrepair_screen1200}"
V164B_EXP="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v164b_v164a_texture_restore_contourlock_screen2k}"
V164A_ITERS="${3:-1200}"
V164B_ITERS="${4:-2000}"
START_CKPT="${5:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v160a_v158a_mixedboundary_cfinish_screen2k/best_ckpt.pth}"
if [ "$#" -ge 5 ]; then
  shift 5
else
  shift "$#"
fi

mkdir -p "$LOG_DIR" "$(dirname "$V164A_EXP")" "$(dirname "$V164B_EXP")"

QUEUE_LOG="$LOG_DIR/v164ab_${QUEUE_TAG}_queue.log"
STATUS_FILE="$LOG_DIR/v164ab_${QUEUE_TAG}_status.log"
STAGE1_LOG="$LOG_DIR/v164a_${QUEUE_TAG}_stage1.log"
STAGE2_LOG="$LOG_DIR/v164b_${QUEUE_TAG}_stage2.log"

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

stats = {
    "support": None,
    "owner_takeover": None,
    "boundary_takeover": None,
    "boundary_focus": None,
    "layer_mean": None,
    "boundary_tag_fraction": None,
    "boundary_opacity_residual_abs_mean": None,
    "boundary_scaling_residual_abs_mean": None,
}

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "[ClarityTexTrunk]" in line:
            for key, pattern in (
                ("support", r"(?<!owner_)support=([0-9.]+)"),
                ("owner_takeover", r"owner_takeover=([0-9.]+)"),
                ("boundary_takeover", r"boundary_takeover=([0-9.]+)"),
                ("boundary_focus", r"boundary_focus=([0-9.]+)"),
            ):
                match = re.search(pattern, line)
                if match:
                    stats[key] = float(match.group(1))
        if "layer_mean=(" in line:
            match = re.search(r"layer_mean=\(([^,]+),([^,]+),([^)]+)\)", line)
            if match:
                stats["layer_mean"] = tuple(float(x) for x in match.groups())
        if "loss/boundary_tag_fraction" in line:
            match = re.search(r"loss/boundary_tag_fraction['\"]?: ([0-9.eE+-]+)", line)
            if match:
                stats["boundary_tag_fraction"] = float(match.group(1))

summary = {
    "stage": stage_name,
    "iteration": metrics.get("iteration"),
    "psnr": metrics.get("psnr"),
    "lpips": metrics.get("lpips"),
    "psnr_fg": metrics.get("psnr_fg"),
    "lpips_fg": metrics.get("lpips_fg"),
    "l1_fg": metrics.get("l1_fg"),
    "ssim_fg": metrics.get("ssim_fg"),
}
summary.update(stats)
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
log_msg "start_ckpt=$START_CKPT"
log_msg "stage1_exp=$V164A_EXP iters=$V164A_ITERS"
log_msg "stage2_exp=$V164B_EXP iters=$V164B_ITERS"

run_stage \
  "v164a_contouralpha_rootrepair" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v164a_v160a_contouralpha.sh" \
  "$STAGE1_LOG" \
  "$V164A_EXP" \
  "$START_CKPT" \
  "$V164A_ITERS" \
  "$@"

V164A_METRICS="$V164A_EXP/best_test_metrics.json"
V164A_CKPT="$V164A_EXP/best_ckpt.pth"
if [ ! -f "$V164A_METRICS" ] || [ ! -f "$V164A_CKPT" ]; then
  log_msg "v164a missing outputs"
  exit 1
fi
log_msg "v164a summary=$(dump_stage_summary v164a "$V164A_METRICS" "$STAGE1_LOG")"

run_stage \
  "v164b_texture_restore_contourlock" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v164b_v164a_texture_restore.sh" \
  "$STAGE2_LOG" \
  "$V164B_EXP" \
  "$V164A_CKPT" \
  "$V164B_ITERS" \
  "$@"

V164B_METRICS="$V164B_EXP/best_test_metrics.json"
V164B_CKPT="$V164B_EXP/best_ckpt.pth"
if [ ! -f "$V164B_METRICS" ] || [ ! -f "$V164B_CKPT" ]; then
  log_msg "v164b missing outputs"
  exit 1
fi
log_msg "v164b summary=$(dump_stage_summary v164b "$V164B_METRICS" "$STAGE2_LOG")"

log_msg "queue done"
log_msg "stage1_log=$STAGE1_LOG"
log_msg "stage2_log=$STAGE2_LOG"
log_msg "stage1_ckpt=$V164A_CKPT"
log_msg "stage2_ckpt=$V164B_CKPT"
