#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
QUEUE_TAG="$(date +%Y%m%d_%H%M%S)"

V157A_EXP="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v157a_v152a_supportmain_screen4k}"
V157B_EXP="${2:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157b_v157a_ownerhf_sharphandoff_screen3k}"
V157C_EXP="${3:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v157c_v157b_signedfinishing_screen2k}"
V157A_ITERS="${4:-4000}"
V157B_ITERS="${5:-3000}"
V157C_ITERS="${6:-2000}"
if [ "$#" -ge 6 ]; then
  shift 6
else
  shift "$#"
fi

mkdir -p "$LOG_DIR" "$(dirname "$V157A_EXP")" "$(dirname "$V157B_EXP")" "$(dirname "$V157C_EXP")"

QUEUE_LOG="$LOG_DIR/v157abc_${QUEUE_TAG}_queue.log"
STATUS_FILE="$LOG_DIR/v157abc_${QUEUE_TAG}_status.log"
STAGE1_LOG="$LOG_DIR/v157a_${QUEUE_TAG}_stage1.log"
STAGE2_LOG="$LOG_DIR/v157b_${QUEUE_TAG}_stage2.log"
STAGE3_LOG="$LOG_DIR/v157c_${QUEUE_TAG}_stage3.log"
V157_SKIP_STAGE1="${V157_SKIP_STAGE1:-0}"
V157A_LOG_PATH="${V157A_LOG_PATH:-$STAGE1_LOG}"

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
    "support": support,
    "owner_takeover": owner_takeover,
    "boundary_takeover": boundary_takeover,
    "layer_mean": layer_mean,
}
print(json.dumps(summary, ensure_ascii=True))
PY
}

check_stage1_gate() {
  local metrics_path="$1"
  local log_path="$2"
  "$PYTHON_BIN" - "$metrics_path" "$log_path" <<'PY'
import json
import re
import sys

metrics_path, log_path = sys.argv[1:3]
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

support = None
owner_takeover = None
boundary_takeover = None
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
        if "layer_mean=(" in line:
            match = re.search(r"layer_mean=\(([^,]+),([^,]+),([^)]+)\)", line)
            if match:
                layer_mean = tuple(float(x) for x in match.groups())

soft_layer = layer_mean[1] if layer_mean else None
passed = (
    metrics.get("lpips_fg", 999.0) <= 0.210
    and support is not None and support >= 0.880
    and soft_layer is not None and soft_layer <= 0.75
    and owner_takeover is not None and owner_takeover <= 0.05
    and boundary_takeover is not None and boundary_takeover <= 0.02
)
sys.exit(0 if passed else 1)
PY
}

check_stage2_gate() {
  local metrics_path="$1"
  local log_path="$2"
  "$PYTHON_BIN" - "$metrics_path" "$log_path" <<'PY'
import json
import re
import sys

metrics_path, log_path = sys.argv[1:3]
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

support = None
owner_takeover = None
boundary_takeover = None

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

passed = (
    metrics.get("lpips_fg", 999.0) <= 0.178
    and support is not None and support >= 0.895
    and owner_takeover is not None and owner_takeover >= 0.55
    and boundary_takeover is not None and boundary_takeover <= 0.05
)
sys.exit(0 if passed else 1)
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
log_msg "stage1_exp=$V157A_EXP"
log_msg "stage2_exp=$V157B_EXP"
log_msg "stage3_exp=$V157C_EXP"

if [ "$V157_SKIP_STAGE1" = "1" ]; then
  log_msg "v157a skip enabled; reuse exp=$V157A_EXP log=$V157A_LOG_PATH"
else
  run_stage \
    "v157a" \
    "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_fresh_v157a_supportmain.sh" \
    "$STAGE1_LOG" \
    "$V157A_EXP" \
    "$V157A_ITERS" \
    "$@"
  V157A_LOG_PATH="$STAGE1_LOG"
fi

V157A_METRICS="$V157A_EXP/best_test_metrics.json"
V157A_CKPT="$V157A_EXP/best_ckpt.pth"
if [ ! -f "$V157A_METRICS" ] || [ ! -f "$V157A_CKPT" ]; then
  log_msg "v157a missing outputs"
  exit 1
fi
if [ ! -f "$V157A_LOG_PATH" ]; then
  log_msg "v157a missing log: $V157A_LOG_PATH"
  exit 1
fi
log_msg "v157a summary=$(dump_stage_summary v157a "$V157A_METRICS" "$V157A_LOG_PATH")"
if ! check_stage1_gate "$V157A_METRICS" "$V157A_LOG_PATH"; then
  log_msg "v157a gate failed; stop before v157b"
  exit 2
fi

run_stage \
  "v157b" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157b_sharphandoff.sh" \
  "$STAGE2_LOG" \
  "$V157B_EXP" \
  "$V157A_CKPT" \
  "$V157B_ITERS" \
  "$@"

V157B_METRICS="$V157B_EXP/best_test_metrics.json"
V157B_CKPT="$V157B_EXP/best_ckpt.pth"
if [ ! -f "$V157B_METRICS" ] || [ ! -f "$V157B_CKPT" ]; then
  log_msg "v157b missing outputs"
  exit 1
fi
log_msg "v157b summary=$(dump_stage_summary v157b "$V157B_METRICS" "$STAGE2_LOG")"
if ! check_stage2_gate "$V157B_METRICS" "$STAGE2_LOG"; then
  log_msg "v157b gate failed; stop before v157c"
  exit 3
fi

run_stage \
  "v157c" \
  "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v157c_signedfinishing.sh" \
  "$STAGE3_LOG" \
  "$V157C_EXP" \
  "$V157B_CKPT" \
  "$V157C_ITERS" \
  "$@"

V157C_METRICS="$V157C_EXP/best_test_metrics.json"
if [ ! -f "$V157C_METRICS" ]; then
  log_msg "v157c missing metrics"
  exit 1
fi
log_msg "v157c summary=$(dump_stage_summary v157c "$V157C_METRICS" "$STAGE3_LOG")"
log_msg "queue done"
log_msg "stage1_log=$STAGE1_LOG"
log_msg "stage2_log=$STAGE2_LOG"
log_msg "stage3_log=$STAGE3_LOG"
