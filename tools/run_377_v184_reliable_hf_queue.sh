#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"

BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
BASE_CKPT="$BASE_EXP/best_ckpt.pth"
DATA_ROOT="$ROOT/data/ZJUMoCap"
LOG_DIR="$ROOT/exp/stageA2/logs/v184_reliable_hf_$RUN_ID"
SUMMARY="$LOG_DIR/summary.tsv"
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

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU_ID" "$1" "$2" <<'PY'
import json
import sys
import time
from pathlib import Path

path, run_id, gpu_id, phase, detail = sys.argv[1:]
data = {
    "run_id": run_id,
    "phase": phase,
    "detail": detail,
    "gpu_id": gpu_id,
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

extract_metrics() {
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

compare_lpips_fg_gate() {
  local candidate="$1"
  local max_worse="${2:-0.010}"
  local baseline_metrics="${3:-$BASE_EXP/best_test_metrics.json}"
  "$PYTHON_BIN" - "$baseline_metrics" "$candidate/best_test_metrics.json" "$max_worse" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
max_worse = float(sys.argv[3])
if not base_path.exists():
    print("missing_baseline_metrics")
    raise SystemExit(2)
if not candidate_path.exists():
    print("missing_candidate_metrics")
    raise SystemExit(2)
base = json.loads(base_path.read_text())
candidate = json.loads(candidate_path.read_text())
base_lpips = float(base.get("lpips_fg", 1e9))
candidate_lpips = float(candidate.get("lpips_fg", 1e9))
delta = candidate_lpips - base_lpips
print(f"baseline={base_path.parent} base_lpips_fg={base_lpips:.6f} candidate_lpips_fg={candidate_lpips:.6f} delta={delta:.6f} max_worse={max_worse:.6f}")
raise SystemExit(0 if delta <= max_worse else 3)
PY
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

printf 'name\tstatus\texp_dir\tlog\textra\n' > "$SUMMARY"
log_msg "v184 reliable HF queue start run_id=$RUN_ID gpu=$GPU_ID"
log_msg "base_exp=$BASE_EXP"
write_status "queue" "started"

V184A_EXP="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v184a_v179c_reliable_hf_supervision_$RUN_ID"
if run_step \
  "v184a_reliable_hf_supervision" \
  "$LOG_DIR/v184a_reliable_hf_supervision.log" \
  bash "$ROOT/tools/start_377_stageA2_v184a_v179c_reliable_hf_supervision.sh" \
    "$V184A_EXP" \
    "$BASE_CKPT" \
    "${V184A_ITERATIONS:-2400}"; then
  metrics="$(extract_metrics "$V184A_EXP")"
  append_summary "v184a_reliable_hf_supervision" "ok" "$V184A_EXP" "$LOG_DIR/v184a_reliable_hf_supervision.log" "$metrics"
else
  append_summary "v184a_reliable_hf_supervision" "failed" "$V184A_EXP" "$LOG_DIR/v184a_reliable_hf_supervision.log" "render_existing_if_any"
fi

if [ -f "$V184A_EXP/best_ckpt.pth" ]; then
  V184A_RENDER_EXP="${V184A_EXP}_render_quick"
  render_eval \
    "v184a_reliable_hf_render" \
    "$V184A_EXP" \
    "$V184A_EXP/best_ckpt.pth" \
    "$V184A_RENDER_EXP" \
    60 \
    "$LOG_DIR/v184a_reliable_hf_render.log"
  append_summary "v184a_reliable_hf_render" "$?" "$V184A_RENDER_EXP" "$LOG_DIR/v184a_reliable_hf_render.log" "quick_render"
  run_contour_diag "v184a_reliable_hf_contour" "$V184A_RENDER_EXP" "$LOG_DIR/v184a_reliable_hf_contour.log"
  append_summary "v184a_reliable_hf_contour" "$?" "$V184A_RENDER_EXP" "$LOG_DIR/v184a_reliable_hf_contour.log" "contour"
  if gate_out="$(compare_lpips_fg_gate "$V184A_EXP" 0.010)"; then
    log_msg "GATE PASS v184a $gate_out"
    append_summary "gate_v184a_vs_v179c" "pass" "$V184A_EXP" "$LOG_DIR/queue.log" "$gate_out"
  else
    gate_status=$?
    log_msg "GATE WARN v184a status=$gate_status $gate_out"
    append_summary "gate_v184a_vs_v179c" "warn" "$V184A_EXP" "$LOG_DIR/queue.log" "$gate_out"
  fi
fi

write_status "queue" "completed"
log_msg "v184 reliable HF queue complete summary=$SUMMARY"
exit 0
