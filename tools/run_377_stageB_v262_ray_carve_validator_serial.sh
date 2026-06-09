#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-stageB_v262_ray_carve_validator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v262_ray_carve_validator_${RUN_ID}}"
mkdir -p "$LOG_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt135710.pth}"

SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
STATUS_JSON="$LOG_DIR/status.json"
VALIDATOR_DIR="$LOG_DIR/v262_ray_carve_validator"

WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
GPU_MAX_USED_MB_START="${GPU_MAX_USED_MB_START:-18000}"
GPU_MAX_UTIL_START="${GPU_MAX_UTIL_START:-65}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tkind\texp_dir\trender_exp\tckpt\tstatus\tdetail\n' > "$SUMMARY"

log_event() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$GPU" "$1" "$2" "$3" | tee -a "$EVENTS"
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" <<'PY' || true
import json, sys, time
from pathlib import Path
path, run_id, gpu, phase, detail = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

summary_row() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$SUMMARY"
}

gpu_stats() {
  nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

wait_for_gpu() {
  local name="$1"
  if [ "$WAIT_FOR_FREE_GPU" != "1" ]; then
    return 0
  fi
  local used util
  while true; do
    read -r used util < <(gpu_stats)
    used="${used:-0}"
    util="${util:-0}"
    if [ "$used" -le "$GPU_MAX_USED_MB_START" ] && [ "$util" -le "$GPU_MAX_UTIL_START" ]; then
      log_event "$name" "gpu_ready" "used=${used}MiB util=${util}%"
      return 0
    fi
    log_event "$name" "gpu_wait" "used=${used}MiB util=${util}% threshold=${GPU_MAX_USED_MB_START}MiB/${GPU_MAX_UTIL_START}%"
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

common_env() {
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
    "$@"
}

check_required() {
  local missing=0
  for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT"; do
    if [ ! -e "$required" ]; then
      echo "missing required path: $required" >&2
      log_event "preflight" "missing" "$required"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    write_status "blocked" "missing required paths; see $EVENTS"
    return 2
  fi
}

run_validator() {
  wait_for_gpu "v262_ray_carve_validator"
  log_event "v262_ray_carve_validator" "start" "base_ckpt=$BASE_CKPT"
  common_env "$PYTHON_BIN" "$ROOT/tools/validate_377_stageB_v262_ray_carve_candidates.py" \
    --config-path "$BASE_EXP/.hydra/config.yaml" \
    --load-ckpt "$BASE_CKPT" \
    --out-dir "$VALIDATOR_DIR" \
    --dataset-root "$DATA_ROOT" \
    --parser-root "$PARSER_ROOT" \
    --compact-mapping "$COMPACT_MAPPING" \
    --candidate-views "${CANDIDATE_VIEWS:-1,2,3,4,5,6,7,8,9,10,11,12}" \
    --eval-views "${EVAL_VIEWS:-21,22,23}" \
    --frames "${FRAMES:-0,570,60}" \
    --max-components-per-view "${MAX_COMPONENTS_PER_VIEW:-6}" \
    --points-per-component "${POINTS_PER_COMPONENT:-2}" \
    --min-component-area "${MIN_COMPONENT_AREA:-18}" \
    --depth-samples "${DEPTH_SAMPLES:-9}" \
    --depth-margin "${DEPTH_MARGIN:-0.060}" \
    --depth-search-radius "${DEPTH_SEARCH_RADIUS:-32.0}" \
    --min-inner-views "${MIN_INNER_VIEWS:-2}" \
    --max-outer-views "${MAX_OUTER_VIEWS:-0}" \
    --min-heldout-inner-views "${MIN_HELDOUT_INNER_VIEWS:-1}" \
    --max-heldout-outer-views "${MAX_HELDOUT_OUTER_VIEWS:-0}" \
    --max-candidates-per-frame "${MAX_CANDIDATES_PER_FRAME:-16}" \
    --topk "${TOPK:-12}" \
    > "$LOG_DIR/v262_ray_carve_validator.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    log_event "v262_ray_carve_validator" "failed" "status=$status"
    summary_row "v262_ray_carve_validator" "diagnostic" "" "" "$BASE_CKPT" "failed" "status=$status"
    return "$status"
  fi
  local candidate_status
  candidate_status="$("$PYTHON_BIN" - "$VALIDATOR_DIR/ray_carve_candidate_summary.json" <<'PY'
import json, sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
print(s.get("status", "unknown"))
PY
)"
  log_event "v262_ray_carve_validator" "$candidate_status" "$VALIDATOR_DIR/ray_carve_candidate_summary.json"
  summary_row "v262_ray_carve_validator" "diagnostic" "" "" "$BASE_CKPT" "$candidate_status" "$VALIDATOR_DIR"
}

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
  echo "GPU=$GPU"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "LOG_DIR=$LOG_DIR"
} | tee "$LOG_DIR/run_info.txt"

write_status "starting" "preflight"
check_required || exit $?
run_validator || true

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "queue" "all_done" "summary=$SUMMARY end=$END_BJT"
write_status "done" "END_BJT=$END_BJT SUMMARY=$SUMMARY"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "EVENTS=$EVENTS"
  echo "STATUS_JSON=$STATUS_JSON"
  echo "VALIDATOR_SUMMARY=$VALIDATOR_DIR/ray_carve_candidate_summary.json"
} | tee -a "$LOG_DIR/run_info.txt"
