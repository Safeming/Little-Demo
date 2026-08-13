#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_ROOT="${PAPER_ROOT:-/remote-home/ming/3dgs-avatar-release-main}"
TRAIN_PY="${TRAIN_PY:-/opt/miniconda3/envs/gaussian_splatting/bin/python}"
TOOL_PY="${TOOL_PY:-/opt/miniconda3/envs/ictrl/bin/python}"
SGGS_REPO="${SGGS_REPO:-/remote-home/ming/SGGS}"
SGGS_HEAD="27b9ed9c9e4c5663deb169247c2339ccafe1c254"
DATASET_ROOT="${DATASET_ROOT:-/remote-home/ming/dataSet}"
BODY_MODELS="${BODY_MODELS:-/remote-home/ming/3dgs-avatar-release-main/body_models}"
AVATAR_ROOT="${AVATAR_ROOT:-/remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap}"
FROZEN_ROOT="${FROZEN_ROOT:-$PAPER_ROOT/exp/external/saga_canonical_five_subject_20260812_120625_bjt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER_ROOT/exp/external/sggs_released_code_canonical_three_subject_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SUBJECTS="${SUBJECTS:-377 386 394}"
ITERATIONS="${ITERATIONS:-30000}"
CANARY_ITERATIONS="${CANARY_ITERATIONS:-100}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
CANARY_ONLY="${CANARY_ONLY:-0}"
SKIP_CANARY="${SKIP_CANARY:-0}"
ETA_BJT="${ETA_BJT:-}"
CURRENT_SUBJECT=""
CURRENT_STAGE="startup"

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

write_state() {
  local status="$1" subject="${2:-}" stage="${3:-}" iteration="${4:-0}"
  "$TOOL_PY" - "$OUTPUT_ROOT/queue_state.json" "$status" "$subject" "$stage" "$iteration" "$ITERATIONS" "$ETA_BJT" <<'PY'
import datetime
import json
import os
import sys

path, status, subject, stage, iteration, iterations, eta = sys.argv[1:]
payload = {
    "status": status,
    "subject": subject,
    "stage": stage,
    "iteration": int(iteration),
    "iterations_per_subject": int(iterations),
    "pid": os.getppid(),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
if eta:
    payload["estimated_completion_bjt"] = eta
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 && "$DRY_RUN" != "1" ]]; then
    write_state failed "$CURRENT_SUBJECT" "$CURRENT_STAGE" 0 || true
  fi
}
trap on_exit EXIT

audit_release() {
  local command=(
    "$TOOL_PY" "$ROOT/tools/audit_sggs_released_code.py"
    --repo "$SGGS_REPO"
    --dataset "$DATASET_ROOT"
    --body-models "$BODY_MODELS"
    --python "$TRAIN_PY"
    --output "$OUTPUT_ROOT/audit/audit.json"
    --skip-network
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  elif [[ ! -f "$OUTPUT_ROOT/audit/audit.json" ]]; then
    mkdir -p "$OUTPUT_ROOT/audit"
    "${command[@]}" 2>&1 | tee "$OUTPUT_ROOT/audit/audit.log"
  fi
}

export_prior() {
  local subject_id="$1"
  local input="$FROZEN_ROOT/CoreView_${subject_id}/frozen_views"
  local output="$OUTPUT_ROOT/CoreView_${subject_id}/prior"
  local command=(
    "$TOOL_PY" "$ROOT/tools/export_sggs_canonical_prior.py"
    --input "$input"
    --output "$output"
    --avatar-data-root "$AVATAR_ROOT"
    --body-models "$BODY_MODELS"
    --sggs-repo "$SGGS_REPO"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  elif [[ ! -f "$output/manifest.json" ]]; then
    mkdir -p "$output"
    "${command[@]}" 2>&1 | tee "$OUTPUT_ROOT/CoreView_${subject_id}_prior.log"
  fi
}

run_training() {
  local subject_id="$1" iterations="$2" output="$3" log="$4"
  local input="$FROZEN_ROOT/CoreView_${subject_id}/frozen_views"
  local prior="$OUTPUT_ROOT/CoreView_${subject_id}/prior"
  local command=(
    env "CUDA_VISIBLE_DEVICES=$GPU" PYTHONUNBUFFERED=1 "$TRAIN_PY"
    "$ROOT/tools/train_sggs_released_code_canonical.py"
    --input "$input"
    --prior "$prior"
    --output "$output"
    --iterations "$iterations"
    --resume auto
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${command[@]}"
  elif [[ -f "$output/COMPLETE" ]]; then
    echo "[$(TZ=Asia/Shanghai date '+%F %T %Z')] reuse completed $output"
  else
    mkdir -p "$output"
    "${command[@]}" 2>&1 | tee "$log"
  fi
}

if [[ "$DRY_RUN" == "1" ]]; then
  audit_release
  for subject_id in $SUBJECTS; do
    export_prior "$subject_id"
  done
  if [[ "$SKIP_CANARY" != "1" ]]; then
    run_training 377 "$CANARY_ITERATIONS" "$OUTPUT_ROOT/canary/CoreView_377" "$OUTPUT_ROOT/canary_377.log"
  fi
  if [[ "$CANARY_ONLY" != "1" ]]; then
    for subject_id in $SUBJECTS; do
      run_training "$subject_id" "$ITERATIONS" "$OUTPUT_ROOT/CoreView_${subject_id}/train_30k" "$OUTPUT_ROOT/CoreView_${subject_id}_train.log"
    done
  fi
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
actual_head="$(git -C "$SGGS_REPO" rev-parse HEAD)"
if [[ "$actual_head" != "$SGGS_HEAD" ]]; then
  echo "unexpected SG-GS HEAD: $actual_head" >&2
  exit 2
fi

CURRENT_STAGE="audit"
write_state running "" "$CURRENT_STAGE" 0
audit_release
for subject_id in $SUBJECTS; do
  CURRENT_SUBJECT="CoreView_${subject_id}"
  CURRENT_STAGE="prior"
  write_state running "$CURRENT_SUBJECT" "$CURRENT_STAGE" 0
  export_prior "$subject_id"
done

if [[ "$SKIP_CANARY" != "1" ]]; then
  CURRENT_SUBJECT="CoreView_377"
  CURRENT_STAGE="canary"
  write_state running "$CURRENT_SUBJECT" "$CURRENT_STAGE" 0
  echo "[$(TZ=Asia/Shanghai date '+%F %T %Z')] CoreView_377 canary started"
  run_training 377 "$CANARY_ITERATIONS" "$OUTPUT_ROOT/canary/CoreView_377" "$OUTPUT_ROOT/canary_377.log"
  ETA_BJT="$("$TOOL_PY" - "$ROOT" "$OUTPUT_ROOT/canary/CoreView_377/metrics.jsonl" "$CANARY_ITERATIONS" "$ITERATIONS" "$OUTPUT_ROOT/eta.json" <<'PY'
import datetime
import json
import sys

root, metrics_path, canary_iterations, formal_iterations, output_path = sys.argv[1:]
sys.path.insert(0, root)
from utils.sggs_released_code_canonical import estimate_queue_seconds

with open(metrics_path, "r", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
estimate = estimate_queue_seconds(
    rows,
    canary_iterations=int(canary_iterations),
    formal_iterations=int(formal_iterations),
    subject_count=3,
    buffer_ratio=0.15,
)
now_utc = datetime.datetime.now(datetime.timezone.utc)
eta_utc = now_utc + datetime.timedelta(seconds=estimate["estimated_seconds"])
bjt = datetime.timezone(datetime.timedelta(hours=8))
estimate["measured_at_utc"] = now_utc.isoformat()
estimate["estimated_completion_bjt"] = eta_utc.astimezone(bjt).isoformat()
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(estimate, handle, indent=2, sort_keys=True)
print(estimate["estimated_completion_bjt"])
PY
)"
  write_state running "$CURRENT_SUBJECT" eta_ready "$CANARY_ITERATIONS"
  echo "Estimated completion BJT: $ETA_BJT"
fi

if [[ "$CANARY_ONLY" == "1" ]]; then
  CURRENT_SUBJECT=""
  CURRENT_STAGE="canary_complete"
  write_state canary_complete "" "$CURRENT_STAGE" "$CANARY_ITERATIONS"
  exit 0
fi

for subject_id in $SUBJECTS; do
  CURRENT_SUBJECT="CoreView_${subject_id}"
  CURRENT_STAGE="train"
  write_state running "$CURRENT_SUBJECT" "$CURRENT_STAGE" 0
  echo "[$(TZ=Asia/Shanghai date '+%F %T %Z')] $CURRENT_SUBJECT training started"
  run_training "$subject_id" "$ITERATIONS" "$OUTPUT_ROOT/$CURRENT_SUBJECT/train_30k" "$OUTPUT_ROOT/${CURRENT_SUBJECT}_train.log"
  echo "[$(TZ=Asia/Shanghai date '+%F %T %Z')] $CURRENT_SUBJECT training completed"
done

date -u '+%FT%TZ' > "$OUTPUT_ROOT/COMPLETE"
CURRENT_SUBJECT=""
CURRENT_STAGE="complete"
write_state complete "" complete "$ITERATIONS"
echo "[$(TZ=Asia/Shanghai date '+%F %T %Z')] all SG-GS canonical subjects completed"
