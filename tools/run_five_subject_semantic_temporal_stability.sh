#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
CAMERA="${CAMERA:-21}"
FRAME_END="${FRAME_END:-570}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/five_subject_semantic_temporal_stability_20260724}"
METHOD_FREEZE="${METHOD_FREEZE:-$ROOT/configs/semantic/frozen_a5_main_method_v1.json}"
ESTIMATED_TOTAL_SECONDS="${ESTIMATED_TOTAL_SECONDS:-14400}"
STATUS_FILE="$OUTPUT_ROOT/queue_status.txt"
PID_FILE="$OUTPUT_ROOT/queue.pid"
SUBJECTS=(377 386 387 393 394)

mkdir -p "$OUTPUT_ROOT/logs"
printf '%s\n' "$$" > "$PID_FILE"

timestamp() {
  TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S BJT'
}

estimated_finish() {
  local finish_epoch
  finish_epoch="$(( $(date +%s) + ESTIMATED_TOTAL_SECONDS ))"
  TZ=Asia/Shanghai date -d "@$finish_epoch" '+%Y-%m-%d %H:%M:%S BJT'
}

status() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

run() {
  printf '[semantic-temporal] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

subject_root() {
  case "$1" in
    377) printf '%s\n' "$ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721" ;;
    386) printf '%s\n' "$ROOT/exp/acceptdata/coreview386_multisubject_strict_20260719" ;;
    387) printf '%s\n' "$ROOT/exp/acceptdata/coreview387_multisubject_strict_20260720" ;;
    393) printf '%s\n' "$ROOT/exp/acceptdata/coreview393_multisubject_strict_20260721" ;;
    394) printf '%s\n' "$ROOT/exp/acceptdata/coreview394_multisubject_strict_20260722" ;;
    *) echo "unsupported subject: $1" >&2; return 2 ;;
  esac
}

subject_complete() {
  local subject="$1" summary="$OUTPUT_ROOT/CoreView_${subject}/summary.json"
  [[ -f "$summary" && -f "$OUTPUT_ROOT/CoreView_${subject}/metrics.csv" ]] || return 1
  "$PYTHON_BIN" - "$summary" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
videos = [Path(value) for value in summary.get("videos", [])]
ok = (
    int(summary.get("metric_row_count", 0)) == 6840
    and int(summary.get("frame_count", 0)) == 570
    and int(summary.get("camera", -1)) == 21
    and summary.get("methods") == ["voting", "a5"]
    and summary.get("parts") == ["hair", "face", "upper", "lower", "shoes", "skin"]
    and summary.get("canonical_selection_fixed_across_frames") is True
    and summary.get("uses_test_parser_for_edit_selection") is False
    and summary.get("uses_test_masks_for_metrics") is True
    and math.isfinite(float(summary.get("elapsed_seconds", float("nan"))))
    and len(videos) == 3
    and all(path.exists() and path.stat().st_size > 0 for path in videos)
    and [path.stem.rsplit("_", 1)[-1] for path in videos] == ["upper", "hair", "shoes"]
)
raise SystemExit(0 if ok else 1)
PY
}

validate_inputs() {
  local subject source path
  for subject in "${SUBJECTS[@]}"; do
    source="$(subject_root "$subject")"
    for path in \
      "$METHOD_FREEZE" \
      "$source/semantic_train_strict/ckpt42000.pth" \
      "$source/assets/test/.hydra/config.yaml" \
      "$source/banks/multiview_voting/part_label_bank.npz" \
      "$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" \
      "$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz"; do
      if [[ "$DRY_RUN" != "1" && ! -e "$path" ]]; then
        echo "missing input: $path" >&2
        return 2
      fi
    done
  done
}

run_subject() {
  local subject="$1" source output log
  source="$(subject_root "$subject")"
  output="$OUTPUT_ROOT/CoreView_${subject}"
  log="$OUTPUT_ROOT/logs/CoreView_${subject}.log"
  if [[ "$DRY_RUN" != "1" ]] && subject_complete "$subject"; then
    status "CoreView_${subject} already complete; reusing temporal outputs"
    return 0
  fi
  status "CoreView_${subject} semantic temporal evaluation started"
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/render_semantic_temporal_stability.py \
    --subject "$subject" \
    --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --a5-bank "$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz" \
    --loso-config "$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" \
    --method-freeze "$METHOD_FREEZE" \
    --checkpoint "$source/semantic_train_strict/ckpt42000.pth" \
    --config "$source/assets/test/.hydra/config.yaml" \
    --output-dir "$output" \
    --camera "$CAMERA" \
    --frame-start 0 --frame-end "$FRAME_END" --frame-step 1 \
    --methods voting a5 \
    --parts hair face upper lower shoes skin \
    --video-parts upper hair shoes \
    --video-fps 25 \
    --screen-threshold 0.2 \
    --edit-strength 1.0 \
    --explicit-binding-render-preset none \
    > "$log" 2>&1
  status "CoreView_${subject} semantic temporal evaluation completed"
}

main() {
  local subject
  : > "$STATUS_FILE"
  status "five-subject semantic temporal queue started pid=$$ dry_run=$DRY_RUN"
  status "estimated_finish_bjt=$(estimated_finish)"
  validate_inputs
  for subject in "${SUBJECTS[@]}"; do
    run_subject "$subject"
  done
  status "semantic temporal aggregate started"
  run "$PYTHON_BIN" tools/summarize_semantic_temporal_stability.py --output-root "$OUTPUT_ROOT" \
    > "$OUTPUT_ROOT/logs/aggregate.log" 2>&1
  status "semantic temporal aggregate completed"
  status "five-subject semantic temporal queue completed status=0"
}

main "$@"
