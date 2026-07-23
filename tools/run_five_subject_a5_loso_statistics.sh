#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
RESUME="${RESUME:-1}"
ESTIMATED_SECONDS="${ESTIMATED_SECONDS:-2700}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723}"
FOOTPRINT_ROOT="${FOOTPRINT_ROOT:-$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723}"
METHOD_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
STATUS_FILE="$OUTPUT_ROOT/queue_status.txt"
PID_FILE="$OUTPUT_ROOT/queue.pid"
SUBJECTS=(377 386 387 393 394)
THRESHOLDS=(0.05 0.1 0.15 0.2 0.25 0.35 0.5)

mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "$$" > "$PID_FILE"

timestamp() {
  TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S BJT'
}

estimated_finish() {
  TZ=Asia/Shanghai date -d "+${ESTIMATED_SECONDS} seconds" '+%Y-%m-%d %H:%M:%S BJT'
}

status() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"
}

run() {
  printf '[five-subject-a5-loso] command:'
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

protocol_path() {
  printf '%s\n' "$ROOT/configs/semantic/coreview${1}_strict_paper_protocol.json"
}

footprint_bank() {
  printf '%s\n' "$FOOTPRINT_ROOT/CoreView_${1}/banks/footprint_evidence_target/part_label_bank.npz"
}

threshold_token() {
  printf '%s\n' "${1//./p}"
}

candidate_dir() {
  printf '%s\n' "$OUTPUT_ROOT/validation_candidates/CoreView_${1}/t$(threshold_token "$2")"
}

fingerprint() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'dry-run-fingerprint\n'
    return
  fi
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from utils.semantic_eval_protocol import file_fingerprint
print(file_fingerprint(sys.argv[1]))
PY
}

validate_inputs() {
  local subject source path
  for subject in "${SUBJECTS[@]}"; do
    source="$(subject_root "$subject")"
    for path in \
      "$METHOD_FREEZE" \
      "$(protocol_path "$subject")" \
      "$source/semantic_train_strict/ckpt42000.pth" \
      "$source/assets/validation/test-view/semantic_editable_assets" \
      "$source/assets/test/test-view/semantic_editable_assets" \
      "$source/banks/raw_trained/part_label_bank.npz" \
      "$source/banks/multiview_voting/part_label_bank.npz" \
      "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
      "$(footprint_bank "$subject")"; do
      if [[ "$DRY_RUN" != "1" && ! -e "$path" ]]; then
        echo "missing input: $path" >&2
        return 2
      fi
    done
  done
}

candidate_complete() {
  local output="$1"
  [[ -s "$output/baseline_summary.csv" && -s "$output/leakage_retention_curve.csv" && -s "$output/matched_retention.csv" ]]
}

main_complete() {
  local output="$1"
  [[ -s "$output/baseline_summary.csv" && -s "$output/matched_retention.csv" && -s "$output/summary.json" ]]
}

evaluate_validation_candidate() {
  local subject="$1" threshold="$2" source output
  source="$(subject_root "$subject")"
  output="$(candidate_dir "$subject" "$threshold")"
  if [[ "$DRY_RUN" != "1" && "$REUSE_COMPLETED" == "1" ]] && candidate_complete "$output"; then
    status "CoreView_${subject} A5 validation threshold=${threshold} reused"
    return
  fi
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$(protocol_path "$subject")" --protocol-split validation \
    --soft-threshold "$threshold" --support-threshold 0.1 --boundary-radius 6 \
    --raw-trained-bank "$source/banks/raw_trained/part_label_bank.npz" \
    --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --footprint-bank "$(footprint_bank "$subject")" \
    --method-freeze "$METHOD_FREEZE" \
    --trained-bank "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
    --checkpoint "$source/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$source/assets/validation/test-view/semantic_editable_assets" \
    --explicit-binding-render-preset none --output-dir "$output" \
    --baselines B1 A5 --retention-reference-baseline B1
}

build_loso_config() {
  local held_out="$1" source output checkpoint evidence footprint checkpoint_fp bank_fp footprint_fp
  local donor threshold report
  local donor_args=()
  source="$(subject_root "$held_out")"
  output="$OUTPUT_ROOT/CoreView_${held_out}/loso_frozen_config.json"
  checkpoint="$source/semantic_train_strict/ckpt42000.pth"
  evidence="$source/banks/voting_evidence_target_support/part_label_bank.npz"
  footprint="$(footprint_bank "$held_out")"
  checkpoint_fp="$(fingerprint "$checkpoint")"
  bank_fp="$(fingerprint "$evidence")"
  footprint_fp="$(fingerprint "$footprint")"
  for donor in "${SUBJECTS[@]}"; do
    [[ "$donor" == "$held_out" ]] && continue
    for threshold in "${THRESHOLDS[@]}"; do
      report="$(candidate_dir "$donor" "$threshold")"
      if [[ "$DRY_RUN" != "1" ]] && ! candidate_complete "$report"; then
        echo "missing A5 donor candidate: $report" >&2
        return 2
      fi
      donor_args+=(--donor-candidate "$donor:$threshold:$report")
    done
  done
  run "$PYTHON_BIN" tools/select_frozen_a5_loso_config.py \
    --protocol "$(protocol_path "$held_out")" --held-out-subject "$held_out" \
    "${donor_args[@]}" --checkpoint-fingerprint "$checkpoint_fp" \
    --bank-fingerprint "$bank_fp" --footprint-bank-fingerprint "$footprint_fp" \
    --method-freeze "$METHOD_FREEZE" --required-retention 0.5 0.6 \
    --max-miou-gap 0.02 --expected-donor-count 4 --output "$output"
  printf '%s\n' "$output"
}

evaluate_test_main() {
  local subject="$1" frozen="$2" source output
  source="$(subject_root "$subject")"
  output="$OUTPUT_ROOT/CoreView_${subject}/main"
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$(protocol_path "$subject")" --protocol-split test \
    --frozen-config "$frozen" \
    --raw-trained-bank "$source/banks/raw_trained/part_label_bank.npz" \
    --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --footprint-bank "$(footprint_bank "$subject")" \
    --method-freeze "$METHOD_FREEZE" \
    --trained-bank "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
    --checkpoint "$source/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$source/assets/test/test-view/semantic_editable_assets" \
    --explicit-binding-render-preset none --output-dir "$output" \
    --baselines B0 B1 B2 B3 B4 A5 --retention-reference-baseline B1
}

main() {
  local subject threshold frozen
  if [[ "$RESUME" != "1" || ! -e "$STATUS_FILE" ]]; then
    : > "$STATUS_FILE"
  fi
  status "five-subject frozen A5 LOSO/statistics queue started pid=$$ dry_run=$DRY_RUN"
  status "estimated_finish_bjt=$(estimated_finish)"
  validate_inputs
  for subject in "${SUBJECTS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      status "CoreView_${subject} A5 validation threshold=${threshold} started"
      evaluate_validation_candidate "$subject" "$threshold"
      status "CoreView_${subject} A5 validation threshold=${threshold} completed"
    done
  done
  for subject in "${SUBJECTS[@]}"; do
    status "CoreView_${subject} four-donor A5 LOSO selection started"
    frozen="$OUTPUT_ROOT/CoreView_${subject}/loso_frozen_config.json"
    build_loso_config "$subject"
    if [[ "$DRY_RUN" != "1" ]]; then
      if [[ -s "$frozen" ]]; then
        :
      else
        status "CoreView_${subject} four-donor A5 LOSO selection failed: missing config"
        return 2
      fi
    fi
    if [[ "$DRY_RUN" != "1" && "$REUSE_COMPLETED" == "1" ]] && \
      main_complete "$OUTPUT_ROOT/CoreView_${subject}/main"; then
      status "CoreView_${subject} frozen A5 test main reused"
      continue
    fi
    status "CoreView_${subject} frozen A5 test main started"
    evaluate_test_main "$subject" "$frozen"
    status "CoreView_${subject} frozen A5 test main completed"
  done
  run "$PYTHON_BIN" tools/summarize_five_subject_a5_loso_statistics.py \
    --output-root "$OUTPUT_ROOT" --bootstrap-repetitions 10000 --bootstrap-seed 20260723
  status "five-subject frozen A5 LOSO/statistics queue completed status=0"
}

main "$@"
