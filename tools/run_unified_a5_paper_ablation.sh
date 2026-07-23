#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
RESUME="${RESUME:-1}"
ESTIMATED_SECONDS="${ESTIMATED_SECONDS:-1800}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/unified_a5_paper_ablation_20260723}"
LOSO_ROOT="$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723"
FULL_A5_ROOT="$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723"
METHOD_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
STATUS_FILE="$OUTPUT_ROOT/queue_status.txt"
PID_FILE="$OUTPUT_ROOT/queue.pid"
SUBJECTS=(377 386 387 393 394)

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
  printf '[unified-a5-ablation] command:'
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

loso_config() {
  printf '%s\n' "$LOSO_ROOT/CoreView_${1}/loso_frozen_config.json"
}

full_a5_bank() {
  printf '%s\n' "$FULL_A5_ROOT/CoreView_${1}/banks/footprint_evidence_target/part_label_bank.npz"
}

variant_bank() {
  printf '%s\n' "$OUTPUT_ROOT/CoreView_${1}/banks/${2}/part_label_bank.npz"
}

bank_complete() {
  local bank="$1"
  [[ -s "$bank" && -s "$(dirname "$bank")/summary.json" ]]
}

stage_complete() {
  local output="$1"
  [[ -s "$output/baseline_summary.csv" && -s "$output/matched_retention.csv" && -s "$output/summary.json" ]]
}

validate_inputs() {
  local subject source path
  for subject in "${SUBJECTS[@]}"; do
    source="$(subject_root "$subject")"
    for path in \
      "$METHOD_FREEZE" \
      "$(protocol_path "$subject")" \
      "$(loso_config "$subject")" \
      "$(full_a5_bank "$subject")" \
      "$source/semantic_train_strict/ckpt42000.pth" \
      "$source/assets/calibration/test-view/semantic_editable_assets" \
      "$source/assets/test/test-view/semantic_editable_assets" \
      "$source/banks/raw_trained/part_label_bank.npz" \
      "$source/banks/multiview_voting/part_label_bank.npz" \
      "$source/banks/voting_evidence_target_support/part_label_bank.npz"; do
      if [[ "$DRY_RUN" != "1" && ! -e "$path" ]]; then
        echo "missing input: $path" >&2
        return 2
      fi
    done
  done
}

materialize_loso_provenance() {
  local subject="$1" source destination
  source="$(loso_config "$subject")"
  destination="$OUTPUT_ROOT/CoreView_${subject}/loso_frozen_config.json"
  mkdir -p "$(dirname "$destination")"
  if [[ "$DRY_RUN" != "1" && -s "$destination" ]] && cmp -s "$source" "$destination"; then
    return
  fi
  run cp -- "$source" "$destination"
}

build_variant_bank() {
  local subject="$1" variant="$2" source output
  source="$(subject_root "$subject")"
  output="$(variant_bank "$subject" "$variant")"
  if [[ "$DRY_RUN" != "1" && "$REUSE_COMPLETED" == "1" ]] && bank_complete "$output"; then
    status "CoreView_${subject} ${variant} bank reused"
    return
  fi
  local geometry_args=()
  if [[ "$variant" == "center_only" ]]; then
    geometry_args=(
      --footprint-radius-scale 0 --min-footprint-radius 0 --max-footprint-radius 0
      --outer-penalty-power 0.2
    )
  elif [[ "$variant" == "no_outer" ]]; then
    geometry_args=(
      --footprint-radius-scale 1.0 --min-footprint-radius 1 --max-footprint-radius 12
      --outer-penalty-power 0
    )
  else
    echo "unsupported variant: $variant" >&2
    return 2
  fi
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/calibrate_evidence_soft_edit_weights.py \
    --part-label-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --checkpoint "$source/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$source/assets/calibration/test-view/semantic_editable_assets" \
    --output "$output" --summary-json "$(dirname "$output")/summary.json" \
    --explicit-binding-render-preset none --mode evidence-calibrated \
    --parts face hair upper lower shoes skin \
    "${geometry_args[@]}" \
    --conflict-penalty-power 1.0 \
    --protocol "$(protocol_path "$subject")" --protocol-split calibration
}

evaluate_stage() {
  local subject="$1" stage="$2" footprint="$3"
  shift 3
  local source output
  source="$(subject_root "$subject")"
  output="$OUTPUT_ROOT/CoreView_${subject}/$stage"
  if [[ "$DRY_RUN" != "1" && "$REUSE_COMPLETED" == "1" ]] && stage_complete "$output"; then
    status "CoreView_${subject} ${stage} evaluation reused"
    return
  fi
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$(protocol_path "$subject")" --protocol-split test \
    --frozen-config "$(loso_config "$subject")" \
    --raw-trained-bank "$source/banks/raw_trained/part_label_bank.npz" \
    --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
    --footprint-bank "$footprint" --method-freeze "$METHOD_FREEZE" \
    --trained-bank "$source/banks/voting_evidence_target_support/part_label_bank.npz" \
    --checkpoint "$source/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$source/assets/test/test-view/semantic_editable_assets" \
    --explicit-binding-render-preset none --output-dir "$output" "$@"
}

main() {
  local subject center_bank no_outer_bank
  if [[ "$RESUME" != "1" || ! -e "$STATUS_FILE" ]]; then
    : > "$STATUS_FILE"
  fi
  status "unified five-subject A5 paper ablation queue started pid=$$ dry_run=$DRY_RUN"
  status "estimated_finish_bjt=$(estimated_finish)"
  validate_inputs
  for subject in "${SUBJECTS[@]}"; do
    materialize_loso_provenance "$subject"
    status "CoreView_${subject} center-only bank started"
    build_variant_bank "$subject" center_only
    status "CoreView_${subject} center-only bank completed"
    status "CoreView_${subject} no-outer bank started"
    build_variant_bank "$subject" no_outer
    status "CoreView_${subject} no-outer bank completed"
    center_bank="$(variant_bank "$subject" center_only)"
    no_outer_bank="$(variant_bank "$subject" no_outer)"

    status "CoreView_${subject} A0-A6 component evaluation started"
    evaluate_stage "$subject" component "$(full_a5_bank "$subject")" \
      --baselines A0 A1 A2 A3 A4 A5 A6 --retention-reference-baseline A0
    status "CoreView_${subject} A0-A6 component evaluation completed"

    status "CoreView_${subject} center-only micro evaluation started"
    evaluate_stage "$subject" center_only "$center_bank" \
      --baselines A0 A4 A5 --retention-reference-baseline A0
    status "CoreView_${subject} center-only micro evaluation completed"

    status "CoreView_${subject} no-outer micro evaluation started"
    evaluate_stage "$subject" no_outer "$no_outer_bank" \
      --baselines A0 A4 A5 --retention-reference-baseline A0
    status "CoreView_${subject} no-outer micro evaluation completed"
  done
  run "$PYTHON_BIN" tools/summarize_unified_a5_paper_ablation.py \
    --output-root "$OUTPUT_ROOT" --bootstrap-repetitions 10000 --bootstrap-seed 20260723
  status "unified five-subject A5 paper ablation queue completed status=0"
}

main "$@"
