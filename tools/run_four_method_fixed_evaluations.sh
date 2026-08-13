#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/root/.config/superpowers/worktrees/3dgs-avatar-release-main/sggs-released-code-canonical}"
PAPER_ROOT="${PAPER_ROOT:-/remote-home/ming/3dgs-avatar-release-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER_ROOT/exp/acceptdata/four_method_paper_evidence_20260813}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$PAPER_ROOT/data/ZJUMoCap}"
METHOD_FREEZE="${METHOD_FREEZE:-$PAPER_ROOT/configs/semantic/frozen_a5_main_method_v1.json}"
SUBJECTS=(377 386 394)
METHODS=(saga gaussian_grouping sggs)

subject_root() {
  case "$1" in
    377) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721" ;;
    386) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview386_multisubject_strict_20260719" ;;
    394) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview394_multisubject_strict_20260722" ;;
    *) echo "unsupported subject: $1" >&2; return 2 ;;
  esac
}

protocol_path() {
  printf '%s\n' "$PAPER_ROOT/configs/semantic/coreview${1}_strict_paper_protocol.json"
}

external_root() {
  case "$1" in
    saga) printf '%s\n' "$PAPER_ROOT/exp/external/saga_canonical_five_subject_20260812_120625_bjt" ;;
    gaussian_grouping) printf '%s\n' "$PAPER_ROOT/exp/external/gaussian_grouping_canonical_three_subject_20260813_0958_bjt" ;;
    sggs) printf '%s\n' "$PAPER_ROOT/exp/external/sggs_released_code_canonical_three_subject_20260813_formal" ;;
    *) echo "unsupported method: $1" >&2; return 2 ;;
  esac
}

external_frozen() {
  local method="$1" subject="$2" root
  root="$(external_root "$method")/CoreView_${subject}/evaluation"
  case "$method" in
    saga) printf '%s\n' "$root/frozen_saga_soft_config.json" ;;
    gaussian_grouping) printf '%s\n' "$root/frozen_gg_loso_config.json" ;;
    sggs) printf '%s\n' "$root/frozen_sggs_loso_config.json" ;;
  esac
}

preset() {
  if [[ "$1" == "377" ]]; then
    printf '%s\n' v338_temporal_selector_grow_only_guard
  else
    printf '%s\n' none
  fi
}

complete() {
  [[ -s "$1/per_view_spatial.csv" && -s "$1/summary.json" ]]
}

run_subject_stage() {
  local stage="$1" subject="$2" source asset_root output_root frozen evidence raw footprint cache method external_bank
  source="$(subject_root "$subject")"
  frozen="$OUTPUT_ROOT/protocol/CoreView_${subject}_a5_shared40k_frozen.json"
  evidence="$source/banks/voting_evidence_target_support/part_label_bank.npz"
  raw="$source/banks/raw_trained/part_label_bank.npz"
  footprint="$PAPER_ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz"
  if [[ "$stage" == "strict" ]]; then
    asset_root="$source/assets/test/test-view/semantic_editable_assets"
    output_root="$OUTPUT_ROOT/strict_shared40k/CoreView_${subject}"
  else
    asset_root="$OUTPUT_ROOT/temporal_assets/CoreView_${subject}/merged"
    output_root="$OUTPUT_ROOT/temporal/evaluations/CoreView_${subject}"
  fi
  cache="$output_root/projection_cache.pkl"
  mkdir -p "$output_root"

  if [[ "$stage" == "strict" && ! -s "$OUTPUT_ROOT/protocol/CoreView_${subject}_a5_operating_point.json" ]]; then
    evaluate "$subject" "$frozen" "$raw" "$asset_root" "$output_root/a5_curve" write "" "B1 A5" \
      --footprint-bank "$footprint"
    "$PYTHON_BIN" "$CODE_ROOT/tools/prepare_four_method_temporal_assets.py" write-operating-point \
      --curve "$output_root/a5_curve/matched_retention.csv" --method a5 --subject "$subject" \
      --output "$OUTPUT_ROOT/protocol/CoreView_${subject}_a5_operating_point.json"
  fi
  if [[ "$stage" == "temporal" && ! -s "$cache" ]]; then
    evaluate "$subject" "$frozen" "$raw" "$asset_root" "$output_root/cache_seed" write \
      "$OUTPUT_ROOT/protocol/CoreView_${subject}_a5_operating_point.json" "B1 A5" \
      --footprint-bank "$footprint"
  fi
  if [[ "$stage" == "strict" && ! -s "$cache" ]]; then
    echo "strict projection cache was not created for CoreView_${subject}" >&2
    return 2
  fi

  if ! complete "$output_root/a5"; then
    evaluate "$subject" "$frozen" "$raw" "$asset_root" "$output_root/a5" read \
      "$OUTPUT_ROOT/protocol/CoreView_${subject}_a5_operating_point.json" "B1 A5" \
      --footprint-bank "$footprint"
  fi
  for method in "${METHODS[@]}"; do
    if complete "$output_root/$method"; then
      continue
    fi
    external_bank="$(external_root "$method")/CoreView_${subject}/train_30k/part_label_bank.npz"
    evaluate "$subject" "$(external_frozen "$method" "$subject")" "$external_bank" \
      "$asset_root" "$output_root/$method" read \
      "$OUTPUT_ROOT/protocol/CoreView_${subject}_${method}_operating_point.json" "B1 B4"
  done
}

# Optional trailing evaluator arguments such as --footprint-bank are appended here.
evaluate() {
  local subject="$1" frozen="$2" raw_bank="$3" asset_root="$4" output="$5" cache_mode="$6"
  local fixed_point="$7" baselines="$8"
  shift 8
  local source checkpoint evidence_bank voting_bank cache stage_root
  source="$(subject_root "$subject")"
  checkpoint="$source/base_train_40k/ckpt40000.pth"
  evidence_bank="$source/banks/voting_evidence_target_support/part_label_bank.npz"
  voting_bank="$source/banks/multiview_voting/part_label_bank.npz"
  cache="${output%/*}/projection_cache.pkl"
  stage_root="${output%/*}"
  local cache_args fixed_args record_args
  if [[ "$cache_mode" == "write" ]]; then cache_args=(--projection-cache-output "$cache"); else cache_args=(--projection-cache-input "$cache"); fi
  if [[ -n "$fixed_point" ]]; then fixed_args=(--fixed-operating-point "$fixed_point" --per-view-spatial-output "$output/per_view_spatial.csv"); else fixed_args=(); fi
  if [[ "$stage_root" == *"/temporal/evaluations/"* ]]; then
    record_args=(--record-list "$OUTPUT_ROOT/protocol/temporal_record_list.json")
  else
    record_args=()
  fi
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=0 BODY_MODELS_ROOT="$PAPER_ROOT/body_models" "$PYTHON_BIN" \
    "$CODE_ROOT/tools/evaluate_semantic_editing_paper_protocol.py" \
    --protocol "$(protocol_path "$subject")" --protocol-split test --frozen-config "$frozen" \
    --raw-trained-bank "$raw_bank" --trained-bank "$evidence_bank" --voting-bank "$voting_bank" \
    --checkpoint "$checkpoint" --asset-root "$asset_root" --config "$source/base_train_40k/.hydra/config.yaml" \
    --dataset-root "$DATASET_ROOT" --subject "CoreView_${subject}" \
    --explicit-binding-render-preset "$(preset "$subject")" --output-dir "$output" \
    --baselines $baselines --retention-reference-baseline B1 \
    "${cache_args[@]}" "${fixed_args[@]}" "${record_args[@]}" "$@"
}

stage="${1:-all}"
case "$stage" in
  strict|temporal)
    for subject in "${SUBJECTS[@]}"; do run_subject_stage "$stage" "$subject"; done
    ;;
  all)
    for subject in "${SUBJECTS[@]}"; do run_subject_stage strict "$subject"; done
    for subject in "${SUBJECTS[@]}"; do run_subject_stage temporal "$subject"; done
    ;;
  *) echo "usage: $0 [strict|temporal|all]" >&2; exit 2 ;;
esac
