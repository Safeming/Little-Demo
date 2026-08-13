#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/root/.config/superpowers/worktrees/3dgs-avatar-release-main/sggs-released-code-canonical}"
PAPER_ROOT="${PAPER_ROOT:-/remote-home/ming/3dgs-avatar-release-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER_ROOT/exp/acceptdata/four_method_paper_evidence_20260813}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
SUBJECTS=(377 386 394)
METHODS=(saga gaussian_grouping sggs a5)
VIEWS=(
  c21_f000180 c21_f000420 c21_f000540
  c22_f000180 c22_f000420 c22_f000540
  c23_f000180 c23_f000420 c23_f000540
)

subject_root() {
  case "$1" in
    377) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721" ;;
    386) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview386_multisubject_strict_20260719" ;;
    394) printf '%s\n' "$PAPER_ROOT/exp/acceptdata/coreview394_multisubject_strict_20260722" ;;
  esac
}

json_value() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

for subject in "${SUBJECTS[@]}"; do
  source="$(subject_root "$subject")"
  protocol="$OUTPUT_ROOT/protocol"
  output="$OUTPUT_ROOT/qualitative/renders/CoreView_${subject}"
  strengths="$output/method_part_strengths.json"
  saga_point="$protocol/CoreView_${subject}_saga_operating_point.json"
  gg_point="$protocol/CoreView_${subject}_gaussian_grouping_operating_point.json"
  sggs_point="$protocol/CoreView_${subject}_sggs_operating_point.json"
  a5_point="$protocol/CoreView_${subject}_a5_operating_point.json"
  mkdir -p "$output"
  "$PYTHON_BIN" "$CODE_ROOT/tools/prepare_four_method_qualitative.py" strengths \
    --operating-point "saga=$saga_point" \
    --operating-point "gaussian_grouping=$gg_point" \
    --operating-point "sggs=$sggs_point" \
    --operating-point "a5=$a5_point" --output "$strengths" > "$output/strengths.json"
  if [[ ! -s "$output/summary.json" ]]; then
    env CUDA_VISIBLE_DEVICES=0 BODY_MODELS_ROOT="$PAPER_ROOT/body_models" "$PYTHON_BIN" \
      "$CODE_ROOT/tools/render_semantic_real_editing_paper_suite.py" \
      --subject "$subject" \
      --raw-bank "$source/banks/raw_trained/part_label_bank.npz" \
      --voting-bank "$source/banks/multiview_voting/part_label_bank.npz" \
      --a5-bank "$PAPER_ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz" \
      --a5-threshold "$(json_value "$a5_point" threshold)" \
      --saga-bank "$PAPER_ROOT/exp/external/saga_canonical_five_subject_20260812_120625_bjt/CoreView_${subject}/train_30k/part_label_bank.npz" \
      --saga-threshold "$(json_value "$saga_point" threshold)" \
      --external-bank gaussian_grouping="$PAPER_ROOT/exp/external/gaussian_grouping_canonical_three_subject_20260813_0958_bjt/CoreView_${subject}/train_30k/part_label_bank.npz" \
      --external-threshold gaussian_grouping="$(json_value "$gg_point" threshold)" \
      --external-bank sggs="$PAPER_ROOT/exp/external/sggs_released_code_canonical_three_subject_20260813_formal/CoreView_${subject}/train_30k/part_label_bank.npz" \
      --external-threshold sggs="$(json_value "$sggs_point" threshold)" \
      --loso-config "$PAPER_ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" \
      --method-freeze "$PAPER_ROOT/configs/semantic/frozen_a5_main_method_v1.json" \
      --checkpoint "$source/base_train_40k/ckpt40000.pth" \
      --asset-root "$source/assets/test/test-view/semantic_editable_assets" \
      --config "$source/base_train_40k/.hydra/config.yaml" \
      --dataset-root "$PAPER_ROOT/data/ZJUMoCap" \
      --output-dir "$output" --views "${VIEWS[@]}" --methods "${METHODS[@]}" \
      --tasks recolor --parts hair shoes --method-part-strengths "$strengths" \
      --explicit-binding-render-preset none > "$output/render.log" 2>&1
  fi
done
