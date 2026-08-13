#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/root/.config/superpowers/worktrees/3dgs-avatar-release-main/sggs-released-code-canonical}"
PAPER_ROOT="${PAPER_ROOT:-/remote-home/ming/3dgs-avatar-release-main}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PAPER_ROOT/exp/acceptdata/four_method_paper_evidence_20260813}"
PARSER_ROOT="${PARSER_ROOT:-/remote-home/ming/dataSet}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SUBJECTS=(377 386 394)
TARGET_RETENTION=0.60
GG_377_RETENTION=0.40

run() {
  printf '[four-method-evidence]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

audit_inputs() {
  local subject
  for subject in "${SUBJECTS[@]}"; do
    run "$PYTHON_BIN" "$CODE_ROOT/tools/prepare_four_method_temporal_assets.py" \
      audit-source --parser-root "$PARSER_ROOT" --subject "$subject"
  done
}

prepare_assets() {
  printf '[four-method-evidence] prepare-assets commands are generated after input audit\n'
}

canary() {
  printf '[four-method-evidence] canary gate target_retention=%s gg377=%s\n' \
    "$TARGET_RETENTION" "$GG_377_RETENTION"
}

significance() {
  run "$PYTHON_BIN" "$CODE_ROOT/tools/summarize_four_method_paper_evidence.py" \
    significance --input "$OUTPUT_ROOT/significance/per_view_long.csv" \
    --output "$OUTPUT_ROOT/significance"
}

qualitative() {
  printf '[four-method-evidence] qualitative stage\n'
}

temporal() {
  run "$PYTHON_BIN" "$CODE_ROOT/tools/summarize_four_method_paper_evidence.py" \
    temporal --input "$OUTPUT_ROOT/temporal/per_frame_long.csv" \
    --output "$OUTPUT_ROOT/temporal"
}

summarize() {
  significance
  temporal
}

verify() {
  printf '[four-method-evidence] verify stage\n'
}

stage="${1:-all}"
case "$stage" in
  audit-inputs) audit_inputs ;;
  prepare-assets) prepare_assets ;;
  canary) canary ;;
  significance) significance ;;
  qualitative) qualitative ;;
  temporal) temporal ;;
  summarize) summarize ;;
  verify) verify ;;
  all)
    audit_inputs
    prepare_assets
    canary
    significance
    qualitative
    temporal
    verify
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
