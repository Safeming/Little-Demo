#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-validate}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUBJECT="${SUBJECT:?set SUBJECT}"
PROTOCOL="${PROTOCOL:?set PROTOCOL}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
BASE_EXP="${BASE_EXP:?set BASE_EXP}"
BASE_CKPT="${BASE_CKPT:?set BASE_CKPT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
FROZEN_TEMPLATE="${FROZEN_TEMPLATE:-$ROOT/exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_a/frozen_validation_config.json}"
COMPACT_MAPPING_FILE="${COMPACT_MAPPING_FILE:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
SEMANTIC_EXP_DIR="${SEMANTIC_EXP_DIR:-$OUTPUT_ROOT/semantic_train_strict}"
SEMANTIC_CKPT="${SEMANTIC_CKPT:-}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

CALIBRATION_EXPORT="$OUTPUT_ROOT/assets/calibration"
VALIDATION_EXPORT="$OUTPUT_ROOT/assets/validation"
TEST_EXPORT="$OUTPUT_ROOT/assets/test"
CALIBRATION_ASSETS="$CALIBRATION_EXPORT/test-view/semantic_editable_assets"
VALIDATION_ASSETS="$VALIDATION_EXPORT/test-view/semantic_editable_assets"
TEST_ASSETS="$TEST_EXPORT/test-view/semantic_editable_assets"
RAW_BANK_DIR="$OUTPUT_ROOT/banks/raw_trained"
VOTING_BANK_DIR="$OUTPUT_ROOT/banks/multiview_voting"
CALIBRATED_BANK_DIR="$OUTPUT_ROOT/banks/voting_evidence_target_support"
RAW_BANK="$RAW_BANK_DIR/part_label_bank.npz"
VOTING_BANK="$VOTING_BANK_DIR/part_label_bank.npz"
CALIBRATED_BANK="$CALIBRATED_BANK_DIR/part_label_bank.npz"
FROZEN_CONFIG="$OUTPUT_ROOT/frozen_validation_config.json"
VALIDATION_EVAL_DIR="$OUTPUT_ROOT/evaluation/validation"
TEST_EVAL_DIR="$OUTPUT_ROOT/evaluation/test"

run() {
  printf '[multisubject-strict] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

protocol_json() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
values = payload[sys.argv[2]][sys.argv[3]]
print("[" + ",".join(str(int(value)) for value in values) + "]")
PY
}

protocol_range() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" <<'PY'
import json, math, sys
values = sorted(int(value) for value in json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]["frame_ids"])
step = 1
if len(values) > 1:
    step = values[1] - values[0]
    for left, right in zip(values[:-1], values[1:]):
        step = math.gcd(step, right - left)
print(f"[{values[0]},{values[-1] + step},{step}]")
PY
}

protocol_count() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" <<'PY'
import json, sys
split = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
print(len(split["camera_ids"]) * len(split["frame_ids"]))
PY
}

SEMANTIC_VIEWS_SPEC="$(protocol_json semantic_train camera_ids)"
SEMANTIC_FRAMES_SPEC="$(protocol_json semantic_train frame_ids)"
CALIBRATION_VIEWS_SPEC="$(protocol_json calibration camera_ids)"
CALIBRATION_FRAMES_SPEC="$(protocol_range calibration)"
VALIDATION_VIEWS_SPEC="$(protocol_json validation camera_ids)"
VALIDATION_FRAMES_SPEC="$(protocol_range validation)"
TEST_VIEWS_SPEC="$(protocol_json test camera_ids)"
TEST_FRAMES_SPEC="$(protocol_range test)"
TEST_RECORD_COUNT="$(protocol_count test)"

resolve_semantic_ckpt() {
  if [[ -n "$SEMANTIC_CKPT" ]]; then
    printf '%s\n' "$SEMANTIC_CKPT"
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$SEMANTIC_EXP_DIR/dry-run-semantic-ckpt.pth"
    return
  fi
  find "$SEMANTIC_EXP_DIR" -maxdepth 1 -type f -name 'ckpt*.pth' -printf '%p\n' 2>/dev/null | sort -V | tail -n 1
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

frozen_value() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$2"
    return
  fi
  "$PYTHON_BIN" - "$FROZEN_CONFIG" "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"][sys.argv[2]])
PY
}

validate() {
  run "$PYTHON_BIN" -c "from utils.semantic_eval_protocol import load_protocol; p=load_protocol(r'$PROTOCOL'); assert p['subject']==r'$SUBJECT'; print('protocol valid')"
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$DATA_ROOT/$SUBJECT" ]]
    [[ -d "$PARSER_ROOT/$SUBJECT/mask_cihp" ]]
    [[ -f "$BASE_EXP/.hydra/config.yaml" ]]
    [[ -f "$BASE_CKPT" ]]
    [[ -f "$FROZEN_TEMPLATE" ]]
  fi
}

semantic_train() {
  run env PYTHON_BIN="$PYTHON_BIN" GPU="$GPU" SUBJECT="$SUBJECT" DATA_ROOT="$DATA_ROOT" \
    PARSER_ROOT="$PARSER_ROOT" COMPACT_MAPPING_FILE="$COMPACT_MAPPING_FILE" \
    BASE_EXP="$BASE_EXP" BASE_CKPT="$BASE_CKPT" EXP_DIR="$SEMANTIC_EXP_DIR" \
    TRAIN_VIEWS_SPEC="$SEMANTIC_VIEWS_SPEC" ALLOWED_CAMERA_IDS="$SEMANTIC_VIEWS_SPEC" \
    ALLOWED_FRAME_IDS="$SEMANTIC_FRAMES_SPEC" TEST_VIEWS_SPEC="$VALIDATION_VIEWS_SPEC" \
    TEST_FRAMES_SPEC="$VALIDATION_FRAMES_SPEC" TRAIN_STEPS="$TRAIN_STEPS" \
    SAVE_ITERATIONS="[$TRAIN_STEPS]" CHECKPOINT_ITERATIONS="[$TRAIN_STEPS]" \
    bash "$ROOT/tools/formal/run_subject_semantic_train.sh"
}

export_assets() {
  local split="$1" output="$2" views="$3" frames="$4" ckpt
  ckpt="$(resolve_semantic_ckpt)"
  run env PYTHON_BIN="$PYTHON_BIN" GPU="$GPU" SUBJECT="$SUBJECT" DATA_ROOT="$DATA_ROOT" \
    PARSER_ROOT="$PARSER_ROOT" COMPACT_MAPPING_FILE="$COMPACT_MAPPING_FILE" \
    BASE_EXP="$SEMANTIC_EXP_DIR" BASE_CKPT="$ckpt" EXP_DIR="$output" \
    TEST_VIEWS_SPEC="$views" TEST_FRAMES_SPEC="$frames" \
    bash "$ROOT/tools/formal/run_subject_semantic_export.sh"
  run "$PYTHON_BIN" -c "from utils.semantic_eval_protocol import load_protocol, prune_asset_records_to_protocol_split; print(prune_asset_records_to_protocol_split(r'$output/test-view/semantic_editable_assets', load_protocol(r'$PROTOCOL'), '$split'))"
}

build_banks() {
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  run "$PYTHON_BIN" tools/semantic_viewer/build_part_label_bank.py --checkpoint "$ckpt" \
    --asset-root "$CALIBRATION_ASSETS" --output "$RAW_BANK" --summary-json "$RAW_BANK_DIR/summary.json" \
    --manifest-json "$RAW_BANK_DIR/manifest.json" --dataset-root "$DATA_ROOT" --subject "$SUBJECT" \
    --explicit-binding-render-preset none \
    --reliability-enable --export-soft-edit-weights
  run "$PYTHON_BIN" tools/semantic_viewer/build_part_label_bank.py --checkpoint "$ckpt" \
    --asset-root "$CALIBRATION_ASSETS" --output "$VOTING_BANK" --summary-json "$VOTING_BANK_DIR/summary.json" \
    --manifest-json "$VOTING_BANK_DIR/manifest.json" --dataset-root "$DATA_ROOT" --subject "$SUBJECT" \
    --explicit-binding-render-preset none \
    --label-bank-source projected-2d-voting --reliability-enable --export-soft-edit-weights
}

calibrate_voting() {
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  run "$PYTHON_BIN" tools/calibrate_evidence_soft_edit_weights.py \
    --part-label-bank "$VOTING_BANK" --checkpoint "$ckpt" --asset-root "$CALIBRATION_ASSETS" \
    --output "$CALIBRATED_BANK" --summary-json "$CALIBRATED_BANK_DIR/summary.json" \
    --explicit-binding-render-preset none \
    --mode support-aware --parts face hair upper lower shoes skin \
    --outer-penalty-power 0.2 --support-penalty-power 0.2 \
    --support-pair face:hair --support-pair hair:face --support-pair upper:skin \
    --support-pair upper:lower --support-pair lower:upper --support-pair lower:skin \
    --support-pair shoes:lower --support-pair shoes:skin --support-pair skin:upper \
    --support-pair skin:lower --protocol "$PROTOCOL" --protocol-split calibration
}

materialize_frozen() {
  local ckpt checkpoint_fp bank_fp
  ckpt="$(resolve_semantic_ckpt)"
  checkpoint_fp="$(fingerprint "$ckpt")"
  bank_fp="$(fingerprint "$CALIBRATED_BANK")"
  run "$PYTHON_BIN" tools/materialize_fixed_semantic_evaluation_config.py \
    --protocol "$PROTOCOL" --template "$FROZEN_TEMPLATE" \
    --checkpoint-fingerprint "$checkpoint_fp" --bank-fingerprint "$bank_fp" \
    --output "$FROZEN_CONFIG"
}

assess_allow_failure() {
  set +e
  "$@"
  local status=$?
  set -e
  echo "assessment_exit=$status"
}

evaluate_validation() {
  local ckpt soft_threshold support_threshold boundary_radius
  ckpt="$(resolve_semantic_ckpt)"
  materialize_frozen
  soft_threshold="$(frozen_value soft_threshold 0.5)"
  support_threshold="$(frozen_value support_threshold 0.3)"
  boundary_radius="$(frozen_value boundary_radius 6)"
  run "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$PROTOCOL" --protocol-split validation --trained-bank "$CALIBRATED_BANK" \
    --voting-bank "$VOTING_BANK" --checkpoint "$ckpt" --asset-root "$VALIDATION_ASSETS" \
    --explicit-binding-render-preset none \
    --soft-threshold "$soft_threshold" --support-threshold "$support_threshold" \
    --boundary-radius "$boundary_radius" --output-dir "$VALIDATION_EVAL_DIR"
  if [[ "$DRY_RUN" != "1" ]]; then
    assess_allow_failure "$PYTHON_BIN" tools/assess_voting_posterior_candidate.py \
      --baseline-summary "$VALIDATION_EVAL_DIR/baseline_summary.csv" \
      --curve "$VALIDATION_EVAL_DIR/leakage_retention_curve.csv" \
      --required-retention 0.5 0.6 --max-miou-gap 0.02 \
      --output "$OUTPUT_ROOT/audit/validation_assessment.json"
  fi
}

evaluate_test() {
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  if [[ "$DRY_RUN" != "1" && ! -f "$FROZEN_CONFIG" ]]; then
    echo "missing frozen config: $FROZEN_CONFIG" >&2
    exit 2
  fi
  run "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$PROTOCOL" --protocol-split test --frozen-config "$FROZEN_CONFIG" \
    --trained-bank "$CALIBRATED_BANK" --voting-bank "$VOTING_BANK" --checkpoint "$ckpt" \
    --asset-root "$TEST_ASSETS" --explicit-binding-render-preset none --output-dir "$TEST_EVAL_DIR"
  if [[ "$DRY_RUN" != "1" ]]; then
    assess_allow_failure "$PYTHON_BIN" tools/assess_voting_posterior_candidate.py \
      --baseline-summary "$TEST_EVAL_DIR/baseline_summary.csv" \
      --curve "$TEST_EVAL_DIR/leakage_retention_curve.csv" \
      --required-retention 0.5 0.6 --max-miou-gap 0.02 \
      --output "$OUTPUT_ROOT/audit/test_assessment.json"
  fi
}

case "$STAGE" in
  validate) validate ;;
  semantic-train) semantic_train ;;
  export-calibration) export_assets calibration "$CALIBRATION_EXPORT" "$CALIBRATION_VIEWS_SPEC" "$CALIBRATION_FRAMES_SPEC" ;;
  export-validation) export_assets validation "$VALIDATION_EXPORT" "$VALIDATION_VIEWS_SPEC" "$VALIDATION_FRAMES_SPEC" ;;
  export-test) export_assets test "$TEST_EXPORT" "$TEST_VIEWS_SPEC" "$TEST_FRAMES_SPEC" ;;
  build-banks) build_banks ;;
  calibrate-voting) calibrate_voting ;;
  evaluate-validation) evaluate_validation ;;
  evaluate-test) evaluate_test ;;
  all)
    validate
    semantic_train
    export_assets calibration "$CALIBRATION_EXPORT" "$CALIBRATION_VIEWS_SPEC" "$CALIBRATION_FRAMES_SPEC"
    export_assets validation "$VALIDATION_EXPORT" "$VALIDATION_VIEWS_SPEC" "$VALIDATION_FRAMES_SPEC"
    build_banks
    calibrate_voting
    evaluate_validation
    export_assets test "$TEST_EXPORT" "$TEST_VIEWS_SPEC" "$TEST_FRAMES_SPEC"
    evaluate_test
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "TEST_RECORD_COUNT=$TEST_RECORD_COUNT"
