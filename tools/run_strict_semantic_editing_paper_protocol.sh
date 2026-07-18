#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-validate}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
PROTOCOL="${PROTOCOL:-$ROOT/configs/semantic/coreview377_strict_paper_protocol.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/strict_semantic_paper_protocol_20260718}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/formal/377_v395_dense_canary_semantic_train_formal_377_v395_dense_canary_semantic_train_v395_dense_canary_selector_batch_20260531_232920_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt139910.pth}"
SEMANTIC_EXP_DIR="${SEMANTIC_EXP_DIR:-$OUTPUT_ROOT/semantic_train_strict}"
SEMANTIC_CKPT="${SEMANTIC_CKPT:-}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
CALIBRATION_PENALTY_POWER="${CALIBRATION_PENALTY_POWER:-0.2}"

CALIBRATION_EXPORT="$OUTPUT_ROOT/assets/calibration"
VALIDATION_EXPORT="$OUTPUT_ROOT/assets/validation"
TEST_EXPORT="$OUTPUT_ROOT/assets/test"
CALIBRATION_ASSETS="$CALIBRATION_EXPORT/test-view/semantic_editable_assets"
VALIDATION_ASSETS="$VALIDATION_EXPORT/test-view/semantic_editable_assets"
TEST_ASSETS="$TEST_EXPORT/test-view/semantic_editable_assets"
RAW_BANK_DIR="$OUTPUT_ROOT/banks/raw_trained"
VOTING_BANK_DIR="$OUTPUT_ROOT/banks/multiview_voting"
CALIBRATED_BANK_DIR="$OUTPUT_ROOT/banks/evidence_target_support"
RAW_BANK="$RAW_BANK_DIR/part_label_bank.npz"
VOTING_BANK="$VOTING_BANK_DIR/part_label_bank.npz"
CALIBRATED_BANK="$CALIBRATED_BANK_DIR/part_label_bank.npz"
VALIDATION_EVAL_DIR="$OUTPUT_ROOT/evaluation/validation"
TEST_EVAL_DIR="$OUTPUT_ROOT/evaluation/test"
FROZEN_CONFIG="$OUTPUT_ROOT/frozen_validation_config.json"

protocol_json() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" "$2" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
values = payload[sys.argv[2]][sys.argv[3]]
print("[" + ",".join(str(int(value)) for value in values) + "]")
PY
}

protocol_range() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" <<'PY'
import json
import math
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
values = sorted(int(value) for value in payload[sys.argv[2]]["frame_ids"])
step = 1
if len(values) > 1:
    step = values[1] - values[0]
    for left, right in zip(values[:-1], values[1:]):
        step = math.gcd(step, right - left)
print(f"[{values[0]},{values[-1] + step},{step}]")
PY
}

protocol_record_count() {
  "$PYTHON_BIN" - "$PROTOCOL" "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
split = payload[sys.argv[2]]
print(len(split["camera_ids"]) * len(split["frame_ids"]))
PY
}

SEMANTIC_VIEWS_SPEC="$(protocol_json semantic_train camera_ids)"
SEMANTIC_FRAMES_CSV="$(protocol_json semantic_train frame_ids)"
CALIBRATION_VIEWS_SPEC="$(protocol_json calibration camera_ids)"
CALIBRATION_FRAMES_SPEC="$(protocol_range calibration)"
VALIDATION_VIEWS_SPEC="$(protocol_json validation camera_ids)"
VALIDATION_FRAMES_SPEC="$(protocol_range validation)"
TEST_VIEWS_SPEC="$(protocol_json test camera_ids)"
TEST_FRAMES_SPEC="$(protocol_range test)"
TEST_RECORD_COUNT="$(protocol_record_count test)"

run() {
  printf '[strict-paper] command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

resolve_semantic_ckpt() {
  if [[ -n "$SEMANTIC_CKPT" ]]; then
    printf '%s\n' "$SEMANTIC_CKPT"
    return
  fi
  find "$SEMANTIC_EXP_DIR" -maxdepth 1 -type f -name 'ckpt*.pth' -printf '%p\n' 2>/dev/null | sort -V | tail -n 1
}

artifact_fingerprint() {
  local path="$1"
  local dry_value="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$dry_value"
    return
  fi
  "$PYTHON_BIN" - "$path" <<'PY'
import sys
from utils.semantic_eval_protocol import file_fingerprint

print(file_fingerprint(sys.argv[1]))
PY
}

frozen_selected_value() {
  local key="$1"
  local dry_value="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$dry_value"
    return
  fi
  "$PYTHON_BIN" - "$FROZEN_CONFIG" "$key" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["selected"][sys.argv[2]])
PY
}

validate() {
  run "$PYTHON_BIN" -c "from utils.semantic_eval_protocol import load_protocol; load_protocol(r'$PROTOCOL'); print('protocol valid')"
  run "$PYTHON_BIN" -c "from utils.semantic_eval_protocol import load_protocol, write_protocol_manifest; write_protocol_manifest(r'$OUTPUT_ROOT/protocol_manifest.json', load_protocol(r'$PROTOCOL'))"
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$BASE_CKPT" ]]
    [[ -d "$DATA_ROOT/CoreView_377" ]]
    [[ -d "$PARSER_ROOT/CoreView_377/mask_cihp" ]]
  fi
}

semantic_train() {
  echo "TRAIN_VIEWS_SPEC=$SEMANTIC_VIEWS_SPEC"
  echo "stageB_semantic_allowed_frame_ids=$SEMANTIC_FRAMES_CSV"
  run env \
    "PYTHON_BIN=$PYTHON_BIN" "GPU=$GPU" "DATA_ROOT=$DATA_ROOT" "PARSER_ROOT=$PARSER_ROOT" \
    "BASE_EXP=$BASE_EXP" "BASE_CKPT=$BASE_CKPT" "EXP_DIR=$SEMANTIC_EXP_DIR" \
    "TRAIN_VIEWS_SPEC=$SEMANTIC_VIEWS_SPEC" "TRAIN_FRAMES_SPEC=[0,570,60]" \
    "TEST_VIEWS_SPEC=$VALIDATION_VIEWS_SPEC" "TEST_FRAMES_SPEC=$VALIDATION_FRAMES_SPEC" \
    "TRAIN_STEPS=$TRAIN_STEPS" "SAVE_ITERATIONS=[$TRAIN_STEPS]" "CHECKPOINT_ITERATIONS=[$TRAIN_STEPS]" \
    "$ROOT/tools/formal/run_377_v338_semantic_train.sh" \
    "++resume.reset_semantic_state_on_resume=true" \
    "++opt.stageB_semantic_allowed_camera_ids=$SEMANTIC_VIEWS_SPEC" \
    "++opt.stageB_semantic_allowed_frame_ids=$SEMANTIC_FRAMES_CSV" \
    "resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,camera_geometry.,texture.shadow_handoff_approved]"
}

export_assets() {
  local role="$1"
  local output="$2"
  local views="$3"
  local frames="$4"
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  if [[ "$DRY_RUN" != "1" && -z "$ckpt" ]]; then
    echo "missing semantic checkpoint" >&2
    exit 2
  fi
  echo "TEST_VIEWS_SPEC=$views role=$role"
  run env \
    "PYTHON_BIN=$PYTHON_BIN" "GPU=$GPU" "DATA_ROOT=$DATA_ROOT" \
    "BASE_EXP=$SEMANTIC_EXP_DIR" "BASE_CKPT=$ckpt" "EXP_DIR=$output" \
    "TEST_VIEWS_SPEC=$views" "TEST_FRAMES_SPEC=$frames" \
    "EXPORT_INTERPRETABILITY=true" "EXPORT_EDITABLE=true" \
    "$ROOT/tools/formal/run_377_signed_geometry_export.sh" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$ROOT/configs/semantic/hulk_cihp_compact_6.json" \
    "++semantic_editable_use_direct_parser=true" \
    "++semantic_editable_export_compact_head=true" \
    "resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,camera_geometry.,texture.shadow_handoff_approved]"
  run "$PYTHON_BIN" -c "from utils.semantic_eval_protocol import load_protocol, prune_asset_records_to_protocol_split; print(prune_asset_records_to_protocol_split(r'$output/test-view/semantic_editable_assets', load_protocol(r'$PROTOCOL'), '$role'))"
}

build_banks() {
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  run "$PYTHON_BIN" tools/semantic_viewer/build_part_label_bank.py \
    --checkpoint "$ckpt" --asset-root "$CALIBRATION_ASSETS" --output "$RAW_BANK" \
    --summary-json "$RAW_BANK_DIR/summary.json" --manifest-json "$RAW_BANK_DIR/manifest.json" \
    --dataset-root "$DATA_ROOT" --subject CoreView_377 --reliability-enable --export-soft-edit-weights
  run "$PYTHON_BIN" tools/semantic_viewer/build_part_label_bank.py \
    --checkpoint "$ckpt" --asset-root "$CALIBRATION_ASSETS" --output "$VOTING_BANK" \
    --summary-json "$VOTING_BANK_DIR/summary.json" --manifest-json "$VOTING_BANK_DIR/manifest.json" \
    --dataset-root "$DATA_ROOT" --subject CoreView_377 --label-bank-source projected-2d-voting \
    --reliability-enable --export-soft-edit-weights
}

calibrate() {
  local ckpt
  ckpt="$(resolve_semantic_ckpt)"
  run "$PYTHON_BIN" tools/calibrate_evidence_soft_edit_weights.py \
    --part-label-bank "$RAW_BANK" --checkpoint "$ckpt" --asset-root "$CALIBRATION_ASSETS" \
    --output "$CALIBRATED_BANK" --parts face hair upper lower shoes skin --mode support-aware \
    --outer-penalty-power "$CALIBRATION_PENALTY_POWER" \
    --support-penalty-power "$CALIBRATION_PENALTY_POWER" \
    --support-pair face:hair --support-pair hair:face --support-pair upper:skin \
    --support-pair upper:lower --support-pair lower:upper --support-pair lower:skin \
    --support-pair shoes:lower --support-pair shoes:skin --support-pair skin:upper --support-pair skin:lower \
    --protocol "$PROTOCOL" --protocol-split calibration \
    --summary-json "$CALIBRATED_BANK_DIR/summary.json"
}

select_validation() {
  local ckpt
  local checkpoint_fp
  local bank_fp
  ckpt="$(resolve_semantic_ckpt)"
  run "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$PROTOCOL" --protocol-split validation --validation-sweep \
    --trained-bank "$CALIBRATED_BANK" --voting-bank "$VOTING_BANK" --checkpoint "$ckpt" \
    --asset-root "$VALIDATION_ASSETS" --output-dir "$VALIDATION_EVAL_DIR"
  checkpoint_fp="$(artifact_fingerprint "$ckpt" dry-run-checkpoint)"
  bank_fp="$(artifact_fingerprint "$CALIBRATED_BANK" dry-run-bank)"
  run "$PYTHON_BIN" tools/select_semantic_editing_validation_config.py \
    --protocol "$PROTOCOL" --candidates "$VALIDATION_EVAL_DIR/validation_candidates.csv" \
    --checkpoint-fingerprint "$checkpoint_fp" --bank-fingerprint "$bank_fp" --output "$FROZEN_CONFIG"
}

evaluate_test() {
  local ckpt
  local soft_threshold
  ckpt="$(resolve_semantic_ckpt)"
  if [[ "$DRY_RUN" != "1" && ! -f "$FROZEN_CONFIG" ]]; then
    echo "missing frozen_validation_config.json: $FROZEN_CONFIG" >&2
    exit 2
  fi
  run "$PYTHON_BIN" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$PROTOCOL" --protocol-split test --frozen-config "$FROZEN_CONFIG" \
    --trained-bank "$CALIBRATED_BANK" --voting-bank "$VOTING_BANK" --checkpoint "$ckpt" \
    --asset-root "$TEST_ASSETS" --output-dir "$TEST_EVAL_DIR"
  soft_threshold="$(frozen_selected_value soft_threshold 0.20)"
  run "$PYTHON_BIN" tools/make_semantic_edit_render_preview.py \
    --part-label-bank "$CALIBRATED_BANK" --checkpoint "$ckpt" --asset-root "$TEST_ASSETS" \
    --output-dir "$TEST_EVAL_DIR/fair_preview" --parts face hair upper lower shoes skin \
    --soft-weight-source auto-target --soft-threshold "$soft_threshold" \
    --max-views "$TEST_RECORD_COUNT" --formal-paper-mode
}

case "$STAGE" in
  validate) validate ;;
  semantic-train) semantic_train ;;
  export-calibration) export_assets calibration "$CALIBRATION_EXPORT" "$CALIBRATION_VIEWS_SPEC" "$CALIBRATION_FRAMES_SPEC" ;;
  export-validation) export_assets validation "$VALIDATION_EXPORT" "$VALIDATION_VIEWS_SPEC" "$VALIDATION_FRAMES_SPEC" ;;
  export-test) export_assets test "$TEST_EXPORT" "$TEST_VIEWS_SPEC" "$TEST_FRAMES_SPEC" ;;
  build-banks) build_banks ;;
  calibrate) calibrate ;;
  select-validation) select_validation ;;
  evaluate-test) evaluate_test ;;
  all)
    validate
    semantic_train
    export_assets calibration "$CALIBRATION_EXPORT" "$CALIBRATION_VIEWS_SPEC" "$CALIBRATION_FRAMES_SPEC"
    export_assets validation "$VALIDATION_EXPORT" "$VALIDATION_VIEWS_SPEC" "$VALIDATION_FRAMES_SPEC"
    export_assets test "$TEST_EXPORT" "$TEST_VIEWS_SPEC" "$TEST_FRAMES_SPEC"
    build_banks
    calibrate
    select_validation
    evaluate_test
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
