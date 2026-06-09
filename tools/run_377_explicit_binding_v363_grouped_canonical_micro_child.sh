#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
RUN_ID="${RUN_ID:-v363_grouped_canonical_micro_child_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v363_grouped_canonical_micro_child_${RUN_ID}}"
ASSET_JSON="${ASSET_JSON:-$LOG_DIR/assets/v363_grouped_canonical_micro_child_asset.json}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"

INNER_JSON="${INNER_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v361_residual_shaped_micro_child_v361_grid32_20260524_184256_bjt/assets/v361_validated_residual_shaped_micro_child_asset.json}"
OUTER_JSON_A="${OUTER_JSON_A:-$ROOT/exp/stageB/logs/377_explicit_binding_v355_validated_paired_rowlocal_asset_v355_validated_pairedrowlocal_fullpairs_20260524/assets/v355_validated_paired_rowlocal_asset.json}"
OUTER_JSON_B="${OUTER_JSON_B:-$ROOT/exp/stageB/logs/377_explicit_binding_v357_paired_split_child_protect_asset_v357_foreground_probe_20260524/assets/v357_validated_paired_split_child_protect_asset.json}"

if [ ! -e "$OUTER_JSON_A" ]; then
  OUTER_JSON_A="$ROOT/exp/stageB/logs/377_explicit_binding_v355_validated_paired_rowlocal_asset_v355_validated_paired_rowlocal_fullpairs_20260524/assets/v355_validated_paired_rowlocal_asset.json"
fi

mkdir -p "$LOG_DIR/assets"

"$PYTHON_BIN" tools/make_377_stageB_v363_grouped_canonical_micro_child_asset.py \
  --inner-json "$INNER_JSON" \
  --outer-json "$OUTER_JSON_A" \
  --outer-json "$OUTER_JSON_B" \
  --out-json "$ASSET_JSON" \
  --child-opacity "${CHILD_OPACITY:-0.04}" \
  --outer-radius-scale "${OUTER_RADIUS_SCALE:-0.70}" \
  --outer-score-scale "${OUTER_SCORE_SCALE:-0.65}" \
  --max-outer-per-inner "${MAX_OUTER_PER_INNER:-2}" \
  ${REQUIRE_OUTER_PAIR:+--require-outer-pair} \
  > "$LOG_DIR/make_v363_asset.log" 2>&1

CANDIDATE_CKPT="$CANDIDATE_CKPT" \
CANDIDATE_VARIANT_NAME="${CANDIDATE_VARIANT_NAME:-candidate_v363_grouped_canonical_micro_child}" \
CANDIDATE_SIGNED_DYNAMIC_COMPONENT_LOCAL_ASSET_JSON="$ASSET_JSON" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="${CHILD_OPACITY:-0.04}" \
CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
RUN_ID="formal_377_v363_grouped_canonical_micro_child_gate_${RUN_ID}" \
"$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
  2>&1 | tee "$LOG_DIR/gate.log"

GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v363_grouped_canonical_micro_child_gate_${RUN_ID}/summary.tsv"
cat > "$LOG_DIR/result.env" <<EOF
RUN_ID=$RUN_ID
LOG_DIR=$LOG_DIR
ASSET_JSON=$ASSET_JSON
CANDIDATE_CKPT=$CANDIDATE_CKPT
GATE_SUMMARY=$GATE_SUMMARY
EOF

echo "RESULT_ENV=$LOG_DIR/result.env"
