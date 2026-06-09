#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
RUN_ID="${RUN_ID:-v362_generalized_split_child_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v362_generalized_split_child_${RUN_ID}}"
V361_ASSET_JSON="${V361_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v361_residual_shaped_micro_child_v361_grid32_20260524_184256_bjt/assets/v361_validated_residual_shaped_micro_child_asset.json}"
V362_ASSET_JSON="${V362_ASSET_JSON:-$LOG_DIR/assets/v362_generalized_split_child_asset.json}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"

mkdir -p "$LOG_DIR/assets"

"$PYTHON_BIN" tools/make_377_stageB_v362_generalized_split_child_asset.py \
  --input-json "$V361_ASSET_JSON" \
  --out-json "$V362_ASSET_JSON" \
  --anchor-knn "${ANCHOR_KNN:-24}" \
  --anchor-radius-scale "${ANCHOR_RADIUS_SCALE:-1.6}" \
  --radius-scale "${RADIUS_SCALE:-1.0}" \
  > "$LOG_DIR/make_v362_asset.log" 2>&1

CANDIDATE_CKPT="$CANDIDATE_CKPT" \
CANDIDATE_VARIANT_NAME="${CANDIDATE_VARIANT_NAME:-candidate_v362_generalized_split_child}" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$V362_ASSET_JSON" \
CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="${CHILD_OPACITY:-0.14}" \
CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
RUN_ID="formal_377_v362_generalized_split_child_gate_${RUN_ID}" \
"$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
  2>&1 | tee "$LOG_DIR/gate.log"

GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v362_generalized_split_child_gate_${RUN_ID}/summary.tsv"
cat > "$LOG_DIR/result.env" <<EOF
RUN_ID=$RUN_ID
LOG_DIR=$LOG_DIR
V362_ASSET_JSON=$V362_ASSET_JSON
CANDIDATE_CKPT=$CANDIDATE_CKPT
GATE_SUMMARY=$GATE_SUMMARY
EOF

echo "RESULT_ENV=$LOG_DIR/result.env"
