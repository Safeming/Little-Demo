#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

V362_ASSET_JSON="${V362_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v362_generalized_split_child_v362_gate_main/assets/v362_generalized_split_child_asset.json}"
BASE_CKPT="${BASE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-formal_377_v362_generalized_split_child_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v362_generalized_split_child_semantic_train_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

if [ ! -e "$V362_ASSET_JSON" ]; then
  echo "missing required path: $V362_ASSET_JSON" >&2
  exit 2
fi

export BASE_CKPT RUN_ID EXP_DIR HYDRA_RUN_DIR

exec "$ROOT/tools/formal/run_377_v338_semantic_train.sh" "$@" \
  "++pipeline.split_child_component_enable=true" \
  "++pipeline.split_child_component_asset_json=$V362_ASSET_JSON" \
  "++pipeline.split_child_component_action_required=false" \
  "++pipeline.split_child_component_opacity=${CHILD_OPACITY:-0.14}" \
  "++pipeline.split_child_component_radius_scale=1.0" \
  "++pipeline.split_child_component_max_children=-1"
