#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

V371_ASSET_JSON="${V371_ASSET_JSON:-}"
BASE_CKPT="${BASE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-formal_377_v371_strength_sweep_grouped_actuator_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v371_strength_sweep_grouped_actuator_semantic_train_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

if [ -z "$V371_ASSET_JSON" ] || [ ! -e "$V371_ASSET_JSON" ]; then
  echo "missing required V371_ASSET_JSON: $V371_ASSET_JSON" >&2
  exit 2
fi

export BASE_CKPT RUN_ID EXP_DIR HYDRA_RUN_DIR

exec "$ROOT/tools/formal/run_377_v338_semantic_train.sh" "$@" \
  "++pipeline.split_child_component_enable=true" \
  "++pipeline.split_child_component_asset_json=$V371_ASSET_JSON" \
  "++pipeline.split_child_component_action_required=false" \
  "++pipeline.split_child_component_opacity=${CHILD_OPACITY:-0.04}" \
  "++pipeline.split_child_component_radius_scale=1.0" \
  "++pipeline.split_child_component_max_children=-1"
