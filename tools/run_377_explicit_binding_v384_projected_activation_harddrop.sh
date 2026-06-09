#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v384_projected_activation_harddrop_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v384_projected_activation_harddrop_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v384_projected_activation_harddrop_${RUN_ID}}"
ASSET_DIR="$LOG_DIR/assets"

BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
SOURCE_CLOSURE_ASSET_JSON="${SOURCE_CLOSURE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v382_post_v374_residual_bundle_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/assets/v382_post_v374_residual_bundle_candidate_asset.json}"
PREFILTER_TSV="${PREFILTER_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_v382_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/action_validation.tsv}"
CANARY_WORST_TSV="${CANARY_WORST_TSV:-$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v383_cumulative_canary_selector_raw_gate_20260529_104022_bjt/worst_frames.tsv}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"

CANARY_TOP_K="${CANARY_TOP_K:-8}"
MAX_CANDIDATES="${MAX_CANDIDATES:-38}"
CHILD_OPACITY="${CHILD_OPACITY:-0.045}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
V384_CLOSURE_ASSET_JSON="$ASSET_DIR/v384_projected_activation_harddrop_candidate_asset.json"
LAUNCH_LOG="$LOG_DIR/v384_launcher.log"

mkdir -p "$LOG_DIR" "$EXP_ROOT" "$ASSET_DIR"
START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+4 hours 45 minutes' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_ASSET_JSON=$BASE_ASSET_JSON
SOURCE_CLOSURE_ASSET_JSON=$SOURCE_CLOSURE_ASSET_JSON
V384_CLOSURE_ASSET_JSON=$V384_CLOSURE_ASSET_JSON
PREFILTER_TSV=$PREFILTER_TSV
CANARY_WORST_TSV=$CANARY_WORST_TSV
CANARY_TOP_K=$CANARY_TOP_K
MAX_CANDIDATES=$MAX_CANDIDATES
EOF

"$PYTHON_BIN" - "$SOURCE_CLOSURE_ASSET_JSON" "$V384_CLOSURE_ASSET_JSON" <<'PY' > "$LOG_DIR/make_v384_asset.log" 2>&1
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
asset = json.loads(src.read_text(encoding="utf-8"))

def patch_item(item):
    pair_role = str(item.get("pair_role", item.get("child_role", "")) or "").strip().lower()
    direction = str(item.get("direction", "") or "").strip().lower()
    scope = str(item.get("scope", item.get("asset_scope", "")) or "").strip().lower()
    is_residual_child = (
        direction in ("inner", "under")
        and scope in ("global", "canonical", "semantic", "all")
        and pair_role in ("inner_residual_supplement", "residual_supplement", "inner_supplement")
    )
    if not is_residual_child:
        return
    item["activation_coord_mode"] = "projected_center"
    item["child_self_protect_drop_on_outer"] = True
    item["v384_projected_activation_harddrop"] = True
    if item.get("activation_screen_x", None) is not None:
        item.setdefault("source_activation_screen_x", item.get("activation_screen_x"))
    if item.get("activation_screen_y", None) is not None:
        item.setdefault("source_activation_screen_y", item.get("activation_screen_y"))
    if item.get("target_screen_x", None) is not None:
        item.setdefault("source_target_screen_x", item.get("target_screen_x"))
    if item.get("target_screen_y", None) is not None:
        item.setdefault("source_target_screen_y", item.get("target_screen_y"))

for key in ("children", "child_actions"):
    for item in asset.get(key, []) or []:
        if isinstance(item, dict):
            patch_item(item)

asset["version"] = "v384_projected_activation_harddrop_candidate_asset"
asset["policy"] = (
    "v382 residual candidate asset with global residual children using runtime projected-center "
    "activation and hard drop on current-frame outer overlap."
)
asset.setdefault("v384_runtime_contract", {})
asset["v384_runtime_contract"].update({
    "activation_coord_mode": "projected_center",
    "child_self_protect_drop_on_outer": True,
})
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(asset, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {dst}")
print(f"children={len(asset.get('children', []))} groups={asset.get('group_count')}")
PY

env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
  GPU="$GPU" \
  PYTHON_BIN="$PYTHON_BIN" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="v384_selector_${RUN_ID}" \
  BASE_ASSET_JSON="$BASE_ASSET_JSON" \
  CLOSURE_ASSET_JSON="$V384_CLOSURE_ASSET_JSON" \
  PREFILTER_TSV="$PREFILTER_TSV" \
  CANARY_WORST_TSV="$CANARY_WORST_TSV" \
  CANARY_TOP_K="$CANARY_TOP_K" \
  MAX_CANDIDATES="$MAX_CANDIDATES" \
  CANDIDATE_CKPT="$CANDIDATE_CKPT" \
  CHILD_OPACITY="$CHILD_OPACITY" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  "$PYTHON_BIN" tools/run_377_explicit_binding_v383_cumulative_canary_selector.py \
  --log-dir "$LOG_DIR/v384_selector" \
  --exp-root "$EXP_ROOT/v384_selector" \
  > "$LAUNCH_LOG" 2>&1

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "LOG_DIR=$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "V384_CLOSURE_ASSET_JSON=$V384_CLOSURE_ASSET_JSON"
echo "LAUNCH_LOG=$LAUNCH_LOG"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
