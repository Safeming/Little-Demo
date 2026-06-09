#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

RUN_ID="${RUN_ID:-v395_dense_canary_selector_batch_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v395_dense_canary_selector_batch_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v395_dense_canary_selector_batch_${RUN_ID}}"

BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
SOURCE_CLOSURE_ASSET_JSON="${SOURCE_CLOSURE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v382_post_v374_residual_bundle_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/assets/v382_post_v374_residual_bundle_candidate_asset.json}"
PREFILTER_TSV="${PREFILTER_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_v382_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/action_validation.tsv}"
CANARY_WORST_TSV="${CANARY_WORST_TSV:-$ROOT/exp/formal/logs/377_v338_raw_contour_gate_formal_377_v394_config_consistent_dense_gate_v394_config_consistent_mainline_gate_20260531_185319_bjt/worst_frames.tsv}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"

CANARY_VARIANT="${CANARY_VARIANT:-candidate_v394_v392_selected_config_consistent_dense}"
CANARY_TOP_K_A="${CANARY_TOP_K_A:-10}"
CANARY_TOP_K_B="${CANARY_TOP_K_B:-14}"
MAX_CANDIDATES_A="${MAX_CANDIDATES_A:-24}"
MAX_CANDIDATES_B="${MAX_CANDIDATES_B:-24}"
CHILD_OPACITY="${CHILD_OPACITY:-0.045}"
SEMANTIC_TRAIN_STEPS="${SEMANTIC_TRAIN_STEPS:-1500}"
MIN_INNER_GAIN="${MIN_INNER_GAIN:-0.5}"

EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
V395_RUNTIME_CLOSURE_ASSET_JSON="$LOG_DIR/assets/v395_runtime_bounded_candidate_asset.json"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_ASSET_JSON" "$SOURCE_CLOSURE_ASSET_JSON" \
  "$PREFILTER_TSV" "$CANARY_WORST_TSV" "$CANDIDATE_CKPT" \
  "$ROOT/tools/run_377_explicit_binding_v387_runtime_bounded_marginal_selector.py" \
  "$ROOT/tools/run_377_explicit_binding_v383_cumulative_canary_selector.py" \
  "$ROOT/tools/formal/run_377_v338_semantic_train.sh"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$EXP_ROOT"
mkdir -p "$LOG_DIR/assets"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+9 hours' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<INFO
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_ASSET_JSON=$BASE_ASSET_JSON
SOURCE_CLOSURE_ASSET_JSON=$SOURCE_CLOSURE_ASSET_JSON
PREFILTER_TSV=$PREFILTER_TSV
CANARY_WORST_TSV=$CANARY_WORST_TSV
CANARY_VARIANT=$CANARY_VARIANT
CANDIDATE_CKPT=$CANDIDATE_CKPT
CANARY_TOP_K_A=$CANARY_TOP_K_A
CANARY_TOP_K_B=$CANARY_TOP_K_B
MAX_CANDIDATES_A=$MAX_CANDIDATES_A
MAX_CANDIDATES_B=$MAX_CANDIDATES_B
SEMANTIC_TRAIN_STEPS=$SEMANTIC_TRAIN_STEPS
CONFIG_CONTRACT=v395_dense_canary_batch_v394_worst_frames_action_selector_then_semantic_train
INFO

"$PYTHON_BIN" - "$SOURCE_CLOSURE_ASSET_JSON" "$V395_RUNTIME_CLOSURE_ASSET_JSON" <<'PY' > "$LOG_DIR/make_runtime_asset.log" 2>&1
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
asset = json.loads(src.read_text(encoding="utf-8"))
patched_children = 0
patched_actions = 0
for key in ("children", "child_actions"):
    for item in asset.get(key, []) or []:
        if not isinstance(item, dict):
            continue
        pair_role = str(item.get("pair_role", item.get("child_role", "")) or "").strip().lower()
        direction = str(item.get("direction", "") or "").strip().lower()
        scope = str(item.get("scope", item.get("asset_scope", "")) or "").strip().lower()
        if direction in ("inner", "under") and scope in ("global", "canonical", "semantic", "all") and pair_role in ("inner_residual_supplement", "residual_supplement", "inner_supplement"):
            item.setdefault("activation_coord_mode", "source_screen")
            item["v395_runtime_bounded_dense_canary_selector"] = True
            patched_children += 1
for key in ("actions", "component_actions"):
    for item in asset.get(key, []) or []:
        if not isinstance(item, dict):
            continue
        pair_role = str(item.get("pair_role", "") or "").strip().lower()
        direction = str(item.get("direction", "") or "").strip().lower()
        scope = str(item.get("scope", item.get("asset_scope", "")) or "").strip().lower()
        if direction in ("outer", "over") and scope in ("global", "canonical", "semantic", "all") and pair_role in ("outer_protect_shrink", "outer_protect", "outer_parent_protect"):
            item.setdefault("activation_coord_mode", "source_screen")
            item["v395_runtime_bounded_dense_canary_selector"] = True
            patched_actions += 1
asset["version"] = "v395_runtime_bounded_candidate_asset"
asset["policy"] = "v382 closure candidates prepared for v395 dense-canary runtime-bounded selector."
asset["v395_runtime_contract"] = {
    "activation_coord_mode": "source_screen",
    "activation_image_allowlist": "selector_injected_per_selected_pair",
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(asset, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {dst}")
print(f"patched_children={patched_children} patched_actions={patched_actions}")
PY

run_selector() {
  local name="$1"
  local script="$2"
  local top_k="$3"
  local max_candidates="$4"
  local closure_asset="$5"
  local include_all_source_images="${6:-false}"
  local sub_log_dir="$LOG_DIR/$name"
  local sub_exp_root="$EXP_ROOT/$name"
  local launch_log="$LOG_DIR/${name}.launcher.log"

  mkdir -p "$sub_log_dir" "$sub_exp_root"
  log_event "${name}_start" "top_k=$top_k max_candidates=$max_candidates"
  env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
    GPU="$GPU" \
    PYTHON_BIN="$PYTHON_BIN" \
    CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
    RUN_ID="${RUN_ID}_${name}" \
    BASE_EXP="$BASE_EXP" \
    BASE_ASSET_JSON="$BASE_ASSET_JSON" \
    CLOSURE_ASSET_JSON="$closure_asset" \
    SOURCE_CLOSURE_ASSET_JSON="$closure_asset" \
    PREFILTER_TSV="$PREFILTER_TSV" \
    CANARY_WORST_TSV="$CANARY_WORST_TSV" \
    CANARY_VARIANT="$CANARY_VARIANT" \
    CANARY_TOP_K="$top_k" \
    MAX_CANDIDATES="$max_candidates" \
    CANDIDATE_CKPT="$CANDIDATE_CKPT" \
    CHILD_OPACITY="$CHILD_OPACITY" \
    MIN_INNER_GAIN="$MIN_INNER_GAIN" \
    INCLUDE_ALL_SOURCE_IMAGES="$include_all_source_images" \
    TRAIN_ON_STRICT_PASS=false \
    "$PYTHON_BIN" "$script" \
      --log-dir "$sub_log_dir" \
      --exp-root "$sub_exp_root" \
    > "$launch_log" 2>&1
  log_event "${name}_done" "$sub_log_dir"
}

run_selector "v395a_runtime_bounded_v394_canary" "$ROOT/tools/run_377_explicit_binding_v387_runtime_bounded_marginal_selector.py" "$CANARY_TOP_K_A" "$MAX_CANDIDATES_A" "$V395_RUNTIME_CLOSURE_ASSET_JSON" "false"
run_selector "v395b_cumulative_v394_canary" "$ROOT/tools/run_377_explicit_binding_v383_cumulative_canary_selector.py" "$CANARY_TOP_K_B" "$MAX_CANDIDATES_B" "$SOURCE_CLOSURE_ASSET_JSON" "false"

"$PYTHON_BIN" - "$LOG_DIR" "$SUMMARY" <<'PY'
import csv
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for name in ["v395a_runtime_bounded_v394_canary", "v395b_cumulative_v394_canary"]:
    path = log_dir / name / "summary.tsv"
    row = {"name": name, "summary": str(path)}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle, delimiter="\t"))
        if data:
            row.update(data[-1])
    rows.append(row)

fields = [
    "name", "raw_gate_status", "selected_candidates", "final_groups",
    "final_asset_json", "raw_gate_summary", "validation_tsv", "summary",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

strict = [row for row in rows if row.get("raw_gate_status") == "strict_pass" and int(row.get("selected_candidates") or 0) > 0]
if strict:
    # Prefer the runtime-bounded selector when both pass because it validates marginal impact.
    strict.sort(key=lambda row: (0 if row["name"].startswith("v395a") else 1, -int(row.get("selected_candidates") or 0)))
    print(strict[0].get("final_asset_json", ""))
else:
    print("")
PY

BEST_ASSET="$("$PYTHON_BIN" - "$SUMMARY" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
strict = [row for row in rows if row.get("raw_gate_status") == "strict_pass" and int(row.get("selected_candidates") or 0) > 0]
if strict:
    strict.sort(key=lambda row: (0 if row["name"].startswith("v395a") else 1, -int(row.get("selected_candidates") or 0)))
    print(strict[0].get("final_asset_json", ""))
else:
    print("")
PY
)"

TRAIN_EXP_DIR=""
if [ -n "$BEST_ASSET" ] && [ -e "$BEST_ASSET" ]; then
  TRAIN_RUN_ID="formal_377_v395_dense_canary_semantic_train_${RUN_ID}"
  TRAIN_EXP_DIR="$ROOT/exp/formal/377_v395_dense_canary_semantic_train_${TRAIN_RUN_ID}"
  TRAIN_LOG="$LOG_DIR/semantic_train.log"
  log_event semantic_train_start "$BEST_ASSET"
  env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
    GPU="$GPU" \
    PYTHON_BIN="$PYTHON_BIN" \
    CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
    BASE_CKPT="$CANDIDATE_CKPT" \
    RUN_ID="$TRAIN_RUN_ID" \
    EXP_DIR="$TRAIN_EXP_DIR" \
    TRAIN_STEPS="$SEMANTIC_TRAIN_STEPS" \
    "$ROOT/tools/formal/run_377_v338_semantic_train.sh" \
      "++pipeline.split_child_component_enable=true" \
      "++pipeline.split_child_component_asset_json=$BEST_ASSET" \
      "++pipeline.split_child_component_action_required=false" \
      "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
      "++pipeline.split_child_component_radius_scale=1.0" \
      "++pipeline.split_child_component_max_children=-1" \
    > "$TRAIN_LOG" 2>&1
  log_event semantic_train_done "$TRAIN_EXP_DIR"
else
  log_event semantic_train_skip "no strict selector asset"
fi

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "BEST_ASSET=$BEST_ASSET"
  echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"
log_event all_done "$SUMMARY"

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "SUMMARY=$SUMMARY"
echo "BEST_ASSET=$BEST_ASSET"
echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
