#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
RUN_ID="${RUN_ID:-v358_child_mask_oracle_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v358_child_mask_oracle_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v358_child_mask_oracle_${RUN_ID}}"

V356_LOG_DIR="${V356_LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v356_validated_split_child_asset_v356_validated_split_child_probe1_20260524}"
V356_EXP_ROOT="${V356_EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v356_validated_split_child_asset_v356_validated_split_child_probe1_20260524}"
V356_SEED_ASSET_JSON="${V356_SEED_ASSET_JSON:-$V356_LOG_DIR/assets/v356_seed_split_child_asset.json}"
V356_ACTION_VALIDATION_TSV="${V356_ACTION_VALIDATION_TSV:-$V356_LOG_DIR/action_validation.tsv}"

ACTION_KEYS="${ACTION_KEYS:-c23_f000060:inner:row169,c23_f000300:inner:row201,c23_f000000:inner:row160}"
MASK_MODE="${MASK_MODE:-ellipse}"
MASK_PAD_PX="${MASK_PAD_PX:-0}"
MASK_SCALE="${MASK_SCALE:-1.0}"
MASK_DILATE_PX="${MASK_DILATE_PX:-0}"

ACTION_VALIDATE_TARGET_MIN_GAIN="${ACTION_VALIDATE_TARGET_MIN_GAIN:-0.25}"
ACTION_VALIDATE_MAX_FG_REGRESS="${ACTION_VALIDATE_MAX_FG_REGRESS:-0.000002}"
ACTION_VALIDATE_MAX_BOUNDARY_REGRESS="${ACTION_VALIDATE_MAX_BOUNDARY_REGRESS:-0.000002}"
ACTION_VALIDATE_MAX_EDGE_REGRESS="${ACTION_VALIDATE_MAX_EDGE_REGRESS:-0.001}"
ACTION_VALIDATE_MAX_COUNT_REGRESS="${ACTION_VALIDATE_MAX_COUNT_REGRESS:-0.0}"
ACTION_VALIDATE_MAX_HARD_REGRESS="${ACTION_VALIDATE_MAX_HARD_REGRESS:-0.000001}"
ACTION_VALIDATE_MAX_OPACITY_REGRESS="${ACTION_VALIDATE_MAX_OPACITY_REGRESS:-0.0}"

SUMMARY="$LOG_DIR/summary.tsv"
ACTION_VALIDATION_TSV="$LOG_DIR/action_validation.tsv"
EVENTS="$LOG_DIR/events.tsv"
ASSET_DIR="$LOG_DIR/assets"
V358_ASSET_JSON="$ASSET_DIR/v358_validated_child_mask_oracle_asset.json"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$V356_SEED_ASSET_JSON" "$V356_ACTION_VALIDATION_TSV"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'component_key\timage_name\tstatus\ttarget_gain\tfg_delta_control\tboundary_delta_control\tedge_delta_control\tinner_delta_control\touter_delta_control\thard_delta_control\topacity_inner_delta_control\topacity_outer_delta_control\tcontrol_exp\tcandidate_exp\n' > "$ACTION_VALIDATION_TSV"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

analyze_render() {
  local label="$1"
  local render_exp="$2"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${label}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 \
    --rgb-close-kernel 5 \
    --opacity-close-kernel 3 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${label}.log" 2>&1
}

append_action_validation_row() {
  local component_key="$1"
  local image_name="$2"
  local control_exp="$3"
  local candidate_exp="$4"
  "$PYTHON_BIN" - \
    "$ACTION_VALIDATION_TSV" "$component_key" "$image_name" "$control_exp" "$candidate_exp" \
    "$ACTION_VALIDATE_TARGET_MIN_GAIN" \
    "$ACTION_VALIDATE_MAX_FG_REGRESS" "$ACTION_VALIDATE_MAX_BOUNDARY_REGRESS" "$ACTION_VALIDATE_MAX_EDGE_REGRESS" \
    "$ACTION_VALIDATE_MAX_COUNT_REGRESS" "$ACTION_VALIDATE_MAX_HARD_REGRESS" "$ACTION_VALIDATE_MAX_OPACITY_REGRESS" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
component_key = sys.argv[2]
image_name = sys.argv[3]
control_exp = Path(sys.argv[4])
candidate_exp = Path(sys.argv[5])
target_min_gain = float(sys.argv[6])
max_fg = float(sys.argv[7])
max_boundary = float(sys.argv[8])
max_edge = float(sys.argv[9])
max_count = float(sys.argv[10])
max_hard = float(sys.argv[11])
max_opacity = float(sys.argv[12])
metrics = ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")

def load_metrics(render_exp):
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8"))
    opacity = json.loads((render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8"))
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    }

control = load_metrics(control_exp)
candidate = load_metrics(candidate_exp)
delta = {key: candidate[key] - control[key] for key in metrics}
target_gain = max(
    -delta["opacity_outer"],
    -delta["outer"],
    -delta["inner"],
    -delta["opacity_inner"],
    -100.0 * delta["edge"],
    -10000.0 * delta["hard"],
)
do_no_harm = (
    delta["fg"] <= max_fg
    and delta["boundary"] <= max_boundary
    and delta["edge"] <= max_edge
    and delta["inner"] <= max_count
    and delta["outer"] <= max_count
    and delta["hard"] <= max_hard
    and delta["opacity_inner"] <= max_opacity
    and delta["opacity_outer"] <= max_opacity
)
status = "keep" if do_no_harm and target_gain >= target_min_gain else "drop"
row = {
    "component_key": component_key,
    "image_name": image_name,
    "status": status,
    "target_gain": target_gain,
    **{f"{key}_delta_control": delta[key] for key in metrics},
    "control_exp": str(control_exp),
    "candidate_exp": str(candidate_exp),
}
with out_path.open("a", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "component_key", "image_name", "status", "target_gain",
        "fg_delta_control", "boundary_delta_control", "edge_delta_control",
        "inner_delta_control", "outer_delta_control", "hard_delta_control",
        "opacity_inner_delta_control", "opacity_outer_delta_control",
        "control_exp", "candidate_exp",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writerow(row)
print(f"{component_key}\t{status}\ttarget_gain={target_gain:.6f}")
PY
}

IFS=',' read -r -a KEYS <<< "$ACTION_KEYS"
for component_key in "${KEYS[@]}"; do
  component_key="$(printf '%s' "$component_key" | xargs)"
  [ -z "$component_key" ] && continue
  safe_key="$(printf '%s' "$component_key" | tr -c 'A-Za-z0-9_' '_')"
  image_name="$(printf '%s' "$component_key" | cut -d: -f1)"
  source_root="$V356_EXP_ROOT/action_validation/$safe_key"
  control_exp="$source_root/control_v345"
  child_exp="$source_root/split_child_only_plus_v345"
  candidate_exp="$EXP_ROOT/action_validation/$safe_key/child_mask_oracle_plus_v345"
  if [ ! -d "$control_exp" ] || [ ! -d "$child_exp" ]; then
    echo "missing v356 action render dirs for $component_key: $source_root" >&2
    exit 3
  fi
  log_event "oracle_compose_start" "$component_key"
  "$PYTHON_BIN" tools/make_377_stageB_v358_child_mask_oracle_render.py \
    --control-exp "$control_exp" \
    --child-exp "$child_exp" \
    --asset-json "$V356_SEED_ASSET_JSON" \
    --component-key "$component_key" \
    --out-exp "$candidate_exp" \
    --mask-mode "$MASK_MODE" \
    --mask-pad-px "$MASK_PAD_PX" \
    --mask-scale "$MASK_SCALE" \
    --mask-dilate-px "$MASK_DILATE_PX" \
    --write-mask \
    > "$LOG_DIR/compose_${safe_key}.log" 2>&1
  analyze_render "action_${safe_key}_child_mask_oracle_plus_v345" "$candidate_exp"
  append_action_validation_row "$component_key" "$image_name" "$control_exp" "$candidate_exp"
  log_event "oracle_compose_done" "$component_key"
done

"$PYTHON_BIN" - "$V356_SEED_ASSET_JSON" "$ACTION_VALIDATION_TSV" "$V358_ASSET_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

seed_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
data = json.loads(seed_path.read_text(encoding="utf-8"))
kept = set()
rows = []
with validation_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        rows.append(row)
        if row.get("status") == "keep":
            kept.add(str(row.get("component_key", "")))
children = []
for child in data.get("children", data.get("actions", [])):
    if str(child.get("component_key", "")) in kept:
        item = dict(child)
        item["split_child_enable"] = True
        item["child_mask_oracle_enable"] = True
        item["split_child_reason"] = "v358_child_mask_oracle_action_validated"
        children.append(item)
payload = {
    "version": "v358_child_mask_oracle_asset",
    "policy": "v358 diagnostic asset: children kept only if post-render child-side component mask oracle passes action-level do-no-harm.",
    "source": {"v356_seed_asset": str(seed_path), "validation_tsv": str(validation_path)},
    "child_count": len(children),
    "children": children,
    "actions": children,
    "action_validation": {"rows": rows, "kept_child_count": len(children), "seed_child_count": len(rows)},
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {out_path} kept_child_count={len(children)}")
PY

"$PYTHON_BIN" - "$SUMMARY" "$ACTION_VALIDATION_TSV" <<'PY'
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
rows = list(csv.DictReader(validation_path.open("r", encoding="utf-8"), delimiter="\t"))
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["actions", "kept", "dropped", "inner_delta_sum", "outer_delta_sum", "opacity_inner_delta_sum", "opacity_outer_delta_sum", "edge_delta_sum", "hard_delta_sum"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerow({
        "actions": len(rows),
        "kept": sum(1 for row in rows if row.get("status") == "keep"),
        "dropped": sum(1 for row in rows if row.get("status") != "keep"),
        "inner_delta_sum": sum(float(row.get("inner_delta_control", 0.0) or 0.0) for row in rows),
        "outer_delta_sum": sum(float(row.get("outer_delta_control", 0.0) or 0.0) for row in rows),
        "opacity_inner_delta_sum": sum(float(row.get("opacity_inner_delta_control", 0.0) or 0.0) for row in rows),
        "opacity_outer_delta_sum": sum(float(row.get("opacity_outer_delta_control", 0.0) or 0.0) for row in rows),
        "edge_delta_sum": sum(float(row.get("edge_delta_control", 0.0) or 0.0) for row in rows),
        "hard_delta_sum": sum(float(row.get("hard_delta_control", 0.0) or 0.0) for row in rows),
    })
PY

log_event "summary_done" "$SUMMARY"
log_event "finished_bjt" "$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "ACTION_VALIDATION_TSV=$ACTION_VALIDATION_TSV"
echo "V358_ASSET_JSON=$V358_ASSET_JSON"
