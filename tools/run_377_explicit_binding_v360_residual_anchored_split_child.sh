#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
RUN_ID="${RUN_ID:-v360_residual_anchored_split_child_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v360_residual_anchored_split_child_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v360_residual_anchored_split_child_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

V356_LOG_DIR="${V356_LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v356_validated_split_child_asset_v356_validated_split_child_probe1_20260524}"
V359_EXP_ROOT="${V359_EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v359_child_footprint_oracle_v359_child_footprint_oracle_fullgain_20260524_143255_bjt}"
V356_SEED_ASSET_JSON="${V356_SEED_ASSET_JSON:-$V356_LOG_DIR/assets/v356_seed_split_child_asset.json}"
V345_SCREEN_DROP_LIST="${V345_SCREEN_DROP_LIST:-$ROOT/exp/stageB/logs/377_explicit_binding_v345_temporal_screen_guard_20260523/assets/v345_combined_screen_guard_drop_images.txt}"

ACTION_VALIDATE_MAX="${ACTION_VALIDATE_MAX:-36}"
OFFSET_SCALES="${OFFSET_SCALES:-0.65,0.85,1.0,1.15}"
RADIUS_SCALES="${RADIUS_SCALES:-0.70,0.90,1.10}"
DEPTH_SCALES="${DEPTH_SCALES:-0.98,1.0,1.02}"
MAX_CANDIDATES_PER_ACTION="${MAX_CANDIDATES_PER_ACTION:-18}"
CHILD_OPACITY="${CHILD_OPACITY:-0.18}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"

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
V360_SEED_ASSET_JSON="$ASSET_DIR/v360_seed_residual_anchored_split_child_asset.json"
V360_ASSET_JSON="$ASSET_DIR/v360_validated_residual_anchored_split_child_asset.json"
V360_CANDIDATES_TSV="$ASSET_DIR/v360_residual_anchored_candidates.tsv"
V360_ACTION_LIST_TSV="$ASSET_DIR/v360_seed_child_actions.tsv"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$CANDIDATE_CKPT" "$DATA_ROOT" "$V356_SEED_ASSET_JSON" "$V359_EXP_ROOT" "$V345_SCREEN_DROP_LIST"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'component_key\tsource_component_key\timage_name\tstatus\ttarget_gain\tfg_delta_control\tboundary_delta_control\tedge_delta_control\tinner_delta_control\touter_delta_control\thard_delta_control\topacity_inner_delta_control\topacity_outer_delta_control\tcontrol_exp\tcandidate_exp\n' > "$ACTION_VALIDATION_TSV"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

POINT_SCREEN_DROP_IMAGES="$(tr '\n' ',' < "$V345_SCREEN_DROP_LIST" | sed 's/,$//')"

log_event "make_seed_asset_start" "$V360_SEED_ASSET_JSON"
"$PYTHON_BIN" tools/make_377_stageB_v360_residual_anchored_split_child_asset.py \
  --seed-asset-json "$V356_SEED_ASSET_JSON" \
  --v359-log-dir "$V359_EXP_ROOT" \
  --checkpoint "$CANDIDATE_CKPT" \
  --dataset-root "$DATA_ROOT" \
  --subject CoreView_377 \
  --offset-scales "$OFFSET_SCALES" \
  --radius-scales "$RADIUS_SCALES" \
  --depth-scales "$DEPTH_SCALES" \
  --max-candidates-per-action "$MAX_CANDIDATES_PER_ACTION" \
  --child-opacity "$CHILD_OPACITY" \
  --out-json "$V360_SEED_ASSET_JSON" \
  --out-candidates-tsv "$V360_CANDIDATES_TSV" \
  > "$LOG_DIR/make_v360_seed_asset.log" 2>&1
log_event "make_seed_asset_done" "$V360_SEED_ASSET_JSON"

"$PYTHON_BIN" - "$V360_SEED_ASSET_JSON" "$V360_ACTION_LIST_TSV" "$ACTION_VALIDATE_MAX" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

asset_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
limit = int(float(sys.argv[3]))
data = json.loads(asset_path.read_text(encoding="utf-8"))
rows = []
for index, child in enumerate(data.get("children", data.get("actions", []))):
    image = str(child.get("image_name", "") or "")
    key = str(child.get("component_key", "") or "")
    source_key = str(child.get("source_component_key", child.get("v360_source_component_key", "")) or "")
    match = re.match(r"c(\d+)_f(\d+)$", image)
    if not image or not key or not source_key or not match:
        continue
    rows.append({
        "action_index": index,
        "component_key": key,
        "source_component_key": source_key,
        "image_name": image,
        "view": int(match.group(1)),
        "frame": int(match.group(2)),
    })
if limit > 0:
    rows = rows[:limit]
with out_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["action_index", "component_key", "source_component_key", "image_name", "view", "frame"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out_path} actions={len(rows)}")
PY

render_and_analyze() {
  local label="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local variant="$4"
  local render_exp="$5"
  shift 5
  log_event "render_${label}_${variant}_start" "$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$CANDIDATE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$views_spec" \
    "dataset.test_frames.view=$frames_spec" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/${label}_${variant}" \
    "wandb_disable=true" \
    "$@" \
    > "$LOG_DIR/render_${label}_${variant}.log" 2>&1
  log_event "render_${label}_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}_${variant}.log" 2>&1
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
    > "$LOG_DIR/boundary_residuals_${label}_${variant}.log" 2>&1
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
    > "$LOG_DIR/opacity_footprint_${label}_${variant}.log" 2>&1
  log_event "analyze_${label}_${variant}_done" "status=0"
}

append_action_validation_row() {
  local component_key="$1"
  local source_component_key="$2"
  local image_name="$3"
  local control_exp="$4"
  local candidate_exp="$5"
  "$PYTHON_BIN" - \
    "$ACTION_VALIDATION_TSV" "$component_key" "$source_component_key" "$image_name" "$control_exp" "$candidate_exp" \
    "$ACTION_VALIDATE_TARGET_MIN_GAIN" \
    "$ACTION_VALIDATE_MAX_FG_REGRESS" "$ACTION_VALIDATE_MAX_BOUNDARY_REGRESS" "$ACTION_VALIDATE_MAX_EDGE_REGRESS" \
    "$ACTION_VALIDATE_MAX_COUNT_REGRESS" "$ACTION_VALIDATE_MAX_HARD_REGRESS" "$ACTION_VALIDATE_MAX_OPACITY_REGRESS" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
component_key = sys.argv[2]
source_component_key = sys.argv[3]
image_name = sys.argv[4]
control_exp = Path(sys.argv[5])
candidate_exp = Path(sys.argv[6])
target_min_gain = float(sys.argv[7])
max_fg = float(sys.argv[8])
max_boundary = float(sys.argv[9])
max_edge = float(sys.argv[10])
max_count = float(sys.argv[11])
max_hard = float(sys.argv[12])
max_opacity = float(sys.argv[13])
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
    "source_component_key": source_component_key,
    "image_name": image_name,
    "status": status,
    "target_gain": target_gain,
    **{f"{key}_delta_control": delta[key] for key in metrics},
    "control_exp": str(control_exp),
    "candidate_exp": str(candidate_exp),
}
with out_path.open("a", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "component_key", "source_component_key", "image_name", "status", "target_gain",
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

while IFS=$'\t' read -r action_index component_key source_component_key image_name view frame; do
  if [ "$action_index" = "action_index" ]; then
    continue
  fi
  safe_key="$(printf '%s' "$component_key" | tr -c 'A-Za-z0-9_' '_')"
  source_safe="$(printf '%s' "$source_component_key" | tr -c 'A-Za-z0-9_' '_')"
  views_spec="[$view]"
  next_frame=$((frame + 1))
  frames_spec="[$frame,$next_frame,1]"
  action_root="$EXP_ROOT/action_validation/$safe_key"
  source_root="$EXP_ROOT/source_controls/$source_safe"
  mkdir -p "$source_root"
  control_exp="$source_root/control_v345"
  if [ ! -d "$control_exp/test-view/renders" ]; then
    render_and_analyze "source_${source_safe}" "$views_spec" "$frames_spec" control_v345 "$control_exp" \
      "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
      "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'"
  fi
  log_event "action_validate_start" "$component_key"
  render_and_analyze "action_${safe_key}" "$views_spec" "$frames_spec" v360_candidate_plus_v345 "$action_root/v360_candidate_plus_v345" \
    "pipeline.compute_cov3D_python=true" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
    "++pipeline.split_child_component_enable=true" \
    "++pipeline.split_child_component_asset_json=$V360_SEED_ASSET_JSON" \
    "++pipeline.split_child_component_action_filter='component_key=$component_key'" \
    "++pipeline.split_child_component_action_required=true" \
    "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
    "++pipeline.split_child_component_radius_scale=1.0" \
    "++pipeline.split_child_component_max_children=1"
  append_action_validation_row "$component_key" "$source_component_key" "$image_name" \
    "$control_exp" "$action_root/v360_candidate_plus_v345"
  log_event "action_validate_done" "$component_key"
done < "$V360_ACTION_LIST_TSV"

"$PYTHON_BIN" - "$V360_SEED_ASSET_JSON" "$ACTION_VALIDATION_TSV" "$V360_ASSET_JSON" <<'PY'
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
validated = []
for child in data.get("children", data.get("actions", [])):
    key = str(child.get("component_key", ""))
    if key in kept:
        item = dict(child)
        item["split_child_enable"] = True
        item["split_child_validation_status"] = "keep"
        item["split_child_reason"] = "v360_residual_anchored_action_validated_do_no_harm"
        validated.append(item)
payload = {
    "version": "v360_validated_residual_anchored_split_child_asset",
    "policy": "v360 keeps residual-anchored split-child candidates that pass single-action raw contour do-no-harm validation.",
    "source": data.get("source", {}),
    "child_count": len(validated),
    "children": validated,
    "actions": validated,
    "action_validation": {"rows": rows, "seed_child_count": len(rows), "kept_child_count": len(validated)},
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {out_path} kept_child_count={len(validated)}")
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

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "summary_done" "$SUMMARY"
log_event "finished_bjt" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "ACTION_VALIDATION_TSV=$ACTION_VALIDATION_TSV"
echo "V360_ASSET_JSON=$V360_ASSET_JSON"
echo "END_BJT=$END_BJT"
