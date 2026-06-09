#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
SOURCE_RAW_GATE_EXP_ROOT="${SOURCE_RAW_GATE_EXP_ROOT:-$ROOT/exp/formal/377_v338_raw_contour_gate_formal_377_v366c_trained_activation_owner_micro_child_raw_gate_20260525_194809_bjt}"
SOURCE_BASELINE_RENDER_EXP="${SOURCE_BASELINE_RENDER_EXP:-$SOURCE_RAW_GATE_EXP_ROOT/baseline_no_preset}"
SOURCE_CURRENT_RENDER_EXP="${SOURCE_CURRENT_RENDER_EXP:-$SOURCE_RAW_GATE_EXP_ROOT/formal_v338_temporal_selector_grow_only_guard}"
RUN_ID="${RUN_ID:-v368_self_protected_residual_grouped_actuator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v368_self_protected_residual_grouped_actuator_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v368_self_protected_residual_grouped_actuator_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"

TOP_FRAMES="${TOP_FRAMES:-30}"
INNER_PER_FRAME="${INNER_PER_FRAME:-2}"
OUTER_PER_INNER="${OUTER_PER_INNER:-1}"
MAX_GROUPS="${MAX_GROUPS:-96}"
MICRO_COUNTS="${MICRO_COUNTS:-3,5}"
RADIUS_SCALES="${RADIUS_SCALES:-0.45,0.60}"
MINOR_SCALES="${MINOR_SCALES:-0.45,0.65}"
DEPTH_SCALES="${DEPTH_SCALES:-1.0}"
COVARIANCE_SCALES="${COVARIANCE_SCALES:-1.0}"
DEPTH_SIGMA_PX="${DEPTH_SIGMA_PX:-1.3}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
CHILD_OPACITY_MODE="${CHILD_OPACITY_MODE:-divide}"
MAX_TOP_IDS="${MAX_TOP_IDS:-8}"
MIN_OWNER_CONSISTENCY="${MIN_OWNER_CONSISTENCY:-0.50}"
REQUIRE_OWNER_MATCH="${REQUIRE_OWNER_MATCH:-false}"
RADIUS_FLOOR="${RADIUS_FLOOR:-0.010}"
RADIUS_PAD="${RADIUS_PAD:-0.006}"
ROW_RADIUS_SCALE="${ROW_RADIUS_SCALE:-1.15}"
OUTER_RADIUS_SCALE="${OUTER_RADIUS_SCALE:-0.62}"
OUTER_SCORE_SCALE="${OUTER_SCORE_SCALE:-0.65}"
ACTIVATION_PAD_PX="${ACTIVATION_PAD_PX:-4.0}"
ACTIVATION_ELLIPSE_SCALE="${ACTIVATION_ELLIPSE_SCALE:-1.15}"
ANCHOR_RADIUS_SCALE="${ANCHOR_RADIUS_SCALE:-0.0}"
RESIDUAL_MASK_ENABLE="${RESIDUAL_MASK_ENABLE:-true}"
RESIDUAL_MIN_MASK_PIXELS="${RESIDUAL_MIN_MASK_PIXELS:-4}"
SELF_PROTECT_INNER_RADIUS_FRACTION="${SELF_PROTECT_INNER_RADIUS_FRACTION:-0.80}"
SELF_PROTECT_SHRINK_FACTOR="${SELF_PROTECT_SHRINK_FACTOR:-0.75}"
SELF_PROTECT_OPACITY_FACTOR="${SELF_PROTECT_OPACITY_FACTOR:-0.50}"
SELF_PROTECT_DROP_ON_OUTER="${SELF_PROTECT_DROP_ON_OUTER:-false}"

GROUP_VALIDATE_MAX="${GROUP_VALIDATE_MAX:-24}"
GROUP_VALIDATE_TARGET_MIN_GAIN="${GROUP_VALIDATE_TARGET_MIN_GAIN:-0.10}"
GROUP_VALIDATE_MAX_FG_REGRESS="${GROUP_VALIDATE_MAX_FG_REGRESS:-0.000002}"
GROUP_VALIDATE_MAX_BOUNDARY_REGRESS="${GROUP_VALIDATE_MAX_BOUNDARY_REGRESS:-0.000002}"
GROUP_VALIDATE_MAX_EDGE_REGRESS="${GROUP_VALIDATE_MAX_EDGE_REGRESS:-0.001}"
GROUP_VALIDATE_MAX_COUNT_REGRESS="${GROUP_VALIDATE_MAX_COUNT_REGRESS:-0.0}"
GROUP_VALIDATE_MAX_HARD_REGRESS="${GROUP_VALIDATE_MAX_HARD_REGRESS:-0.000001}"
GROUP_VALIDATE_MAX_OPACITY_REGRESS="${GROUP_VALIDATE_MAX_OPACITY_REGRESS:-0.0}"

RAW_GATE_ENABLE="${RAW_GATE_ENABLE:-true}"
TRAIN_ON_STRICT_PASS="${TRAIN_ON_STRICT_PASS:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

SUMMARY="$LOG_DIR/summary.tsv"
GROUP_VALIDATION_TSV="$LOG_DIR/group_validation.tsv"
EVENTS="$LOG_DIR/events.tsv"
ASSET_DIR="$LOG_DIR/assets"
V368_BOOTSTRAP_ASSET_JSON="$ASSET_DIR/v368_bootstrap_self_protected_residual_grouped_actuator_asset.json"
V368_BOOTSTRAP_CANDIDATES_TSV="$ASSET_DIR/v368_bootstrap_self_protected_residual_grouped_actuator_groups.tsv"
V368_SEED_ASSET_JSON="$ASSET_DIR/v368_seed_self_protected_residual_grouped_actuator_asset.json"
V368_SEED_CANDIDATES_TSV="$ASSET_DIR/v368_seed_self_protected_residual_grouped_actuator_groups.tsv"
V368_ASSET_JSON="$ASSET_DIR/v368_validated_self_protected_residual_grouped_actuator_asset.json"
V368_ACTION_LIST_TSV="$ASSET_DIR/v368_seed_action_groups.tsv"
V368_RENDERER_SPACE_NPZ="$ASSET_DIR/v368_renderer_space_gaussians.npz"
V368_RENDERER_SPACE_TSV="$ASSET_DIR/v368_renderer_space_gaussians.tsv"
RAW_GATE_LOG="$LOG_DIR/v368_raw_gate.launcher.log"
TRAIN_LAUNCH_LOG="$LOG_DIR/v368_semantic_train.launcher.log"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$SOURCE_BASELINE_RENDER_EXP/diagnostics/boundary_residuals/boundary_residual_samples.csv" \
  "$SOURCE_CURRENT_RENDER_EXP/diagnostics/boundary_residuals/boundary_residual_samples.csv" \
  "$COMPONENT_CSV" "$POINT_CSV"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'pair_id\tsource_component_key\timage_name\tmicro_count\touter_action_count\tstatus\ttarget_gain\tfg_delta_control\tboundary_delta_control\tedge_delta_control\tinner_delta_control\touter_delta_control\thard_delta_control\topacity_inner_delta_control\topacity_outer_delta_control\tcontrol_exp\tcandidate_exp\n' > "$GROUP_VALIDATION_TSV"

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

make_asset() {
  local out_json="$1"
  local out_tsv="$2"
  shift 2
  local args=(
    --baseline-render-exp "$SOURCE_BASELINE_RENDER_EXP"
    --current-render-exp "$SOURCE_CURRENT_RENDER_EXP"
    --checkpoint "$CANDIDATE_CKPT"
    --component-csv "$COMPONENT_CSV"
    --point-csv "$POINT_CSV"
    --dataset-root "$DATA_ROOT"
    --subject CoreView_377
    --top-frames "$TOP_FRAMES"
    --inner-per-frame "$INNER_PER_FRAME"
    --outer-per-inner "$OUTER_PER_INNER"
    --max-groups "$MAX_GROUPS"
    --micro-counts "$MICRO_COUNTS"
    --radius-scales "$RADIUS_SCALES"
    --minor-scales "$MINOR_SCALES"
    --depth-scales "$DEPTH_SCALES"
    --covariance-scales "$COVARIANCE_SCALES"
    --depth-sigma-px "$DEPTH_SIGMA_PX"
    --child-opacity "$CHILD_OPACITY"
    --child-opacity-mode "$CHILD_OPACITY_MODE"
    --max-top-ids "$MAX_TOP_IDS"
    --min-owner-consistency "$MIN_OWNER_CONSISTENCY"
    --radius-floor "$RADIUS_FLOOR"
    --radius-pad "$RADIUS_PAD"
    --row-radius-scale "$ROW_RADIUS_SCALE"
    --outer-radius-scale "$OUTER_RADIUS_SCALE"
    --outer-score-scale "$OUTER_SCORE_SCALE"
    --activation-pad-px "$ACTIVATION_PAD_PX"
    --activation-ellipse-scale "$ACTIVATION_ELLIPSE_SCALE"
    --anchor-radius-scale "$ANCHOR_RADIUS_SCALE"
    --residual-min-mask-pixels "$RESIDUAL_MIN_MASK_PIXELS"
    --self-protect-inner-radius-fraction "$SELF_PROTECT_INNER_RADIUS_FRACTION"
    --self-protect-shrink-factor "$SELF_PROTECT_SHRINK_FACTOR"
    --self-protect-opacity-factor "$SELF_PROTECT_OPACITY_FACTOR"
    --out-json "$out_json"
    --out-candidates-tsv "$out_tsv"
    "$@"
  )
  if [ "$RESIDUAL_MASK_ENABLE" = "true" ]; then
    args+=(--residual-mask-enable)
  else
    args+=(--no-residual-mask-enable)
  fi
  if [ "$REQUIRE_OWNER_MATCH" = "true" ]; then
    args+=(--require-owner-match)
  fi
  if [ "$SELF_PROTECT_DROP_ON_OUTER" = "true" ]; then
    args+=(--self-protect-drop-on-outer)
  fi
  "$PYTHON_BIN" tools/make_377_stageB_v368_self_protected_residual_grouped_actuator_asset.py "${args[@]}"
}

write_action_list() {
  local asset_json="$1"
  local action_list_tsv="$2"
  local limit="$3"
  "$PYTHON_BIN" - "$asset_json" "$action_list_tsv" "$limit" <<'PY'
import csv
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

asset_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
limit = int(float(sys.argv[3]))
data = json.loads(asset_path.read_text(encoding="utf-8"))
groups = []
for index, group in enumerate(data.get("action_groups", [])):
    image = str(group.get("image_name", group.get("source_image_name", "")) or "")
    pair_id = str(group.get("pair_id", "") or "")
    source_key = str(group.get("source_component_key", "") or "")
    match = re.match(r"c(\d+)_f(\d+)$", image)
    if not image or not pair_id or not source_key or not match:
        continue
    groups.append({
        "action_index": index,
        "pair_id": pair_id,
        "source_component_key": source_key,
        "image_name": image,
        "view": int(match.group(1)),
        "frame": int(match.group(2)),
        "micro_count": int(group.get("micro_count", 0) or 0),
        "outer_action_count": int(group.get("outer_action_count", 0) or 0),
        "score": float(group.get("frame_score", 0.0) or 0.0),
    })
groups.sort(key=lambda row: row["score"], reverse=True)
by_image = defaultdict(deque)
for row in groups:
    by_image[row["image_name"]].append(row)
ordered = []
while by_image:
    for image in sorted(list(by_image)):
        bucket = by_image.get(image)
        if not bucket:
            by_image.pop(image, None)
            continue
        ordered.append(bucket.popleft())
        if limit > 0 and len(ordered) >= limit:
            break
    if limit > 0 and len(ordered) >= limit:
        break
if limit > 0:
    ordered = ordered[:limit]
with out_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "action_index", "pair_id", "source_component_key", "image_name", "view", "frame",
        "micro_count", "outer_action_count", "score",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(ordered)
print(f"wrote {out_path} groups={len(ordered)}")
PY
}

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
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --band-width 7 --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}_${variant}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --render-support-threshold 0.025 --close-kernel 5 \
    --band-width 7 --search-band-width 24 --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${label}_${variant}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --render-support-threshold 0.025 --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 --rgb-close-kernel 5 \
    --opacity-close-kernel 3 --band-width 7 --search-band-width 24 --topk 16 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${label}_${variant}.log" 2>&1
  log_event "analyze_${label}_${variant}_done" "status=0"
}

append_group_validation_row() {
  local pair_id="$1"
  local source_component_key="$2"
  local image_name="$3"
  local micro_count="$4"
  local outer_action_count="$5"
  local control_exp="$6"
  local candidate_exp="$7"
  "$PYTHON_BIN" - \
    "$GROUP_VALIDATION_TSV" "$pair_id" "$source_component_key" "$image_name" "$micro_count" "$outer_action_count" \
    "$control_exp" "$candidate_exp" \
    "$GROUP_VALIDATE_TARGET_MIN_GAIN" \
    "$GROUP_VALIDATE_MAX_FG_REGRESS" "$GROUP_VALIDATE_MAX_BOUNDARY_REGRESS" "$GROUP_VALIDATE_MAX_EDGE_REGRESS" \
    "$GROUP_VALIDATE_MAX_COUNT_REGRESS" "$GROUP_VALIDATE_MAX_HARD_REGRESS" "$GROUP_VALIDATE_MAX_OPACITY_REGRESS" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
pair_id = sys.argv[2]
source_component_key = sys.argv[3]
image_name = sys.argv[4]
micro_count = int(float(sys.argv[5]))
outer_action_count = int(float(sys.argv[6]))
control_exp = Path(sys.argv[7])
candidate_exp = Path(sys.argv[8])
target_min_gain = float(sys.argv[9])
max_fg = float(sys.argv[10])
max_boundary = float(sys.argv[11])
max_edge = float(sys.argv[12])
max_count = float(sys.argv[13])
max_hard = float(sys.argv[14])
max_opacity = float(sys.argv[15])
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
    -delta["inner"],
    -delta["opacity_inner"],
    -delta["outer"],
    -delta["opacity_outer"],
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
    "pair_id": pair_id,
    "source_component_key": source_component_key,
    "image_name": image_name,
    "micro_count": micro_count,
    "outer_action_count": outer_action_count,
    "status": status,
    "target_gain": target_gain,
    **{f"{key}_delta_control": delta[key] for key in metrics},
    "control_exp": str(control_exp),
    "candidate_exp": str(candidate_exp),
}
with out_path.open("a", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "pair_id", "source_component_key", "image_name", "micro_count", "outer_action_count", "status",
        "target_gain", "fg_delta_control", "boundary_delta_control", "edge_delta_control",
        "inner_delta_control", "outer_delta_control", "hard_delta_control",
        "opacity_inner_delta_control", "opacity_outer_delta_control", "control_exp", "candidate_exp",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writerow(row)
print(f"{pair_id}\t{status}\ttarget_gain={target_gain:.6f}")
PY
}

log_event "make_bootstrap_asset_start" "$V368_BOOTSTRAP_ASSET_JSON"
make_asset "$V368_BOOTSTRAP_ASSET_JSON" "$V368_BOOTSTRAP_CANDIDATES_TSV" \
  > "$LOG_DIR/make_v368_bootstrap_asset.log" 2>&1
write_action_list "$V368_BOOTSTRAP_ASSET_JSON" "$V368_ACTION_LIST_TSV" -1
log_event "make_bootstrap_asset_done" "$V368_BOOTSTRAP_ASSET_JSON"

log_event "export_renderer_space_start" "$V368_RENDERER_SPACE_NPZ"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/export_renderer_space_gaussians.py \
  --config "$BASE_EXP/.hydra/config.yaml" \
  --checkpoint "$CANDIDATE_CKPT" \
  --action-list-tsv "$V368_ACTION_LIST_TSV" \
  --out-npz "$V368_RENDERER_SPACE_NPZ" \
  --out-tsv "$V368_RENDERER_SPACE_TSV" \
  --exp-dir "$EXP_ROOT/renderer_space_export" \
  --explicit-binding-render-preset v338_temporal_selector_grow_only_guard \
  --dataset-root "$DATA_ROOT" \
  --subject CoreView_377 \
  --train-views "$TRAIN_VIEWS_SPEC" \
  --train-frames "$TRAIN_FRAMES_SPEC" \
  > "$LOG_DIR/export_renderer_space_gaussians.log" 2>&1
log_event "export_renderer_space_done" "$V368_RENDERER_SPACE_NPZ"

log_event "make_seed_asset_start" "$V368_SEED_ASSET_JSON"
make_asset "$V368_SEED_ASSET_JSON" "$V368_SEED_CANDIDATES_TSV" \
  --renderer-space-cache "$V368_RENDERER_SPACE_NPZ" \
  > "$LOG_DIR/make_v368_seed_asset.log" 2>&1
write_action_list "$V368_SEED_ASSET_JSON" "$V368_ACTION_LIST_TSV" "$GROUP_VALIDATE_MAX"
log_event "make_seed_asset_done" "$V368_SEED_ASSET_JSON"

while IFS=$'\t' read -r action_index pair_id source_component_key image_name view frame micro_count outer_action_count score; do
  if [ "$action_index" = "action_index" ]; then
    continue
  fi
  safe_pair="$(printf '%s' "$pair_id" | tr -c 'A-Za-z0-9_' '_')"
  views_spec="[$view]"
  next_frame=$((frame + 1))
  frames_spec="[$frame,$next_frame,1]"
  group_root="$EXP_ROOT/group_validation/$safe_pair"
  control_exp="$group_root/control_v338"
  candidate_exp="$group_root/v368_grouped_actuator"
  log_event "group_validate_start" "$pair_id"
  render_and_analyze "group_${safe_pair}" "$views_spec" "$frames_spec" control_v338 "$control_exp" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"
  render_and_analyze "group_${safe_pair}" "$views_spec" "$frames_spec" v368_grouped_actuator "$candidate_exp" \
    "pipeline.compute_cov3D_python=true" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++pipeline.split_child_component_enable=true" \
    "++pipeline.split_child_component_asset_json=$V368_SEED_ASSET_JSON" \
    "++pipeline.split_child_component_action_filter='pair_id=$pair_id'" \
    "++pipeline.split_child_component_action_required=true" \
    "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
    "++pipeline.split_child_component_radius_scale=1.0" \
    "++pipeline.split_child_component_max_children=-1"
  append_group_validation_row "$pair_id" "$source_component_key" "$image_name" "$micro_count" "$outer_action_count" \
    "$control_exp" "$candidate_exp"
  log_event "group_validate_done" "$pair_id"
done < "$V368_ACTION_LIST_TSV"

"$PYTHON_BIN" - "$V368_SEED_ASSET_JSON" "$GROUP_VALIDATION_TSV" "$V368_ASSET_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

seed_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
data = json.loads(seed_path.read_text(encoding="utf-8"))
rows = []
best_by_source = {}
with validation_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        rows.append(row)
        if row.get("status") != "keep":
            continue
        source = str(row.get("source_component_key", "") or "")
        if not source:
            continue
        try:
            score = (
                float(row.get("target_gain", 0.0) or 0.0),
                -float(row.get("inner_delta_control", 0.0) or 0.0),
                -float(row.get("outer_delta_control", 0.0) or 0.0),
                -float(row.get("opacity_outer_delta_control", 0.0) or 0.0),
                -float(row.get("hard_delta_control", 0.0) or 0.0),
            )
        except Exception:
            continue
        current = best_by_source.get(source)
        if current is None or score > current["score"]:
            best_by_source[source] = {"score": score, "pair_id": str(row.get("pair_id", ""))}
kept = {item["pair_id"] for item in best_by_source.values() if item.get("pair_id")}
groups = [group for group in data.get("action_groups", []) if str(group.get("pair_id", "")) in kept]
children = [child for child in data.get("children", []) if str(child.get("pair_id", "")) in kept]
actions = [action for action in data.get("actions", []) if str(action.get("pair_id", "")) in kept]
payload = {
    **{key: value for key, value in data.items() if key not in ("action_groups", "children", "actions")},
    "version": "v368_validated_residual_driven_grouped_actuator_asset",
    "policy": (
        "Validated v368 asset: one best action-level raw-contour-safe grouped actuator per source residual component."
    ),
    "group_count": len(groups),
    "child_count": len(children),
    "action_count": len(actions),
    "action_groups": groups,
    "children": children,
    "actions": actions,
    "group_validation": {
        "rows": rows,
        "validated_group_count": len(rows),
        "kept_group_count": sum(1 for row in rows if row.get("status") == "keep"),
        "selected_group_count": len(groups),
        "selected_pair_ids": sorted(kept),
    },
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {out_path} selected_groups={len(groups)} children={len(children)} actions={len(actions)}")
PY

"$PYTHON_BIN" - "$SUMMARY" "$GROUP_VALIDATION_TSV" "$V368_ASSET_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
asset_path = Path(sys.argv[3])
rows = list(csv.DictReader(validation_path.open("r", encoding="utf-8"), delimiter="\t"))
asset = json.loads(asset_path.read_text(encoding="utf-8"))
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "groups", "kept", "selected", "children", "outer_actions",
        "inner_delta_sum", "outer_delta_sum", "opacity_inner_delta_sum",
        "opacity_outer_delta_sum", "edge_delta_sum", "hard_delta_sum",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerow({
        "groups": len(rows),
        "kept": sum(1 for row in rows if row.get("status") == "keep"),
        "selected": int(asset.get("group_count", 0) or 0),
        "children": int(asset.get("child_count", 0) or 0),
        "outer_actions": int(asset.get("action_count", 0) or 0),
        "inner_delta_sum": sum(float(row.get("inner_delta_control", 0.0) or 0.0) for row in rows),
        "outer_delta_sum": sum(float(row.get("outer_delta_control", 0.0) or 0.0) for row in rows),
        "opacity_inner_delta_sum": sum(float(row.get("opacity_inner_delta_control", 0.0) or 0.0) for row in rows),
        "opacity_outer_delta_sum": sum(float(row.get("opacity_outer_delta_control", 0.0) or 0.0) for row in rows),
        "edge_delta_sum": sum(float(row.get("edge_delta_control", 0.0) or 0.0) for row in rows),
        "hard_delta_sum": sum(float(row.get("hard_delta_control", 0.0) or 0.0) for row in rows),
    })
PY
log_event "summary_done" "$SUMMARY"

RAW_GATE_STATUS="not_run"
RAW_GATE_SUMMARY=""
if [ "$RAW_GATE_ENABLE" = "true" ]; then
  selected_groups="$("$PYTHON_BIN" - "$V368_ASSET_JSON" <<'PY'
import json, sys
data=json.loads(open(sys.argv[1], encoding="utf-8").read())
print(int(data.get("group_count", 0) or 0))
PY
)"
  if [ "$selected_groups" -gt 0 ]; then
    RAW_GATE_RUN_ID="formal_377_v368_self_protected_residual_grouped_actuator_raw_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
    log_event "raw_gate_start" "$RAW_GATE_RUN_ID"
    env GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
      BASE_CKPT="$BASE_CKPT" \
      CANDIDATE_CKPT="$CANDIDATE_CKPT" \
      CANDIDATE_VARIANT_NAME="candidate_v368_self_protected_residual_grouped_actuator" \
      CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
      CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$V368_ASSET_JSON" \
      CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
      CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY="$CHILD_OPACITY" \
      CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
      CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
      RUN_ID="$RAW_GATE_RUN_ID" \
      "$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh" \
      > "$RAW_GATE_LOG" 2>&1
    RAW_GATE_SUMMARY="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${RAW_GATE_RUN_ID}/summary.tsv"
    RAW_GATE_STATUS="$("$PYTHON_BIN" - "$RAW_GATE_SUMMARY" <<'PY'
import csv, sys
path=sys.argv[1]
status="missing"
with open(path, encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row.get("variant") == "candidate_v368_self_protected_residual_grouped_actuator":
            status = row.get("status", "")
print(status)
PY
)"
    log_event "raw_gate_done" "status=$RAW_GATE_STATUS summary=$RAW_GATE_SUMMARY"
  else
    RAW_GATE_STATUS="no_selected_groups"
    log_event "raw_gate_skip" "$RAW_GATE_STATUS"
  fi
fi

TRAIN_PID=""
TRAIN_EXP_DIR=""
TRAIN_EST_END_BJT=""
if [ "$TRAIN_ON_STRICT_PASS" = "true" ] && [ "$RAW_GATE_STATUS" = "strict_pass" ]; then
  TRAIN_RUN_ID="formal_377_v368_self_protected_residual_grouped_actuator_semantic_train_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')"
  TRAIN_EXP_DIR="$ROOT/exp/formal/377_v368_self_protected_residual_grouped_actuator_semantic_train_${TRAIN_RUN_ID}"
  TRAIN_SCRIPT="$LOG_DIR/v368_semantic_train.launch.sh"
  log_event "train_start" "$TRAIN_EXP_DIR"
  cat > "$TRAIN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
env GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \\
  BASE_CKPT="$CANDIDATE_CKPT" \\
  RUN_ID="$TRAIN_RUN_ID" \\
  EXP_DIR="$TRAIN_EXP_DIR" \\
  TRAIN_STEPS="$TRAIN_STEPS" \\
  "$ROOT/tools/formal/run_377_v338_semantic_train.sh" \\
  "++pipeline.split_child_component_enable=true" \\
  "++pipeline.split_child_component_asset_json=$V368_ASSET_JSON" \\
  "++pipeline.split_child_component_action_required=false" \\
  "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \\
  "++pipeline.split_child_component_radius_scale=1.0" \\
  "++pipeline.split_child_component_max_children=-1"
EOF
  chmod +x "$TRAIN_SCRIPT"
  setsid -f "$TRAIN_SCRIPT" > "$TRAIN_LAUNCH_LOG" 2>&1 < /dev/null
  sleep 2
  TRAIN_PID="$(pgrep -f "$TRAIN_SCRIPT" | tail -n 1 || true)"
  if [ -z "$TRAIN_PID" ]; then
    TRAIN_PID="$(pgrep -f "$TRAIN_EXP_DIR" | tail -n 1 || true)"
  fi
  echo "$TRAIN_PID" > "$LOG_DIR/train.pid"
  if [ -z "$TRAIN_PID" ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    log_event "train_launch_failed" "pid=$TRAIN_PID log=$TRAIN_LAUNCH_LOG"
    exit 3
  fi
  TRAIN_EST_END_BJT="$(TZ=Asia/Shanghai date -d '+60 minutes' '+%F %T BJT')"
  log_event "train_launched" "pid=$TRAIN_PID est_end=$TRAIN_EST_END_BJT script=$TRAIN_SCRIPT log=$TRAIN_LAUNCH_LOG"
else
  log_event "train_skip" "raw_gate_status=$RAW_GATE_STATUS"
fi

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "finished_bjt" "$END_BJT"

echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "GROUP_VALIDATION_TSV=$GROUP_VALIDATION_TSV"
echo "V368_ASSET_JSON=$V368_ASSET_JSON"
echo "RAW_GATE_STATUS=$RAW_GATE_STATUS"
echo "RAW_GATE_SUMMARY=$RAW_GATE_SUMMARY"
echo "TRAIN_PID=$TRAIN_PID"
echo "TRAIN_EXP_DIR=$TRAIN_EXP_DIR"
echo "TRAIN_EST_END_BJT=$TRAIN_EST_END_BJT"
echo "END_BJT=$END_BJT"
