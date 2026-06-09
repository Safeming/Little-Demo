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
RUN_ID="${RUN_ID:-v357_paired_split_child_protect_probe1_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v357_paired_split_child_protect_asset_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v357_paired_split_child_protect_asset_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
WINDOW_SPECS="${WINDOW_SPECS:-c23_f298_301|[23]|[298,301,1];c22_f240_242|[22]|[240,242,1];c21_f480_482|[21]|[480,482,1];c23_f060_062|[23]|[60,62,1];c21_f120_121|[21]|[120,121,1];c21_f540_541|[21]|[540,541,1];c21_f420_421|[21]|[420,421,1];c22_f120_121|[22]|[120,121,1];c21_f300_301|[21]|[300,301,1]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"
V356_LOG_DIR="${V356_LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v356_validated_split_child_asset_v356_validated_split_child_probe1_20260524}"
V356_SEED_JSON="${V356_SEED_JSON:-$V356_LOG_DIR/assets/v356_seed_split_child_asset.json}"
V356_ACTION_VALIDATION_TSV="${V356_ACTION_VALIDATION_TSV:-$V356_LOG_DIR/action_validation.tsv}"
V345_SCREEN_DROP_LIST="${V345_SCREEN_DROP_LIST:-$ROOT/exp/stageB/logs/377_explicit_binding_v345_temporal_screen_guard_20260523/assets/v345_combined_screen_guard_drop_images.txt}"

MIN_TARGET_GAIN="${MIN_TARGET_GAIN:-0.25}"
MIN_OUTER_HURT="${MIN_OUTER_HURT:-1.0}"
OUTER_CANDIDATES_PER_CHILD="${OUTER_CANDIDATES_PER_CHILD:-2}"
MAX_PAIRS="${MAX_PAIRS:-12}"
MAX_TOP_IDS="${MAX_TOP_IDS:-8}"
MIN_OWNER_CONSISTENCY="${MIN_OWNER_CONSISTENCY:-0.50}"
REQUIRE_OWNER_MATCH="${REQUIRE_OWNER_MATCH:-false}"
OUTER_MODE="${OUTER_MODE:-paired_local_3d_intersect}"
OUTER_SCORE_SCALE="${OUTER_SCORE_SCALE:-1.35}"
RADIUS_FLOOR="${RADIUS_FLOOR:-0.010}"
RADIUS_PAD="${RADIUS_PAD:-0.006}"
ROW_RADIUS_SCALE="${ROW_RADIUS_SCALE:-1.15}"
CHILD_RADIUS_SCALE="${CHILD_RADIUS_SCALE:-0.38}"
CHILD_OPACITY="${CHILD_OPACITY:-0.18}"
CLUSTER_ENABLE="${CLUSTER_ENABLE:-true}"
CLUSTER_MIN_POINTS="${CLUSTER_MIN_POINTS:-4}"
CLUSTER_RADIUS_MAX="${CLUSTER_RADIUS_MAX:-0.18}"
CLUSTER_OWNER_GATE="${CLUSTER_OWNER_GATE:-true}"

PAIR_VALIDATE_ENABLE="${PAIR_VALIDATE_ENABLE:-true}"
PAIR_VALIDATE_MAX="${PAIR_VALIDATE_MAX:-8}"
PAIR_VALIDATE_TARGET_MIN_GAIN="${PAIR_VALIDATE_TARGET_MIN_GAIN:-0.25}"
PAIR_VALIDATE_MAX_FG_REGRESS="${PAIR_VALIDATE_MAX_FG_REGRESS:-0.000002}"
PAIR_VALIDATE_MAX_BOUNDARY_REGRESS="${PAIR_VALIDATE_MAX_BOUNDARY_REGRESS:-0.000002}"
PAIR_VALIDATE_MAX_EDGE_REGRESS="${PAIR_VALIDATE_MAX_EDGE_REGRESS:-0.001}"
PAIR_VALIDATE_MAX_COUNT_REGRESS="${PAIR_VALIDATE_MAX_COUNT_REGRESS:-0.0}"
PAIR_VALIDATE_MAX_HARD_REGRESS="${PAIR_VALIDATE_MAX_HARD_REGRESS:-0.000001}"
PAIR_VALIDATE_MAX_OPACITY_REGRESS="${PAIR_VALIDATE_MAX_OPACITY_REGRESS:-0.0}"

SUMMARY="$LOG_DIR/summary.tsv"
WINDOW_SUMMARY="$LOG_DIR/window_summary.tsv"
PAIR_VALIDATION_TSV="$LOG_DIR/pair_validation.tsv"
EVENTS="$LOG_DIR/events.tsv"
ASSET_DIR="$LOG_DIR/assets"
V357_SEED_ASSET_JSON="$ASSET_DIR/v357_seed_paired_split_child_protect_asset.json"
V357_ASSET_JSON="$ASSET_DIR/v357_validated_paired_split_child_protect_asset.json"
V357_CANDIDATES_TSV="$ASSET_DIR/v357_paired_split_child_protect_candidates.tsv"
V357_PAIR_LIST_TSV="$ASSET_DIR/v357_seed_pairs.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$COMPONENT_CSV" "$POINT_CSV" "$V356_SEED_JSON" "$V356_ACTION_VALIDATION_TSV" "$V345_SCREEN_DROP_LIST"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'pair_id\timage_name\tstatus\ttarget_gain\tfg_delta_control\tboundary_delta_control\tedge_delta_control\tinner_delta_control\touter_delta_control\thard_delta_control\topacity_inner_delta_control\topacity_outer_delta_control\tcontrol_exp\tcandidate_exp\n' > "$PAIR_VALIDATION_TSV"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

MAKE_ARGS=()
[ "$REQUIRE_OWNER_MATCH" = "true" ] && MAKE_ARGS+=(--require-owner-match)
[ "$CLUSTER_ENABLE" = "true" ] && MAKE_ARGS+=(--cluster-enable)
[ "$CLUSTER_OWNER_GATE" = "true" ] && MAKE_ARGS+=(--cluster-owner-gate)

log_event "make_seed_asset_start" "$V357_SEED_ASSET_JSON"
"$PYTHON_BIN" tools/make_377_stageB_v357_paired_split_child_protect_asset.py \
  --v356-seed-json "$V356_SEED_JSON" \
  --v356-action-validation-tsv "$V356_ACTION_VALIDATION_TSV" \
  --component-csv "$COMPONENT_CSV" \
  --point-csv "$POINT_CSV" \
  --min-target-gain "$MIN_TARGET_GAIN" \
  --min-outer-hurt "$MIN_OUTER_HURT" \
  --outer-candidates-per-child "$OUTER_CANDIDATES_PER_CHILD" \
  --max-pairs "$MAX_PAIRS" \
  --max-top-ids "$MAX_TOP_IDS" \
  --min-owner-consistency "$MIN_OWNER_CONSISTENCY" \
  --outer-mode "$OUTER_MODE" \
  --outer-score-scale "$OUTER_SCORE_SCALE" \
  --radius-floor "$RADIUS_FLOOR" \
  --radius-pad "$RADIUS_PAD" \
  --row-radius-scale "$ROW_RADIUS_SCALE" \
  --cluster-min-points "$CLUSTER_MIN_POINTS" \
  --cluster-radius-max "$CLUSTER_RADIUS_MAX" \
  --out-json "$V357_SEED_ASSET_JSON" \
  --out-candidates-tsv "$V357_CANDIDATES_TSV" \
  "${MAKE_ARGS[@]}" \
  > "$LOG_DIR/make_v357_seed_asset.log" 2>&1
log_event "make_seed_asset_done" "$V357_SEED_ASSET_JSON"

"$PYTHON_BIN" - "$V357_SEED_ASSET_JSON" "$V357_PAIR_LIST_TSV" "$PAIR_VALIDATE_MAX" <<'PY'
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
for index, pair in enumerate(data.get("pairs", [])):
    pair_id = str(pair.get("pair_id", "") or "")
    image = str(pair.get("image_name", "") or "")
    match = re.match(r"c(\d+)_f(\d+)$", image)
    if not pair_id or not match:
        continue
    rows.append({"pair_index": index, "pair_id": pair_id, "image_name": image, "view": int(match.group(1)), "frame": int(match.group(2))})
if limit > 0:
    rows = rows[:limit]
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["pair_index", "pair_id", "image_name", "view", "frame"], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out_path} pairs={len(rows)}")
PY

POINT_SCREEN_DROP_IMAGES="$(tr '\n' ',' < "$V345_SCREEN_DROP_LIST" | sed 's/,$//')"
COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

render_and_analyze() {
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local variant="$4"
  local ckpt="$5"
  local render_exp="$6"
  shift 6

  log_event "render_${window}_${variant}_start" "$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
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
    "hydra.run.dir=$HYDRA_RUN_ROOT/${window}_${variant}" \
    "wandb_disable=true" \
    "$@" \
    > "$LOG_DIR/render_${window}_${variant}.log" 2>&1
  log_event "render_${window}_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${window}_${variant}.log" 2>&1

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
    > "$LOG_DIR/boundary_residuals_${window}_${variant}.log" 2>&1

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
    > "$LOG_DIR/opacity_footprint_${window}_${variant}.log" 2>&1
  log_event "analyze_${window}_${variant}_done" "status=0"
}

append_pair_validation_row() {
  local pair_id="$1"
  local image_name="$2"
  local control_exp="$3"
  local candidate_exp="$4"
  "$PYTHON_BIN" - \
    "$PAIR_VALIDATION_TSV" "$pair_id" "$image_name" "$control_exp" "$candidate_exp" \
    "$PAIR_VALIDATE_TARGET_MIN_GAIN" \
    "$PAIR_VALIDATE_MAX_FG_REGRESS" "$PAIR_VALIDATE_MAX_BOUNDARY_REGRESS" "$PAIR_VALIDATE_MAX_EDGE_REGRESS" \
    "$PAIR_VALIDATE_MAX_COUNT_REGRESS" "$PAIR_VALIDATE_MAX_HARD_REGRESS" "$PAIR_VALIDATE_MAX_OPACITY_REGRESS" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
pair_id = sys.argv[2]
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
    "pair_id": pair_id,
    "image_name": image_name,
    "status": status,
    "target_gain": target_gain,
    **{f"{key}_delta_control": delta[key] for key in metrics},
    "control_exp": str(control_exp),
    "candidate_exp": str(candidate_exp),
}
with out_path.open("a", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "pair_id", "image_name", "status", "target_gain",
        "fg_delta_control", "boundary_delta_control", "edge_delta_control",
        "inner_delta_control", "outer_delta_control", "hard_delta_control",
        "opacity_inner_delta_control", "opacity_outer_delta_control",
        "control_exp", "candidate_exp",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writerow(row)
print(f"{pair_id}\t{status}\ttarget_gain={target_gain:.6f}")
PY
}

if [ "$PAIR_VALIDATE_ENABLE" = "true" ]; then
  while IFS=$'\t' read -r pair_index pair_id image_name view frame; do
    frame="${frame//$'\r'/}"
    view="${view//$'\r'/}"
    if [ "$pair_index" = "pair_index" ]; then
      continue
    fi
    safe_key="$(printf '%s' "$pair_id" | tr -c 'A-Za-z0-9_' '_')"
    views_spec="[$view]"
    next_frame=$((frame + 1))
    frames_spec="[$frame,$next_frame,1]"
    pair_root="$EXP_ROOT/pair_validation/$safe_key"
    log_event "pair_validate_start" "$pair_id"
    render_and_analyze "pair_${safe_key}" "$views_spec" "$frames_spec" control_v345 "$CANDIDATE_CKPT" "$pair_root/control_v345" \
      "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
      "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'"

    render_and_analyze "pair_${safe_key}" "$views_spec" "$frames_spec" paired_split_child_protect_plus_v345 "$CANDIDATE_CKPT" "$pair_root/paired_split_child_protect_plus_v345" \
      "pipeline.compute_cov3D_python=true" \
      "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
      "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
      "++pipeline.split_child_component_enable=true" \
      "++pipeline.split_child_component_asset_json=$V357_SEED_ASSET_JSON" \
      "++pipeline.split_child_component_action_filter='pair_id=$pair_id'" \
      "++pipeline.split_child_component_action_required=true" \
      "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
      "++pipeline.split_child_component_radius_scale=$CHILD_RADIUS_SCALE" \
      "++pipeline.split_child_component_max_children=1" \
      "++explicit_binding_adopted_signed_dynamic_component_local_asset_json=$V357_SEED_ASSET_JSON" \
      "++pipeline.covariance_signed_dynamic_component_action_filter='pair_id=$pair_id'" \
      "++pipeline.covariance_signed_dynamic_component_action_required=true"

    append_pair_validation_row "$pair_id" "$image_name" \
      "$pair_root/control_v345" \
      "$pair_root/paired_split_child_protect_plus_v345"
    log_event "pair_validate_done" "$pair_id"
  done < "$V357_PAIR_LIST_TSV"
fi

"$PYTHON_BIN" - "$V357_SEED_ASSET_JSON" "$PAIR_VALIDATION_TSV" "$V357_ASSET_JSON" <<'PY'
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
if validation_path.exists():
    with validation_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row)
            if row.get("status") == "keep":
                kept.add(str(row.get("pair_id", "")))
data["pairs"] = [pair for pair in data.get("pairs", []) if str(pair.get("pair_id", "")) in kept]
data["children"] = [child for child in data.get("children", []) if str(child.get("pair_id", "")) in kept]
data["actions"] = [action for action in data.get("actions", []) if str(action.get("pair_id", "")) in kept]
data["version"] = "v357_validated_paired_split_child_parent_protect_asset"
data["policy"] = (
    "v357 keeps only paired split-child + parent outer row-local protect actions that pass "
    "single-pair raw contour do-no-harm validation against v338+v345."
)
data["pair_count"] = len(data.get("pairs", []))
data["child_count"] = len(data.get("children", []))
data["action_count"] = len(data.get("actions", []))
data["pair_validation"] = {
    "validation_tsv": str(validation_path),
    "seed_pair_count": len(rows),
    "kept_pair_count": len(data.get("pairs", [])),
    "rows": rows,
}
out_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {out_path} kept_pair_count={len(data.get('pairs', []))} actions={len(data.get('actions', []))}")
PY
log_event "validated_asset_done" "$V357_ASSET_JSON"

run_window() {
  local window="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local window_root="$EXP_ROOT/$window"

  render_and_analyze "$window" "$views_spec" "$frames_spec" baseline_no_preset "$BASE_CKPT" "$window_root/baseline_no_preset" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=false" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
    "++pipeline.covariance_signed_center_offset_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

  render_and_analyze "$window" "$views_spec" "$frames_spec" formal_v338_current "$CANDIDATE_CKPT" "$window_root/formal_v338_current" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v345_screen_guard "$CANDIDATE_CKPT" "$window_root/v345_screen_guard" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'"

  render_and_analyze "$window" "$views_spec" "$frames_spec" v357_validated_paired_split_child_protect_plus_v345 "$CANDIDATE_CKPT" "$window_root/v357_validated_paired_split_child_protect_plus_v345" \
    "pipeline.compute_cov3D_python=true" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_adopted_signed_point_screen_actuator_drop_images='$POINT_SCREEN_DROP_IMAGES'" \
    "++pipeline.split_child_component_enable=true" \
    "++pipeline.split_child_component_asset_json=$V357_ASSET_JSON" \
    "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
    "++pipeline.split_child_component_radius_scale=$CHILD_RADIUS_SCALE" \
    "++explicit_binding_adopted_signed_dynamic_component_local_asset_json=$V357_ASSET_JSON"
}

IFS=';' read -r -a WINDOWS <<< "$WINDOW_SPECS"
for item in "${WINDOWS[@]}"; do
  IFS='|' read -r window views frames <<< "$item"
  run_window "$window" "$views" "$frames"
done

"$PYTHON_BIN" - "$SUMMARY" "$WINDOW_SUMMARY" "$EXP_ROOT" "${WINDOWS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
window_summary_path = Path(sys.argv[2])
exp_root = Path(sys.argv[3])
windows = [item.split("|")[0] for item in sys.argv[4:]]
variants = [
    "baseline_no_preset",
    "formal_v338_current",
    "v345_screen_guard",
    "v357_validated_paired_split_child_protect_plus_v345",
]
metrics = ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")

def load_metrics(render_exp):
    render_exp = Path(render_exp)
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8"))
    opacity = json.loads((render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8"))
    return {
        "samples": int(contour.get("n_samples", residual.get("n_samples", opacity.get("n_samples", 0)))),
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    }

def status_for(delta):
    strict = all(delta[k] <= 0.0 for k in metrics)
    probe = (
        delta["fg"] <= 0.00002
        and delta["boundary"] <= 0.00002
        and delta["edge"] <= 0.01
        and delta["inner"] <= 2.0
        and delta["outer"] <= 2.0
        and delta["hard"] <= 0.00025
        and delta["opacity_inner"] <= 2.0
        and delta["opacity_outer"] <= 2.0
    )
    return "strict_pass" if strict else ("probe_pass" if probe else "regresses")

window_rows = []
totals = {variant: {"samples": 0, **{k: 0.0 for k in metrics}} for variant in variants}
for window in windows:
    loaded = {variant: load_metrics(exp_root / window / variant) for variant in variants}
    base = loaded["baseline_no_preset"]
    for variant, data in loaded.items():
        delta = {k: data[k] - base[k] for k in metrics}
        row = {
            "window": window,
            "variant": variant,
            "samples": data["samples"],
            **{k: data[k] for k in metrics},
            **{f"{k}_delta_base": delta[k] for k in metrics},
            "status": "baseline" if variant == "baseline_no_preset" else status_for(delta),
        }
        window_rows.append(row)
        totals[variant]["samples"] += data["samples"]
        for key in metrics:
            totals[variant][key] += data[key] * data["samples"]

base_total = totals["baseline_no_preset"]
base_samples = max(1, base_total["samples"])
base_mean = {k: base_total[k] / base_samples for k in metrics}
rows = []
for variant in variants:
    samples = max(1, totals[variant]["samples"])
    data = {"samples": totals[variant]["samples"], **{k: totals[variant][k] / samples for k in metrics}}
    delta = {k: data[k] - base_mean[k] for k in metrics}
    rows.append({
        "variant": variant,
        "samples": data["samples"],
        **{k: data[k] for k in metrics},
        **{f"{k}_delta_base": delta[k] for k in metrics},
        "status": "baseline" if variant == "baseline_no_preset" else status_for(delta),
    })

with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
with window_summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(window_rows)
PY

log_event "summary_done" "$SUMMARY"
log_event "finished_bjt" "$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "WINDOW_SUMMARY=$WINDOW_SUMMARY"
echo "PAIR_VALIDATION_TSV=$PAIR_VALIDATION_TSV"
echo "V357_ASSET_JSON=$V357_ASSET_JSON"
echo "V357_CANDIDATES_TSV=$V357_CANDIDATES_TSV"
