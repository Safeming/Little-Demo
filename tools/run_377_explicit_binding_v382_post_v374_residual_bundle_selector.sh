#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v382_post_v374_residual_bundle_selector_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v382_post_v374_residual_bundle_selector_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v382_post_v374_residual_bundle_selector_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"
ASSET_DIR="$LOG_DIR/assets"
EVENTS="$LOG_DIR/events.tsv"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
SOURCE_RAW_GATE_EXP_ROOT="${SOURCE_RAW_GATE_EXP_ROOT:-$ROOT/exp/formal/377_v338_raw_contour_gate_formal_377_v381_closure_raw_selector_raw_gate_20260528_160855_bjt}"
SOURCE_BASELINE_RENDER_EXP="${SOURCE_BASELINE_RENDER_EXP:-$SOURCE_RAW_GATE_EXP_ROOT/baseline_no_preset}"
SOURCE_CURRENT_RENDER_EXP="${SOURCE_CURRENT_RENDER_EXP:-$SOURCE_RAW_GATE_EXP_ROOT/candidate_v381_closure_raw_selector}"
BASE_ASSET_JSON="${BASE_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"

TOP_FRAMES="${TOP_FRAMES:-30}"
INNER_PER_FRAME="${INNER_PER_FRAME:-6}"
OUTER_PER_INNER="${OUTER_PER_INNER:-1}"
MAX_GROUPS="${MAX_GROUPS:-180}"
MAX_GROUPS_PER_IMAGE="${MAX_GROUPS_PER_IMAGE:-18}"
MAX_GROUPS_PER_SOURCE="${MAX_GROUPS_PER_SOURCE:-4}"
MAX_GROUPS_PER_TARGET="${MAX_GROUPS_PER_TARGET:-2}"
MICRO_COUNTS="${MICRO_COUNTS:-4,6}"
RADIUS_SCALES="${RADIUS_SCALES:-0.55,0.72}"
MINOR_SCALES="${MINOR_SCALES:-0.55,0.75}"
DEPTH_SCALES="${DEPTH_SCALES:-1.0}"
COVARIANCE_SCALES="${COVARIANCE_SCALES:-1.0}"
CHILD_OPACITY="${CHILD_OPACITY:-0.045}"
CHILD_OPACITY_MODE="${CHILD_OPACITY_MODE:-divide}"
RESIDUAL_TARGETS_PER_INNER="${RESIDUAL_TARGETS_PER_INNER:-3}"
RESIDUAL_MIN_MASK_PIXELS="${RESIDUAL_MIN_MASK_PIXELS:-3}"
RESIDUAL_MIN_GATE_OVERLAP="${RESIDUAL_MIN_GATE_OVERLAP:-1}"
RESIDUAL_GATE_PAD_PX="${RESIDUAL_GATE_PAD_PX:-26.0}"
STRENGTH_SWEEP_VARIANTS="${STRENGTH_SWEEP_VARIANTS:-base,co2_cr1.2_sp,co3_cr1.3_sp}"
SELECTOR_MAX_ACTIONS="${SELECTOR_MAX_ACTIONS:-240}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

BOOTSTRAP_JSON="$ASSET_DIR/v382_bootstrap_post_v374_residual_bundle_asset.json"
BOOTSTRAP_TSV="$ASSET_DIR/v382_bootstrap_post_v374_residual_bundle_groups.tsv"
ACTION_LIST_TSV="$ASSET_DIR/v382_action_groups.tsv"
RENDERER_SPACE_NPZ="$ASSET_DIR/v382_renderer_space_gaussians.npz"
RENDERER_SPACE_TSV="$ASSET_DIR/v382_renderer_space_gaussians.tsv"
SEED_JSON="$ASSET_DIR/v382_seed_post_v374_residual_bundle_asset.json"
SEED_TSV="$ASSET_DIR/v382_seed_post_v374_residual_bundle_groups.tsv"
CANDIDATE_JSON="$ASSET_DIR/v382_post_v374_residual_bundle_candidate_asset.json"
SELECTOR_LOG="$LOG_DIR/v382_selector.launcher.log"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$CANDIDATE_CKPT" "$DATA_ROOT" "$SOURCE_BASELINE_RENDER_EXP/diagnostics/boundary_residuals/boundary_residual_samples.csv" "$SOURCE_CURRENT_RENDER_EXP/diagnostics/boundary_residuals/boundary_residual_samples.csv" "$BASE_ASSET_JSON" "$COMPONENT_CSV" "$POINT_CSV"; do
  [ -e "$required" ] || { echo "missing required path: $required" >&2; exit 2; }
done

mkdir -p "$LOG_DIR" "$EXP_ROOT" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
log_event(){ printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"; }

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

write_action_list() {
  local asset_json="$1"
  local out_tsv="$2"
  "$PYTHON_BIN" - "$asset_json" "$out_tsv" <<'PY'
import csv, json, re, sys
from pathlib import Path
asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = []
for idx, group in enumerate(asset.get("action_groups", [])):
    image = str(group.get("image_name", group.get("source_image_name", "")) or "")
    m = re.fullmatch(r"c(\d+)_f(\d+)", image)
    if not m:
        continue
    rows.append({
        "action_index": idx,
        "pair_id": str(group.get("pair_id", "")),
        "source_component_key": str(group.get("source_component_key", "")),
        "image_name": image,
        "view": int(m.group(1)),
        "frame": int(m.group(2)),
        "score": float(group.get("frame_score", 0.0) or 0.0),
        "target_score": float(group.get("residual_target_score", 0.0) or 0.0),
    })
rows.sort(key=lambda r: (r["image_name"], -r["score"], -r["target_score"], r["pair_id"]))
with Path(sys.argv[2]).open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["action_index","pair_id","source_component_key","image_name","view","frame","score","target_score"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {sys.argv[2]} groups={len(rows)}")
PY
}

make_asset() {
  local out_json="$1"
  local out_tsv="$2"
  shift 2
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v369_residual_component_multi_micro_grouped_actuator_asset.py \
    --baseline-render-exp "$SOURCE_BASELINE_RENDER_EXP" \
    --current-render-exp "$SOURCE_CURRENT_RENDER_EXP" \
    --checkpoint "$CANDIDATE_CKPT" \
    --component-csv "$COMPONENT_CSV" \
    --point-csv "$POINT_CSV" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --top-frames "$TOP_FRAMES" \
    --inner-per-frame "$INNER_PER_FRAME" \
    --outer-per-inner "$OUTER_PER_INNER" \
    --max-groups "$MAX_GROUPS" \
    --max-groups-per-image "$MAX_GROUPS_PER_IMAGE" \
    --max-groups-per-source "$MAX_GROUPS_PER_SOURCE" \
    --max-groups-per-target "$MAX_GROUPS_PER_TARGET" \
    --micro-counts "$MICRO_COUNTS" \
    --radius-scales "$RADIUS_SCALES" \
    --minor-scales "$MINOR_SCALES" \
    --depth-scales "$DEPTH_SCALES" \
    --covariance-scales "$COVARIANCE_SCALES" \
    --child-opacity "$CHILD_OPACITY" \
    --child-opacity-mode "$CHILD_OPACITY_MODE" \
    --residual-mask-enable \
    --residual-targets-per-inner "$RESIDUAL_TARGETS_PER_INNER" \
    --residual-min-mask-pixels "$RESIDUAL_MIN_MASK_PIXELS" \
    --residual-min-gate-overlap "$RESIDUAL_MIN_GATE_OVERLAP" \
    --residual-gate-pad-px "$RESIDUAL_GATE_PAD_PX" \
    --out-json "$out_json" \
    --out-candidates-tsv "$out_tsv" \
    "$@"
}

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+2 hours 30 minutes' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
SOURCE_CURRENT_RENDER_EXP=$SOURCE_CURRENT_RENDER_EXP
BASE_ASSET_JSON=$BASE_ASSET_JSON
EOF

log_event "make_bootstrap_start" "$BOOTSTRAP_JSON"
make_asset "$BOOTSTRAP_JSON" "$BOOTSTRAP_TSV" > "$LOG_DIR/make_v382_bootstrap_asset.log" 2>&1
write_action_list "$BOOTSTRAP_JSON" "$ACTION_LIST_TSV" > "$LOG_DIR/write_v382_action_list.log" 2>&1
log_event "make_bootstrap_done" "$BOOTSTRAP_JSON"

group_count="$("$PYTHON_BIN" - "$BOOTSTRAP_JSON" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("group_count", 0) or 0))
PY
)"
if [ "$group_count" -le 0 ]; then
  log_event "no_candidates" "$BOOTSTRAP_JSON"
  echo "LOG_DIR=$LOG_DIR"; echo "STATUS=no_candidates"; echo "EST_END_BJT=$EST_END_BJT"; exit 0
fi

log_event "export_renderer_space_start" "$RENDERER_SPACE_NPZ"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/export_renderer_space_gaussians.py \
  --config "$BASE_EXP/.hydra/config.yaml" \
  --checkpoint "$CANDIDATE_CKPT" \
  --action-list-tsv "$ACTION_LIST_TSV" \
  --out-npz "$RENDERER_SPACE_NPZ" \
  --out-tsv "$RENDERER_SPACE_TSV" \
  --exp-dir "$EXP_ROOT/renderer_space_export" \
  --explicit-binding-render-preset v338_temporal_selector_grow_only_guard \
  --dataset-root "$DATA_ROOT" \
  --subject CoreView_377 \
  --train-views "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
  --train-frames "[0,570,60]" \
  > "$LOG_DIR/export_v382_renderer_space.log" 2>&1
log_event "export_renderer_space_done" "$RENDERER_SPACE_NPZ"

log_event "make_seed_start" "$SEED_JSON"
make_asset "$SEED_JSON" "$SEED_TSV" --renderer-space-cache "$RENDERER_SPACE_NPZ" > "$LOG_DIR/make_v382_seed_asset.log" 2>&1
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/expand_377_stageB_v371_strength_sweep_asset.py \
  --in-json "$SEED_JSON" \
  --out-json "$CANDIDATE_JSON" \
  --variants "$STRENGTH_SWEEP_VARIANTS" \
  > "$LOG_DIR/expand_v382_strength_sweep.log" 2>&1
log_event "make_seed_done" "$CANDIDATE_JSON"

log_event "selector_start" "$CANDIDATE_JSON"
env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT \
  GPU="$GPU" PYTHON_BIN="$PYTHON_BIN" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  RUN_ID="v382_selector_${RUN_ID}" \
  BASE_ASSET_JSON="$BASE_ASSET_JSON" \
  CLOSURE_ASSET_JSON="$CANDIDATE_JSON" \
  CANDIDATE_CKPT="$CANDIDATE_CKPT" \
  MAX_ACTIONS="$SELECTOR_MAX_ACTIONS" \
  MIN_INNER_GAIN=0.5 \
  MAX_OUTER_REGRESS=0.0 \
  MAX_OPACITY_OUTER_REGRESS=0.0 \
  CHILD_OPACITY="$CHILD_OPACITY" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  "$PYTHON_BIN" tools/run_377_explicit_binding_v381_closure_raw_selector.py \
  > "$SELECTOR_LOG" 2>&1
log_event "selector_done" "$SELECTOR_LOG"

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "LOG_DIR=$LOG_DIR"
echo "EXP_ROOT=$EXP_ROOT"
echo "CANDIDATE_JSON=$CANDIDATE_JSON"
echo "SELECTOR_LOG=$SELECTOR_LOG"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
