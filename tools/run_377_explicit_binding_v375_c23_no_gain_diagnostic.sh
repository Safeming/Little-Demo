#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
RUN_ID="${RUN_ID:-v375_c23_no_gain_diag_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v375_c23_no_gain_diagnostic_${RUN_ID}}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v375_c23_no_gain_diagnostic_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
ASSET_DIR="$LOG_DIR/assets"
EVENTS="$LOG_DIR/events.tsv"
ACTION_LIST="$ASSET_DIR/v375_c23_actions.tsv"
SUMMARY="$LOG_DIR/summary.tsv"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
SEED_ASSET_JSON="${SEED_ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/assets/v371_seed_residual_component_multi_micro_grouped_actuator_asset.json}"
VALIDATION_TSV="${VALIDATION_TSV:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_v373b_quota_rr_full_20260527_152104_bjt/group_validation.tsv}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"
MAX_ACTIONS="${MAX_ACTIONS:-12}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$CANDIDATE_CKPT" "$DATA_ROOT" "$SEED_ASSET_JSON" "$VALIDATION_TSV" "$COMPONENT_CSV" "$POINT_CSV"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$EXP_ROOT" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

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
  "STAGEB_SPLIT_CHILD_DEBUG=1"
)

log_event "write_action_list_start" "$ACTION_LIST"
"$PYTHON_BIN" - "$VALIDATION_TSV" "$ACTION_LIST" "$MAX_ACTIONS" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(float(sys.argv[3]))
rows = []
with src.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if not str(row.get("image_name", "")).startswith("c23_"):
            continue
        rows.append(row)
rows.sort(key=lambda r: (float(r.get("target_gain") or 0.0), r.get("image_name", ""), r.get("pair_id", "")))
if limit > 0:
    rows = rows[:limit]
with dst.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["pair_id", "image_name"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
print(f"wrote {dst} rows={len(rows)}")
PY
log_event "write_action_list_done" "$ACTION_LIST"

printf 'pair_id\timage_name\tappended\tactivation_fail\tpair_fail\tself_protect_drop\tinput_children\tinput_pair_actions\tdebug_log\n' > "$SUMMARY"
while IFS=$'\t' read -r pair_id image_name; do
  if [ "$pair_id" = "pair_id" ]; then
    continue
  fi
  view="${image_name#c}"
  view="${view%%_*}"
  frame="${image_name##*_f}"
  frame_int="$((10#$frame))"
  frame_next="$((frame_int + 1))"
  safe_pair="$(printf '%s' "$pair_id" | tr -c 'A-Za-z0-9_' '_')"
  render_exp="$EXP_ROOT/$safe_pair"
  render_log="$LOG_DIR/render_${safe_pair}.log"
  log_event "diagnose_start" "$pair_id"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$CANDIDATE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
    "dataset.train_frames=[0,570,60]" \
    "dataset.test_views.view=[$view]" \
    "dataset.test_frames.view=[$frame_int,$frame_next,1]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=false" \
    "++render_export_refine=false" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
    "++pipeline.split_child_component_enable=true" \
    "++pipeline.split_child_component_asset_json=$SEED_ASSET_JSON" \
    "++pipeline.split_child_component_action_filter=pair_id=$pair_id" \
    "++pipeline.split_child_component_action_required=true" \
    "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
    "++pipeline.split_child_component_radius_scale=1.0" \
    "++pipeline.split_child_component_max_children=-1" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$safe_pair" \
    "wandb_disable=true" \
    > "$render_log" 2>&1 || true
  "$PYTHON_BIN" - "$SUMMARY" "$pair_id" "$image_name" "$render_log" <<'PY'
import ast
import csv
import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
pair_id = sys.argv[2]
image_name = sys.argv[3]
log_path = Path(sys.argv[4])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
counters = {}
for match in re.finditer(r"counters=(\{.*?\})(?: rows=|$)", text):
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        continue
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            try:
                counters[key] = counters.get(key, 0) + int(value)
            except Exception:
                pass
keys = ["appended", "activation_fail", "pair_fail", "self_protect_drop", "input_children", "input_pair_actions"]
with summary.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow([pair_id, image_name] + [counters.get(key, 0) for key in keys] + [str(log_path)])
PY
  log_event "diagnose_done" "$pair_id"
done < "$ACTION_LIST"

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "finished_bjt" "$END_BJT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
