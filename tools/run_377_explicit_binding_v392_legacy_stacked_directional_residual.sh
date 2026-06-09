#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

RUN_ID="${RUN_ID:-v392_legacy_stacked_directional_residual_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v392_legacy_stacked_directional_residual_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v392_legacy_stacked_directional_residual_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$ROOT/exp/formal/377_v388_raw_config_boundary_residual_tune_v388_raw_config_boundary_residual_tune_20260530_171041_bjt/best_ckpt.pth}"
ASSET_JSON="${ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v387_runtime_bounded_marginal_selector_v387_runtime_bounded_marginal_selector_20260530_094841_bjt/v387_selector/assets/v387_final_runtime_bounded_marginal_selector_asset.json}"

BOUNDARY_DIRECTION_TAG_ENABLE="${BOUNDARY_DIRECTION_TAG_ENABLE:-true}"
BOUNDARY_DIRECTION_CONFLICT_MODE="${BOUNDARY_DIRECTION_CONFLICT_MODE:-freeze}"
BOUNDARY_DIRECTION_TAG_TOPK_RATIO="${BOUNDARY_DIRECTION_TAG_TOPK_RATIO:-0.032}"
BOUNDARY_DIRECTION_TAG_MIN_RATIO="${BOUNDARY_DIRECTION_TAG_MIN_RATIO:-0.0}"
BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL="${BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL:-40}"
BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER="${BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER:-1700}"
BOUNDARY_DIRECTION_REG_UNION_ENABLE="${BOUNDARY_DIRECTION_REG_UNION_ENABLE:-false}"
BOUNDARY_DIRECTION_DOMINANCE_MARGIN="${BOUNDARY_DIRECTION_DOMINANCE_MARGIN:-1.15}"
BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS="${BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS:-0.025}"
BOUNDARY_GROW_OPACITY_RESIDUAL_LR="${BOUNDARY_GROW_OPACITY_RESIDUAL_LR:-0.000020}"
BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR="${BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR:-0.000020}"
EXTRA_TRAIN_ARGS_VALUE="${EXTRA_TRAIN_ARGS_VALUE:-++model.gaussian.directional_boundary_residual_enable=true ++model.gaussian.directional_boundary_include_legacy_residual_enable=true ++model.gaussian.directional_boundary_residual_sign_clamp_enable=true ++model.gaussian.directional_boundary_residual_conflict_mode=freeze ++opt.directional_boundary_residual_enable=true ++opt.directional_boundary_freeze_base_gaussian_for_boundary_loss=true ++opt.directional_boundary_grow_residual_scale=1.0 ++opt.directional_boundary_shrink_residual_scale=1.0 ++opt.directional_boundary_grow_scaling_residual_scale=0.0 ++opt.directional_boundary_shrink_scaling_residual_scale=0.0 ++opt.boundary_opacity_residual_lr=0.0 ++opt.boundary_scaling_residual_lr=0.0 ++opt.boundary_grow_opacity_residual_lr=$BOUNDARY_GROW_OPACITY_RESIDUAL_LR ++opt.boundary_shrink_opacity_residual_lr=$BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR ++opt.boundary_grow_scaling_residual_lr=0.0 ++opt.boundary_shrink_scaling_residual_lr=0.0 ++opt.boundary_direction_tag_enable=$BOUNDARY_DIRECTION_TAG_ENABLE ++opt.boundary_direction_conflict_mode=$BOUNDARY_DIRECTION_CONFLICT_MODE ++opt.boundary_direction_tag_topk_ratio=$BOUNDARY_DIRECTION_TAG_TOPK_RATIO ++opt.boundary_direction_tag_min_ratio=$BOUNDARY_DIRECTION_TAG_MIN_RATIO ++opt.boundary_direction_tag_update_interval=$BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL ++opt.boundary_direction_tag_update_until_iter=$BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER ++opt.boundary_direction_reg_union_enable=$BOUNDARY_DIRECTION_REG_UNION_ENABLE ++opt.boundary_direction_dominance_gate_enable=true ++opt.boundary_direction_dominance_margin=$BOUNDARY_DIRECTION_DOMINANCE_MARGIN ++opt.boundary_direction_dominance_min_abs=$BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS ++opt.boundary_tag_topk_ratio=$BOUNDARY_DIRECTION_TAG_TOPK_RATIO ++opt.boundary_tag_min_ratio=$BOUNDARY_DIRECTION_TAG_MIN_RATIO}"

TRAIN_STEPS="${TRAIN_STEPS:-2000}"
TEST_INTERVAL="${TEST_INTERVAL:-250}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[250,500,750,1000,1250,1500,1750,2000]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[250,500,750,1000,1250,1500,1750,2000]}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

V388_TRAIN_SCRIPT="$ROOT/tools/run_377_explicit_binding_v388_raw_config_boundary_residual_tune.sh"
RAW_GATE_SCRIPT="$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh"
SELECTOR_SUMMARY="$LOG_DIR/v392_raw_contour_checkpoint_summary.tsv"
SELECTED_JSON="$LOG_DIR/v392_selected_checkpoint.json"
EVENTS="$LOG_DIR/events.tsv"

for required in \
  "$PYTHON_BIN" "$V388_TRAIN_SCRIPT" "$RAW_GATE_SCRIPT" "$BASE_EXP/.hydra/config.yaml" \
  "$BASE_CKPT" "$ASSET_JSON" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_DIR" "$LOG_DIR" "$HYDRA_RUN_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+125 minutes' '+%F %T BJT')"
cat > "$LOG_DIR/run_info.txt" <<INFO
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
ASSET_JSON=$ASSET_JSON
EXP_DIR=$EXP_DIR
LOG_DIR=$LOG_DIR
TRAIN_STEPS=$TRAIN_STEPS
TEST_INTERVAL=$TEST_INTERVAL
SAVE_ITERATIONS=$SAVE_ITERATIONS
CHECKPOINT_ITERATIONS=$CHECKPOINT_ITERATIONS
BOUNDARY_DIRECTION_TAG_ENABLE=$BOUNDARY_DIRECTION_TAG_ENABLE
BOUNDARY_DIRECTION_CONFLICT_MODE=$BOUNDARY_DIRECTION_CONFLICT_MODE
BOUNDARY_DIRECTION_TAG_TOPK_RATIO=$BOUNDARY_DIRECTION_TAG_TOPK_RATIO
BOUNDARY_DIRECTION_TAG_MIN_RATIO=$BOUNDARY_DIRECTION_TAG_MIN_RATIO
BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL=$BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL
BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER=$BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER
BOUNDARY_DIRECTION_REG_UNION_ENABLE=$BOUNDARY_DIRECTION_REG_UNION_ENABLE
BOUNDARY_DIRECTION_DOMINANCE_MARGIN=$BOUNDARY_DIRECTION_DOMINANCE_MARGIN
BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS=$BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS
BOUNDARY_GROW_OPACITY_RESIDUAL_LR=$BOUNDARY_GROW_OPACITY_RESIDUAL_LR
BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR=$BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR
CONFIG_CONTRACT=v388_best_legacy_residual_stacked_with_dominance_gated_directional_opacity_residual
INFO

log_event train_start "$EXP_DIR"
RUN_ID="$RUN_ID" \
EXP_DIR="$EXP_DIR" \
LOG_DIR="$LOG_DIR" \
HYDRA_RUN_DIR="$HYDRA_RUN_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
GPU="$GPU" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
DATA_ROOT="$DATA_ROOT" \
BASE_EXP="$BASE_EXP" \
BASE_CKPT="$BASE_CKPT" \
ASSET_JSON="$ASSET_JSON" \
TRAIN_STEPS="$TRAIN_STEPS" \
TEST_INTERVAL="$TEST_INTERVAL" \
SAVE_ITERATIONS="$SAVE_ITERATIONS" \
CHECKPOINT_ITERATIONS="$CHECKPOINT_ITERATIONS" \
TRAIN_VIEWS_SPEC="$TRAIN_VIEWS_SPEC" \
TRAIN_FRAMES_SPEC="$TRAIN_FRAMES_SPEC" \
TEST_VIEWS_SPEC="$TEST_VIEWS_SPEC" \
TEST_FRAMES_SPEC="$TEST_FRAMES_SPEC" \
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS_VALUE" \
"$V388_TRAIN_SCRIPT"
log_event train_done "$EXP_DIR"

printf 'checkpoint\titeration\tstatus\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\topacity_inner_delta\topacity_outer_delta\traw_gate_summary\n' > "$SELECTOR_SUMMARY"

mapfile -t checkpoints < <(find "$EXP_DIR" -maxdepth 1 -type f -name 'ckpt*.pth' | sort)
if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "no checkpoints found in $EXP_DIR" >&2
  exit 3
fi

for ckpt in "${checkpoints[@]}"; do
  ckpt_name="$(basename "$ckpt" .pth)"
  iter="${ckpt_name#ckpt}"
  gate_run_id="formal_377_v392_${ckpt_name}_raw_gate_${RUN_ID}"
  gate_log="$LOG_DIR/raw_gate_${ckpt_name}.log"
  gate_summary="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${gate_run_id}/summary.tsv"
  variant="candidate_v392_${ckpt_name}_directional_split_opacity_residual"

  log_event raw_gate_start "$ckpt"
  RUN_ID="$gate_run_id" \
  CANDIDATE_VARIANT_NAME="$variant" \
  CANDIDATE_CKPT="$ckpt" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
  CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY=0.045 \
  CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
  CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
  GPU="$GPU" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  BASE_EXP="$BASE_EXP" \
  "$RAW_GATE_SCRIPT" > "$gate_log" 2>&1
  log_event raw_gate_done "$gate_summary"

  "$PYTHON_BIN" - "$SELECTOR_SUMMARY" "$ckpt" "$iter" "$gate_summary" <<'PY'
import csv
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
ckpt = sys.argv[2]
iteration = sys.argv[3]
summary = Path(sys.argv[4])

with summary.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
candidate = rows[-1]
with out_path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow([
        ckpt,
        iteration,
        candidate["status"],
        candidate["fg_delta_base"],
        candidate["boundary_delta_base"],
        candidate["edge_delta_base"],
        candidate["inner_delta_base"],
        candidate["outer_delta_base"],
        candidate["hard_delta_base"],
        candidate["opacity_inner_delta_base"],
        candidate["opacity_outer_delta_base"],
        str(summary),
    ])
PY
done

"$PYTHON_BIN" - "$SELECTOR_SUMMARY" "$SELECTED_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
selected_path = Path(sys.argv[2])
rows = []
with summary_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        for key in [
            "fg_delta", "boundary_delta", "edge_delta", "inner_delta", "outer_delta",
            "hard_delta", "opacity_inner_delta", "opacity_outer_delta",
        ]:
            row[key] = float(row[key])
        rows.append(row)

strict = [
    row for row in rows
    if row["status"] == "strict_pass"
    and row["fg_delta"] <= 0.0
    and row["boundary_delta"] <= 0.0
    and row["edge_delta"] <= 0.0
]
pool = strict or [row for row in rows if row["status"] == "strict_pass"] or rows

def rank_key(row):
    inner_shortfall = max(row["inner_delta"] - (-5.4333), 0.0)
    return (
        inner_shortfall,
        row["hard_delta"],
        row["outer_delta"],
        row["opacity_outer_delta"],
        row["edge_delta"],
    )

selected = min(pool, key=rank_key) if pool else None
payload = {
    "selector": "strict_raw_contour_then_inner_v392_directional_split_opacity",
    "summary": str(summary_path),
    "selected": selected,
    "num_candidates": len(rows),
    "num_strict_edge_safe": len(strict),
}
selected_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event all_done "$SELECTED_JSON"

echo "EXP_DIR=$EXP_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "SELECTOR_SUMMARY=$SELECTOR_SUMMARY"
echo "SELECTED_JSON=$SELECTED_JSON"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
