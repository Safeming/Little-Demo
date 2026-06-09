#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

RUN_ID="${RUN_ID:-v396_generalized_boundary_controller_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v396_generalized_boundary_controller_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v396_generalized_boundary_controller_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$ROOT/exp/formal/377_v392_legacy_stacked_directional_residual_v392_legacy_stacked_directional_residual_20260531_125104_bjt/ckpt140160.pth}"
ASSET_JSON="${ASSET_JSON:-$ROOT/exp/stageB/logs/377_explicit_binding_v387_runtime_bounded_marginal_selector_v387_runtime_bounded_marginal_selector_20260530_094841_bjt/v387_selector/assets/v387_final_runtime_bounded_marginal_selector_asset.json}"

TRAIN_STEPS="${TRAIN_STEPS:-1500}"
TEST_INTERVAL="${TEST_INTERVAL:-250}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-[250,500,750,1000,1250,1500]}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-[250,500,750,1000,1250,1500]}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

BOUNDARY_DIRECTION_TAG_ENABLE="${BOUNDARY_DIRECTION_TAG_ENABLE:-true}"
BOUNDARY_DIRECTION_CONFLICT_MODE="${BOUNDARY_DIRECTION_CONFLICT_MODE:-freeze}"
BOUNDARY_DIRECTION_TAG_TOPK_RATIO="${BOUNDARY_DIRECTION_TAG_TOPK_RATIO:-0.040}"
BOUNDARY_DIRECTION_TAG_MIN_RATIO="${BOUNDARY_DIRECTION_TAG_MIN_RATIO:-0.0}"
BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL="${BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL:-40}"
BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER="${BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER:-1200}"
BOUNDARY_DIRECTION_REG_UNION_ENABLE="${BOUNDARY_DIRECTION_REG_UNION_ENABLE:-false}"
BOUNDARY_DIRECTION_DOMINANCE_MARGIN="${BOUNDARY_DIRECTION_DOMINANCE_MARGIN:-1.20}"
BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS="${BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS:-0.035}"
BOUNDARY_GROW_OPACITY_RESIDUAL_LR="${BOUNDARY_GROW_OPACITY_RESIDUAL_LR:-0.000024}"
BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR="${BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR:-0.000018}"

V392_INNER_FLOOR="${V392_INNER_FLOOR:--5.4333}"
V392_OUTER_FLOOR="${V392_OUTER_FLOOR:--1.6333}"
V392_HARD_FLOOR="${V392_HARD_FLOOR:--0.00023074}"
V392_OPACITY_OUTER_FLOOR="${V392_OPACITY_OUTER_FLOOR:--26.8667}"
V392_UNDER_JACCARD_FLOOR="${V392_UNDER_JACCARD_FLOOR:-0.95}"
V392_OVER_JACCARD_FLOOR="${V392_OVER_JACCARD_FLOOR:-0.95}"
V392_ADOPTED_LOST_MAX="${V392_ADOPTED_LOST_MAX:-0}"

SELECTOR_NAME="${SELECTOR_NAME:-v396}"
SUPPORT_BANK_DIAGNOSTIC_ENABLE="${SUPPORT_BANK_DIAGNOSTIC_ENABLE:-false}"
SUPPORT_BANK_SELECTOR_ENABLE="${SUPPORT_BANK_SELECTOR_ENABLE:-false}"
SUPPORT_BANK_TRAIN_ENABLE="${SUPPORT_BANK_TRAIN_ENABLE:-false}"
SUPPORT_BANK_BASELINE_CKPT="${SUPPORT_BANK_BASELINE_CKPT:-$BASE_CKPT}"
SUPPORT_BANK_SUMMARY="${SUPPORT_BANK_SUMMARY:-$LOG_DIR/support_bank_summary.tsv}"
BAD_FRAME_SELECTOR_ENABLE="${BAD_FRAME_SELECTOR_ENABLE:-false}"
BAD_FRAME_OUTER_VETO="${BAD_FRAME_OUTER_VETO:-5.0}"
BAD_FRAME_HARD_VETO="${BAD_FRAME_HARD_VETO:-0.0}"
BAD_FRAME_HARD_PENALTY="${BAD_FRAME_HARD_PENALTY:-0.00005}"
BAD_FRAME_FG_POSITIVE_MAX="${BAD_FRAME_FG_POSITIVE_MAX:-0}"
BAD_FRAME_BOUNDARY_POSITIVE_MAX="${BAD_FRAME_BOUNDARY_POSITIVE_MAX:-0}"
BAD_FRAME_EDGE_POSITIVE_MAX="${BAD_FRAME_EDGE_POSITIVE_MAX:-0}"
BAD_FRAME_SELECTOR_MODE="${BAD_FRAME_SELECTOR_MODE:-veto}"
BAD_FRAME_OUTER_HARD_VETO="${BAD_FRAME_OUTER_HARD_VETO:-$BAD_FRAME_OUTER_VETO}"
BAD_FRAME_HARD_HARD_VETO="${BAD_FRAME_HARD_HARD_VETO:-$BAD_FRAME_HARD_VETO}"
BAD_FRAME_IMAGE_SUMMARY="${BAD_FRAME_IMAGE_SUMMARY:-$LOG_DIR/bad_frame_image_summary.tsv}"
V399_UNDER_EFFECTIVE_RATIO="${V399_UNDER_EFFECTIVE_RATIO:-}"
V399_UNDER_NEW_ONLY_RATIO="${V399_UNDER_NEW_ONLY_RATIO:-}"
V399_OVER_EFFECTIVE_RATIO="${V399_OVER_EFFECTIVE_RATIO:-}"
V399_OVER_NEW_ONLY_RATIO="${V399_OVER_NEW_ONLY_RATIO:-}"
STABLE_WINDOW_TARGET="${STABLE_WINDOW_TARGET:-3}"
INHERITED_EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

EXTRA_TRAIN_ARGS_VALUE="${EXTRA_TRAIN_ARGS_VALUE:-++model.gaussian.directional_boundary_residual_enable=true ++model.gaussian.directional_boundary_include_legacy_residual_enable=true ++model.gaussian.directional_boundary_residual_sign_clamp_enable=true ++model.gaussian.directional_boundary_residual_conflict_mode=freeze ++opt.directional_boundary_residual_enable=true ++opt.directional_boundary_freeze_base_gaussian_for_boundary_loss=true ++opt.directional_boundary_grow_residual_scale=1.0 ++opt.directional_boundary_shrink_residual_scale=1.0 ++opt.directional_boundary_grow_scaling_residual_scale=0.0 ++opt.directional_boundary_shrink_scaling_residual_scale=0.0 ++opt.boundary_opacity_residual_lr=0.0 ++opt.boundary_scaling_residual_lr=0.0 ++opt.boundary_grow_opacity_residual_lr=$BOUNDARY_GROW_OPACITY_RESIDUAL_LR ++opt.boundary_shrink_opacity_residual_lr=$BOUNDARY_SHRINK_OPACITY_RESIDUAL_LR ++opt.boundary_grow_scaling_residual_lr=0.0 ++opt.boundary_shrink_scaling_residual_lr=0.0 ++opt.boundary_direction_tag_enable=$BOUNDARY_DIRECTION_TAG_ENABLE ++opt.boundary_direction_conflict_mode=$BOUNDARY_DIRECTION_CONFLICT_MODE ++opt.boundary_direction_tag_topk_ratio=$BOUNDARY_DIRECTION_TAG_TOPK_RATIO ++opt.boundary_direction_tag_min_ratio=$BOUNDARY_DIRECTION_TAG_MIN_RATIO ++opt.boundary_direction_tag_update_interval=$BOUNDARY_DIRECTION_TAG_UPDATE_INTERVAL ++opt.boundary_direction_tag_update_until_iter=$BOUNDARY_DIRECTION_TAG_UPDATE_UNTIL_ITER ++opt.boundary_direction_reg_union_enable=$BOUNDARY_DIRECTION_REG_UNION_ENABLE ++opt.boundary_direction_dominance_gate_enable=true ++opt.boundary_direction_dominance_margin=$BOUNDARY_DIRECTION_DOMINANCE_MARGIN ++opt.boundary_direction_dominance_min_abs=$BOUNDARY_DIRECTION_DOMINANCE_MIN_ABS ++opt.boundary_tag_topk_ratio=$BOUNDARY_DIRECTION_TAG_TOPK_RATIO ++opt.boundary_tag_min_ratio=$BOUNDARY_DIRECTION_TAG_MIN_RATIO ++opt.boundary_signed_shrink_loss_scale=1.05 ++opt.boundary_signed_grow_loss_scale=0.34 ++opt.boundary_signed_share_gain=1.04 ++opt.boundary_signed_shrink_share_gain=1.16 ++opt.lambda_silhouette_outer_shell=[0.010,1,0.016,600,0.020,1200,0.024] ++opt.lambda_silhouette_upper_torso_outer_shell=[0.010,1,0.020,600,0.024,1200,0.026] ++opt.lambda_silhouette_shoulder_arm_outer_shell=[0.008,1,0.016,600,0.020,1200,0.022] ++opt.lambda_silhouette_inner=[0.005,1,0.010,600,0.013,1200,0.015]}"
if [ "$SUPPORT_BANK_TRAIN_ENABLE" = "true" ]; then
  EXTRA_TRAIN_ARGS_VALUE="$EXTRA_TRAIN_ARGS_VALUE ++opt.boundary_support_bank_enable=true ++opt.boundary_support_bank_key=image ++opt.boundary_support_bank_ema=0.72 ++opt.boundary_support_bank_score_threshold=0.50 ++opt.boundary_support_bank_promote_init_iter=80 ++opt.boundary_support_bank_promote_interval=40 ++opt.boundary_support_bank_promote_until_iter=1200 ++opt.boundary_support_bank_min_hits=2 ++opt.boundary_support_bank_min_views=2 ++opt.boundary_support_bank_min_frames=1 ++opt.boundary_support_bank_promote_threshold=0.60 ++opt.boundary_support_bank_dominance_margin=$BOUNDARY_DIRECTION_DOMINANCE_MARGIN ++opt.boundary_support_bank_verbose=true"
fi
if [ -n "$INHERITED_EXTRA_TRAIN_ARGS" ]; then
  EXTRA_TRAIN_ARGS_VALUE="$EXTRA_TRAIN_ARGS_VALUE $INHERITED_EXTRA_TRAIN_ARGS"
fi

V388_TRAIN_SCRIPT="$ROOT/tools/run_377_explicit_binding_v388_raw_config_boundary_residual_tune.sh"
RAW_GATE_SCRIPT="$ROOT/tools/formal/run_377_v338_raw_contour_gate.sh"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$LOG_DIR/${SELECTOR_NAME}_raw_contour_checkpoint_summary.tsv}"
SELECTED_JSON="${SELECTED_JSON:-$LOG_DIR/${SELECTOR_NAME}_selected_checkpoint.json}"
SELECTED_CHECKPOINT_PATH_TXT="${SELECTED_CHECKPOINT_PATH_TXT:-$LOG_DIR/selected_checkpoint_path.txt}"
SELECTED_CHECKPOINT_METRICS_JSON="${SELECTED_CHECKPOINT_METRICS_JSON:-$LOG_DIR/selected_checkpoint_metrics.json}"
SELECTED_CKPT_LINK="${SELECTED_CKPT_LINK:-$EXP_DIR/selected_ckpt.pth}"
SELECTOR_SCHEMA_NAME="${SELECTOR_SCHEMA_NAME:-v396_strict_raw_contour_v392_safety_floor_then_inner}"
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
EST_END_BJT="$(TZ=Asia/Shanghai date -d '+150 minutes' '+%F %T BJT')"
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
V392_INNER_FLOOR=$V392_INNER_FLOOR
V392_OUTER_FLOOR=$V392_OUTER_FLOOR
V392_HARD_FLOOR=$V392_HARD_FLOOR
V392_OPACITY_OUTER_FLOOR=$V392_OPACITY_OUTER_FLOOR
V392_UNDER_JACCARD_FLOOR=$V392_UNDER_JACCARD_FLOOR
V392_OVER_JACCARD_FLOOR=$V392_OVER_JACCARD_FLOOR
V392_ADOPTED_LOST_MAX=$V392_ADOPTED_LOST_MAX
SUPPORT_BANK_TRAIN_ENABLE=$SUPPORT_BANK_TRAIN_ENABLE
SUPPORT_BANK_DIAGNOSTIC_ENABLE=$SUPPORT_BANK_DIAGNOSTIC_ENABLE
SUPPORT_BANK_SELECTOR_ENABLE=$SUPPORT_BANK_SELECTOR_ENABLE
SUPPORT_BANK_BASELINE_CKPT=$SUPPORT_BANK_BASELINE_CKPT
SUPPORT_BANK_SUMMARY=$SUPPORT_BANK_SUMMARY
BAD_FRAME_SELECTOR_ENABLE=$BAD_FRAME_SELECTOR_ENABLE
BAD_FRAME_OUTER_VETO=$BAD_FRAME_OUTER_VETO
BAD_FRAME_HARD_VETO=$BAD_FRAME_HARD_VETO
BAD_FRAME_HARD_PENALTY=$BAD_FRAME_HARD_PENALTY
BAD_FRAME_FG_POSITIVE_MAX=$BAD_FRAME_FG_POSITIVE_MAX
BAD_FRAME_BOUNDARY_POSITIVE_MAX=$BAD_FRAME_BOUNDARY_POSITIVE_MAX
BAD_FRAME_EDGE_POSITIVE_MAX=$BAD_FRAME_EDGE_POSITIVE_MAX
BAD_FRAME_SELECTOR_MODE=$BAD_FRAME_SELECTOR_MODE
BAD_FRAME_OUTER_HARD_VETO=$BAD_FRAME_OUTER_HARD_VETO
BAD_FRAME_HARD_HARD_VETO=$BAD_FRAME_HARD_HARD_VETO
BAD_FRAME_IMAGE_SUMMARY=$BAD_FRAME_IMAGE_SUMMARY
V399_UNDER_EFFECTIVE_RATIO=$V399_UNDER_EFFECTIVE_RATIO
V399_UNDER_NEW_ONLY_RATIO=$V399_UNDER_NEW_ONLY_RATIO
V399_OVER_EFFECTIVE_RATIO=$V399_OVER_EFFECTIVE_RATIO
V399_OVER_NEW_ONLY_RATIO=$V399_OVER_NEW_ONLY_RATIO
STABLE_WINDOW_TARGET=$STABLE_WINDOW_TARGET
CONFIG_CONTRACT=v396_continue_v392_signed_boundary_controller_with_raw_gate_checkpoint_selector
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

printf 'checkpoint\titeration\tstatus\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\topacity_inner_delta\topacity_outer_delta\traw_gate_summary\tbad_frame_summary\n' > "$SELECTOR_SUMMARY"

mapfile -t checkpoints < <(find "$EXP_DIR" -maxdepth 1 -type f -name 'ckpt*.pth' | sort)
if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "no checkpoints found in $EXP_DIR" >&2
  exit 3
fi

for ckpt in "${checkpoints[@]}"; do
  ckpt_name="$(basename "$ckpt" .pth)"
  iter="${ckpt_name#ckpt}"
  gate_run_id="formal_377_v396_${ckpt_name}_raw_gate_${RUN_ID}"
  gate_log="$LOG_DIR/raw_gate_${ckpt_name}.log"
  gate_log_dir="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${gate_run_id}"
  gate_exp_root="$ROOT/exp/formal/377_v338_raw_contour_gate_${gate_run_id}"
  gate_summary="$gate_log_dir/summary.tsv"
  gate_worst="$gate_log_dir/worst_frames.tsv"
  variant="candidate_v396_${ckpt_name}_generalized_boundary_controller"

  log_event raw_gate_start "$ckpt"
  RUN_ID="$gate_run_id" \
  EXP_ROOT="$gate_exp_root" \
  LOG_DIR="$gate_log_dir" \
  HYDRA_RUN_ROOT="$gate_log_dir/hydra_runtime" \
  CANDIDATE_VARIANT_NAME="$variant" \
  CANDIDATE_CKPT="$ckpt" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE=true \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON="$ASSET_JSON" \
  CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED=false \
  CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY=0.045 \
  CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE=1.0 \
  CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN=-1 \
  CANDIDATE_DIRECTIONAL_BOUNDARY_RESIDUAL_ENABLE=true \
  CANDIDATE_DIRECTIONAL_BOUNDARY_INCLUDE_LEGACY_RESIDUAL_ENABLE=true \
  CANDIDATE_DIRECTIONAL_BOUNDARY_RESIDUAL_SIGN_CLAMP_ENABLE=true \
  CANDIDATE_DIRECTIONAL_BOUNDARY_RESIDUAL_CONFLICT_MODE=freeze \
  GPU="$GPU" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  BASE_EXP="$BASE_EXP" \
  "$RAW_GATE_SCRIPT" > "$gate_log" 2>&1
  log_event raw_gate_done "$gate_summary"

  "$PYTHON_BIN" - "$SELECTOR_SUMMARY" "$ckpt" "$iter" "$gate_summary" "$gate_worst" <<'PY'
import csv
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
ckpt = sys.argv[2]
iteration = sys.argv[3]
summary = Path(sys.argv[4])
bad_frame_summary = Path(sys.argv[5])

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
        str(bad_frame_summary),
    ])
PY
  if [ "$SUPPORT_BANK_DIAGNOSTIC_ENABLE" = "true" ]; then
    "$PYTHON_BIN" tools/analyze_377_boundary_support_bank.py \
      --baseline "$SUPPORT_BANK_BASELINE_CKPT" \
      --checkpoint "$ckpt" \
      --out-tsv "$LOG_DIR/support_bank_${ckpt_name}.tsv" \
      --out-json "$LOG_DIR/support_bank_${ckpt_name}.json"
    if [ ! -s "$SUPPORT_BANK_SUMMARY" ]; then
      cat "$LOG_DIR/support_bank_${ckpt_name}.tsv" > "$SUPPORT_BANK_SUMMARY"
    else
      tail -n +2 "$LOG_DIR/support_bank_${ckpt_name}.tsv" >> "$SUPPORT_BANK_SUMMARY"
    fi
  fi
done

"$PYTHON_BIN" - "$SELECTOR_SUMMARY" "$SELECTED_JSON" "$V392_INNER_FLOOR" "$V392_OUTER_FLOOR" "$V392_HARD_FLOOR" "$V392_OPACITY_OUTER_FLOOR" "$SUPPORT_BANK_SELECTOR_ENABLE" "$SUPPORT_BANK_SUMMARY" "$V392_UNDER_JACCARD_FLOOR" "$V392_OVER_JACCARD_FLOOR" "$V392_ADOPTED_LOST_MAX" "$BAD_FRAME_SELECTOR_ENABLE" "$BAD_FRAME_OUTER_VETO" "$BAD_FRAME_HARD_VETO" "$BAD_FRAME_HARD_PENALTY" "$BAD_FRAME_FG_POSITIVE_MAX" "$BAD_FRAME_BOUNDARY_POSITIVE_MAX" "$BAD_FRAME_EDGE_POSITIVE_MAX" "$STABLE_WINDOW_TARGET" "$SELECTED_CHECKPOINT_PATH_TXT" "$SELECTED_CHECKPOINT_METRICS_JSON" "$SELECTED_CKPT_LINK" "$SELECTOR_SCHEMA_NAME" "$BAD_FRAME_SELECTOR_MODE" "$BAD_FRAME_OUTER_HARD_VETO" "$BAD_FRAME_HARD_HARD_VETO" "$BAD_FRAME_IMAGE_SUMMARY" "$V399_UNDER_EFFECTIVE_RATIO" "$V399_UNDER_NEW_ONLY_RATIO" "$V399_OVER_EFFECTIVE_RATIO" "$V399_OVER_NEW_ONLY_RATIO" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
selected_path = Path(sys.argv[2])
inner_floor = float(sys.argv[3])
outer_floor = float(sys.argv[4])
hard_floor = float(sys.argv[5])
opacity_outer_floor = float(sys.argv[6])
support_selector_enable = str(sys.argv[7]).lower() == "true"
support_summary = Path(sys.argv[8])
under_floor = float(sys.argv[9])
over_floor = float(sys.argv[10])
lost_max = int(float(sys.argv[11]))
bad_frame_selector_enable = str(sys.argv[12]).lower() == "true"
bad_frame_outer_veto = float(sys.argv[13])
bad_frame_hard_veto = float(sys.argv[14])
bad_frame_hard_penalty = float(sys.argv[15])
bad_frame_fg_positive_max = int(float(sys.argv[16]))
bad_frame_boundary_positive_max = int(float(sys.argv[17]))
bad_frame_edge_positive_max = int(float(sys.argv[18]))
stable_window_target = int(float(sys.argv[19]))
selected_path_txt = Path(sys.argv[20])
selected_metrics_json = Path(sys.argv[21])
selected_ckpt_link = Path(sys.argv[22])
selector_schema_name = str(sys.argv[23])
bad_frame_selector_mode = str(sys.argv[24]).lower()
bad_frame_outer_hard_veto = float(sys.argv[25])
bad_frame_hard_hard_veto = float(sys.argv[26])
bad_frame_image_summary = Path(sys.argv[27])
under_effective_ratio_arg = str(sys.argv[28])
under_new_only_ratio_arg = str(sys.argv[29])
over_effective_ratio_arg = str(sys.argv[30])
over_new_only_ratio_arg = str(sys.argv[31])


def _float_value(row, key):
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _optional_float(value):
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except ValueError:
        return None


under_effective_ratio = _optional_float(under_effective_ratio_arg)
under_new_only_ratio = _optional_float(under_new_only_ratio_arg)
over_effective_ratio = _optional_float(over_effective_ratio_arg)
over_new_only_ratio = _optional_float(over_new_only_ratio_arg)


def load_bad_frame_stats(path):
    path = Path(path)
    stats = {
        "worst_frames_summary": str(path),
        "bad_frame_max_outer_delta": 0.0,
        "bad_frame_max_hard_delta": 0.0,
        "bad_frame_hard_penalty_count": 0,
        "bad_frame_fg_positive_count": 0,
        "bad_frame_boundary_positive_count": 0,
        "bad_frame_edge_positive_count": 0,
        "bad_frame_reject_reasons": [],
        "bad_frame_hard_veto_pass": True,
        "bad_frame_penalty": 0.0,
        "bad_frame_penalty_reasons": [],
        "candidate_bad_frame_rows": [],
    }
    if not path.exists():
        stats["bad_frame_reject_reasons"].append("missing_worst_frames")
        return stats
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not str(row.get("variant", "")).startswith("candidate_"):
                continue
            stats["candidate_bad_frame_rows"].append(dict(row))
            outer = _float_value(row, "outer_delta")
            hard = _float_value(row, "hard_delta")
            fg = _float_value(row, "fg_delta")
            boundary = _float_value(row, "boundary_delta")
            edge = _float_value(row, "edge_delta")
            stats["bad_frame_max_outer_delta"] = max(stats["bad_frame_max_outer_delta"], outer)
            stats["bad_frame_max_hard_delta"] = max(stats["bad_frame_max_hard_delta"], hard)
            stats["bad_frame_hard_penalty_count"] += int(hard > bad_frame_hard_penalty)
            stats["bad_frame_fg_positive_count"] += int(fg > 0.0)
            stats["bad_frame_boundary_positive_count"] += int(boundary > 0.0)
            stats["bad_frame_edge_positive_count"] += int(edge > 0.0)
    if bad_frame_selector_mode == "penalty":
        if stats["bad_frame_max_outer_delta"] > bad_frame_outer_hard_veto:
            stats["bad_frame_hard_veto_pass"] = False
            stats["bad_frame_reject_reasons"].append("outer_hard_veto")
        if stats["bad_frame_max_hard_delta"] > bad_frame_hard_hard_veto:
            stats["bad_frame_hard_veto_pass"] = False
            stats["bad_frame_reject_reasons"].append("hard_hard_veto")
        if stats["bad_frame_max_hard_delta"] > 0.0:
            stats["bad_frame_penalty"] += stats["bad_frame_max_hard_delta"] * 10000.0
            stats["bad_frame_penalty_reasons"].append("hard_delta_positive_penalty")
        if stats["bad_frame_hard_penalty_count"] > 0:
            stats["bad_frame_penalty"] += stats["bad_frame_hard_penalty_count"] * 0.25
        if stats["bad_frame_fg_positive_count"] > 0:
            stats["bad_frame_penalty"] += stats["bad_frame_fg_positive_count"] * 0.05
            stats["bad_frame_penalty_reasons"].append("fg_positive_count_penalty")
        if stats["bad_frame_boundary_positive_count"] > 0:
            stats["bad_frame_penalty"] += stats["bad_frame_boundary_positive_count"] * 0.05
            stats["bad_frame_penalty_reasons"].append("boundary_positive_count_penalty")
        if stats["bad_frame_edge_positive_count"] > 0:
            stats["bad_frame_penalty"] += stats["bad_frame_edge_positive_count"] * 0.05
            stats["bad_frame_penalty_reasons"].append("edge_positive_count_penalty")
    else:
        if stats["bad_frame_max_outer_delta"] > bad_frame_outer_veto:
            stats["bad_frame_reject_reasons"].append("outer_veto")
        if stats["bad_frame_max_hard_delta"] > bad_frame_hard_veto:
            stats["bad_frame_reject_reasons"].append("hard_veto")
        if stats["bad_frame_fg_positive_count"] > bad_frame_fg_positive_max:
            stats["bad_frame_reject_reasons"].append("fg_positive_count")
        if stats["bad_frame_boundary_positive_count"] > bad_frame_boundary_positive_max:
            stats["bad_frame_reject_reasons"].append("boundary_positive_count")
        if stats["bad_frame_edge_positive_count"] > bad_frame_edge_positive_max:
            stats["bad_frame_reject_reasons"].append("edge_positive_count")
    return stats

rows = []
with summary_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        for key in [
            "fg_delta", "boundary_delta", "edge_delta", "inner_delta", "outer_delta",
            "hard_delta", "opacity_inner_delta", "opacity_outer_delta",
        ]:
            row[key] = float(row[key])
        rows.append(row)

support_by_ckpt = {}
if support_selector_enable and support_summary.exists():
    with support_summary.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = support_by_ckpt.setdefault(row["checkpoint"], {})
            direction = row["direction"]
            item[f"{direction}_jaccard"] = float(row["jaccard"])
            item[f"{direction}_adopted_lost"] = int(float(row["adopted_lost"]))
            item[f"{direction}_effective_count"] = int(float(row["effective_count"]))
            item[f"{direction}_new_only"] = int(float(row["new_only"]))
            item[f"{direction}_adopted_count"] = int(float(row["adopted_count"]))


def _saturation(count, ratio, point_count):
    if ratio is None or point_count <= 0:
        return None
    limit = max(1, int(point_count * ratio))
    return float(count) / float(limit)

for row in rows:
    support = support_by_ckpt.get(row["checkpoint"], {})
    row.update(support)
    point_count = max(
        int(row.get("under_effective_count", 0) or 0),
        int(row.get("over_effective_count", 0) or 0),
    )
    row["support_under_effective_cap_saturation"] = _saturation(
        int(row.get("under_effective_count", 0) or 0),
        under_effective_ratio,
        point_count,
    )
    row["support_over_effective_cap_saturation"] = _saturation(
        int(row.get("over_effective_count", 0) or 0),
        over_effective_ratio,
        point_count,
    )
    row["support_under_cap_saturation"] = _saturation(
        int(row.get("under_new_only", 0) or 0),
        under_new_only_ratio,
        point_count,
    )
    row["support_over_cap_saturation"] = _saturation(
        int(row.get("over_new_only", 0) or 0),
        over_new_only_ratio,
        point_count,
    )
    row["support_cap_saturated"] = any(
        value is not None and value >= 0.999
        for value in (
            row["support_under_effective_cap_saturation"],
            row["support_over_effective_cap_saturation"],
            row["support_under_cap_saturation"],
            row["support_over_cap_saturation"],
        )
    )
    bad_frame_stats = load_bad_frame_stats(row.get("bad_frame_summary", ""))
    row.update(bad_frame_stats)
    if bad_frame_selector_mode == "penalty":
        row["bad_frame_gate_pass"] = (
            not bad_frame_selector_enable
            or row.get("bad_frame_hard_veto_pass", True)
        )
    else:
        row["bad_frame_gate_pass"] = (
            not bad_frame_selector_enable
            or len(bad_frame_stats["bad_frame_reject_reasons"]) == 0
        )
    row["support_floor_pass"] = (
        not support_selector_enable
        or (
            row.get("under_adopted_lost", 999999) <= lost_max
            and row.get("over_adopted_lost", 999999) <= lost_max
        )
    )

def failure_reasons(row):
    reasons = []
    if not (
        row["status"] == "strict_pass"
        and row["fg_delta"] <= 0.0
        and row["boundary_delta"] <= 0.0
        and row["edge_delta"] <= 0.0
    ):
        reasons.append("fg_boundary_edge_regression")
    if not row.get("support_floor_pass", True):
        reasons.append("support_floor_miss")
    if not (
        row["outer_delta"] <= outer_floor
        and row["hard_delta"] <= hard_floor
        and row["opacity_outer_delta"] <= opacity_outer_floor
    ):
        reasons.append("v392_floor_miss")
    if not row.get("bad_frame_gate_pass", True):
        reasons.append("bad_frame_veto")
    if row.get("support_cap_saturated", False):
        reasons.append("cap_saturation")
    return reasons

for row in rows:
    row["failure_reasons"] = failure_reasons(row)

failure_reason_counts = {}
for row in rows:
    for reason in row.get("failure_reasons", []):
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1

image_failure_aggregate = {}
for row in rows:
    for bad_row in row.get("candidate_bad_frame_rows", []):
        image = str(bad_row.get("image", ""))
        item = image_failure_aggregate.setdefault(image, {
            "image": image,
            "count": 0,
            "worsen_score_sum": 0.0,
            "worsen_score_max": 0.0,
            "outer_delta_max": 0.0,
            "hard_delta_max": 0.0,
            "fg_positive_count": 0,
            "boundary_positive_count": 0,
            "edge_positive_count": 0,
        })
        item["count"] += 1
        worsen = _float_value(bad_row, "worsen_score")
        item["worsen_score_sum"] += worsen
        item["worsen_score_max"] = max(item["worsen_score_max"], worsen)
        item["outer_delta_max"] = max(item["outer_delta_max"], _float_value(bad_row, "outer_delta"))
        item["hard_delta_max"] = max(item["hard_delta_max"], _float_value(bad_row, "hard_delta"))
        item["fg_positive_count"] += int(_float_value(bad_row, "fg_delta") > 0.0)
        item["boundary_positive_count"] += int(_float_value(bad_row, "boundary_delta") > 0.0)
        item["edge_positive_count"] += int(_float_value(bad_row, "edge_delta") > 0.0)

top_bad_frame_images = sorted(
    image_failure_aggregate.values(),
    key=lambda item: (-item["worsen_score_sum"], item["image"]),
)[:10]
bad_frame_image_summary.parent.mkdir(parents=True, exist_ok=True)
with bad_frame_image_summary.open("w", encoding="utf-8", newline="") as handle:
    fields = [
        "image", "count", "worsen_score_sum", "worsen_score_max",
        "outer_delta_max", "hard_delta_max",
        "fg_positive_count", "boundary_positive_count", "edge_positive_count",
    ]
    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
    writer.writeheader()
    for item in sorted(image_failure_aggregate.values(), key=lambda x: (-x["worsen_score_sum"], x["image"])):
        writer.writerow(item)

for row in rows:
    row.pop("candidate_bad_frame_rows", None)

strict = [
    row for row in rows
    if row["status"] == "strict_pass"
    and row["fg_delta"] <= 0.0
    and row["boundary_delta"] <= 0.0
    and row["edge_delta"] <= 0.0
]
v392_safe = [
    row for row in strict
    if row["support_floor_pass"]
    and row.get("bad_frame_gate_pass", True)
    and row["outer_delta"] <= outer_floor
    and row["hard_delta"] <= hard_floor
    and row["opacity_outer_delta"] <= opacity_outer_floor
]
stable = [
    row for row in v392_safe
    if row.get("bad_frame_gate_pass", True)
]
stable_window_pass = len(stable) >= stable_window_target
stable_ids = {id(row) for row in stable}


def _iteration(row):
    try:
        return int(float(row.get("iteration", 0)))
    except ValueError:
        return 0


ordered_rows = sorted(rows, key=_iteration)
stable_neighbors = set()
for prev_row, next_row in zip(ordered_rows, ordered_rows[1:]):
    if prev_row in stable and next_row in stable:
        stable_neighbors.add(_iteration(prev_row))
        stable_neighbors.add(_iteration(next_row))
for row in rows:
    row["stable_window_member"] = _iteration(row) in stable_neighbors

if support_selector_enable:
    pool = stable or v392_safe
else:
    pool = stable or v392_safe or strict or [row for row in rows if row["status"] == "strict_pass"] or rows

def rank_key(row):
    safety_miss = (
        max(row["outer_delta"] - outer_floor, 0.0) * 10.0
        + max(row["hard_delta"] - hard_floor, 0.0) * 10000.0
        + max(row["opacity_outer_delta"] - opacity_outer_floor, 0.0) * 0.05
    )
    inner_miss = max(row["inner_delta"] - inner_floor, 0.0)
    return (
        0 if row.get("stable_window_member", False) else 1,
        safety_miss,
        row.get("bad_frame_penalty", 0.0),
        inner_miss,
        row["inner_delta"],
        row["hard_delta"],
        row["outer_delta"],
        row["opacity_outer_delta"],
        row["edge_delta"],
    )

selected = min(pool, key=rank_key) if pool else None
selected_support = selected or {}
artifact_mode = None
if selected is not None:
    selected_checkpoint = Path(selected["checkpoint"])
    selected_path_txt.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    selected_ckpt_link.parent.mkdir(parents=True, exist_ok=True)
    selected_path_txt.write_text(str(selected_checkpoint) + "\n", encoding="utf-8")
    if selected_ckpt_link.exists() or selected_ckpt_link.is_symlink():
        selected_ckpt_link.unlink()
    try:
        selected_ckpt_link.symlink_to(selected_checkpoint)
        artifact_mode = "symlink"
    except OSError:
        import shutil
        shutil.copy2(selected_checkpoint, selected_ckpt_link)
        artifact_mode = "copy"

payload = {
    "selector": selector_schema_name,
    "summary": str(summary_path),
    "selected": selected,
    "reject_reason": None if selected is not None else "no_checkpoint_passed_v392_metric_and_adopted_support_floors",
    "counts": {
        "num_candidates": len(rows),
        "num_strict_edge_safe": len(strict),
        "num_v392_safety_floor": len(v392_safe),
        "num_support_floor": len([row for row in rows if row.get("support_floor_pass")]),
        "num_bad_frame_gate": len([row for row in rows if row.get("bad_frame_gate_pass", True)]),
        "num_stable_window_pass": len(stable),
    },
    "num_candidates": len(rows),
    "num_strict_edge_safe": len(strict),
    "num_v392_safety_floor": len(v392_safe),
    "num_support_floor": len([row for row in rows if row.get("support_floor_pass")]),
    "num_stable_window_pass": len(stable),
    "stable_window_target": stable_window_target,
    "stable_window_pass": stable_window_pass,
    "stability_warning": None if stable_window_pass else "fewer_than_target_stable_checkpoints",
    "artifacts": {
        "selected_checkpoint_path_txt": str(selected_path_txt),
        "selected_checkpoint_metrics_json": str(selected_metrics_json),
        "selected_ckpt": str(selected_ckpt_link),
        "selected_artifact_mode": artifact_mode,
    },
    "v392_floor": {
        "inner_delta": inner_floor,
        "outer_delta": outer_floor,
        "hard_delta": hard_floor,
        "opacity_outer_delta": opacity_outer_floor,
        "adopted_lost_max": lost_max,
    },
    "support_diagnostics": {
        "jaccard_reference": "diagnostic_only",
        "under_jaccard_floor_legacy_config": under_floor,
        "over_jaccard_floor_legacy_config": over_floor,
        "selected_under_jaccard": selected_support.get("under_jaccard"),
        "selected_over_jaccard": selected_support.get("over_jaccard"),
        "selected_under_new_only": selected_support.get("under_new_only"),
        "selected_over_new_only": selected_support.get("over_new_only"),
        "selected_under_adopted_lost": selected_support.get("under_adopted_lost"),
        "selected_over_adopted_lost": selected_support.get("over_adopted_lost"),
    },
    "failure_reason_counts": failure_reason_counts,
    "cap_diagnostics": {
        "selected_support_cap_saturated": selected_support.get("support_cap_saturated"),
        "selected_support_under_effective_cap_saturation": selected_support.get("support_under_effective_cap_saturation"),
        "selected_support_over_effective_cap_saturation": selected_support.get("support_over_effective_cap_saturation"),
        "selected_support_under_cap_saturation": selected_support.get("support_under_cap_saturation"),
        "selected_support_over_cap_saturation": selected_support.get("support_over_cap_saturation"),
        "under_effective_ratio": under_effective_ratio,
        "under_new_only_ratio": under_new_only_ratio,
        "over_effective_ratio": over_effective_ratio,
        "over_new_only_ratio": over_new_only_ratio,
    },
    "bad_frame_diagnostics": {
        "selected_bad_frame_gate_pass": selected_support.get("bad_frame_gate_pass"),
        "selected_bad_frame_reject_reasons": selected_support.get("bad_frame_reject_reasons"),
        "selected_bad_frame_hard_veto_pass": selected_support.get("bad_frame_hard_veto_pass"),
        "selected_bad_frame_penalty": selected_support.get("bad_frame_penalty"),
        "selected_bad_frame_penalty_reasons": selected_support.get("bad_frame_penalty_reasons"),
        "selected_bad_frame_max_outer_delta": selected_support.get("bad_frame_max_outer_delta"),
        "selected_bad_frame_max_hard_delta": selected_support.get("bad_frame_max_hard_delta"),
        "selected_bad_frame_fg_positive_count": selected_support.get("bad_frame_fg_positive_count"),
        "selected_bad_frame_boundary_positive_count": selected_support.get("bad_frame_boundary_positive_count"),
        "selected_bad_frame_edge_positive_count": selected_support.get("bad_frame_edge_positive_count"),
        "bad_frame_image_summary": str(bad_frame_image_summary),
        "top_bad_frame_images": top_bad_frame_images,
    },
}
selected_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
selected_metrics_json.write_text(json.dumps({
    "selected": selected,
    "support_diagnostics": payload.get("support_diagnostics", {}),
    "cap_diagnostics": payload.get("cap_diagnostics", {}),
    "failure_reason_counts": payload.get("failure_reason_counts", {}),
    "bad_frame_diagnostics": payload.get("bad_frame_diagnostics", {}),
    "stable_window": {
        "num_stable_window_pass": payload.get("num_stable_window_pass", 0),
        "stable_window_target": payload.get("stable_window_target", 0),
        "stable_window_pass": payload.get("stable_window_pass", False),
    },
    "artifacts": payload.get("artifacts", {}),
}, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SELECTOR_SUMMARY=$SELECTOR_SUMMARY"
  echo "SELECTED_JSON=$SELECTED_JSON"
} >> "$LOG_DIR/run_info.txt"
log_event all_done "$SELECTED_JSON"

echo "EXP_DIR=$EXP_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "SELECTOR_SUMMARY=$SELECTOR_SUMMARY"
echo "SELECTED_JSON=$SELECTED_JSON"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
