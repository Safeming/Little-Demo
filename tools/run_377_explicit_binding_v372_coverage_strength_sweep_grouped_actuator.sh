#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# v372 keeps the v371 action-level raw validation loop, but pushes coverage:
# more source frames/rows/targets plus finer strength/radius variants.
RUN_ID="${RUN_ID:-v372_coverage_strength_sweep_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"

export RUN_ID
export RUN_LABEL="${RUN_LABEL:-v372}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-v372_coverage_strength_sweep_grouped_actuator}"
export RAW_GATE_CANDIDATE_VARIANT="${RAW_GATE_CANDIDATE_VARIANT:-candidate_v372_coverage_strength_sweep_grouped_actuator}"
export TRAIN_RUN_PREFIX="${TRAIN_RUN_PREFIX:-formal_377_v372_coverage_strength_sweep_grouped_actuator_semantic_train}"
export TRAIN_EXP_PREFIX="${TRAIN_EXP_PREFIX:-377_v372_coverage_strength_sweep_grouped_actuator_semantic_train}"
export EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v372_coverage_strength_sweep_grouped_actuator_${RUN_ID}}"
export LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v372_coverage_strength_sweep_grouped_actuator_${RUN_ID}}"
export TOP_FRAMES="${TOP_FRAMES:-45}"
export INNER_PER_FRAME="${INNER_PER_FRAME:-5}"
export OUTER_PER_INNER="${OUTER_PER_INNER:-1}"
export MAX_GROUPS="${MAX_GROUPS:-320}"
export RESIDUAL_TARGETS_PER_INNER="${RESIDUAL_TARGETS_PER_INNER:-4}"
export RESIDUAL_MIN_MASK_PIXELS="${RESIDUAL_MIN_MASK_PIXELS:-3}"
export RESIDUAL_MIN_GATE_OVERLAP="${RESIDUAL_MIN_GATE_OVERLAP:-1}"
export RESIDUAL_GATE_PAD_PX="${RESIDUAL_GATE_PAD_PX:-22.0}"

export MICRO_COUNTS="${MICRO_COUNTS:-3,5}"
export RADIUS_SCALES="${RADIUS_SCALES:-0.42,0.55,0.68}"
export MINOR_SCALES="${MINOR_SCALES:-0.42,0.60,0.78}"
export COVARIANCE_SCALES="${COVARIANCE_SCALES:-1.0}"
export CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
export CHILD_OPACITY_MODE="${CHILD_OPACITY_MODE:-divide}"

# Keep co3/co4 winners from v371, but insert weaker/finer variants to avoid
# trading inner gains for outer/opacity regressions.
export STRENGTH_SWEEP_VARIANTS="${STRENGTH_SWEEP_VARIANTS:-base,co2_cr1.2_np,co2.5_cr1.2_np,co2.5_cr1.4_np,co3_cr1.2_np,co3_cr1.4_np,co3_cr1.5_np,co4_cr1.4_np,co4_cr1.5_np}"

export GROUP_VALIDATE_MAX="${GROUP_VALIDATE_MAX:-240}"
export GROUP_VALIDATE_TARGET_MIN_GAIN="${GROUP_VALIDATE_TARGET_MIN_GAIN:-0.10}"
export GROUP_VALIDATE_MAX_FG_REGRESS="${GROUP_VALIDATE_MAX_FG_REGRESS:-0.000002}"
export GROUP_VALIDATE_MAX_BOUNDARY_REGRESS="${GROUP_VALIDATE_MAX_BOUNDARY_REGRESS:-0.000002}"
export GROUP_VALIDATE_MAX_EDGE_REGRESS="${GROUP_VALIDATE_MAX_EDGE_REGRESS:-0.001}"
export GROUP_VALIDATE_MAX_COUNT_REGRESS="${GROUP_VALIDATE_MAX_COUNT_REGRESS:-0.0}"
export GROUP_VALIDATE_MAX_HARD_REGRESS="${GROUP_VALIDATE_MAX_HARD_REGRESS:-0.000001}"
export GROUP_VALIDATE_MAX_OPACITY_REGRESS="${GROUP_VALIDATE_MAX_OPACITY_REGRESS:-0.0}"

export RAW_GATE_ENABLE="${RAW_GATE_ENABLE:-true}"
export TRAIN_ON_STRICT_PASS="${TRAIN_ON_STRICT_PASS:-true}"
export TRAIN_STEPS="${TRAIN_STEPS:-2000}"

exec "$ROOT/tools/run_377_explicit_binding_v371_strength_sweep_grouped_actuator.sh" "$@"
