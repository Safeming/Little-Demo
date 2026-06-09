#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# v373 fixes the v372 failure mode: seed candidates are quota-limited per
# image/source/target so c21 high-score frames cannot consume the full pool.
RUN_ID="${RUN_ID:-v373_quota_coverage_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"

export RUN_ID
export RUN_LABEL="${RUN_LABEL:-v373}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-v373_quota_coverage_grouped_actuator}"
export RAW_GATE_CANDIDATE_VARIANT="${RAW_GATE_CANDIDATE_VARIANT:-candidate_v373_quota_coverage_grouped_actuator}"
export TRAIN_RUN_PREFIX="${TRAIN_RUN_PREFIX:-formal_377_v373_quota_coverage_grouped_actuator_semantic_train}"
export TRAIN_EXP_PREFIX="${TRAIN_EXP_PREFIX:-377_v373_quota_coverage_grouped_actuator_semantic_train}"
export EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v373_quota_coverage_grouped_actuator_${RUN_ID}}"
export LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v373_quota_coverage_grouped_actuator_${RUN_ID}}"

export TOP_FRAMES="${TOP_FRAMES:-30}"
export INNER_PER_FRAME="${INNER_PER_FRAME:-4}"
export OUTER_PER_INNER="${OUTER_PER_INNER:-1}"
export MAX_GROUPS="${MAX_GROUPS:-360}"
export MAX_GROUPS_PER_IMAGE="${MAX_GROUPS_PER_IMAGE:-18}"
export MAX_GROUPS_PER_SOURCE="${MAX_GROUPS_PER_SOURCE:-6}"
export MAX_GROUPS_PER_TARGET="${MAX_GROUPS_PER_TARGET:-3}"

export RESIDUAL_TARGETS_PER_INNER="${RESIDUAL_TARGETS_PER_INNER:-3}"
export RESIDUAL_MIN_MASK_PIXELS="${RESIDUAL_MIN_MASK_PIXELS:-3}"
export RESIDUAL_MIN_GATE_OVERLAP="${RESIDUAL_MIN_GATE_OVERLAP:-1}"
export RESIDUAL_GATE_PAD_PX="${RESIDUAL_GATE_PAD_PX:-22.0}"

export MICRO_COUNTS="${MICRO_COUNTS:-3}"
export RADIUS_SCALES="${RADIUS_SCALES:-0.42,0.55}"
export MINOR_SCALES="${MINOR_SCALES:-0.42,0.60}"
export COVARIANCE_SCALES="${COVARIANCE_SCALES:-1.0}"
export CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
export CHILD_OPACITY_MODE="${CHILD_OPACITY_MODE:-divide}"

export STRENGTH_SWEEP_VARIANTS="${STRENGTH_SWEEP_VARIANTS:-base,co3_cr1.2_np,co3_cr1.5_np,co4_cr1.5_np}"

export GROUP_VALIDATE_MAX="${GROUP_VALIDATE_MAX:-180}"
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
