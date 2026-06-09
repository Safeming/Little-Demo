#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ID="${RUN_ID:-v322_boundary_color_locked_paired_signed_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"

export RUN_ID
export TRAIN_ITERS="${TRAIN_ITERS:-100}"
export TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-50,100}"
export FEATURE_LR="${FEATURE_LR:-0.00002}"
export TEXTURE_LR="${TEXTURE_LR:-0.0}"

export BOUNDARY_COLOR_PROTECT_ENABLE="${BOUNDARY_COLOR_PROTECT_ENABLE:-true}"
export BOUNDARY_COLOR_PROTECT_VERBOSE="${BOUNDARY_COLOR_PROTECT_VERBOSE:-false}"
export SCREEN_COLOR_PROTECT_ENABLE="${SCREEN_COLOR_PROTECT_ENABLE:-true}"
export SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH="${SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH:-17}"
export SCREEN_COLOR_PROTECT_OUTER_START="${SCREEN_COLOR_PROTECT_OUTER_START:-1}"
export SCREEN_COLOR_PROTECT_OUTER_END="${SCREEN_COLOR_PROTECT_OUTER_END:-38}"
export SCREEN_COLOR_PROTECT_RADIUS_PAD="${SCREEN_COLOR_PROTECT_RADIUS_PAD:-5}"
export SCREEN_COLOR_PROTECT_MIN_RADIUS="${SCREEN_COLOR_PROTECT_MIN_RADIUS:-0.0}"
export SCREEN_COLOR_PROTECT_MAX_POINTS="${SCREEN_COLOR_PROTECT_MAX_POINTS:-0}"

export EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v322_boundary_color_locked_paired_signed_${RUN_ID}}"
export LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v322_boundary_color_locked_paired_signed_${RUN_ID}}"

exec bash "$ROOT/tools/run_377_explicit_binding_v321_teacher_locked_paired_signed_train.sh"
