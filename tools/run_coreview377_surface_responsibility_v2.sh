#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_RUN="${BASE_RUN:-$ROOT/exp/zero_train_to_v395/coreview377_surface_responsibility_v2_20260716_bjt}"
RUN="${RUN:-$BASE_RUN/run_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')_bjt}"
BASE_LAUNCHER="$ROOT/exp/zero_train_to_v395/coreview377_neutral_longhorizon_20260710_bjt/launch_neutral_longhorizon.sh"
SEED="${SEED:-20260716}"
SMOKE="${SMOKE:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-12000}"
CONVERTER_LR_MAX_STEPS="${CONVERTER_LR_MAX_STEPS:-64000}"
SMOKE_CONVERTER_LR_MAX_STEPS=64
TESTS="[1000,2000,3000,4000,5000,6500,8000,10000,12000]"
SAVES="[5000,8000,12000]"
DIAGNOSTIC_ITERS="5000 8000 12000"
EXTRA_OPTIONS="stageA_377_multiview_explicit_hq_fromzero_surface_responsibility_v2"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-} ++dataset.init_point_count=80000 ++dataset.init_sampling_mode=surface_carrier_v1 ++dataset.init_surface_seed=$SEED ++dataset.init_surface_head_fraction=0.10 ++dataset.init_surface_shoulder_fraction=0.22 ++dataset.init_surface_hand_fraction=0.08 ++dataset.init_surface_tangent_scale_factor=1.8 ++dataset.init_surface_normal_scale_ratio=0.25 opt.densify_until_iter=0 opt.densify_from_iter=1000000 opt.densification_interval=1000000 opt.opacity_reset_interval=1000000 opt.percent_dense=0.0 opt.densify_prune_min_points=80000"

if [[ "$SMOKE" == "1" ]]; then
  EXTRA_OVERRIDES="$EXTRA_OVERRIDES opt.local_anchor_tether_log_interval=5 opt.local_anchor_tether_metric_gate_enable=false ++opt.surface_carrier_metric_gate_enable=false"
fi

mkdir -p "$RUN/logs"
printf '%s\n' "$RUN" > "$BASE_RUN/latest_run.txt"
printf '%s\n' "$$" > "$RUN/logs/surface_responsibility_v2.pid"

export BASE_RUN
export RUN
export SEED
export SMOKE
export TRAIN_ITERS
export CONVERTER_LR_MAX_STEPS
export SMOKE_CONVERTER_LR_MAX_STEPS
export TESTS
export SAVES
export DIAGNOSTIC_ITERS
export EXTRA_OPTIONS
export EXTRA_OVERRIDES

echo "[surface-responsibility-v2] RUN=$RUN"
echo "[surface-responsibility-v2] PIPELINE_START_BJT=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
exec "$BASE_LAUNCHER"
