#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export EXPORT_INTERPRETABILITY="${EXPORT_INTERPRETABILITY:-true}"
export EXPORT_EDITABLE="${EXPORT_EDITABLE:-true}"
export EXPORT_OPACITY="${EXPORT_OPACITY:-false}"
export RUN_ID="${RUN_ID:-formal_377_signed_geometry_export_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
export EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_signed_geometry_export_${RUN_ID}}"

exec "$ROOT/tools/run_377_formal_signed_geometry_render.sh" "$@"
