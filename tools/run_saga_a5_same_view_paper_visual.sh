#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/exp/acceptdata/saga_a5_same_view_paper_visual_20260812}"
STATUS_FILE="$OUTPUT_ROOT/queue_status.txt"

mkdir -p "$OUTPUT_ROOT"
timestamp() { TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S BJT'; }
status() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$STATUS_FILE"; }

args=(
  tools/make_saga_a5_same_view_paper_visual.py
  --repo-root "$ROOT"
  --output-root "$OUTPUT_ROOT"
  --python-bin "$PYTHON_BIN"
  --gpu "$GPU"
)
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

status "SAGA/A5 same-view paper visualization started dry_run=$DRY_RUN"
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$OUTPUT_ROOT/orchestrator.log"
status "SAGA/A5 same-view paper visualization completed"
