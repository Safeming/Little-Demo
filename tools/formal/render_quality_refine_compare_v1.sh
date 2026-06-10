#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
SUBJECT="${SUBJECT:-CoreView_377}"
BEFORE_RENDER_DIR="${BEFORE_RENDER_DIR:-$ROOT/exp/acceptdata/v395_dense_canary_semantic_20260609_export/test-view/renders}"
AFTER_RENDER_DIR="${AFTER_RENDER_DIR:?set AFTER_RENDER_DIR to refined test-view/renders directory}"
RUN_ID="${RUN_ID:-render_quality_compare_v1_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
OUT_DIR="${OUT_DIR:-$ROOT/exp/acceptdata/${RUN_ID}}"

mkdir -p "$OUT_DIR/before" "$OUT_DIR/after"

"$PYTHON_BIN" tools/analyze_render_quality_edges.py \
  --render-dir "$BEFORE_RENDER_DIR" \
  --gt-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.jpg" \
  --mask-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.png" \
  --out-dir "$OUT_DIR/before" \
  --band-width 7 \
  --topk 12

"$PYTHON_BIN" tools/analyze_render_quality_edges.py \
  --render-dir "$AFTER_RENDER_DIR" \
  --gt-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.jpg" \
  --mask-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.png" \
  --out-dir "$OUT_DIR/after" \
  --band-width 7 \
  --topk 12

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
before = json.loads((out_dir / "before" / "render_quality_summary.json").read_text(encoding="utf-8"))
after = json.loads((out_dir / "after" / "render_quality_summary.json").read_text(encoding="utf-8"))

keys = [
    "mean_foreground_l1",
    "mean_boundary_l1",
    "mean_interior_l1",
    "mean_edge_symmetric_dist_px",
    "mean_render_minus_gt_luma_fg",
    "mean_halo_luma_outside",
    "mean_hard_score",
]
delta = {}
for key in keys:
    delta[key] = after.get(key, 0.0) - before.get(key, 0.0)

report = {
    "before": before,
    "after": after,
    "delta_after_minus_before": delta,
    "acceptance_hint": {
        "boundary_l1_should_drop": delta["mean_boundary_l1"] < 0,
        "foreground_l1_should_not_increase_more_than_0_003": delta["mean_foreground_l1"] <= 0.003,
        "halo_should_drop_or_stay": delta["mean_halo_luma_outside"] <= 0.0,
        "hard_score_should_drop": delta["mean_hard_score"] < 0,
    },
}
(out_dir / "compare_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["delta_after_minus_before"], indent=2))
PY

echo "COMPARE_OUT_DIR=$OUT_DIR"
