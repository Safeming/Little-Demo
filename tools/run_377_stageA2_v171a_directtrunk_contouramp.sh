#!/usr/bin/env bash
set -uo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PY="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
EXP_DIR="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v171a_v170e_directtrunk_contouramp_screen2500"
START_CKPT="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v170e_v168a_tagfocus_photo_contour_screen3000/best_ckpt.pth"

mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$LOG_DIR/v171a_${RUN_ID}_directtrunk_contouramp.log"
RENDER_LOG="$LOG_DIR/v171a_${RUN_ID}_directtrunk_contouramp_render.log"
SUMMARY="$LOG_DIR/v171a_${RUN_ID}_summary.tsv"
RENDER_EXP="${EXP_DIR}_render_bestcmp"

printf '%s\n' "$EXP_DIR" > "$LOG_DIR/v171a_latest_exp.txt"
printf '%s\n' "$TRAIN_LOG" > "$LOG_DIR/v171a_latest_log.txt"
printf '%s\n' "$RENDER_LOG" > "$LOG_DIR/v171a_latest_render_log.txt"
printf '%s\n' "$SUMMARY" > "$LOG_DIR/v171a_latest_summary.txt"
printf 'tag\tstatus\titeration\tlpips_fg\tl1_fg\tpsnr_fg\tssim_fg\texp\tlog\trender_exp\n' > "$SUMMARY"

echo "[$(date '+%F %T')] v171a train start"
echo "exp=$EXP_DIR"
echo "log=$TRAIN_LOG"

bash "$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v171a_v170e_directtrunk_contouramp.sh" \
  "$EXP_DIR" \
  "$START_CKPT" \
  2500 \
  > "$TRAIN_LOG" 2>&1
TRAIN_STATUS=$?

echo "[$(date '+%F %T')] v171a train finished status=$TRAIN_STATUS"

if [ "$TRAIN_STATUS" -eq 0 ] && [ -f "$EXP_DIR/best_ckpt.pth" ] && [ -f "$EXP_DIR/.hydra/config.yaml" ]; then
  echo "[$(date '+%F %T')] v171a render start -> $RENDER_EXP"
  "$PY" "$ROOT/render.py" \
    --config-path "$EXP_DIR/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$EXP_DIR/best_ckpt.pth" \
    "exp_dir=$RENDER_EXP" \
    wandb_disable=true \
    dataset.preload=false \
    > "$RENDER_LOG" 2>&1
  RENDER_STATUS=$?
  echo "[$(date '+%F %T')] v171a render finished status=$RENDER_STATUS"
else
  echo "[$(date '+%F %T')] v171a render skipped"
fi

"$PY" - "$TRAIN_STATUS" "$EXP_DIR" "$TRAIN_LOG" "$RENDER_EXP" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

status, exp_dir, train_log, render_exp, summary = sys.argv[1:]
metrics_path = Path(exp_dir) / "best_test_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    row = [
        "v171a",
        "ok" if status == "0" else f"train_failed_{status}",
        str(metrics.get("iteration", "")),
        f'{float(metrics.get("lpips_fg", float("nan"))):.9f}',
        f'{float(metrics.get("l1_fg", float("nan"))):.9f}',
        f'{float(metrics.get("psnr_fg", float("nan"))):.6f}',
        f'{float(metrics.get("ssim_fg", float("nan"))):.9f}',
        exp_dir,
        train_log,
        render_exp,
    ]
else:
    row = [
        "v171a",
        "ok" if status == "0" else f"train_failed_{status}",
        "",
        "",
        "",
        "",
        "",
        exp_dir,
        train_log,
        render_exp,
    ]

with open(summary, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY

cat "$SUMMARY"
