#!/usr/bin/env bash
set -uo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PY="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
START_CKPT="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v170e_v168a_tagfocus_photo_contour_screen3000/best_ckpt.pth"
START_SCRIPT="$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v169a_v168a_hardcontour_boundaryactuator.sh"

mkdir -p "$LOG_DIR"

BATCH_ID="${V172_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="$LOG_DIR/v172_batch_${BATCH_ID}_summary.tsv"
REGISTRY="$LOG_DIR/v172_batch_${BATCH_ID}_runs.txt"
RUNNER_LOG="$LOG_DIR/v172_batch_${BATCH_ID}.log"

printf '%s\n' "$RUNNER_LOG" > "$LOG_DIR/v172_latest_runner_log.txt"
printf '%s\n' "$SUMMARY" > "$LOG_DIR/v172_latest_summary.txt"
printf 'tag\tstatus\titeration\tlpips_fg\tl1_fg\tpsnr_fg\tssim_fg\texp\tlog\trender_exp\n' > "$SUMMARY"
: > "$REGISTRY"

run_render() {
  local tag="$1"
  local exp_dir="$2"
  local render_exp="${exp_dir}_render_bestcmp"
  local render_log="$LOG_DIR/${tag}_${BATCH_ID}_render.log"

  printf '%s\n' "$render_log" > "$LOG_DIR/v172_latest_render_log.txt"
  if [ ! -f "$exp_dir/best_ckpt.pth" ] || [ ! -f "$exp_dir/.hydra/config.yaml" ]; then
    echo "[$(date '+%F %T')] render skipped for $tag: missing best_ckpt or hydra config"
    return 0
  fi

  echo "[$(date '+%F %T')] render $tag -> $render_exp"
  "$PY" "$ROOT/render.py" \
    --config-path "$exp_dir/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$exp_dir/best_ckpt.pth" \
    "exp_dir=$render_exp" \
    wandb_disable=true \
    dataset.preload=false \
    > "$render_log" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] render $tag finished status=$status"
  return 0
}

append_metric_summary() {
  local tag="$1"
  local status="$2"
  local exp_dir="$3"
  local train_log="$4"
  local render_exp="${exp_dir}_render_bestcmp"

  "$PY" - "$tag" "$status" "$exp_dir" "$train_log" "$render_exp" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

tag, status, exp_dir, train_log, render_exp, summary = sys.argv[1:]
metrics_path = Path(exp_dir) / "best_test_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    row = [
        tag,
        status,
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
    row = [tag, status, "", "", "", "", "", exp_dir, train_log, render_exp]
with open(summary, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

run_one() {
  local tag="$1"
  local iterations="$2"
  local option_name="$3"
  local slug="$4"

  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${tag}_v170e_${slug}_screen${iterations}"
  local train_log="$LOG_DIR/${tag}_${BATCH_ID}_${slug}.log"

  printf '%s\t%s\t%s\t%s\n' "$tag" "$exp_dir" "$train_log" "$option_name" >> "$REGISTRY"
  printf '%s\n' "$exp_dir" > "$LOG_DIR/v172_latest_exp.txt"
  printf '%s\n' "$train_log" > "$LOG_DIR/v172_latest_log.txt"

  echo "[$(date '+%F %T')] start $tag iterations=$iterations option=$option_name"
  bash "$START_SCRIPT" \
    "$exp_dir" \
    "$START_CKPT" \
    "$iterations" \
    --option stageA_377_multiview_explicit_hq_v170e_v168a_tagfocus_photo_contour_v1 \
    --option "$option_name" \
    > "$train_log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "[$(date '+%F %T')] train failed $tag status=$status log=$train_log"
    append_metric_summary "$tag" "train_failed_$status" "$exp_dir" "$train_log"
    return 0
  fi

  echo "[$(date '+%F %T')] train complete $tag"
  run_render "$tag" "$exp_dir"
  append_metric_summary "$tag" "ok" "$exp_dir" "$train_log"
  echo "[$(date '+%F %T')] done $tag"
}

echo "[$(date '+%F %T')] v172 batch started"
echo "summary=$SUMMARY"
echo "registry=$REGISTRY"

run_one "v172a" 1 "stageA_377_multiview_explicit_hq_v172a_v170e_layersharpen_probe_v1" "layersharpen_probe"
run_one "v172b" 1500 "stageA_377_multiview_explicit_hq_v172b_v170e_layersharpen_boundaryoverride_v1" "layersharpen_boundaryoverride"

echo "[$(date '+%F %T')] v172 batch finished"
cat "$SUMMARY"
