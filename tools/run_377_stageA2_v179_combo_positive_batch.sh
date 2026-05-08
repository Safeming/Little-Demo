#!/usr/bin/env bash
set -uo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PY="/opt/miniconda3/envs/anim/bin/python"
LOG_DIR="$ROOT/exp/stageA2/logs"
START_SCRIPT="$ROOT/tools/start_377_stageA2_multiview_explicit_rootfix_resume_v169a_v168a_hardcontour_boundaryactuator.sh"
CKPT_V178E="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v178e_v177a_k6_probe_screen3000/best_ckpt.pth"
CKPT_V178C="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v178c_v177a_k4_photo_consistency_screen3000/best_ckpt.pth"

mkdir -p "$LOG_DIR"

BATCH_ID="${V179_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY="$LOG_DIR/v179_batch_${BATCH_ID}_summary.tsv"
REGISTRY="$LOG_DIR/v179_batch_${BATCH_ID}_runs.txt"
RUNNER_LOG="$LOG_DIR/v179_batch_${BATCH_ID}.log"

printf '%s\n' "$RUNNER_LOG" > "$LOG_DIR/v179_latest_runner_log.txt"
printf '%s\n' "$SUMMARY" > "$LOG_DIR/v179_latest_summary.txt"
printf '%s\n' "$BATCH_ID" > "$LOG_DIR/v179_latest_run_id.txt"
printf 'tag\tstatus\titeration\tlpips_fg\tl1_fg\tpsnr_fg\tssim_fg\texp\tlog\trender_exp\n' > "$SUMMARY"
: > "$REGISTRY"

run_render() {
  local tag="$1"
  local exp_dir="$2"
  local render_exp="${exp_dir}_render_bestcmp"
  local render_log="$LOG_DIR/${tag}_${BATCH_ID}_render.log"

  printf '%s\n' "$render_log" > "$LOG_DIR/v179_latest_render_log.txt"
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
  local base_tag="$2"
  local start_ckpt="$3"
  local iterations="$4"
  local option_name="$5"
  local slug="$6"

  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_${tag}_${base_tag}_${slug}_screen${iterations}"
  local train_log="$LOG_DIR/${tag}_${BATCH_ID}_${slug}.log"

  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$exp_dir" "$train_log" "$option_name" "$start_ckpt" >> "$REGISTRY"
  printf '%s\n' "$exp_dir" > "$LOG_DIR/v179_latest_exp.txt"
  printf '%s\n' "$train_log" > "$LOG_DIR/v179_latest_log.txt"

  if [ ! -f "$start_ckpt" ]; then
    echo "[$(date '+%F %T')] missing start checkpoint for $tag: $start_ckpt"
    append_metric_summary "$tag" "missing_ckpt" "$exp_dir" "$train_log"
    return 0
  fi

  echo "[$(date '+%F %T')] start $tag iterations=$iterations option=$option_name"
  bash "$START_SCRIPT" \
    "$exp_dir" \
    "$start_ckpt" \
    "$iterations" \
    --option stageA_377_multiview_explicit_hq_v170e_v168a_tagfocus_photo_contour_v1 \
    --option stageA_377_multiview_explicit_hq_v175a_v170e_hfsource_edgesupervise_v1 \
    --option stageA_377_multiview_explicit_hq_v177a_v176a_sameframe_k4_v1 \
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

echo "[$(date '+%F %T')] v179 combo-positive batch started"
echo "summary=$SUMMARY"
echo "registry=$REGISTRY"
echo "ckpt_v178e=$CKPT_V178E"
echo "ckpt_v178c=$CKPT_V178C"

run_one "v179a" "v178e" "$CKPT_V178E" 3000 "stageA_377_multiview_explicit_hq_v179a_v178e_k6_camquality_focus_v1" "k6_camquality_focus"
run_one "v179b" "v178e" "$CKPT_V178E" 3000 "stageA_377_multiview_explicit_hq_v179b_v178e_k6_camquality_tightphoto_v1" "k6_camquality_tightphoto"
run_one "v179c" "v178c" "$CKPT_V178C" 3000 "stageA_377_multiview_explicit_hq_v179c_v178c_k6_camquality_tightphoto_v1" "k6_camquality_tightphoto"

echo "[$(date '+%F %T')] v179 combo-positive batch finished"
cat "$SUMMARY"
