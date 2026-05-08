#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

BASE_EXP="$ROOT/exp/377_multiview_explicit_hq_rootfix_resume_v179c_v178c_k6_camquality_tightphoto_screen3000"
START_CKPT="${START_CKPT:-$BASE_EXP/best_ckpt.pth}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageA2/logs/v185_replay_parity_$RUN_ID}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_case() {
  local label="$1"
  local local_context="$2"
  local exp_dir="$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v185_${label}_$RUN_ID"
  local metrics_path="$LOG_DIR/${label}_metrics.json"
  local log_path="$LOG_DIR/${label}.log"

  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] start $label local_context=$local_context" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" env PYTHONUNBUFFERED=1 "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$ROOT/data/ZJUMoCap" \
    "dataset.preload=false" \
    "start_checkpoint=$START_CKPT" \
    "exp_dir=$exp_dir" \
    "wandb_disable=true" \
    "opt.iterations=3000" \
    "++opt.diagnostic_validate_at_start=true" \
    "++opt.diagnostic_validate_interval_iteration=$local_context" \
    "++opt.diagnostic_validation_metrics_path=$metrics_path" \
    "++opt.diagnostic_exit_after_start_validation=true" \
    "++validation_image_log_limit=0" \
    > "$log_path" 2>&1
  echo "[$(TZ=Asia/Shanghai date '+%F %T BJT')] done $label metrics=$metrics_path log=$log_path" | tee -a "$LOG_DIR/queue.log"
}

run_case "global_schedule_replay" "global"
run_case "local3000_replay" "3000"
run_case "local0_resume_start" "0"

"$PYTHON_BIN" - "$LOG_DIR" "$BASE_EXP/best_test_metrics.json" <<'PY'
import json
import os
import sys

log_dir, baseline_path = sys.argv[1:3]
with open(baseline_path) as f:
    baseline = json.load(f)

rows = []
for label in ("global_schedule_replay", "local3000_replay", "local0_resume_start"):
    metrics_path = os.path.join(log_dir, f"{label}_metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    best_eval = metrics.get("best_eval") or metrics.get("test") or {}
    rows.append({
        "label": label,
        "psnr": best_eval.get("psnr"),
        "lpips": best_eval.get("lpips"),
        "psnr_fg": best_eval.get("psnr_fg"),
        "lpips_fg": best_eval.get("lpips_fg"),
        "delta_lpips_fg_vs_v179c": (
            best_eval.get("lpips_fg") - baseline.get("lpips_fg")
            if best_eval.get("lpips_fg") is not None else None
        ),
    })

summary_path = os.path.join(log_dir, "summary.tsv")
with open(summary_path, "w") as f:
    f.write("label\tpsnr\tlpips\tpsnr_fg\tlpips_fg\tdelta_lpips_fg_vs_v179c\n")
    for row in rows:
        f.write(
            "{label}\t{psnr:.6f}\t{lpips:.6f}\t{psnr_fg:.6f}\t{lpips_fg:.6f}\t{delta_lpips_fg_vs_v179c:.6f}\n".format(
                **row
            )
        )

print(f"summary={summary_path}")
for row in rows:
    print(
        "{label}: lpips_fg={lpips_fg:.6f}, delta={delta_lpips_fg_vs_v179c:.6f}".format(
            **row
        )
    )
PY
