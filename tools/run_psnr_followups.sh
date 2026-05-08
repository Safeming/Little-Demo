#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/anim/bin/python}"
LOG_DIR="$ROOT_DIR/exp/logs"
mkdir -p "$LOG_DIR"

run_and_log() {
  local name="$1"
  shift
  echo "[RUN] $name"
  echo "[CMD] $*"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

# Experiment B: relaxed regularization 15k
run_and_log psnr_relaxreg_15k \
  "$PYTHON_BIN" train.py \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  opt.iterations=15000 \
  opt.lambda_binding_temporal=0.03 \
  opt.lambda_binding_semantic=0.05 \
  opt.lambda_binding_surface=0.1 \
  opt.lambda_binding_canonical=0.3 \
  save_iterations=[5000,10000,15000] \
  checkpoint_iterations=[5000,10000,15000] \
  test_interval=5000 \
  tag=bodycloth_v41_relaxreg_15k

# Experiment C1: evaluate 30k ckpt10000
run_and_log psnr_render_30k_ckpt10000 \
  "$PYTHON_BIN" render.py \
  mode=test \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  load_ckpt=exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_30k_psnr-0312-0909/ckpt10000.pth \
  opt.iterations=10000 \
  +exp_dir=exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_30k_psnr_ckpt10000_eval

# Experiment C2: evaluate 30k ckpt20000
run_and_log psnr_render_30k_ckpt20000 \
  "$PYTHON_BIN" render.py \
  mode=test \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  load_ckpt=exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_30k_psnr-0312-0909/ckpt20000.pth \
  opt.iterations=20000 \
  +exp_dir=exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_30k_psnr_ckpt20000_eval

echo "[DONE] Follow-up PSNR experiments finished. Logs are under $LOG_DIR"
