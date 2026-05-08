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

# A4: w/o temporal
run_and_log ablate_notemporal_train \
  "$PYTHON_BIN" train.py \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  opt.iterations=15000 \
  opt.lambda_binding_temporal=0.0 \
  model.deformer.rigid.temporal_momentum=0.0 \
  model.deformer.rigid.temporal_cache_size=1 \
  save_iterations=[5000,10000,15000] \
  checkpoint_iterations=[5000,10000,15000] \
  test_interval=5000 \
  tag=ablate_notemporal_15k

LATEST_NOTEMP=$(ls -dt exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_notemporal_15k-* | head -n 1)
echo "[INFO] A4 dir: $LATEST_NOTEMP"

run_and_log ablate_notemporal_render \
  "$PYTHON_BIN" render.py \
  mode=test \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  load_ckpt="$LATEST_NOTEMP/ckpt15000.pth" \
  opt.iterations=15000 \
  +exp_dir="$LATEST_NOTEMP"

run_and_log ablate_notemporal_interp \
  "$PYTHON_BIN" tools/run_full_interpretability_pipeline.py \
  --main-exp "$LATEST_NOTEMP" \
  --copy-assets

# A5: w/o semantic
run_and_log ablate_nosemantic_train \
  "$PYTHON_BIN" train.py \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  opt.iterations=15000 \
  opt.lambda_binding_semantic=0.0 \
  model.deformer.rigid.semantic_skinning_weight=0.0 \
  model.deformer.rigid.semantic_normal_weight=0.0 \
  model.deformer.rigid.semantic_prior_weight=0.0 \
  save_iterations=[5000,10000,15000] \
  checkpoint_iterations=[5000,10000,15000] \
  test_interval=5000 \
  tag=ablate_nosemantic_15k

LATEST_NOSEM=$(ls -dt exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-ablate_nosemantic_15k-* | head -n 1)
echo "[INFO] A5 dir: $LATEST_NOSEM"

run_and_log ablate_nosemantic_render \
  "$PYTHON_BIN" render.py \
  mode=test \
  dataset=zjumocap_377_mono \
  rigid=explicit_binding \
  non_rigid=hashgrid \
  pose_correction=direct \
  texture=shallow_mlp \
  wandb_disable=true \
  load_ckpt="$LATEST_NOSEM/ckpt15000.pth" \
  opt.iterations=15000 \
  +exp_dir="$LATEST_NOSEM"

run_and_log ablate_nosemantic_interp \
  "$PYTHON_BIN" tools/run_full_interpretability_pipeline.py \
  --main-exp "$LATEST_NOSEM" \
  --skip-video \
  --copy-assets

echo "[DONE] Ablation follow-up experiments finished. Logs are under $LOG_DIR"
