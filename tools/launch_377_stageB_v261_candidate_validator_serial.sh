#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-stageB_v261_candidate_validator_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_v261_candidate_validator_${RUN_ID}}"
mkdir -p "$LOG_DIR"

NOHUP_LOG="$LOG_DIR/nohup.log"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"

BASE_EXP_DEFAULT="$ROOT/exp/stageB/377_hulk_light_v233d_shoes_preserve_control_stageB_compact_v233_skincloth_20260512_161652_bjt"
BASE_CKPT_DEFAULT="$BASE_EXP_DEFAULT/ckpt135710.pth"
BASE_RENDER_EXP_DEFAULT="$ROOT/exp/stageB/377_hulk_light_v233d_stageC_softer_stageC_v235_v236_20260513_114036_bjt_render_compact_final"

env \
  RUN_ID="$RUN_ID" \
  GPU="$GPU" \
  LOG_DIR="$LOG_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_EXP="${BASE_EXP:-$BASE_EXP_DEFAULT}" \
  BASE_CKPT="${BASE_CKPT:-$BASE_CKPT_DEFAULT}" \
  BASE_ITER="${BASE_ITER:-135710}" \
  BASE_RENDER_EXP="${BASE_RENDER_EXP:-$BASE_RENDER_EXP_DEFAULT}" \
  WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}" \
  RUN_SHORT_TRAIN="${RUN_SHORT_TRAIN:-0}" \
  CANDIDATE_VIEWS="${CANDIDATE_VIEWS:-1,2,3,4,5,6,7,8,9,10,11,12}" \
  CANDIDATE_FRAMES="${CANDIDATE_FRAMES:-0,570,60}" \
  EVAL_VIEWS="${EVAL_VIEWS:-21,22,23}" \
  EVAL_FRAMES="${EVAL_FRAMES:-0,570,60}" \
  MIN_ACCEPTED_CANDIDATES="${MIN_ACCEPTED_CANDIDATES:-2}" \
  MIN_INNER_OUTER_RATIO="${MIN_INNER_OUTER_RATIO:-1.35}" \
  MIN_INNER_HIT_PIXELS="${MIN_INNER_HIT_PIXELS:-12.0}" \
  setsid bash "$ROOT/tools/run_377_stageB_v261_candidate_validator_serial.sh" \
  > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/launcher.pid"

{
  echo "RUN_ID=$RUN_ID"
  echo "PID=$PID"
  echo "GPU=$GPU"
  echo "LOG_DIR=$LOG_DIR"
  echo "NOHUP_LOG=$NOHUP_LOG"
  echo "BASE_CKPT=${BASE_CKPT:-$BASE_CKPT_DEFAULT}"
  echo "BASE_RENDER_EXP=${BASE_RENDER_EXP:-$BASE_RENDER_EXP_DEFAULT}"
  echo "RUN_SHORT_TRAIN=${RUN_SHORT_TRAIN:-0}"
  echo "EVENTS=$LOG_DIR/events.tsv"
  echo "SUMMARY=$LOG_DIR/summary.tsv"
  echo "STATUS_JSON=$LOG_DIR/status.json"
  echo "START_BJT=$(TZ=Asia/Shanghai date '+%F %T BJT')"
} | tee "$LOG_DIR/launcher_info.txt"
