#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote-home/ming/3dgs-avatar-release-main
PYTHON=/opt/miniconda3/envs/ictrl/bin/python
GPU=0
SUBJECT=377
SOURCE="$ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721"
OUTPUT_ROOT="$ROOT/exp/acceptdata/a7_renderer_objective_v3_canary_377"
RUN_ROOT="$OUTPUT_ROOT/run_377_v3"
EVIDENCE="$OUTPUT_ROOT/evidence/377/evidence.npz"
CANDIDATE_ROOT="$OUTPUT_ROOT/candidates/377"
A5_BANK="$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
A5_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
A7_CONTRACT="$ROOT/configs/semantic/frozen_a7_renderer_objective_v3_canary_377.json"
LOSO_CONFIG="$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_377/loso_frozen_config.json"
CAMERAS=(17 18 19 20)

mkdir -p "$RUN_ROOT/logs" "$(dirname "$EVIDENCE")" "$CANDIDATE_ROOT"
cd "$ROOT"

log_event() {
  printf '%s %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')" "$*" \
    | tee -a "$RUN_ROOT/progress.log"
}

log_event "A7 v3 renderer-objective start commit=$(git rev-parse HEAD) gpu=$GPU"

if [[ ! -s "$EVIDENCE" ]]; then
  log_event "evidence start samples=456 sequence_schema=3"
  start_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/build_renderer_aligned_temporal_evidence.py \
    --config "$SOURCE/assets/test/.hydra/config.yaml" \
    --checkpoint "$SOURCE/semantic_train_strict/ckpt42000.pth" \
    --a5-bank "$A5_BANK" \
    --method-freeze "$A5_FREEZE" \
    --a7-contract "$A7_CONTRACT" \
    --output "$EVIDENCE" >"$RUN_ROOT/logs/evidence.log" 2>&1
  log_event "evidence done elapsed_seconds=$(( $(date +%s) - start_epoch ))"
else
  log_event "evidence skip reason=complete"
fi

if [[ ! -s "$CANDIDATE_ROOT/candidate_index.json" ]]; then
  log_event "candidates start count=2 policies=bounded_damping_005,bounded_retention_010"
  start_epoch="$(date +%s)"
  "$PYTHON" tools/calibrate_renderer_objective_a7_weights.py \
    --a5-bank "$A5_BANK" \
    --evidence "$EVIDENCE" \
    --method-freeze "$A5_FREEZE" \
    --a7-contract "$A7_CONTRACT" \
    --output-dir "$CANDIDATE_ROOT" >"$RUN_ROOT/logs/candidates.log" 2>&1
  log_event "candidates done elapsed_seconds=$(( $(date +%s) - start_epoch ))"
else
  log_event "candidates skip reason=complete"
fi

mapfile -t CANDIDATES < <(
  "$PYTHON" - "$CANDIDATE_ROOT/candidate_index.json" <<'PY'
import json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for candidate_id in payload["validation_shortlist"]:
    print(candidate_id)
PY
)
if [[ "${#CANDIDATES[@]}" -ne 2 ]]; then
  log_event "failed reason=expected_two_valid_candidates actual=${#CANDIDATES[@]}"
  exit 2
fi
log_event "validation shortlist=${CANDIDATES[*]}"

for candidate in "${CANDIDATES[@]}"; do
  bank="$CANDIDATE_ROOT/$candidate/part_label_bank.npz"
  for camera in "${CAMERAS[@]}"; do
    output="$OUTPUT_ROOT/validation/$SUBJECT/$candidate/temporal/c$camera"
    log="$RUN_ROOT/logs/${candidate}_c${camera}.log"
    if [[ -s "$output/summary.json" && -s "$output/metrics.csv" ]]; then
      log_event "temporal skip candidate=$candidate camera=c$camera reason=complete"
      continue
    fi
    log_event "temporal start candidate=$candidate camera=c$camera"
    start_epoch="$(date +%s)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/render_semantic_temporal_stability.py \
      --subject "$SUBJECT" \
      --voting-bank "$SOURCE/banks/multiview_voting/part_label_bank.npz" \
      --a5-bank "$A5_BANK" \
      --a7-bank "$bank" \
      --a7-contract "$A7_CONTRACT" \
      --loso-config "$LOSO_CONFIG" \
      --method-freeze "$A5_FREEZE" \
      --checkpoint "$SOURCE/semantic_train_strict/ckpt42000.pth" \
      --config "$SOURCE/assets/test/.hydra/config.yaml" \
      --output-dir "$output" \
      --camera "$camera" \
      --frame-start 0 --frame-end 570 --frame-step 5 \
      --methods a5 a7 --no-videos \
      --adaptive-strength-grid 0.50 0.525 0.55 1.0 >"$log" 2>&1
    log_event "temporal done candidate=$candidate camera=c$camera elapsed_seconds=$(( $(date +%s) - start_epoch ))"
  done

  spatial="$OUTPUT_ROOT/validation/$SUBJECT/$candidate/spatial"
  log="$RUN_ROOT/logs/${candidate}_spatial.log"
  if [[ -s "$spatial/summary.json" && -s "$spatial/spatial_guard_metrics.csv" ]]; then
    log_event "spatial skip candidate=$candidate reason=complete"
  else
    log_event "spatial start candidate=$candidate"
    start_epoch="$(date +%s)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/evaluate_semantic_editing_paper_protocol.py \
      --protocol "$ROOT/configs/semantic/coreview377_strict_paper_protocol.json" \
      --protocol-split validation \
      --frozen-config "$LOSO_CONFIG" \
      --raw-trained-bank "$SOURCE/banks/raw_trained/part_label_bank.npz" \
      --trained-bank "$SOURCE/banks/voting_evidence_target_support/part_label_bank.npz" \
      --voting-bank "$SOURCE/banks/multiview_voting/part_label_bank.npz" \
      --footprint-bank "$A5_BANK" \
      --a7-bank "$bank" \
      --a7-contract "$A7_CONTRACT" \
      --method-freeze "$A5_FREEZE" \
      --checkpoint "$SOURCE/semantic_train_strict/ckpt42000.pth" \
      --asset-root "$SOURCE/assets/validation/test-view/semantic_editable_assets" \
      --explicit-binding-render-preset none \
      --output-dir "$spatial" \
      --baselines A5 A7 --retention-reference-baseline A5 >"$log" 2>&1
    log_event "spatial done candidate=$candidate elapsed_seconds=$(( $(date +%s) - start_epoch ))"
  fi
done

log_event "A7 v3 renderer-objective complete"
touch "$RUN_ROOT/.done"
