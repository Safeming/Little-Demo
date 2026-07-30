#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote-home/ming/3dgs-avatar-release-main
PYTHON=/opt/miniconda3/envs/ictrl/bin/python
GPU=0
SUBJECT=377
SOURCE="$ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721"
OUTPUT_ROOT="$ROOT/exp/acceptdata/a7_dual_evidence_v5_4_canary_377"
RUN_ROOT="$OUTPUT_ROOT/run_377_v5_4"
EVIDENCE="$ROOT/exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz"
CANDIDATE_ROOT="$OUTPUT_ROOT/candidates/377"
A5_BANK="$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
V4_BANK="$ROOT/exp/acceptdata/a7_sparse_robust_v4_canary_377/candidates/377/sparse_robust_loco_v4/part_label_bank.npz"
A5_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
A7_CONTRACT="$ROOT/configs/semantic/frozen_a7_dual_evidence_v5_4_canary_377.json"
LOSO_CONFIG="$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_377/loso_frozen_config.json"
EXPECTED_CANDIDATE=dual_evidence_camera_time_v5_4

mkdir -p "$RUN_ROOT/logs" "$CANDIDATE_ROOT"
rm -f "$RUN_ROOT/.done" "$RUN_ROOT/.failed"
cd "$ROOT"

log_event() {
  printf '%s %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')" "$*" \
    | tee -a "$RUN_ROOT/progress.log"
}

V5_4_PATHS=(
  configs/semantic/frozen_a7_dual_evidence_v5_4_canary_377.json
  tools/calibrate_constrained_a7_weights.py
  tools/summarize_a7_v5_1_audit.py
  tools/summarize_a7_v5_4_development.py
  utils/constrained_sparse_temporal_optimizer.py
  utils/frozen_semantic_method.py
  exp/acceptdata/a7_dual_evidence_v5_4_canary_377/run_377_v5_4_development.sh
)
if ! git diff --quiet -- "${V5_4_PATHS[@]}" || ! git diff --cached --quiet -- "${V5_4_PATHS[@]}"; then
  log_event "failed reason=uncommitted_v5_4_implementation"
  exit 2
fi
if [[ ! -s "$EVIDENCE" ]]; then
  log_event "failed reason=missing_frozen_evidence path=$EVIDENCE"
  exit 2
fi

log_event "A7 v5.4 camera-time stability start commit=$(git rev-parse HEAD) gpu=$GPU paper_test_eligible=false"
if [[ ! -s "$CANDIDATE_ROOT/candidate_index.json" ]]; then
  log_event "capacity start folds=48 blocks=6 stability_count=36"
  start_epoch="$(date +%s)"
  "$PYTHON" tools/calibrate_constrained_a7_weights.py \
    --a5-bank "$A5_BANK" --v4-bank "$V4_BANK" --evidence "$EVIDENCE" \
    --method-freeze "$A5_FREEZE" --a7-contract "$A7_CONTRACT" \
    --output-dir "$CANDIDATE_ROOT" >"$RUN_ROOT/logs/capacity.log" 2>&1
  log_event "capacity done elapsed_seconds=$(( $(date +%s) - start_epoch ))"
else
  log_event "capacity skip reason=complete"
fi

if ! candidate="$($PYTHON - "$CANDIDATE_ROOT/candidate_index.json" "$A7_CONTRACT" "$A5_FREEZE" <<'PY'
import sys
from pathlib import Path
from tools.summarize_a7_v5_1_audit import load_validated_candidate
from utils.frozen_semantic_method import load_a7_temporal_contract

index, contract_path, method_freeze = map(Path, sys.argv[1:])
contract = load_a7_temporal_contract(contract_path, method_freeze)
candidate, _ = load_validated_candidate(index, contract)
print(candidate["candidate_id"])
PY
)"; then
  log_event "failed reason=capacity_gate_failure"
  touch "$RUN_ROOT/.failed"
  exit 2
fi
if [[ "$candidate" != "$EXPECTED_CANDIDATE" ]]; then
  log_event "failed reason=unexpected_candidate candidate=$candidate"
  touch "$RUN_ROOT/.failed"
  exit 2
fi
bank="$CANDIDATE_ROOT/$candidate/part_label_bank.npz"

"$PYTHON" - "$A7_CONTRACT" "$EVIDENCE" "$CANDIDATE_ROOT/candidate_index.json" \
  "$bank" "$RUN_ROOT/freeze_manifest.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

contract, evidence, index, bank, output = map(Path, sys.argv[1:])
manifest = {
    "schema_version": 1,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "contract_sha256": sha256(contract),
    "evidence_sha256": sha256(evidence),
    "candidate_index_sha256": sha256(index),
    "candidate_bank_sha256": sha256(bank),
    "camera_time_fold_count": 48,
    "temporal_block_count": 6,
    "minimum_stability_selection_count": 36,
    "retrospective_rendering": False,
    "paper_test_eligible": False,
}
if output.is_file():
    existing = json.loads(output.read_text(encoding="utf-8"))
    if any(existing.get(key) != value for key, value in manifest.items()):
        raise SystemExit("freeze manifest mismatch")
else:
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY
log_event "development freeze candidate=$candidate paper_test_eligible=false"

spatial="$OUTPUT_ROOT/development/$SUBJECT/$candidate/spatial_test"
log="$RUN_ROOT/logs/${candidate}_spatial_test.log"
if [[ -s "$spatial/summary.json" && -s "$spatial/spatial_guard_metrics.csv" ]]; then
  log_event "spatial skip reason=complete"
else
  if [[ -e "$spatial/summary.json" || -e "$spatial/spatial_guard_metrics.csv" ]]; then
    log_event "failed reason=stale_partial_output stage=spatial"
    touch "$RUN_ROOT/.failed"
    exit 2
  fi
  log_event "spatial start split=test"
  start_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$ROOT/configs/semantic/coreview377_strict_paper_protocol.json" \
    --protocol-split test --frozen-config "$LOSO_CONFIG" \
    --raw-trained-bank "$SOURCE/banks/raw_trained/part_label_bank.npz" \
    --trained-bank "$SOURCE/banks/voting_evidence_target_support/part_label_bank.npz" \
    --voting-bank "$SOURCE/banks/multiview_voting/part_label_bank.npz" \
    --footprint-bank "$A5_BANK" --a7-bank "$bank" --a7-contract "$A7_CONTRACT" \
    --method-freeze "$A5_FREEZE" --checkpoint "$SOURCE/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$SOURCE/assets/test/test-view/semantic_editable_assets" \
    --explicit-binding-render-preset none --output-dir "$spatial" \
    --baselines A5 A7 --retention-reference-baseline A5 >"$log" 2>&1
  log_event "spatial done elapsed_seconds=$(( $(date +%s) - start_epoch ))"
fi

summary="$RUN_ROOT/development_summary.json"
log_event "development summary start output=$summary"
if ! "$PYTHON" tools/summarize_a7_v5_4_development.py \
  --candidate-index "$CANDIDATE_ROOT/candidate_index.json" \
  --a7-contract "$A7_CONTRACT" --method-freeze "$A5_FREEZE" \
  --spatial-root "$spatial" --output "$summary" >"$RUN_ROOT/logs/development_summary.log" 2>&1; then
  log_event "failed reason=development_gate_failure summary=$summary"
  touch "$RUN_ROOT/.failed"
  exit 2
fi

log_event "A7 v5.4 development complete paper_test_eligible=false"
touch "$RUN_ROOT/.done"
