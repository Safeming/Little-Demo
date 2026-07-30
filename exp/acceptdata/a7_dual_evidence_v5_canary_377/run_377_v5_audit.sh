#!/usr/bin/env bash
set -euo pipefail

ROOT=/remote-home/ming/3dgs-avatar-release-main
PYTHON=/opt/miniconda3/envs/ictrl/bin/python
GPU=0
SUBJECT=377
SOURCE="$ROOT/exp/acceptdata/coreview377_multisubject_strict_20260721"
OUTPUT_ROOT="$ROOT/exp/acceptdata/a7_dual_evidence_v5_canary_377"
RUN_ROOT="$OUTPUT_ROOT/run_377_v5"
EVIDENCE="$OUTPUT_ROOT/evidence/377/evidence.npz"
CANDIDATE_ROOT="$OUTPUT_ROOT/candidates/377"
A5_BANK="$ROOT/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
V4_BANK="$ROOT/exp/acceptdata/a7_sparse_robust_v4_canary_377/candidates/377/sparse_robust_loco_v4/part_label_bank.npz"
A5_FREEZE="$ROOT/configs/semantic/frozen_a5_main_method_v1.json"
A7_CONTRACT="$ROOT/configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json"
LOSO_CONFIG="$ROOT/exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_377/loso_frozen_config.json"
CAMERAS=(21 22 23)

mkdir -p "$RUN_ROOT/logs" "$(dirname "$EVIDENCE")" "$CANDIDATE_ROOT"
cd "$ROOT"

log_event() {
  printf '%s %s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')" "$*" \
    | tee -a "$RUN_ROOT/progress.log"
}

V5_PATHS=(
  configs/semantic/frozen_a7_dual_evidence_v5_canary_377.json
  tools/build_renderer_aligned_temporal_evidence.py
  tools/calibrate_constrained_a7_weights.py
  utils/constrained_sparse_temporal_optimizer.py
  utils/frozen_semantic_method.py
  utils/renderer_aligned_temporal_evidence.py
  exp/acceptdata/a7_dual_evidence_v5_canary_377/run_377_v5_audit.sh
)
if ! git diff --quiet -- "${V5_PATHS[@]}" || ! git diff --cached --quiet -- "${V5_PATHS[@]}"; then
  log_event "failed reason=uncommitted_v5_implementation"
  exit 2
fi

log_event "A7 v5 dual-evidence constrained start commit=$(git rev-parse HEAD) gpu=$GPU"

if [[ ! -s "$EVIDENCE" ]]; then
  log_event "evidence start samples=456 sequence_schema=4 channels=edit,selection"
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
  log_event "capacity start seed=hair:a5,lower:v4 constraints=target,soft_iou,visibility"
  start_epoch="$(date +%s)"
  "$PYTHON" tools/calibrate_constrained_a7_weights.py \
    --a5-bank "$A5_BANK" \
    --v4-bank "$V4_BANK" \
    --evidence "$EVIDENCE" \
    --method-freeze "$A5_FREEZE" \
    --a7-contract "$A7_CONTRACT" \
    --output-dir "$CANDIDATE_ROOT" >"$RUN_ROOT/logs/capacity.log" 2>&1
  log_event "capacity done elapsed_seconds=$(( $(date +%s) - start_epoch ))"
else
  log_event "capacity skip reason=complete"
fi

candidate="$($PYTHON - "$CANDIDATE_ROOT/candidate_index.json" <<'PY'
import json, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
shortlist = payload["validation_shortlist"]
if len(shortlist) != 1:
    raise SystemExit(f"expected one valid v5 candidate, got {len(shortlist)}")
print(shortlist[0])
PY
)"
bank="$CANDIDATE_ROOT/$candidate/part_label_bank.npz"

"$PYTHON" - "$A7_CONTRACT" "$EVIDENCE" "$CANDIDATE_ROOT/candidate_index.json" "$bank" "$RUN_ROOT/freeze_manifest.json" <<'PY'
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

def command_sha256(command):
    return hashlib.sha256(subprocess.check_output(command)).hexdigest()

contract, evidence, index, bank, output = map(Path, sys.argv[1:])
identity = {
    "schema_version": 1,
    "frozen_before_audit": True,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "tracked_worktree_diff_sha256": command_sha256(["git", "diff", "--binary", "HEAD"]),
    "worktree_status_sha256": command_sha256(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    ),
    "contract_sha256": sha256(contract),
    "evidence_sha256": sha256(evidence),
    "candidate_index_sha256": sha256(index),
    "candidate_bank_sha256": sha256(bank),
    "audit_cameras": [21, 22, 23],
}
if output.is_file():
    existing = json.loads(output.read_text(encoding="utf-8"))
    if any(existing.get(key) != value for key, value in identity.items()):
        raise SystemExit("freeze manifest mismatch")
else:
    payload = {**identity, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
PY
log_event "audit freeze candidate=$candidate manifest=$RUN_ROOT/freeze_manifest.json cameras=${CAMERAS[*]}"

for camera in "${CAMERAS[@]}"; do
  output="$OUTPUT_ROOT/audit/$SUBJECT/$candidate/temporal/c$camera"
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

spatial="$OUTPUT_ROOT/audit/$SUBJECT/$candidate/spatial_test"
log="$RUN_ROOT/logs/${candidate}_spatial_test.log"
if [[ -s "$spatial/summary.json" && -s "$spatial/spatial_guard_metrics.csv" ]]; then
  log_event "spatial skip candidate=$candidate split=test reason=complete"
else
  log_event "spatial start candidate=$candidate split=test"
  start_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/evaluate_semantic_editing_paper_protocol.py \
    --protocol "$ROOT/configs/semantic/coreview377_strict_paper_protocol.json" \
    --protocol-split test \
    --frozen-config "$LOSO_CONFIG" \
    --raw-trained-bank "$SOURCE/banks/raw_trained/part_label_bank.npz" \
    --trained-bank "$SOURCE/banks/voting_evidence_target_support/part_label_bank.npz" \
    --voting-bank "$SOURCE/banks/multiview_voting/part_label_bank.npz" \
    --footprint-bank "$A5_BANK" \
    --a7-bank "$bank" \
    --a7-contract "$A7_CONTRACT" \
    --method-freeze "$A5_FREEZE" \
    --checkpoint "$SOURCE/semantic_train_strict/ckpt42000.pth" \
    --asset-root "$SOURCE/assets/test/test-view/semantic_editable_assets" \
    --explicit-binding-render-preset none \
    --output-dir "$spatial" \
    --baselines A5 A7 --retention-reference-baseline A5 >"$log" 2>&1
  log_event "spatial done candidate=$candidate split=test elapsed_seconds=$(( $(date +%s) - start_epoch ))"
fi

log_event "A7 v5 dual-evidence constrained audit complete"
touch "$RUN_ROOT/.done"
