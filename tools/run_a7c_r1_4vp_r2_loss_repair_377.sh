#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUT="${1:-${ROOT}/exp/acceptdata/a7c_r1_4vp_r2_loss_scale_repair_377_v1}"
PYTHON="/opt/miniconda3/envs/ictrl/bin/python"
CONTRACT="${ROOT}/configs/semantic/a7c_r1_4vp_r2_loss_scale_repair_377_v1.json"
PROBE="${ROOT}/exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz"
EVIDENCE="${ROOT}/exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz"
A5_BANK="${ROOT}/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
TEACHER="${ROOT}/exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz"
R12B="${ROOT}/exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training"
POSE="${ROOT}/data/ZJUMoCap/CoreView_377/models"
WITNESS="${ROOT}/exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/witness"
TEACHERS="${ROOT}/exp/acceptdata/a7c_r1_4vp_oracle_distilled_view_pose_377_v1/teachers"
NEAREST="${ROOT}/exp/acceptdata/a7c_r1_4vp_oracle_distilled_view_pose_377_v1/nearest_neighbor"
AUDIT="${OUT}/audit"

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/runner.log") 2>&1
printf '%s\n' "$$" > "${OUT}/runner.pid"

mark_terminal() {
  local marker="$1"
  rm -f "${OUT}/.completed" "${OUT}/.rejected" "${OUT}/.failed"
  touch "${OUT}/.${marker}"
}

write_error_summary() {
  local status="$1"
  "${PYTHON}" - "${OUT}" "${status}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "summary.json"
payload = {}
if path.is_file():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
payload.update({"verdict": "TRAINING_ERROR", "execution_status": "TRAINING_ERROR",
                "runner_status": int(sys.argv[2]), "deployment_eligible": False,
                "teacher_eligible": False, "paper_test_eligible": False})
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
temporary.replace(path)
PY
}

on_error() {
  local status="$?"
  trap - ERR
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/ended_utc.txt"
  write_error_summary "${status}" || true
  mark_terminal failed
  exit "${status}"
}

required_outputs_complete() {
  local count=0 marker fold
  for marker in completed rejected failed; do
    [[ -f "${OUT}/.${marker}" ]] && count=$((count + 1))
  done
  [[ "${count}" -eq 1 ]] || return 1
  for path in summary.json training/summary.json models_frozen.json audit/held_block_summary.json started_utc.txt ended_utc.txt runner.pid runner.log; do
    [[ -f "${OUT}/${path}" ]] || return 1
  done
  for fold in 0 1 2 3 4 5; do
    [[ -f "${OUT}/training/fold_${fold}/model.pt" ]] || return 1
    [[ -f "${OUT}/training/fold_${fold}/predictions.npz" ]] || return 1
    [[ -f "${OUT}/training/fold_${fold}/summary.json" ]] || return 1
  done
}

if required_outputs_complete; then
  exit 0
fi

mark_terminal failed
trap on_error ERR
[[ -f "${OUT}/started_utc.txt" ]] || date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/started_utc.txt"

"${PYTHON}" - "${ROOT}" "${CONTRACT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, contract_path = Path(sys.argv[1]), Path(sys.argv[2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

observed_contract = digest(contract_path)
expected_contract = "01b67f556d56bd04006a15117ea7499f196f7911946b1fa500c16a527bf22ee1"
if observed_contract != expected_contract:
    raise ValueError(f"contract fingerprint mismatch: {observed_contract}")

def verify(relative, expected):
    path = root / relative
    observed = digest(path)
    if observed != expected:
        raise ValueError(f"source fingerprint mismatch: {relative}")

for path_key, hash_key in (
    ("source_design", "source_design_sha256"),
    ("source_r1_1_contract", "source_r1_1_contract_sha256"),
    ("source_probe", "source_probe_sha256"),
    ("source_teacher", "source_teacher_sha256"),
    ("source_evidence", "source_evidence_sha256"),
    ("source_a5_bank", "source_a5_bank_sha256"),
):
    verify(contract[path_key], contract[hash_key])
for paths_key, hashes_key in (
    ("source_r1_2b_predictions", "source_r1_2b_prediction_sha256"),
    ("source_r1_3g_witness_predictions", "source_r1_3g_witness_prediction_sha256"),
):
    for relative, expected in zip(contract[paths_key], contract[hashes_key]):
        verify(relative, expected)
teachers = root / contract["source_teachers_dir"]
for relative, expected in contract["source_teacher_artifacts"].items():
    observed = digest(teachers / relative)
    if observed != expected:
        raise ValueError(f"teacher fingerprint mismatch: {relative}")
nearest = root / contract["source_nearest_neighbor_dir"]
for fold, expected in enumerate(contract["source_nearest_neighbor_prediction_sha256"]):
    path = nearest / f"fold_{fold}/predictions.npz"
    if digest(path) != expected:
        raise ValueError(f"nearest-neighbor fingerprint mismatch: fold {fold}")
PY

"${PYTHON}" - "${ROOT}" "${POSE}" "${CONTRACT}" "${TEACHER}" <<'PY'
import json
import sys
import numpy as np
from pathlib import Path
from utils.a7c_r1_4vp_r2_runtime import pose_manifest_sha256
root, pose, contract_path, teacher_path = map(Path, sys.argv[1:5])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
with np.load(teacher_path, allow_pickle=False) as source:
    frames = np.unique(source["frame_index"])
observed = pose_manifest_sha256(pose, frames, root)
if observed != contract["source_pose_manifest_sha256"]:
    raise ValueError(f"pose manifest fingerprint mismatch: {observed}")
PY

if [[ ! -f "${OUT}/models_frozen.json" ]]; then
  "${PYTHON}" "${ROOT}/tools/train_a7c_r1_4vp_r2_loss_repair.py" \
    --contract "${CONTRACT}" --probe "${PROBE}" --a5-bank "${A5_BANK}" \
    --teacher "${TEACHER}" --teachers-dir "${TEACHERS}" \
    --r1-2b-training-dir "${R12B}" --pose-model-dir "${POSE}" \
    --output-dir "${OUT}" --device cuda
fi
[[ -f "${OUT}/models_frozen.json" ]]

if "${PYTHON}" "${ROOT}/tools/audit_a7c_r1_4vp_r2_loss_repair.py" \
  --contract "${CONTRACT}" --evidence "${EVIDENCE}" --a5-bank "${A5_BANK}" \
  --teacher "${TEACHER}" --witness-dir "${WITNESS}" \
  --nearest-neighbor-dir "${NEAREST}" --frozen-root "${OUT}" \
  --output-dir "${AUDIT}"; then
  audit_status=0
else
  audit_status=$?
fi
if [[ "${audit_status}" -ne 0 && "${audit_status}" -ne 2 ]]; then
  false
fi

verdict="$("${PYTHON}" - "${OUT}/summary.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["verdict"])
PY
)"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/ended_utc.txt"
trap - ERR
case "${verdict}" in
  CANARY_PROMOTED) mark_terminal completed ;;
  CANARY_NEGATIVE) mark_terminal rejected ;;
  TRAINING_ERROR) mark_terminal failed; exit 1 ;;
  *) write_error_summary 1; mark_terminal failed; exit 1 ;;
esac
