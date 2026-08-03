#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/exp/acceptdata/a7c_r1_2a_quotient_compositor_377_v1}"
PYTHON="/opt/miniconda3/envs/anim/bin/python"
CONTRACT="${ROOT}/configs/semantic/a7c_r1_2a_quotient_compositor_377_v1.json"
PROBE="${ROOT}/exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz"
TEACHER="${ROOT}/exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz"
EVIDENCE="${ROOT}/exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz"
A5_BANK="${ROOT}/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
TRAINING="${OUT}/training"
AUDIT="${OUT}/audit"

mkdir -p "${OUT}"
if [[ -f "${OUT}/.done" || -f "${OUT}/.rejected" ]]; then
  exit 0
fi
rm -f "${OUT}/.failed"
trap 'touch "${OUT}/.failed"' ERR
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/started_utc.txt"

check_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]]
}

check_sha "${PROBE}" "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411"
check_sha "${TEACHER}" "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8"
check_sha "${EVIDENCE}" "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4"
check_sha "${A5_BANK}" "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3"

complete=1
for fold in 0 1 2 3 4 5; do
  [[ -f "${TRAINING}/fold_${fold}/model.pt" ]] || complete=0
  [[ -f "${TRAINING}/fold_${fold}/predictions.npz" ]] || complete=0
  [[ -f "${TRAINING}/fold_${fold}/summary.json" ]] || complete=0
done
[[ -f "${OUT}/training/final/model.pt" ]] || complete=0
[[ -f "${TRAINING}/final/predictions.npz" ]] || complete=0
[[ -f "${TRAINING}/final/summary.json" ]] || complete=0
[[ -f "${TRAINING}/training_summary.json" ]] || complete=0

if [[ "${complete}" -eq 0 ]]; then
  "${PYTHON}" "${ROOT}/tools/train_a7c_r1_2a_quotient_compositor.py" \
    --contract "${CONTRACT}" \
    --probe "${PROBE}" \
    --evidence "${EVIDENCE}" \
    --a5-bank "${A5_BANK}" \
    --teacher "${TEACHER}" \
    --output-dir "${TRAINING}" \
    --device cuda
fi

set +e
"${PYTHON}" "${ROOT}/tools/audit_a7c_r1_2a_quotient_compositor.py" \
  --contract "${CONTRACT}" \
  --evidence "${EVIDENCE}" \
  --a5-bank "${A5_BANK}" \
  --teacher "${TEACHER}" \
  --training-dir "${TRAINING}" \
  --output-dir "${AUDIT}"
audit_status=$?
set -e

trap - ERR
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/ended_utc.txt"
if [[ "${audit_status}" -eq 0 ]]; then
  touch "${OUT}/.done"
elif [[ "${audit_status}" -eq 2 ]]; then
  touch "${OUT}/.rejected"
else
  touch "${OUT}/.failed"
  exit "${audit_status}"
fi
