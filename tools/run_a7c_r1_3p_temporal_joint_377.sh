#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1}"
PYTHON="/opt/miniconda3/envs/ictrl/bin/python"
CONTRACT="${ROOT}/configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json"
DESIGN="${ROOT}/docs/superpowers/specs/2026-08-04-a7c-r1-3p-temporal-joint-projection-design.md"
PARENT="${ROOT}/configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json"
SOURCE_TRAINING="${ROOT}/exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training"
TRAINING_SUMMARY="${SOURCE_TRAINING}/training_summary.json"
PROBE="${ROOT}/exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz"
TEACHER="${ROOT}/exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz"
EVIDENCE="${ROOT}/exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz"
A5_BANK="${ROOT}/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
PROJECTION="${OUT}/projection"
AUDIT="${PROJECTION}/audit"
ORACLE="${OUT}/oracle"

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/runner.log") 2>&1
printf '%s\n' "$$" > "${OUT}/runner.pid"

mark_terminal() {
  local marker="$1"
  rm -f "${OUT}/.completed" "${OUT}/.rejected" "${OUT}/.failed"
  touch "${OUT}/.${marker}"
}

required_outputs_complete() {
  local terminal_count=0
  local marker
  for marker in completed rejected failed; do
    [[ -f "${OUT}/.${marker}" ]] && terminal_count=$((terminal_count + 1))
  done
  [[ "${terminal_count}" -eq 1 ]] || return 1
  [[ -f "${PROJECTION}/summary.json" ]] || return 1
  [[ -f "${AUDIT}/held_block_summary.json" ]] || return 1
  [[ -f "${ORACLE}/records.json" ]] || return 1
  [[ -f "${ORACLE}/summary.json" ]] || return 1
  [[ -f "${OUT}/started_utc.txt" ]] || return 1
  [[ -f "${OUT}/ended_utc.txt" ]] || return 1
  [[ -f "${OUT}/runner.pid" ]] || return 1
  [[ -f "${OUT}/runner.log" ]] || return 1
  for fold in 0 1 2 3 4 5; do
    [[ -f "${PROJECTION}/fold_${fold}/predictions.npz" ]] || return 1
    [[ -f "${PROJECTION}/fold_${fold}/segment_certificates.json" ]] || return 1
  done
  return 0
}

if required_outputs_complete; then
  exit 0
fi

mark_terminal failed
trap 'mark_terminal failed' ERR
if [[ ! -f "${OUT}/started_utc.txt" ]]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/started_utc.txt"
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]]
}

check_sha "${CONTRACT}" "a62d99f65d1358d2b985db3c5dec5221396a7fb1c8cbf287abc8943788f4c61c"
check_sha "${DESIGN}" "6204b23695f79c955e45af1b222a61b92209b556aac8f736db5faa21fd5ba9b2"
check_sha "${PARENT}" "e2825c1d59e96ff2ea6124bfa1defafb62c73c04a64519c889e909be9ef2f9b5"
check_sha "${TRAINING_SUMMARY}" "8a604cd5df7407b8b559adcea11304c93de2d9272c0bb8f1f60ca8b4f8efc46d"
check_sha "${PROBE}" "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411"
check_sha "${TEACHER}" "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8"
check_sha "${EVIDENCE}" "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4"
check_sha "${A5_BANK}" "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3"

prediction_hashes=(
  "5e53226483194c26ec46e7da08602ee0b72818076d72e3bcafb6190328516dee"
  "95a56b28b50f538f0e7128954700233904fbb5eaf8762fa242e699284e3f4300"
  "87e5db5244e1d6bff8342cd2e84b33aaeb4901e0669e4e041c0adc90c36a26e8"
  "160e54f8e1b0e45d6fef2bcc800817278cd0dc3293d82db701436f4fc8fa7758"
  "d348a458812105e356855909df06893edb404e484944ffbd21a0eeba69a7a25d"
  "8c1ef0e5c5fe9b4001e2829c7da536811b6de4cef1761ef878b8adc141ff357c"
)
projection_complete=1
for fold in 0 1 2 3 4 5; do
  check_sha "${SOURCE_TRAINING}/fold_${fold}/predictions.npz" "${prediction_hashes[fold]}"
  [[ -f "${PROJECTION}/fold_${fold}/predictions.npz" ]] || projection_complete=0
  [[ -f "${PROJECTION}/fold_${fold}/segment_certificates.json" ]] || projection_complete=0
done
[[ -f "${PROJECTION}/summary.json" ]] || projection_complete=0

if [[ "${projection_complete}" -eq 0 ]]; then
  "${PYTHON}" "${ROOT}/tools/project_a7c_r1_3p_temporal_joint.py" \
    --contract "${CONTRACT}" \
    --probe "${PROBE}" \
    --a5-bank "${A5_BANK}" \
    --teacher "${TEACHER}" \
    --source-training-dir "${SOURCE_TRAINING}" \
    --output-dir "${PROJECTION}"
fi

trap - ERR
set +e
"${PYTHON}" "${ROOT}/tools/audit_a7c_r1_3p_temporal_joint_projection.py" \
  --contract "${CONTRACT}" \
  --evidence "${EVIDENCE}" \
  --a5-bank "${A5_BANK}" \
  --teacher "${TEACHER}" \
  --projection-dir "${PROJECTION}" \
  --output-dir "${AUDIT}"
audit_status=$?
set -e
if [[ "${audit_status}" -ne 0 && "${audit_status}" -ne 2 ]]; then
  mark_terminal failed
  exit "${audit_status}"
fi
trap 'mark_terminal failed' ERR

oracle_complete=0
if [[ -f "${ORACLE}/records.json" && -f "${ORACLE}/summary.json" ]]; then
  if rg -q '"execution_status": "COMPLETED"' "${ORACLE}/summary.json"; then
    oracle_complete=1
  fi
fi
if [[ "${oracle_complete}" -eq 0 ]]; then
  "${PYTHON}" "${ROOT}/tools/evaluate_a7c_r1_3p_feasibility_oracle.py" \
    --contract "${CONTRACT}" \
    --probe "${PROBE}" \
    --evidence "${EVIDENCE}" \
    --a5-bank "${A5_BANK}" \
    --teacher "${TEACHER}" \
    --output-dir "${ORACLE}"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/ended_utc.txt"
trap - ERR
if [[ "${audit_status}" -eq 0 ]]; then
  mark_terminal completed
else
  mark_terminal rejected
fi
