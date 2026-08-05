#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUT="${1:-${ROOT}/exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1}"
PYTHON="/opt/miniconda3/envs/ictrl/bin/python"
CONTRACT="${ROOT}/configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json"
DESIGN="${ROOT}/docs/superpowers/specs/2026-08-05-a7c-r1-3g-exact-global-aggregate-oracle-design.md"
SOURCE_CONTRACT="${ROOT}/configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json"
SOURCE_RECORDS="${ROOT}/exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/records.json"
SOURCE_SUMMARY="${ROOT}/exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/summary.json"
PROBE="${ROOT}/exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz"
EVIDENCE="${ROOT}/exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz"
A5_BANK="${ROOT}/exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz"
TEACHER="${ROOT}/exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz"
WITNESS="${OUT}/witness"
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
status = int(sys.argv[2])
summary_path = root / "summary.json"
payload = {}
if summary_path.is_file():
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
audit_path = root / "audit/held_block_summary.json"
if audit_path.is_file():
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        audit = {}
    if audit.get("execution_status") == "ORACLE_ERROR":
        payload["error_type"] = audit.get("error_type", "AuditError")
        payload["error"] = audit.get("error", "exact aggregate audit failed")
payload.setdefault("error_type", "RunnerError")
payload.setdefault("error", f"R1.3-G runner exited with status {status}")
payload.update({
    "execution_status": "ORACLE_ERROR",
    "verdict": "ORACLE_ERROR",
    "aggregate_audit_opened": audit_path.is_file(),
    "deployment_eligible": False,
    "teacher_eligible": False,
    "paper_test_eligible": False,
})
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
temporary.replace(summary_path)
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
  local terminal_count=0
  local marker
  for marker in completed rejected failed; do
    [[ -f "${OUT}/.${marker}" ]] && terminal_count=$((terminal_count + 1))
  done
  [[ "${terminal_count}" -eq 1 ]] || return 1
  [[ -f "${OUT}/summary.json" ]] || return 1
  [[ -f "${OUT}/records.json" ]] || return 1
  [[ -f "${WITNESS}/summary.json" ]] || return 1
  [[ -f "${AUDIT}/held_block_summary.json" ]] || return 1
  [[ -f "${OUT}/started_utc.txt" ]] || return 1
  [[ -f "${OUT}/ended_utc.txt" ]] || return 1
  [[ -f "${OUT}/runner.pid" ]] || return 1
  [[ -f "${OUT}/runner.log" ]] || return 1
  for fold in 0 1 2 3 4 5; do
    [[ -f "${WITNESS}/fold_${fold}/predictions.npz" ]] || return 1
    [[ -f "${WITNESS}/fold_${fold}/certificates.json" ]] || return 1
  done
  if [[ -f "${OUT}/.completed" ]]; then
    rg -q '"verdict": "CERTIFIED_FEASIBLE"' "${OUT}/summary.json"
  elif [[ -f "${OUT}/.rejected" ]]; then
    rg -q '"verdict": "UNRESOLVED"' "${OUT}/summary.json"
  else
    rg -q '"verdict": "ORACLE_ERROR"' "${OUT}/summary.json"
  fi
}

if required_outputs_complete; then
  exit 0
fi

mark_terminal failed
trap on_error ERR
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

check_sha "${CONTRACT}" "0ed0d588ab4a89abfa50d3213a84dc4e055ecd2d800c1bfc7bc154d3bf927bbb"
check_sha "${DESIGN}" "839a7624848a56e4d15b3e81e3b146a0ac897f3eadf17c982381083bc2949d92"
check_sha "${SOURCE_CONTRACT}" "a62d99f65d1358d2b985db3c5dec5221396a7fb1c8cbf287abc8943788f4c61c"
check_sha "${SOURCE_RECORDS}" "7a4d3998408a67cb4754d2bcb799e9e4e2ed8518b23fa3254055c4b88f9d3ce8"
check_sha "${SOURCE_SUMMARY}" "82c1d019002a7ee980d0b13c583bf023ae0df5aac8f5d383e513d2cbff12c5c5"
check_sha "${PROBE}" "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411"
check_sha "${EVIDENCE}" "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4"
check_sha "${A5_BANK}" "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3"
check_sha "${TEACHER}" "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8"

replay_complete=1
[[ -f "${OUT}/records.json" ]] || replay_complete=0
[[ -f "${WITNESS}/summary.json" ]] || replay_complete=0
for fold in 0 1 2 3 4 5; do
  [[ -f "${WITNESS}/fold_${fold}/predictions.npz" ]] || replay_complete=0
  [[ -f "${WITNESS}/fold_${fold}/certificates.json" ]] || replay_complete=0
done

if [[ "${replay_complete}" -eq 0 ]]; then
  "${PYTHON}" "${ROOT}/tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py" \
    --contract "${CONTRACT}" \
    --source-records "${SOURCE_RECORDS}" \
    --probe "${PROBE}" \
    --evidence "${EVIDENCE}" \
    --a5-bank "${A5_BANK}" \
    --teacher "${TEACHER}" \
    --output-dir "${OUT}"
fi

set +e
"${PYTHON}" "${ROOT}/tools/audit_a7c_r1_3g_exact_aggregate_oracle.py" \
  --contract "${CONTRACT}" \
  --evidence "${EVIDENCE}" \
  --a5-bank "${A5_BANK}" \
  --teacher "${TEACHER}" \
  --witness-dir "${WITNESS}" \
  --output-dir "${AUDIT}"
audit_status=$?
set -e
if [[ "${audit_status}" -ne 0 && "${audit_status}" -ne 2 ]]; then
  false
fi

verdict="$("${PYTHON}" - "${OUT}" "${audit_status}" <<'PY'
import json
import sys
from pathlib import Path

from utils.a7c_exact_aggregate_oracle import classify_exact_replay

root = Path(sys.argv[1])
audit_status = int(sys.argv[2])
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("execution_status") != "REPLAY_COMPLETED":
    raise ValueError("root replay summary is incomplete")
audit = json.loads(
    (root / "audit/held_block_summary.json").read_text(encoding="utf-8")
)
audit_passed = bool(audit["summary"]["passed"])
if audit_passed != (audit_status == 0):
    raise ValueError("audit status and summary disagree")
verdict = classify_exact_replay(
    replay_complete=True,
    audit_passed=audit_passed,
)
summary.update({
    "execution_status": "COMPLETED",
    "verdict": verdict,
    "aggregate_audit_opened": True,
    "exact_aggregate_summary": audit["summary"],
    "certificate_maximum_primal_violation": audit[
        "certificate_maximum_primal_violation"
    ],
    "witness_fingerprints": audit["witness_fingerprints"],
    "deployment_eligible": False,
    "teacher_eligible": False,
    "paper_test_eligible": False,
})
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
temporary.replace(summary_path)
print(verdict)
PY
)"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/ended_utc.txt"
trap - ERR
case "${verdict}" in
  CERTIFIED_FEASIBLE)
    mark_terminal completed
    ;;
  UNRESOLVED)
    mark_terminal rejected
    ;;
  ORACLE_ERROR)
    mark_terminal failed
    exit 1
    ;;
  *)
    write_error_summary 1
    mark_terminal failed
    exit 1
    ;;
esac
