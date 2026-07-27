#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

REAL_ROOT="exp/acceptdata/five_subject_real_editing_matched_strength_20260723"
REAL_FORMAL="${REAL_ROOT}/aggregate/formal_coverage_constrained"
TEMPORAL_ROOT="exp/acceptdata/five_subject_semantic_temporal_stability_20260724"
TEMPORAL_FORMAL="${TEMPORAL_ROOT}/aggregate/formal_matched_retention"
FLICKER_FORMAL="${TEMPORAL_ROOT}/aggregate/formal_flicker_diagnostic"
FAILURE_ROOT="exp/acceptdata/formal_semantic_failure_cases_20260727"
STATUS_FILE="${FAILURE_ROOT}/queue_status.txt"
LOG_FILE="${FAILURE_ROOT}/queue.log"
MARKDOWN_PATH="docs/正式论文失败案例与时序诊断_20260727.md"

mkdir -p "${REAL_FORMAL}" "${FLICKER_FORMAL}" "${FAILURE_ROOT}"

bjt_now() {
  TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S BJT'
}

write_status() {
  printf 'status=%s\nstart_bjt=%s\nend_bjt=%s\n' "$1" "${START_BJT}" "$2" > "${STATUS_FILE}"
}

on_error() {
  write_status "failed" "$(bjt_now)"
}
trap on_error ERR

run_python() {
  /opt/miniconda3/bin/conda run -n ictrl python "$@"
}

START_BJT="$(bjt_now)"
write_status "running" ""

{
  run_python tools/summarize_semantic_real_editing_coverage_constrained.py \
    --input-root "${REAL_ROOT}" \
    --output-dir "${REAL_FORMAL}" \
    --subjects 377 386 387 393 394 \
    --tasks recolor removal texture \
    --parts hair face upper lower shoes skin \
    --retentions 0.25 0.50 \
    --coverage-threshold 0.80 \
    --bootstrap-repetitions 10000 \
    --bootstrap-seed 20260727

  run_python tools/summarize_semantic_temporal_flicker_diagnostic.py \
    --input-csv "${TEMPORAL_ROOT}/aggregate/all_metrics.csv" \
    --output-dir "${FLICKER_FORMAL}" \
    --subjects 377 386 387 393 394 \
    --parts hair face upper lower shoes skin \
    --retentions 0.25 0.50 \
    --coverage-threshold 0.80 \
    --bootstrap-repetitions 10000 \
    --bootstrap-seed 20260727

  run_python tools/build_semantic_paper_failure_report.py \
    --real-dir "${REAL_FORMAL}" \
    --temporal-part-csv "${TEMPORAL_FORMAL}/part_table.csv" \
    --flicker-dir "${FLICKER_FORMAL}" \
    --output-dir "${FAILURE_ROOT}" \
    --markdown-path "${MARKDOWN_PATH}" \
    --real-asset-dir "${REAL_ROOT%_matched_strength_20260723}_paper_20260723/aggregate" \
    --temporal-video-root "${TEMPORAL_ROOT}"
} > "${LOG_FILE}" 2>&1

write_status "completed" "$(bjt_now)"
trap - ERR
