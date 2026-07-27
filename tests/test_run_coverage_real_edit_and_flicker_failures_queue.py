from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_queue_runs_all_statistics_and_records_beijing_status():
    script = REPO_ROOT / "tools/run_coverage_real_edit_and_flicker_failures_queue.sh"
    text = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "summarize_semantic_real_editing_coverage_constrained.py" in text
    assert "summarize_semantic_temporal_flicker_diagnostic.py" in text
    assert "build_semantic_paper_failure_report.py" in text
    assert "conda run -n ictrl python" in text
    assert "TZ=Asia/Shanghai" in text
    assert "queue_status.txt" in text
