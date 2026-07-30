from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_dual_evidence_v5_4_canary_377/run_377_v5_4_development.sh"
)


def test_v5_4_runner_uses_capacity_and_spatial_without_consuming_retrospective_views():
    text = RUNNER.read_text(encoding="utf-8")

    assert "frozen_a7_dual_evidence_v5_4_canary_377.json" in text
    assert "dual_evidence_camera_time_v5_4" in text
    assert "summarize_a7_v5_4_development.py" in text
    assert "development_summary.json" in text
    assert "paper_test_eligible=false" in text
    assert "render_semantic_temporal_stability.py" not in text
    assert "RETROSPECTIVE_CAMERAS" not in text
    assert "c21" not in text and "c22" not in text and "c23" not in text
    assert "stale_partial_output" in text
    assert 'rm -f "$RUN_ROOT/.done" "$RUN_ROOT/.failed"' in text
    assert text.index("development_summary.json") < text.index('touch "$RUN_ROOT/.done"')
