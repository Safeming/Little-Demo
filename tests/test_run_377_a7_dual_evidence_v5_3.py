from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_dual_evidence_v5_3_canary_377/run_377_v5_3_development.sh"
)


def test_v5_3_runner_uses_eight_camera_capacity_and_retrospective_diagnostics():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'OUTPUT_ROOT="$ROOT/exp/acceptdata/a7_dual_evidence_v5_3_canary_377"' in text
    assert 'EVIDENCE="$OUTPUT_ROOT/evidence/377/evidence.npz"' in text
    assert "frozen_a7_dual_evidence_v5_3_canary_377.json" in text
    assert "dual_evidence_constrained_v5_3" in text
    assert "VALIDATION_CAMERAS" not in text
    assert "RETROSPECTIVE_CAMERAS=(21 22 23)" in text
    assert "/retrospective/c$camera" in text
    assert "summarize_a7_v5_3_development.py" in text
    assert "development_summary.json" in text
    assert "paper_test_eligible=false" in text
    assert "stale_partial_output" in text
    assert 'rm -f "$RUN_ROOT/.done" "$RUN_ROOT/.failed"' in text
    assert text.index("development_summary.json") < text.index(
        'touch "$RUN_ROOT/.done"'
    )
