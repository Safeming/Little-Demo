from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_dual_evidence_v5_2_canary_377/run_377_v5_2_development.sh"
)


def test_v5_2_runner_separates_validation_from_retrospective_diagnostics():
    text = RUNNER.read_text(encoding="utf-8")

    assert "a7_dual_evidence_v5_canary_377/evidence/377/evidence.npz" in text
    assert "frozen_a7_dual_evidence_v5_2_canary_377.json" in text
    assert "dual_evidence_constrained_v5_2" in text
    assert "VALIDATION_CAMERAS=(17 18 19 20)" in text
    assert "RETROSPECTIVE_CAMERAS=(21 22 23)" in text
    assert "/validation/c$camera" in text
    assert "/retrospective/c$camera" in text
    assert "summarize_a7_v5_2_development.py" in text
    assert "development_summary.json" in text
    assert "stale_partial_output" in text
    assert 'rm -f "$RUN_ROOT/.done" "$RUN_ROOT/.failed"' in text
    assert text.index("development_summary.json") < text.index(
        'touch "$RUN_ROOT/.done"'
    )
