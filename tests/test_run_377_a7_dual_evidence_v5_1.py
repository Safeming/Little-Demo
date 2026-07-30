from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_dual_evidence_v5_1_canary_377/run_377_v5_1_audit.sh"
)


def test_v5_1_runner_reuses_frozen_v5_evidence_before_test_audit():
    text = RUNNER.read_text(encoding="utf-8")

    assert "a7_dual_evidence_v5_canary_377/evidence/377/evidence.npz" in text
    assert "build_renderer_aligned_temporal_evidence.py" not in text
    assert "frozen_a7_dual_evidence_v5_1_canary_377.json" in text
    assert "dual_evidence_constrained_v5_1" in text
    assert "CAMERAS=(21 22 23)" in text
    assert "freeze_manifest.json" in text
    assert "--protocol-split test" in text
    assert "assets/test/test-view/semantic_editable_assets" in text
    assert "summarize_a7_v5_1_audit.py" in text
    assert "promotion_summary.json" in text
    assert "stale_partial_output" in text
    assert 'rm -f "$RUN_ROOT/.done" "$RUN_ROOT/.failed"' in text
    assert "load_validated_candidate" in text
    assert text.index("promotion_summary.json") < text.index('touch "$RUN_ROOT/.done"')
