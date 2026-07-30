from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_dual_evidence_v5_canary_377/run_377_v5_audit.sh"
)


def test_v5_runner_freezes_dual_evidence_candidate_before_test_audit():
    text = RUNNER.read_text(encoding="utf-8")

    assert "build_renderer_aligned_temporal_evidence.py" in text
    assert "calibrate_constrained_a7_weights.py" in text
    assert "frozen_a7_dual_evidence_v5_canary_377.json" in text
    assert "CAMERAS=(21 22 23)" in text
    assert "--protocol-split test" in text
    assert "assets/test/test-view/semantic_editable_assets" in text
    assert "freeze_manifest.json" in text
    assert "freeze manifest mismatch" in text
    assert text.index("freeze_manifest.json") < text.index("for camera in")
    assert "camera=c17" not in text
    assert "camera=c18" not in text
    assert "camera=c19" not in text
    assert "camera=c20" not in text
