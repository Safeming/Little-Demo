from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_renderer_objective_v3_canary_377/run_377_v3_validation.sh"
)


def test_v3_runner_is_isolated_and_uses_only_registered_validation_cameras():
    text = RUNNER.read_text(encoding="utf-8")

    assert "a7_renderer_objective_v3_canary_377" in text
    assert "frozen_a7_renderer_objective_v3_canary_377.json" in text
    assert "tools/build_renderer_aligned_temporal_evidence.py" in text
    assert "tools/calibrate_renderer_objective_a7_weights.py" in text
    assert "CAMERAS=(17 18 19 20)" in text
    assert "expected_two_valid_candidates" in text
    assert "run_377_v3/.done" not in text
    assert 'touch "$RUN_ROOT/.done"' in text
    assert "camera=21" not in text
    assert "camera=22" not in text
    assert "camera=23" not in text
