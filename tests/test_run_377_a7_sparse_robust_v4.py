from pathlib import Path


RUNNER = Path(
    "exp/acceptdata/a7_sparse_robust_v4_canary_377/run_377_v4_validation.sh"
)


def test_v4_runner_reuses_v3_evidence_and_runs_one_registered_candidate():
    text = RUNNER.read_text(encoding="utf-8")

    assert "a7_sparse_robust_v4_canary_377" in text
    assert "a7_renderer_objective_v3_canary_377/evidence/377/evidence.npz" in text
    assert "calibrate_sparse_robust_a7_weights.py" in text
    assert "frozen_a7_sparse_robust_v4_canary_377.json" in text
    assert "CAMERAS=(17 18 19 20)" in text
    assert "expected_one_valid_candidate" in text
    assert 'touch "$RUN_ROOT/.done"' in text
    assert "build_renderer_aligned_temporal_evidence.py" not in text
    assert "camera=21" not in text
    assert "camera=22" not in text
    assert "camera=23" not in text
