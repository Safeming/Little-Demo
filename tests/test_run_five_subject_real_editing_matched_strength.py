from pathlib import Path


SCRIPT = Path("tools/run_five_subject_real_editing_matched_strength.sh")


def test_matched_queue_declares_five_subjects_views_and_strengths():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "SUBJECTS=(377 386 387 393 394)" in source
    assert "STRENGTHS=(0.2 0.4 0.6 0.8 1.0)" in source
    for camera in (21, 22, 23):
        for frame in (180, 420, 540):
            assert f"c{camera}_f{frame:06d}" in source


def test_matched_queue_uses_full_matrix_metrics_only_and_loso_assets():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--edit-strengths \"${STRENGTHS[@]}\"" in source
    assert "--metrics-only" in source
    assert "--methods raw_hard voting a5" in source
    assert "--tasks recolor removal texture" in source
    assert "--parts hair face upper lower shoes skin" in source
    assert "frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" in source


def test_matched_queue_is_restartable_and_runs_final_summarizer():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "subject_complete" in source
    assert "metric_row_count\", 0)) == 2430" in source
    assert "tools/summarize_semantic_real_editing_matched_strength.py" in source
    assert "estimated_finish_bjt=" in source
    assert "TZ=Asia/Shanghai" in source
    assert "matched-strength queue completed status=0" in source
