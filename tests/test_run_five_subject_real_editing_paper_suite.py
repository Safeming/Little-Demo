from pathlib import Path


SCRIPT = Path("tools/run_five_subject_real_editing_paper_suite.sh")


def test_queue_declares_five_subjects_and_all_test_records():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "SUBJECTS=(377 386 387 393 394)" in source
    for camera in (21, 22, 23):
        for frame in (180, 420, 540):
            assert f"c{camera}_f{frame:06d}" in source


def test_queue_routes_loso_and_frozen_a5_assets():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" in source
    assert "frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz" in source
    assert "configs/semantic/frozen_a5_main_method_v1.json" in source


def test_queue_runs_full_method_task_part_matrix_and_summarizer():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--methods raw_hard voting a5" in source
    assert "--tasks recolor removal texture" in source
    assert "--parts hair face upper lower shoes skin" in source
    assert "tools/summarize_semantic_real_editing_paper_suite.py" in source


def test_queue_is_restartable_and_records_beijing_eta():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "subject_complete" in source
    assert "estimated_finish_bjt=" in source
    assert "TZ=Asia/Shanghai" in source
    assert "queue completed status=0" in source
