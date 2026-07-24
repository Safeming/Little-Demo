from pathlib import Path
import subprocess


SCRIPT = Path("tools/run_five_subject_semantic_temporal_stability.sh")


def test_temporal_queue_routes_five_frozen_subjects_and_protocol():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "SUBJECTS=(377 386 387 393 394)" in source
    assert "coreview377_multisubject_strict_20260721" in source
    assert "coreview386_multisubject_strict_20260719" in source
    assert "coreview387_multisubject_strict_20260720" in source
    assert "coreview393_multisubject_strict_20260721" in source
    assert "coreview394_multisubject_strict_20260722" in source
    assert "frozen_a5_five_subject_loso_stats_20260723/CoreView_${subject}/loso_frozen_config.json" in source
    assert "frozen_a5_five_subject_main_20260723/CoreView_${subject}/banks/footprint_evidence_target/part_label_bank.npz" in source
    assert "configs/semantic/frozen_a5_main_method_v1.json" in source


def test_temporal_queue_fixes_camera_frames_rows_and_videos():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'CAMERA="${CAMERA:-21}"' in source
    assert 'FRAME_END="${FRAME_END:-570}"' in source
    assert "--frame-start 0 --frame-end \"$FRAME_END\" --frame-step 1" in source
    assert "metric_row_count\", 0)) == 6840" in source
    assert '"upper", "hair", "shoes"' in source
    assert "tools/summarize_semantic_temporal_stability.py" in source
    assert "TZ=Asia/Shanghai" in source


def test_temporal_queue_shell_syntax_is_valid():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
