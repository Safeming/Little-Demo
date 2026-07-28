import os
from pathlib import Path
import subprocess


SCRIPT = Path("tools/run_a7_temporal_reliability_queue.sh")


def test_queue_text_freezes_subjects_protocol_and_stage_order():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "SUBJECTS=(377 386 387 393 394)" in script
    assert "EVIDENCE_CAMERAS=c01,c05,c09,c13" in script
    assert "VALIDATION_CAMERAS=(17 18 19 20)" in script
    assert "FRAME_START=0" in script
    assert "FRAME_END=570" in script
    assert "FRAME_STEP=5" in script
    assert "frozen_a5_main_method_v1.json" in script
    assert "frozen_a7_temporal_reliable_v1.json" in script
    assert "DEFAULT_STAGE=validation" in script
    stages = (
        "canary",
        "evidence",
        "candidates",
        "validation",
        "loso-freeze",
        "retrospective-c21",
        "frozen-c22-c23",
        "paper-tables",
    )
    positions = [script.index(stage) for stage in stages]
    assert positions == sorted(positions)


def test_queue_has_fingerprint_checked_resume_states_and_provenance():
    script = SCRIPT.read_text(encoding="utf-8")

    for marker in (".running", ".done", ".failed"):
        assert marker in script
    for field in (
        "started_utc",
        "started_bjt",
        "finished_utc",
        "finished_bjt",
        "pid",
        "command",
        "log",
        "output_fingerprint",
        "command_fingerprint",
    ):
        assert field in script
    assert "recorded_output_fingerprint" in script
    assert "current_output_fingerprint" in script
    assert "recorded_command_fingerprint" in script
    assert "current_command_fingerprint" in script


def test_post_validation_stages_have_freeze_and_read_only_guards():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "require_all_selected_configs" in script
    assert "verify_loso_freeze_manifest" in script
    assert "aggregate/loso_freeze_manifest.json" in script
    assert "-w \"$manifest\"" in script
    assert "--allow-post-validation" in script


def test_queue_contains_no_training_or_checkpoint_mutation_commands():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "train.py" not in script
    assert "semantic-train" not in script
    assert "rm -f \"$checkpoint\"" not in script
    assert "mv \"$checkpoint\"" not in script
    assert "cp \"$checkpoint\"" not in script


def test_validation_dry_run_prints_order_samples_and_paths_without_writes(tmp_path):
    output_root = tmp_path / "a7-output"
    env = dict(os.environ)
    env["OUTPUT_ROOT"] = str(output_root)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--stage", "validation", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert not output_root.exists()
    assert "dry_run=true" in result.stdout
    assert "gpu_job_order=377,386,387,393,394" in result.stdout
    for subject in ("377", "386", "387", "393", "394"):
        assert f"subject={subject} stage=evidence samples=456" in result.stdout
        assert f"subject={subject} stage=validation samples=912" in result.stdout
        assert f"evidence/{subject}/evidence.npz" in result.stdout
        assert f"validation/{subject}/<candidate_id>" in result.stdout
    assert "CUDA_VISIBLE_DEVICES" in result.stdout
