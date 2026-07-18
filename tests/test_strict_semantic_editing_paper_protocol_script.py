import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_strict_semantic_editing_paper_protocol.sh"


def test_strict_runner_dry_run_wires_protocol_splits_and_fair_preview(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUTPUT_ROOT": str(tmp_path / "out"),
            "BASE_EXP": str(tmp_path / "base"),
            "BASE_CKPT": str(tmp_path / "base" / "ckpt.pth"),
            "SEMANTIC_CKPT": str(tmp_path / "semantic" / "ckpt.pth"),
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "all"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout

    assert "TRAIN_VIEWS_SPEC=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" in output
    assert "stageB_semantic_allowed_frame_ids=[0,120,240,360,480]" in output
    assert "TEST_VIEWS_SPEC=[17,18,19,20]" in output
    assert "--protocol-split calibration" in output
    assert "--protocol-split validation" in output
    assert "--protocol-split test" in output
    assert "--formal-paper-mode" in output
    assert "--screen-mask-composite" not in output
    assert "--checkpoint-fingerprint dry-run-checkpoint" in output
    assert "--bank-fingerprint dry-run-bank" in output


def test_strict_runner_declares_all_required_stages():
    text = SCRIPT.read_text(encoding="utf-8")

    for stage in (
        "validate",
        "semantic-train",
        "export-calibration",
        "export-validation",
        "export-test",
        "build-banks",
        "calibrate",
        "select-validation",
        "evaluate-test",
        "all",
    ):
        assert stage in text
    assert "frozen_validation_config.json" in text
    assert "from utils.semantic_eval_protocol import file_fingerprint" in text
    assert "--checkpoint-fingerprint checkpoint" not in text
    assert "--bank-fingerprint bank" not in text
