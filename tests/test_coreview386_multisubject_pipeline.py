import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_coreview386_dataset_uses_multiview_paper_split():
    payload = OmegaConf.load(ROOT / "configs/dataset/zjumocap_386_multiview_hq.yaml")

    assert payload.dataset.subject == "CoreView_386"
    assert [str(value) for value in payload.dataset.train_views] == [
        str(value) for value in range(1, 21)
    ]
    assert [str(value) for value in payload.dataset.val_views] == ["21", "22", "23"]
    assert list(payload.dataset.train_frames) == [0, 570, 1]
    assert list(payload.dataset.img_hw) == [768, 768]


def test_coreview386_protocol_matches_cross_subject_split():
    payload = json.loads(
        (ROOT / "configs/semantic/coreview386_strict_paper_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["protocol_name"] == "coreview386_strict_paper_v1"
    assert payload["subject"] == "CoreView_386"
    assert payload["semantic_train"] == {
        "camera_ids": list(range(1, 17)),
        "frame_ids": [0, 120, 240, 360, 480],
    }
    assert payload["calibration"] == payload["semantic_train"]
    assert payload["validation"] == {
        "camera_ids": [17, 18, 19, 20],
        "frame_ids": [60, 300],
    }
    assert payload["test"] == {
        "camera_ids": [21, 22, 23],
        "frame_ids": [180, 420, 540],
    }


@pytest.mark.parametrize(
    ("subject_id", "train_frame_end"),
    [("387", 570), ("392", 550)],
)
def test_additional_subject_configs_follow_strict_multisubject_protocol(
    subject_id, train_frame_end
):
    subject = f"CoreView_{subject_id}"
    dataset = OmegaConf.load(
        ROOT / f"configs/dataset/zjumocap_{subject_id}_multiview_hq.yaml"
    )
    protocol = json.loads(
        (
            ROOT
            / f"configs/semantic/coreview{subject_id}_strict_paper_protocol.json"
        ).read_text(encoding="utf-8")
    )

    assert dataset.dataset.subject == subject
    assert [str(value) for value in dataset.dataset.train_views] == [
        str(value) for value in range(1, 21)
    ]
    assert [str(value) for value in dataset.dataset.val_views] == ["21", "22", "23"]
    assert list(dataset.dataset.train_frames) == [0, train_frame_end, 1]
    assert protocol["subject"] == subject
    assert protocol["semantic_train"] == {
        "camera_ids": list(range(1, 17)),
        "frame_ids": [0, 120, 240, 360, 480],
    }
    assert protocol["calibration"] == protocol["semantic_train"]
    assert protocol["validation"] == {
        "camera_ids": [17, 18, 19, 20],
        "frame_ids": [60, 300],
    }
    assert protocol["test"] == {
        "camera_ids": [21, 22, 23],
        "frame_ids": [180, 420, 540],
    }


def test_subject_semantic_launchers_do_not_load_377_geometry():
    train_text = (ROOT / "tools/formal/run_subject_semantic_train.sh").read_text(
        encoding="utf-8"
    )
    export_text = (ROOT / "tools/formal/run_subject_semantic_export.sh").read_text(
        encoding="utf-8"
    )

    assert 'SUBJECT="${SUBJECT:?set SUBJECT}"' in train_text
    assert "assets/adopted_geometry/377" not in train_text
    assert "explicit_binding_render_preset" not in train_text
    assert '"dataset.subject=$SUBJECT"' in train_text
    assert "stageB_semantic_adapter_only_train=true" in train_text

    assert 'SUBJECT="${SUBJECT:?set SUBJECT}"' in export_text
    assert "assets/adopted_geometry/377" not in export_text
    assert "explicit_binding_render_preset" not in export_text
    assert '"dataset.subject=$SUBJECT"' in export_text
    assert "export_semantic_editable_assets=$EXPORT_EDITABLE" in export_text


def test_multisubject_orchestrator_uses_frozen_scheme_a():
    text = (ROOT / "tools/run_multisubject_strict_semantic_protocol.sh").read_text(
        encoding="utf-8"
    )

    for stage in (
        "validate",
        "semantic-train",
        "export-calibration",
        "export-validation",
        "export-test",
        "build-banks",
        "calibrate-voting",
        "evaluate-validation",
        "select-validation-guard",
        "evaluate-test",
        "all",
    ):
        assert stage in text
    assert "select_semantic_editing_validation_config.py" not in text
    assert "materialize_fixed_semantic_evaluation_config.py" in text
    assert "select_guarded_semantic_validation_config.py" in text
    assert "--b5-fallback-threshold" in text
    assert "select_validation_guard" in text
    assert '--part-label-bank "$VOTING_BANK"' in text
    assert text.count("--explicit-binding-render-preset none") == 7
    assert "--protocol-split calibration" in text
    assert "--protocol-split test" in text
    assert '--frozen-config "$FROZEN_CONFIG"' in text


def test_multisubject_orchestrator_all_dry_run_needs_no_generated_checkpoint(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SUBJECT": "CoreView_386",
        "PROTOCOL": str(ROOT / "configs/semantic/coreview386_strict_paper_protocol.json"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "BASE_EXP": str(tmp_path / "base"),
        "BASE_CKPT": str(tmp_path / "base/ckpt40000.pth"),
    }

    result = subprocess.run(
        ["bash", "tools/run_multisubject_strict_semantic_protocol.sh", "all"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "export-calibration" not in result.stderr
    assert "dry-run-semantic-ckpt.pth" in result.stdout
