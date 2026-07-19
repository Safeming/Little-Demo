import json
from pathlib import Path

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
