import json
from pathlib import Path

import pytest


def _module():
    from utils import semantic_eval_protocol

    return semantic_eval_protocol


def _protocol():
    return {
        "protocol_name": "strict_test_v1",
        "subject": "CoreView_377",
        "semantic_train": {
            "camera_ids": [1, 2],
            "frame_ids": [0, 120],
        },
        "calibration": {
            "camera_ids": [1, 2],
            "frame_ids": [0, 120],
        },
        "validation": {
            "camera_ids": [17],
            "frame_ids": [60],
        },
        "test": {
            "camera_ids": [21],
            "frame_ids": [180],
        },
        "parts": ["face", "hair"],
        "allowed_adjacency": {"face": ["hair"], "hair": ["face"]},
        "validation_grid": {
            "soft_thresholds": [0.1, 0.2],
            "support_thresholds": [0.1],
            "boundary_radii": [0, 2],
        },
    }


def test_load_protocol_normalizes_and_validates(tmp_path):
    module = _module()
    path = tmp_path / "protocol.json"
    payload = _protocol()
    payload["semantic_train"]["camera_ids"] = ["2", "1", "1"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    protocol = module.load_protocol(path)

    assert protocol["semantic_train"]["camera_ids"] == [1, 2]
    assert protocol["semantic_train"]["frame_ids"] == [0, 120]


def test_validate_protocol_rejects_camera_overlap():
    module = _module()
    protocol = _protocol()
    protocol["validation"]["camera_ids"] = [2, 17]

    with pytest.raises(ValueError, match="camera overlap.*semantic_train.*validation"):
        module.validate_protocol(protocol)


def test_validate_protocol_requires_calibration_subset_of_semantic_train():
    module = _module()
    protocol = _protocol()
    protocol["calibration"]["camera_ids"] = [1, 3]

    with pytest.raises(ValueError, match="calibration records must be a subset"):
        module.validate_protocol(protocol)


def test_select_protocol_records_returns_exact_split_records():
    module = _module()
    records = [
        {"image_name": "c01_f000000", "cam_id": 1, "frame_id": 0},
        {"image_name": "c02_f000120", "cam_id": 2, "frame_id": 120},
        {"image_name": "c17_f000060", "cam_id": 17, "frame_id": 60},
        {"image_name": "c21_f000180", "cam_id": 21, "frame_id": 180},
    ]

    selected = module.select_protocol_records(records, _protocol(), "validation")

    assert [record["image_name"] for record in selected] == ["c17_f000060"]
    module.assert_record_set_matches_split(selected, _protocol()["validation"])


def test_assert_record_set_matches_split_rejects_out_of_split_record():
    module = _module()
    records = [{"image_name": "c21_f000180", "cam_id": 21, "frame_id": 180}]

    with pytest.raises(ValueError, match="outside protocol split.*c21_f000180"):
        module.assert_record_set_matches_split(records, _protocol()["validation"])


def test_protocol_fingerprint_is_stable_across_key_order():
    module = _module()
    protocol = _protocol()
    reversed_protocol = dict(reversed(list(protocol.items())))

    assert module.protocol_fingerprint(protocol) == module.protocol_fingerprint(reversed_protocol)


def test_record_fingerprint_is_stable_across_record_order():
    module = _module()
    records = [
        {"image_name": "b", "cam_id": 2, "frame_id": 120},
        {"image_name": "a", "cam_id": 1, "frame_id": 0},
    ]

    assert module.record_fingerprint(records) == module.record_fingerprint(list(reversed(records)))


def test_validate_frozen_config_rejects_wrong_protocol_or_checkpoint():
    module = _module()
    protocol = _protocol()
    frozen = {
        "protocol_fingerprint": module.protocol_fingerprint(protocol),
        "checkpoint_fingerprint": "checkpoint-a",
    }

    module.validate_frozen_config(frozen, protocol=protocol, checkpoint_fingerprint="checkpoint-a")

    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        module.validate_frozen_config(frozen, protocol=protocol, checkpoint_fingerprint="checkpoint-b")


def test_write_protocol_provenance_records_selected_records(tmp_path):
    module = _module()
    protocol = _protocol()
    records = [{"image_name": "c17_f000060", "cam_id": 17, "frame_id": 60}]

    path = module.write_protocol_provenance(
        tmp_path,
        protocol,
        records,
        split_name="validation",
        source_asset_root=Path("/tmp/assets"),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["split_name"] == "validation"
    assert payload["selected_record_names"] == ["c17_f000060"]
    assert payload["protocol_fingerprint"] == module.protocol_fingerprint(protocol)
    assert payload["record_fingerprint"] == module.record_fingerprint(records)


def test_file_fingerprint_hashes_file_content(tmp_path):
    module = _module()
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"semantic-checkpoint")
    right.write_bytes(b"semantic-checkpoint")

    assert module.file_fingerprint(left) == module.file_fingerprint(right)
    right.write_bytes(b"different")
    assert module.file_fingerprint(left) != module.file_fingerprint(right)


def test_write_protocol_manifest_records_normalized_protocol_and_fingerprint(tmp_path):
    module = _module()

    path = module.write_protocol_manifest(tmp_path / "protocol_manifest.json", _protocol())
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["protocol_name"] == "strict_test_v1"
    assert payload["protocol_fingerprint"] == module.protocol_fingerprint(_protocol())
    assert payload["splits"]["test"]["record_count"] == 1


def test_prune_asset_records_to_protocol_split_removes_extra_records_and_files(tmp_path):
    module = _module()
    asset_root = tmp_path / "assets"
    mask_dir = asset_root / "compact_head_masks" / "face"
    mask_dir.mkdir(parents=True)
    keep = mask_dir / "render_c21_f000180.png"
    remove = mask_dir / "render_c21_f000300.png"
    keep.write_bytes(b"keep")
    remove.write_bytes(b"remove")
    (asset_root / "view_records.json").write_text(
        json.dumps(
            [
                {
                    "image_name": "c21_f000180",
                    "cam_id": 21,
                    "frame_id": 180,
                    "compact_head_mask_files": {"face": str(keep.relative_to(asset_root))},
                },
                {
                    "image_name": "c21_f000300",
                    "cam_id": 21,
                    "frame_id": 300,
                    "compact_head_mask_files": {"face": str(remove.relative_to(asset_root))},
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = module.prune_asset_records_to_protocol_split(asset_root, _protocol(), "test")

    records = json.loads((asset_root / "view_records.json").read_text(encoding="utf-8"))
    assert [record["image_name"] for record in records] == ["c21_f000180"]
    assert keep.exists()
    assert not remove.exists()
    assert summary["removed_record_names"] == ["c21_f000300"]
