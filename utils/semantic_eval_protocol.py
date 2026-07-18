from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


PROTOCOL_SPLITS = ("semantic_train", "calibration", "validation", "test")


def _canonical_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def _normalize_int_list(values, *, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list")
    try:
        normalized = sorted({int(value) for value in values})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integers") from exc
    return normalized


def _normalize_split(split, *, name: str) -> dict:
    if not isinstance(split, dict):
        raise ValueError(f"{name} must be an object")
    return {
        "camera_ids": _normalize_int_list(split.get("camera_ids"), field=f"{name}.camera_ids"),
        "frame_ids": _normalize_int_list(split.get("frame_ids"), field=f"{name}.frame_ids"),
    }


def _record_key(record: dict) -> tuple[int, int]:
    try:
        return int(record["cam_id"]), int(record["frame_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"record is missing integer cam_id/frame_id: {record}") from exc


def _split_keys(split: dict) -> set[tuple[int, int]]:
    return {
        (int(camera_id), int(frame_id))
        for camera_id in split["camera_ids"]
        for frame_id in split["frame_ids"]
    }


def normalize_protocol(protocol: dict) -> dict:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    normalized = dict(protocol)
    for split_name in PROTOCOL_SPLITS:
        normalized[split_name] = _normalize_split(protocol.get(split_name), name=split_name)
    name = str(protocol.get("protocol_name", "")).strip()
    subject = str(protocol.get("subject", "")).strip()
    if not name:
        raise ValueError("protocol_name must be non-empty")
    if not subject:
        raise ValueError("subject must be non-empty")
    normalized["protocol_name"] = name
    normalized["subject"] = subject
    validate_protocol(normalized)
    return normalized


def load_protocol(path: Path | str) -> dict:
    path = Path(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return normalize_protocol(protocol)


def validate_protocol(protocol: dict) -> None:
    normalized_splits = {
        split_name: _normalize_split(protocol.get(split_name), name=split_name)
        for split_name in PROTOCOL_SPLITS
    }
    disjoint_names = ("semantic_train", "validation", "test")
    for index, left_name in enumerate(disjoint_names):
        left_cameras = set(normalized_splits[left_name]["camera_ids"])
        left_keys = _split_keys(normalized_splits[left_name])
        for right_name in disjoint_names[index + 1 :]:
            right_cameras = set(normalized_splits[right_name]["camera_ids"])
            camera_overlap = sorted(left_cameras & right_cameras)
            if camera_overlap:
                raise ValueError(
                    f"camera overlap between {left_name} and {right_name}: {camera_overlap}"
                )
            record_overlap = sorted(left_keys & _split_keys(normalized_splits[right_name]))
            if record_overlap:
                raise ValueError(
                    f"record overlap between {left_name} and {right_name}: {record_overlap}"
                )
    calibration_keys = _split_keys(normalized_splits["calibration"])
    semantic_train_keys = _split_keys(normalized_splits["semantic_train"])
    outside = sorted(calibration_keys - semantic_train_keys)
    if outside:
        raise ValueError(
            "calibration records must be a subset of semantic_train records: "
            f"{outside[:8]}"
        )


def assert_no_forbidden_overlap(protocol: dict) -> None:
    validate_protocol(protocol)


def assert_record_set_matches_split(records: Iterable[dict], split: dict) -> None:
    normalized_split = _normalize_split(split, name="selected")
    allowed = _split_keys(normalized_split)
    outside = []
    for record in records:
        if _record_key(record) not in allowed:
            outside.append(str(record.get("image_name", _record_key(record))))
    if outside:
        raise ValueError(f"records outside protocol split: {', '.join(outside[:8])}")


def select_protocol_records(records: Iterable[dict], protocol: dict, split_name: str) -> list[dict]:
    if split_name not in PROTOCOL_SPLITS:
        raise ValueError(f"unknown protocol split: {split_name}")
    normalized = normalize_protocol(protocol)
    allowed = _split_keys(normalized[split_name])
    selected = [record for record in records if _record_key(record) in allowed]
    selected.sort(key=lambda record: (_record_key(record), str(record.get("image_name", ""))))
    if not selected:
        raise ValueError(f"no records match protocol split: {split_name}")
    assert_record_set_matches_split(selected, normalized[split_name])
    return selected


def protocol_fingerprint(protocol: dict) -> str:
    normalized = normalize_protocol(protocol)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def record_fingerprint(records: Iterable[dict]) -> str:
    canonical_records = sorted(
        [
            {
            "image_name": str(record.get("image_name", "")),
            "cam_id": _record_key(record)[0],
            "frame_id": _record_key(record)[1],
            }
            for record in records
        ],
        key=lambda record: (record["cam_id"], record["frame_id"], record["image_name"]),
    )
    return hashlib.sha256(_canonical_json(canonical_records).encode("utf-8")).hexdigest()


def validate_frozen_config(
    frozen_config: dict,
    *,
    protocol: dict,
    checkpoint_fingerprint: str,
) -> None:
    expected_protocol = protocol_fingerprint(protocol)
    actual_protocol = str(frozen_config.get("protocol_fingerprint", ""))
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"protocol fingerprint mismatch: expected {expected_protocol}, got {actual_protocol}"
        )
    actual_checkpoint = str(frozen_config.get("checkpoint_fingerprint", ""))
    if actual_checkpoint != str(checkpoint_fingerprint):
        raise ValueError(
            "checkpoint fingerprint mismatch: "
            f"expected {checkpoint_fingerprint}, got {actual_checkpoint}"
        )


def write_protocol_provenance(
    output_dir: Path | str,
    protocol: dict,
    selected_records: Iterable[dict],
    *,
    split_name: str,
    source_asset_root: Path | str,
    frozen_config: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_protocol(protocol)
    records = list(selected_records)
    assert_record_set_matches_split(records, normalized[split_name])
    payload = {
        "protocol_name": normalized["protocol_name"],
        "subject": normalized["subject"],
        "split_name": str(split_name),
        "source_asset_root": str(Path(source_asset_root)),
        "protocol_fingerprint": protocol_fingerprint(normalized),
        "record_fingerprint": record_fingerprint(records),
        "selected_record_names": sorted(str(record.get("image_name", "")) for record in records),
        "selected_records": [
            {
                "image_name": str(record.get("image_name", "")),
                "cam_id": _record_key(record)[0],
                "frame_id": _record_key(record)[1],
            }
            for record in sorted(records, key=lambda item: (_record_key(item), str(item.get("image_name", ""))))
        ],
        "frozen_config": frozen_config,
    }
    path = output_dir / "protocol_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
