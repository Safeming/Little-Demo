from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def frozen_method_fingerprint(payload: Mapping) -> str:
    canonical = {
        str(key): value for key, value in payload.items() if not str(key).startswith("_")
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_contract(payload: Mapping) -> None:
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("frozen semantic method schema_version must be 1")
    if str(payload.get("status", "")) != "frozen":
        raise ValueError("frozen semantic method status must be frozen")
    if str(payload.get("primary_method", "")) != "A5":
        raise ValueError("frozen semantic method primary_method must be A5")
    extensions = [str(value) for value in payload.get("extension_methods", [])]
    if extensions != ["A6"]:
        raise ValueError("frozen semantic method extension_methods must be [A6]")
    components = payload.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("frozen semantic method requires components")
    footprint = components.get("footprint_evidence_calibration")
    if not isinstance(footprint, Mapping):
        raise ValueError("frozen semantic method requires footprint evidence calibration")
    if str(footprint.get("mode", "")) != "evidence-calibrated":
        raise ValueError("A5 calibration mode must be evidence-calibrated")
    if str(footprint.get("output_field", "")) != "soft_edit_weights":
        raise ValueError("A5 output field must be soft_edit_weights")
    extension = components.get("target_support_extension")
    if not isinstance(extension, Mapping) or str(extension.get("method", "")) != "A6":
        raise ValueError("frozen semantic method requires the A6 target/support extension")
    reporting = payload.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError("frozen semantic method requires reporting metadata")
    if str(reporting.get("main_table_method", "")) != "A5":
        raise ValueError("main table method must be A5")
    if [str(value) for value in reporting.get("ablation_only", [])] != ["A6"]:
        raise ValueError("A6 must be ablation-only")


def load_frozen_semantic_method(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen semantic method must be a JSON object")
    _validate_contract(payload)
    payload["_source"] = str(source.resolve())
    payload["_fingerprint"] = frozen_method_fingerprint(payload)
    return payload


_A7_REQUIRED_VALUES = {
    "schema_version": 1,
    "freeze_id": "a7_temporal_reliable_v1",
    "status": "frozen",
    "base_method": "A5",
    "output_field": "soft_edit_weights",
    "runtime_state": False,
    "retrain_avatar": False,
    "checkpoint_mutation": False,
    "evidence_cameras": ["c01", "c05", "c09", "c13"],
    "evidence_frame_start": 0,
    "evidence_frame_end": 570,
    "evidence_frame_stride": 5,
    "validation_cameras": ["c17", "c18", "c19", "c20"],
    "validation_frame_stride": 5,
    "retrospective_test_cameras": ["c21"],
    "frozen_test_cameras": ["c22", "c23"],
    "parts": ["hair", "face", "upper", "lower", "shoes", "skin"],
}


def validate_a7_temporal_contract(
    payload: Mapping,
    *,
    base_method: Mapping,
) -> None:
    _validate_contract(base_method)
    for field in ("evidence_cameras", "validation_cameras"):
        if "c21" in [str(value) for value in payload.get(field, [])]:
            raise ValueError(f"A7 contract must not use c21 in {field}")

    for field, expected in _A7_REQUIRED_VALUES.items():
        if payload.get(field) != expected:
            raise ValueError(f"A7 contract {field} must be {expected!r}")

    base_fingerprint = frozen_method_fingerprint(base_method)
    if payload.get("base_method_freeze_fingerprint") != base_fingerprint:
        raise ValueError("A7 contract base A5 fingerprint does not match base method")

    min_pair_support = payload.get("min_pair_support")
    if not isinstance(min_pair_support, int) or min_pair_support <= 0:
        raise ValueError("A7 contract min_pair_support must be a positive integer")
    minimum_coverage = payload.get("minimum_evidence_support_coverage")
    if not isinstance(minimum_coverage, (int, float)) or not 0 <= minimum_coverage <= 1:
        raise ValueError(
            "A7 contract minimum_evidence_support_coverage must be in [0, 1]"
        )


def load_a7_temporal_contract(
    path: str | Path,
    base_method_path: str | Path,
) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A7 temporal contract must be a JSON object")
    base_method = load_frozen_semantic_method(base_method_path)
    validate_a7_temporal_contract(payload, base_method=base_method)
    payload["_source"] = str(source.resolve())
    payload["_fingerprint"] = frozen_method_fingerprint(payload)
    payload["_base_method_source"] = base_method["_source"]
    return payload


def _require_matrix(bank: Mapping | None, field: str, *, owner: str) -> None:
    if bank is None:
        raise ValueError(f"{owner} requires a bank containing {field}")
    if field not in bank:
        raise ValueError(f"{owner} bank is missing required field: {field}")
    matrix = np.asarray(bank[field])
    if matrix.ndim != 2:
        raise ValueError(f"{owner} field {field} must be a 2D matrix")


def validate_frozen_method_assets(
    frozen: Mapping,
    *,
    requested_methods: Sequence[str],
    footprint_bank: Mapping | None,
    evidence_bank: Mapping | None,
) -> None:
    _validate_contract(frozen)
    requested = {str(value) for value in requested_methods}
    components = frozen["components"]
    if str(frozen["primary_method"]) in requested:
        if footprint_bank is None:
            raise ValueError("A5 frozen main method requires a footprint bank")
        field = str(components["footprint_evidence_calibration"]["output_field"])
        _require_matrix(footprint_bank, field, owner="A5 footprint")
    if "A6" in requested:
        extension = components["target_support_extension"]
        _require_matrix(
            evidence_bank,
            str(extension["target_field"]),
            owner="A6 evidence",
        )
        _require_matrix(
            evidence_bank,
            str(extension["support_field"]),
            owner="A6 evidence",
        )
