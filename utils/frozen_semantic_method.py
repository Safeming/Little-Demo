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
    "max_weight_scale_from_posterior": 1.0,
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

    freeze_id = str(payload.get("freeze_id", ""))
    if freeze_id not in {
        "a7_temporal_reliable_v1",
        "a7_temporal_reliable_v1_1",
        "a7_renderer_aligned_v2_canary_377",
        "a7_renderer_objective_v3_canary_377",
        "a7_sparse_robust_v4_canary_377",
    }:
        raise ValueError("unsupported A7 contract freeze_id")
    required_values = dict(_A7_REQUIRED_VALUES)
    required_values["freeze_id"] = freeze_id
    for field, expected in required_values.items():
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
    if freeze_id == "a7_temporal_reliable_v1_1":
        expected_policy = {
            "boundary_dominance_margin": 0.2,
            "minimum_carrier_support_ratio": 0.5,
            "minimum_carrier_existing_weight": 0.2,
            "carrier_ranking": "reliability_support_target_posterior",
        }
        for field, expected in expected_policy.items():
            if payload.get(field) != expected:
                raise ValueError(f"A7 v1.1 contract {field} must be {expected!r}")
    if freeze_id == "a7_renderer_aligned_v2_canary_377":
        expected_policy = {
            "subject": "377",
            "evidence_mode": "renderer_aligned",
            "renderer_attribution": "colors_gradient",
            "coverage_freeze_threshold": 0.8,
            "frozen_parts": ["face", "upper", "shoes", "skin"],
            "selection_threshold": 0.2,
            "preserve_a5_selection_topology": True,
            "renderer_contribution_epsilon": 1.0e-8,
            "renderer_boundary_radius": 6,
            "minimum_carrier_support_ratio": 0.5,
            "minimum_carrier_existing_weight": 0.2,
            "carrier_ranking": "reliability_support_target_posterior",
        }
        for field, expected in expected_policy.items():
            if payload.get(field) != expected:
                raise ValueError(f"A7 v2 contract {field} must be {expected!r}")
    if freeze_id == "a7_renderer_objective_v3_canary_377":
        expected_policy = {
            "subject": "377",
            "evidence_mode": "renderer_aligned_sequence",
            "renderer_attribution": "colors_gradient",
            "coverage_freeze_threshold": 0.8,
            "frozen_parts": ["face", "upper", "shoes", "skin"],
            "selection_threshold": 0.2,
            "preserve_a5_selection_topology": True,
            "renderer_contribution_epsilon": 1.0e-8,
            "renderer_boundary_radius": 6,
            "minimum_carrier_support_ratio": 0.5,
            "minimum_carrier_existing_weight": 0.2,
            "carrier_ranking": "reliability_support_target_posterior",
            "lambda_outer": 1.0,
            "lambda_boundary": 0.5,
            "lambda_target": 0.25,
            "maximum_weight_above_a5": 0.0,
            "candidate_policies": [
                {
                    "name": "bounded_damping_005",
                    "minimum_weight_ratio_from_a5": 0.95,
                    "rho": 0.95,
                    "restore_target_mass": False,
                    "maximum_part_weight_l1_from_a5": 12.0,
                },
                {
                    "name": "bounded_retention_010",
                    "minimum_weight_ratio_from_a5": 0.9,
                    "rho": 0.95,
                    "restore_target_mass": True,
                    "maximum_part_weight_l1_from_a5": 12.0,
                },
            ],
        }
        for field, expected in expected_policy.items():
            if payload.get(field) != expected:
                raise ValueError(f"A7 v3 contract {field} must be {expected!r}")
    if freeze_id == "a7_sparse_robust_v4_canary_377":
        expected_policy = {
            "subject": "377",
            "evidence_mode": "renderer_aligned_sequence_sparse_robust",
            "source_evidence_sha256": "17142db0063bb63b84b8f0a777e9ca9ec21c1d4084608aa791f79e8e595ab965",
            "source_evidence_contract_fingerprint": "816c975183a868c97ab156d866d1411d03df1265e9ad98f885cf50f3d637e508",
            "processed_parts": ["hair", "lower"],
            "frozen_parts": ["face", "upper", "shoes", "skin"],
            "selection_threshold": 0.2,
            "preserve_a5_selection_topology": True,
            "maximum_weight_above_a5": 0.0,
            "coordinate_reduction_fractions": [0.05, 0.1],
            "maximum_changed_fraction": 0.2,
            "minimum_camera_target_ratio": 0.98,
            "objective_mean_weight": 0.25,
            "objective_absolute_adjacent_weight": 0.05,
            "loco_all_folds_required": True,
            "minimum_active_temporal_gain": 0.005,
            "maximum_spatial_burden_worsening": 0.02,
            "maximum_part_soft_iou_drop": 0.01,
            "maximum_macro_miou_drop": 0.01,
            "maximum_micro_iou_drop": 0.005,
        }
        for field, expected in expected_policy.items():
            if payload.get(field) != expected:
                raise ValueError(f"A7 v4 contract {field} must be {expected!r}")


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


def _file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_scalar_string(payload: Mapping, key: str) -> str:
    if key not in payload:
        raise ValueError(f"A7 bank is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"A7 bank {key} must be a scalar string")
    return str(value)


def validate_a7_bank_against_contract(
    bank: Mapping,
    *,
    contract: Mapping,
    a5_bank_path: str | Path,
) -> dict:
    if _mapping_scalar_string(bank, "method_id") != "A7":
        raise ValueError("A7 bank method_id must be A7")
    if _mapping_scalar_string(bank, "base_method") != "A5":
        raise ValueError("A7 bank base_method must be A5")
    expected_base_sha = _file_sha256(a5_bank_path)
    if _mapping_scalar_string(bank, "base_bank_sha256") != expected_base_sha:
        raise ValueError("A7 bank base A5 bank SHA-256 mismatch")
    if _mapping_scalar_string(bank, "base_method_freeze_fingerprint") != str(
        contract["base_method_freeze_fingerprint"]
    ):
        raise ValueError("A7 bank base method freeze fingerprint mismatch")
    if _mapping_scalar_string(bank, "a7_contract_fingerprint") != str(
        contract["_fingerprint"]
    ):
        raise ValueError("A7 bank contract fingerprint mismatch")
    return {
        "canonical_selection_fixed_across_frames": True,
        "base_a5_bank_sha256": expected_base_sha,
        "base_method_freeze_fingerprint": str(
            contract["base_method_freeze_fingerprint"]
        ),
        "a7_contract_fingerprint": str(contract["_fingerprint"]),
        "a7_bank_fingerprint": _mapping_scalar_string(
            bank, "output_bank_fingerprint"
        ),
    }


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
