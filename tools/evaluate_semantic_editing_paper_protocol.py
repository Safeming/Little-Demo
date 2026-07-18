#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from utils.part_label_bank import PART_NAMES, compute_semantic_margin, compute_soft_edit_weights


BASELINE_SPECS = OrderedDict(
    (
        ("B0", {"name": "parser_oracle", "oracle": True, "persistent_asset": False}),
        ("B1", {"name": "projected_multiview_voting", "oracle": False, "persistent_asset": True}),
        ("B2", {"name": "hard_trained_label", "oracle": False, "persistent_asset": True}),
        ("B3", {"name": "raw_semantic_probability", "oracle": False, "persistent_asset": True}),
        ("B4", {"name": "confidence_margin", "oracle": False, "persistent_asset": True}),
        ("B5", {"name": "evidence_target_support", "oracle": False, "persistent_asset": True}),
    )
)


def _one_hot_labels(labels, *, part_index: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int16).reshape(-1)
    return (values == int(part_index)).astype(np.float32)


def _required_matrix(bank: dict, field: str, *, part_index: int) -> np.ndarray:
    if field not in bank:
        raise ValueError(f"baseline requires bank field: {field}")
    matrix = np.asarray(bank[field], dtype=np.float32)
    if matrix.ndim != 2 or not (0 <= int(part_index) < matrix.shape[1]):
        raise ValueError(f"{field} must be a 2D part-weight matrix")
    return matrix[:, int(part_index)].astype(np.float32, copy=False)


def resolve_baseline_point_weights(
    baseline: str,
    *,
    trained_bank: dict,
    voting_bank: dict | None,
    part_index: int,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    baseline = str(baseline)
    if baseline not in BASELINE_SPECS:
        raise ValueError(f"unknown baseline: {baseline}")
    if baseline == "B0":
        raise ValueError("B0 parser oracle is a screen-space baseline without Gaussian point weights")
    if baseline == "B1":
        if voting_bank is None:
            raise ValueError("B1 requires a projected multi-view voting bank")
        labels = voting_bank.get("editable_label", voting_bank.get("part_label"))
        if labels is None:
            raise ValueError("B1 voting bank requires editable_label or part_label")
        weights = _one_hot_labels(labels, part_index=part_index)
        support = None
        weight_field = "voting_editable_label"
    elif baseline == "B2":
        labels = trained_bank.get("editable_label", trained_bank.get("part_label"))
        if labels is None:
            raise ValueError("B2 trained bank requires editable_label or part_label")
        weights = _one_hot_labels(labels, part_index=part_index)
        support = None
        weight_field = "editable_label"
    elif baseline == "B3":
        weights = _required_matrix(trained_bank, "semantic_probs", part_index=part_index)
        support = None
        weight_field = "semantic_probs"
    elif baseline == "B4":
        if "semantic_probs" not in trained_bank or "confidence" not in trained_bank:
            raise ValueError("B4 requires semantic_probs and confidence")
        probs = np.asarray(trained_bank["semantic_probs"], dtype=np.float32)
        margin = np.asarray(
            trained_bank.get("semantic_margin", compute_semantic_margin(probs)),
            dtype=np.float32,
        )
        reliable = trained_bank.get("reliable_mask", np.ones((probs.shape[0],), dtype=np.uint8))
        matrix = compute_soft_edit_weights(
            semantic_probs=probs,
            confidence=trained_bank["confidence"],
            semantic_margin=margin,
            reliable_mask=reliable,
        )
        weights = matrix[:, int(part_index)].astype(np.float32, copy=False)
        support = None
        weight_field = "confidence_margin_recomputed"
    else:
        weights = _required_matrix(trained_bank, "edit_target_weights", part_index=part_index)
        support = _required_matrix(trained_bank, "edit_support_weights", part_index=part_index)
        weight_field = "edit_target_weights"
    metadata = {
        "baseline": baseline,
        **BASELINE_SPECS[baseline],
        "weight_field": weight_field,
    }
    return weights, support, metadata


def resolve_parser_oracle_prediction(part_masks: dict[str, np.ndarray], part: str) -> np.ndarray:
    if part not in part_masks:
        raise ValueError(f"parser oracle is missing part mask: {part}")
    return np.asarray(part_masks[part], dtype=np.float32)


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_reports(output_dir: Path | str, result: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reports = (
        ("baseline_summary.csv", "baseline_summary"),
        ("per_part_metrics.csv", "per_part"),
        ("per_view_metrics.csv", "per_view"),
        ("leakage_retention_curve.csv", "curve"),
        ("matched_retention.csv", "matched_retention"),
    )
    for filename, key in reports:
        _write_csv(output_dir / filename, list(result.get(key, [])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate B0-B5 strict semantic editing baselines.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--trained-bank", required=True, type=Path)
    parser.add_argument("--voting-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINE_SPECS), choices=list(BASELINE_SPECS))
    parser.add_argument("--parts", nargs="+", default=list(PART_NAMES), choices=list(PART_NAMES))
    return parser.parse_args(argv)


def main() -> int:
    raise SystemExit("scene evaluation wiring is provided by the strict protocol runner")


if __name__ == "__main__":
    raise SystemExit(main())
