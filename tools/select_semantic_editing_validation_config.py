#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.semantic_eval_protocol import load_protocol, protocol_fingerprint


def select_validation_candidate(rows, *, minimum_retention: float = 0.60) -> dict:
    eligible = [
        dict(row)
        for row in rows
        if float(row.get("aggregate_target_retention", 0.0)) >= float(minimum_retention)
    ]
    if not eligible:
        raise ValueError(
            f"no validation candidate reaches target retention >= {float(minimum_retention):.4f}"
        )
    eligible.sort(
        key=lambda row: (
            float(row["mean_actionable_footprint_leakage"]),
            -float(row.get("mean_boundary_f1", 0.0)),
            int(row.get("boundary_radius", 0)),
            float(row.get("soft_threshold", 0.0)),
            -float(row.get("allowed_support_fraction", 0.0)),
            float(row.get("actionable_support_fraction", 0.0)),
            float(row.get("support_threshold", 0.0)),
        )
    )
    return eligible[0]


def write_frozen_validation_config(
    path: Path | str,
    candidate: dict,
    *,
    protocol: dict,
    checkpoint_fingerprint: str,
    bank_fingerprint: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_name": str(protocol["protocol_name"]),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "bank_fingerprint": str(bank_fingerprint),
        "selection_objective": [
            "min_mean_actionable_footprint_leakage",
            "max_mean_boundary_f1",
            "min_boundary_radius",
            "max_allowed_support_fraction",
            "min_actionable_support_fraction",
        ],
        "selected": dict(candidate),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("candidates", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError("validation input JSON must contain a candidate list")
        return [dict(row) for row in payload]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a strict semantic validation configuration.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--checkpoint-fingerprint", required=True)
    parser.add_argument("--bank-fingerprint", required=True)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    selected = select_validation_candidate(
        _read_rows(args.candidates),
        minimum_retention=float(args.minimum_retention),
    )
    path = write_frozen_validation_config(
        args.output,
        selected,
        protocol=protocol,
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        bank_fingerprint=args.bank_fingerprint,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
