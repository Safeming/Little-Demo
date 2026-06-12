#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_part_label_bank


def _copy_array(value):
    arr = np.asarray(value)
    return arr.copy()


def _scalar_str(bank: dict, key: str, default: str = "") -> str:
    if key not in bank:
        return default
    return str(np.asarray(bank[key]).item())


def _scalar_int(bank: dict, key: str, default: int = 0) -> int:
    if key not in bank:
        return int(default)
    return int(np.asarray(bank[key]).item())


def _require_soft_weights(bank: dict, *, name: str) -> np.ndarray:
    if "soft_edit_weights" not in bank:
        raise ValueError(f"{name} bank does not contain soft_edit_weights")
    weights = np.asarray(bank["soft_edit_weights"], dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != len(PART_NAMES):
        raise ValueError(f"{name} soft_edit_weights must have shape [N, {len(PART_NAMES)}]")
    return weights


def _validate_parts(parts: tuple[str, ...] | list[str]) -> list[str]:
    if not parts:
        raise ValueError("at least one override part is required")
    invalid = [part for part in parts if part not in PART_NAMES]
    if invalid:
        raise ValueError(f"unknown override part(s): {invalid}")
    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(str(part))
    return deduped


def build_hybrid_bank(
    base_bank: dict,
    override_bank: dict,
    *,
    parts: tuple[str, ...] | list[str],
    source_type: str = "hybrid_soft_channels",
) -> tuple[dict, dict]:
    override_parts = _validate_parts(parts)
    base_weights = _require_soft_weights(base_bank, name="base")
    override_weights = _require_soft_weights(override_bank, name="override")
    if base_weights.shape[0] != override_weights.shape[0]:
        raise ValueError(
            "base and override soft_edit_weights point count mismatch: "
            f"{base_weights.shape[0]} != {override_weights.shape[0]}"
        )

    hybrid = {key: _copy_array(value) for key, value in base_bank.items()}
    hybrid_weights = base_weights.copy()
    channel_deltas = {}
    for part in override_parts:
        part_index = PART_NAMES.index(part)
        before = hybrid_weights[:, part_index].copy()
        hybrid_weights[:, part_index] = override_weights[:, part_index]
        channel_deltas[part] = {
            "base_weight_sum": float(np.sum(before)),
            "override_weight_sum": float(np.sum(override_weights[:, part_index])),
            "mean_absolute_delta": float(np.mean(np.abs(override_weights[:, part_index] - before))),
        }
    hybrid["soft_edit_weights"] = hybrid_weights.astype(np.float32, copy=False)

    source_suffix = "_".join(override_parts)
    hybrid["source_type"] = np.array(f"{source_type}_{source_suffix}")
    summary = {
        "mode": "soft_channels",
        "point_count": int(base_weights.shape[0]),
        "part_names": list(PART_NAMES),
        "override_parts": override_parts,
        "unchanged_parts": [part for part in PART_NAMES if part not in override_parts],
        "source_type": str(hybrid["source_type"]),
        "base_source_type": _scalar_str(base_bank, "source_type"),
        "override_source_type": _scalar_str(override_bank, "source_type"),
        "channel_deltas": channel_deltas,
    }
    return hybrid, summary


def save_hybrid_bank(path: Path, hybrid: dict) -> None:
    save_part_label_bank(
        path,
        part_label=hybrid["part_label"],
        confidence=hybrid["confidence"],
        vote_count=hybrid["vote_count"],
        per_part_votes=hybrid["per_part_votes"],
        visible_vote_count=hybrid["visible_vote_count"],
        conflict_count=hybrid["conflict_count"],
        source_checkpoint=_scalar_str(hybrid, "source_checkpoint"),
        source_asset_root=_scalar_str(hybrid, "source_asset_root"),
        source_iteration=_scalar_int(hybrid, "source_iteration"),
        semantic_probs=hybrid.get("semantic_probs"),
        semantic_margin=hybrid.get("semantic_margin"),
        reliable_mask=hybrid.get("reliable_mask"),
        editable_label=hybrid.get("editable_label"),
        soft_edit_weights=hybrid["soft_edit_weights"],
        neighbor_fill_mask=hybrid.get("neighbor_fill_mask"),
        source_type=_scalar_str(hybrid, "source_type", "hybrid_soft_channels"),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hybrid semantic part label bank by replacing selected soft channels.")
    parser.add_argument("--base-bank", required=True, type=Path)
    parser.add_argument("--override-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", required=True, choices=list(PART_NAMES))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--copy-base-sidecars", action="store_true")
    parser.add_argument("--source-type", default="hybrid_soft_channels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_bank = load_part_label_bank(args.base_bank)
    override_bank = load_part_label_bank(args.override_bank)
    hybrid, summary = build_hybrid_bank(
        base_bank,
        override_bank,
        parts=tuple(args.parts),
        source_type=str(args.source_type),
    )
    summary["base_bank"] = str(args.base_bank)
    summary["override_bank"] = str(args.override_bank)
    summary["output"] = str(args.output)
    save_hybrid_bank(args.output, hybrid)

    if args.copy_base_sidecars:
        output_dir = args.output.parent
        for name in ("manifest.json", "summary.json", "part_label_preview.ply"):
            src = args.base_bank.parent / name
            if src.exists():
                shutil.copy2(src, output_dir / name)

    if args.summary_json is not None:
        _write_json(args.summary_json, summary)
    if args.manifest_json is not None:
        _write_json(
            args.manifest_json,
            {
                "part_label_bank": str(args.output),
                "source_type": summary["source_type"],
                "base_bank": str(args.base_bank),
                "override_bank": str(args.override_bank),
                "override_parts": summary["override_parts"],
                "unchanged_parts": summary["unchanged_parts"],
                "soft_edit_weight_field": "soft_edit_weights",
                "soft_edit_part_names": list(PART_NAMES),
            },
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
