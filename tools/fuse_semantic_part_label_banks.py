#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.part_label_bank import (
    PART_NAMES,
    compute_semantic_margin,
    compute_semantic_reliable_mask,
    compute_soft_edit_weights,
    finalize_trained_semantic_probs,
    load_part_label_bank,
    save_part_label_bank,
)
from utils.semantic_eval_protocol import file_fingerprint
from utils.semantic_posterior_fusion import fuse_semantic_posteriors


def _scalar(bank: dict, key: str):
    return np.asarray(bank[key]).item()


def _validate_sources(trained: dict, voting: dict) -> None:
    for name, bank in (("trained", trained), ("voting", voting)):
        if "semantic_probs" not in bank:
            raise ValueError(f"{name} bank must contain semantic_probs")
        if tuple(str(value) for value in np.asarray(bank["part_names"]).tolist()) != PART_NAMES:
            raise ValueError(f"{name} bank part names do not match canonical order")
    if np.asarray(trained["semantic_probs"]).shape != np.asarray(voting["semantic_probs"]).shape:
        raise ValueError("trained and voting bank point counts or channels differ")
    for key in ("source_checkpoint", "source_iteration"):
        if _scalar(trained, key) != _scalar(voting, key):
            raise ValueError(f"trained and voting bank {key} values differ")


def fuse_part_label_banks(
    trained_path: Path,
    voting_path: Path,
    output_path: Path,
    *,
    voting_alpha: float,
) -> dict:
    trained = load_part_label_bank(trained_path)
    voting = load_part_label_bank(voting_path)
    _validate_sources(trained, voting)
    fused_probs = fuse_semantic_posteriors(
        trained["semantic_probs"],
        voting["semantic_probs"],
        voting_alpha=float(voting_alpha),
    )
    finalized = finalize_trained_semantic_probs(fused_probs, PART_NAMES)
    finalized.pop("source_type", None)
    margin = compute_semantic_margin(finalized["semantic_probs"])
    reliable = compute_semantic_reliable_mask(
        part_label=finalized["part_label"],
        confidence=finalized["confidence"],
        semantic_margin=margin,
    )
    editable_label = finalized["part_label"].copy()
    editable_label[reliable == 0] = -1
    soft_weights = compute_soft_edit_weights(
        semantic_probs=finalized["semantic_probs"],
        confidence=finalized["confidence"],
        semantic_margin=margin,
        reliable_mask=reliable,
        reliable_floor=0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_part_label_bank(
        output_path,
        **finalized,
        semantic_margin=margin,
        reliable_mask=reliable,
        editable_label=editable_label,
        soft_edit_weights=soft_weights,
        source_checkpoint=str(_scalar(trained, "source_checkpoint")),
        source_asset_root=str(_scalar(voting, "source_asset_root")),
        source_iteration=int(_scalar(trained, "source_iteration")),
        source_type="fused_trained_voting_semantic_probs",
    )
    arrays = load_part_label_bank(output_path)
    trained_fingerprint = file_fingerprint(trained_path)
    voting_fingerprint = file_fingerprint(voting_path)
    arrays.update(
        {
            "trained_bank_fingerprint": np.array(trained_fingerprint),
            "voting_bank_fingerprint": np.array(voting_fingerprint),
            "fusion_alpha": np.array(float(voting_alpha), dtype=np.float32),
        }
    )
    np.savez_compressed(output_path, **arrays)
    return {
        "output": str(output_path),
        "point_count": int(fused_probs.shape[0]),
        "fusion_alpha": float(voting_alpha),
        "trained_bank_fingerprint": trained_fingerprint,
        "voting_bank_fingerprint": voting_fingerprint,
        "reliable_count": int(np.sum(reliable > 0)),
        "source_type": "fused_trained_voting_semantic_probs",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse trained and voting semantic posterior banks.")
    parser.add_argument("--trained-bank", required=True, type=Path)
    parser.add_argument("--voting-bank", required=True, type=Path)
    parser.add_argument("--voting-alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = fuse_part_label_banks(
        args.trained_bank,
        args.voting_bank,
        args.output,
        voting_alpha=float(args.voting_alpha),
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
