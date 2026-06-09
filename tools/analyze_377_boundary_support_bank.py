#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.boundary_support_bank import (  # noqa: E402
    boundary_residual_support_stats,
    boundary_support_overlap_stats,
)


def _load_model_tuple(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, (tuple, list)) or len(ckpt) < 1:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    model = ckpt[0]
    if not isinstance(model, (tuple, list)):
        raise ValueError(f"Unexpected GaussianModel capture format: {path}")
    return model


def _binding_state_from_model_tuple(model):
    for idx in (22, 18, 13):
        if len(model) > idx and isinstance(model[idx], dict):
            return model[idx]
    return {}


def _extract(path: Path):
    model = _load_model_tuple(path)
    if len(model) < 10:
        raise ValueError(f"Checkpoint has no boundary direction tags: {path}")
    binding_state = _binding_state_from_model_tuple(model)
    under = model[8].detach().reshape(-1).float()
    over = model[9].detach().reshape(-1).float()
    grow = model[12].detach().reshape(-1, 1).float() if len(model) >= 14 else torch.zeros((under.shape[0], 1))
    shrink = model[13].detach().reshape(-1, 1).float() if len(model) >= 14 else torch.zeros((under.shape[0], 1))
    adopted_under = binding_state.get("boundary_adopted_under_tag", under)
    adopted_over = binding_state.get("boundary_adopted_over_tag", over)
    persistent_under = binding_state.get("boundary_persistent_under_tag", torch.zeros_like(under))
    persistent_over = binding_state.get("boundary_persistent_over_tag", torch.zeros_like(over))
    return {
        "under": under,
        "over": over,
        "grow": grow,
        "shrink": shrink,
        "adopted_under": adopted_under.detach().reshape(-1).float(),
        "adopted_over": adopted_over.detach().reshape(-1).float(),
        "persistent_under": persistent_under.detach().reshape(-1).float(),
        "persistent_over": persistent_over.detach().reshape(-1).float(),
        "has_bank": "boundary_support_bank_version" in binding_state,
    }


def _row(base_name: str, ckpt_name: str, direction: str, base_tensor, cand_tensor, cand):
    stats = boundary_support_overlap_stats(base_tensor, cand_tensor)
    residual = boundary_residual_support_stats(cand["under"], cand["over"], cand["grow"], cand["shrink"])
    return {
        "base": base_name,
        "checkpoint": ckpt_name,
        "direction": direction,
        "has_bank": int(bool(cand["has_bank"])),
        **stats,
        **residual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    base = _extract(args.baseline)
    rows = []
    for ckpt in args.checkpoint:
        cand = _extract(ckpt)
        rows.append(_row(str(args.baseline), str(ckpt), "under", base["under"], cand["under"], cand))
        rows.append(_row(str(args.baseline), str(ckpt), "over", base["over"], cand["over"], cand))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out_tsv": str(args.out_tsv), "out_json": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
