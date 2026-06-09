#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _owner_from_source(child: dict) -> dict:
    out = {}
    for key in (
        "owner_layer",
        "owner_layer_id",
        "owner_region",
        "owner_region_id",
        "owner_joint",
        "owner_joint_id",
    ):
        if child.get(key, None) is not None:
            out[key] = child[key]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert validated image-local split-child assets into generalized canonical semantic split-child assets."
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--anchor-knn", default=24, type=int)
    parser.add_argument("--anchor-radius-scale", default=1.6, type=float)
    parser.add_argument("--child-opacity", default=None, type=float)
    parser.add_argument("--radius-scale", default=1.0, type=float)
    parser.add_argument("--require-owner", action="store_true")
    args = parser.parse_args()

    data = _load_json(args.input_json)
    rows = data.get("children") or data.get("actions") or []
    children = []
    skipped = 0
    for index, child in enumerate(rows):
        if not isinstance(child, dict):
            skipped += 1
            continue
        owner = _owner_from_source(child)
        if args.require_owner and not owner:
            skipped += 1
            continue
        item = dict(child)
        item.update(owner)
        source_key = str(item.get("source_component_key", item.get("component_key", f"child_{index}")))
        item["component_key"] = f"v362_global:{source_key}:m{index}"
        item["source_component_key"] = source_key
        item["scope"] = "global"
        item["asset_scope"] = "global"
        item["image_name"] = ""
        item["split_child_enable"] = True
        item["anchor_mode"] = "semantic_local_frame"
        item["child_pose_mode"] = "semantic_local_frame"
        item["anchor_local_frame"] = True
        item["rotate_covariance_with_anchor"] = True
        item["anchor_owner_gate"] = True
        item["owner_gate"] = True
        item["anchor_knn"] = int(args.anchor_knn)
        item["anchor_min_points"] = 3
        try:
            radius = float(item.get("canonical_radius", 0.0) or 0.0)
        except Exception:
            radius = 0.0
        if radius > 0.0:
            item["anchor_radius"] = radius * float(args.anchor_radius_scale)
        if args.child_opacity is not None:
            item["child_opacity"] = max(0.0, min(float(args.child_opacity), 1.0))
        item["child_radius_scale"] = float(args.radius_scale)
        item["reason"] = "v362_generalized_from_validated_micro_child"
        item["v362_generalized_from_image_name"] = child.get("image_name", "")
        item["v362_generalized_from_component_key"] = child.get("component_key", "")
        children.append(item)

    payload = {
        "version": "v362_generalized_split_child_asset",
        "policy": (
            "Global canonical split-child asset. Children are selected for every view/frame and bound "
            "through semantic owner-gated canonical KNN local frames instead of image_name keyed patches."
        ),
        "source": {
            "input_json": str(args.input_json),
            "input_version": data.get("version", ""),
        },
        "thresholds": {
            "anchor_knn": int(args.anchor_knn),
            "anchor_radius_scale": float(args.anchor_radius_scale),
            "child_opacity": args.child_opacity,
            "radius_scale": float(args.radius_scale),
            "require_owner": bool(args.require_owner),
        },
        "child_count": len(children),
        "skipped_count": skipped,
        "children": children,
        "actions": children,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out_json} child_count={len(children)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
