#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROW_RE = re.compile(r":row(?P<row>\d+)")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(data: dict, *keys: str) -> list[dict]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _row_id(item: dict) -> int | None:
    value = item.get("row_index", item.get("csv_row_index", None))
    if value is not None and str(value).strip() != "":
        try:
            return int(float(value))
        except Exception:
            pass
    for key in ("source_component_key", "component_key", "key"):
        text = str(item.get(key, "") or "")
        match = ROW_RE.search(text)
        if match:
            return int(match.group("row"))
    return None


def _source_key(item: dict) -> str:
    return str(item.get("source_component_key", item.get("component_key", item.get("key", ""))) or "").strip()


def _group_key(item: dict, index: int) -> str:
    for key in ("pair_id", "action_group_key", "v361_group_id"):
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    source = _source_key(item) or f"inner_{index}"
    return f"{source}:singleton:{index}"


def _copy_owner(src: dict, dst: dict) -> None:
    for key in (
        "owner_gate",
        "owner_layer",
        "owner_layer_id",
        "owner_region",
        "owner_region_id",
        "owner_joint",
        "owner_joint_id",
    ):
        if src.get(key, None) is not None:
            dst[key] = src[key]


def _globalize_child(child: dict, index: int, opacity: float | None) -> dict:
    out = dict(child)
    source = _source_key(child) or f"child_{index}"
    out["component_key"] = f"v363_global_inner:{source}:m{index}"
    out["source_component_key"] = source
    out["scope"] = "global"
    out["asset_scope"] = "global"
    out["image_name"] = ""
    out["split_child_enable"] = True
    out["direction"] = "inner"
    out["anchor_mode"] = "top_ids_translation"
    out["child_pose_mode"] = "top_ids_translation"
    out["anchor_local_frame"] = False
    out["rotate_covariance_with_anchor"] = False
    out["pair_required"] = False
    out["anchor_owner_gate"] = True
    out["owner_gate"] = True
    out["activation_required"] = True
    out["activation_direction"] = "inner"
    out["activation_pad_px"] = 4.0
    out["activation_ellipse_scale"] = 1.15
    out["activation_min_area"] = 1.0
    out["activation_owner_gate"] = True
    out["activation_owner_primary_only"] = True
    out.setdefault("anchor_knn", 24)
    out.setdefault("anchor_min_points", 3)
    try:
        radius = float(out.get("canonical_radius", 0.0) or 0.0)
    except Exception:
        radius = 0.0
    if radius > 0.0:
        out.setdefault("anchor_radius", radius * 1.6)
    if opacity is not None:
        out["child_opacity"] = max(0.0, min(float(opacity), 1.0))
    out["reason"] = "v363_grouped_global_inner_micro_child"
    return out


def _globalize_outer(action: dict, pair_id: str, index: int, radius_scale: float, score_scale: float) -> dict:
    out = dict(action)
    source = _source_key(action) or f"outer_{index}"
    out["component_key"] = f"v363_global_outer:{source}:a{index}"
    out["source_component_key"] = source
    out["scope"] = "global"
    out["asset_scope"] = "global"
    out["image_name"] = ""
    out["direction"] = "outer"
    out["pair_id"] = pair_id
    out["pair_role"] = "outer_protect_shrink"
    out["mode"] = "paired_local_3d_intersect"
    out["semantic_override"] = True
    out["targeted_only"] = True
    out["owner_gate"] = True
    out["activation_required"] = True
    out["activation_direction"] = "outer"
    out["activation_pad_px"] = 4.0
    out["activation_ellipse_scale"] = 1.15
    out["activation_min_area"] = 1.0
    out["activation_owner_gate"] = True
    out["activation_owner_primary_only"] = True
    out["anchor_mode"] = "semantic_local_frame"
    out["anchor_local_frame"] = True
    out["anchor_owner_gate"] = True
    out.setdefault("anchor_knn", 24)
    out.setdefault("anchor_min_points", 3)
    out["top_ids_enable"] = False
    out["top_ids_only"] = False
    out["local_3d_fallback_top_ids"] = True
    try:
        out["score_scale"] = float(out.get("score_scale", 1.0) or 1.0) * float(score_scale)
    except Exception:
        out["score_scale"] = float(score_scale)
    for key in ("canonical_radius", "canonical_radius_outer"):
        try:
            value = float(out.get(key, 0.0) or 0.0)
        except Exception:
            value = 0.0
        if value > 0.0:
            out[key] = value * float(radius_scale)
    if out.get("canonical_radius_outer", None) is None and out.get("canonical_radius", None) is not None:
        out["canonical_radius_outer"] = out["canonical_radius"]
    out["reason"] = "v363_grouped_global_outer_protect"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a grouped global canonical asset with inner micro-children and paired outer protect actions."
    )
    parser.add_argument("--inner-json", required=True, type=Path)
    parser.add_argument("--outer-json", action="append", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--child-opacity", default=0.04, type=float)
    parser.add_argument("--outer-radius-scale", default=0.70, type=float)
    parser.add_argument("--outer-score-scale", default=0.65, type=float)
    parser.add_argument("--max-outer-per-inner", default=2, type=int)
    parser.add_argument("--require-outer-pair", action="store_true")
    args = parser.parse_args()

    inner_data = _load_json(args.inner_json)
    inner_rows = _rows(inner_data, "children", "actions")
    outer_rows: list[dict] = []
    for path in args.outer_json:
        data = _load_json(path)
        outer_rows.extend(_rows(data, "actions", "children"))

    outer_by_row: dict[int, list[dict]] = {}
    for action in outer_rows:
        direction = str(action.get("direction", "") or "").strip().lower()
        if direction not in ("outer", "over"):
            continue
        row = _row_id(action)
        if row is None:
            continue
        outer_by_row.setdefault(row, []).append(action)

    grouped_inner: dict[str, list[dict]] = {}
    for idx, child in enumerate(inner_rows):
        direction = str(child.get("direction", "inner") or "inner").strip().lower()
        if direction not in ("inner", "under"):
            continue
        grouped_inner.setdefault(_group_key(child, idx), []).append(child)

    children = []
    actions = []
    pairs = []
    used_outer = set()
    for group_idx, (inner_group_key, group_children) in enumerate(grouped_inner.items()):
        if not group_children:
            continue
        child = group_children[0]
        row = _row_id(child)
        candidates = []
        if row is not None:
            for delta in (5, 4, 3, 2, 1, 0, 6, 7, 8):
                candidates.extend(outer_by_row.get(row + delta, []))
                if len(candidates) >= max(int(args.max_outer_per_inner), 0):
                    break
        selected = []
        for action in candidates:
            key = (action.get("component_key"), action.get("row_index"))
            if key in used_outer:
                continue
            selected.append(action)
            used_outer.add(key)
            if len(selected) >= max(int(args.max_outer_per_inner), 0):
                break
        if bool(args.require_outer_pair) and not selected:
            continue
        pair_id = f"v363:{inner_group_key}"
        group_global_children = []
        for child_offset, group_child in enumerate(group_children):
            global_child = _globalize_child(group_child, len(children) + child_offset, args.child_opacity)
            global_child["pair_id"] = pair_id
            global_child["source_group_key"] = inner_group_key
            _copy_owner(group_child, global_child)
            group_global_children.append(global_child)
        children.extend(group_global_children)
        pair = {
            "pair_id": pair_id,
            "inner_child_keys": [item["component_key"] for item in group_global_children],
            "outer_action_keys": [],
            "source_inner_row": row,
            "source_group_key": inner_group_key,
            "micro_count": len(group_global_children),
        }
        for outer_idx, action in enumerate(selected):
            global_outer = _globalize_outer(action, pair_id, len(actions), args.outer_radius_scale, args.outer_score_scale)
            global_outer["source_group_key"] = inner_group_key
            actions.append(global_outer)
            pair["outer_action_keys"].append(global_outer["component_key"])
        pairs.append(pair)

    payload = {
        "version": "v363_grouped_canonical_micro_child_asset",
        "policy": (
            "Global grouped asset: residual-shaped inner micro-children are paired with outer "
            "protect/shrink actions consumed by the existing signed dynamic local actuator."
        ),
        "source": {
            "inner_json": str(args.inner_json),
            "outer_json": [str(path) for path in args.outer_json],
        },
        "thresholds": {
            "child_opacity": float(args.child_opacity),
            "outer_radius_scale": float(args.outer_radius_scale),
            "outer_score_scale": float(args.outer_score_scale),
            "max_outer_per_inner": int(args.max_outer_per_inner),
        },
        "child_count": len(children),
        "action_count": len(actions),
        "pair_count": len(pairs),
        "children": children,
        "actions": actions,
        "pairs": pairs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out_json} children={len(children)} outer_actions={len(actions)} pairs={len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
