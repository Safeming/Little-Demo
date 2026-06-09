#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _image_view(image_name: str) -> str:
    return str(image_name or "").split("_", 1)[0]


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _selected_pairs_from_validation(path: Path, views: set[str], max_per_view: int) -> set[str]:
    if not path.exists():
        return set()
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "keep":
                continue
            if views and _image_view(row.get("image_name", "")) not in views:
                continue
            rows.append(row)
    rows.sort(
        key=lambda row: (
            _image_view(row.get("image_name", "")),
            -_float(row, "target_gain"),
            _float(row, "outer_delta_control"),
            _float(row, "opacity_outer_delta_control"),
            _float(row, "hard_delta_control"),
            row.get("pair_id", ""),
        )
    )
    out: set[str] = set()
    counts: dict[str, int] = {}
    for row in rows:
        view = _image_view(row.get("image_name", ""))
        if max_per_view > 0 and counts.get(view, 0) >= max_per_view:
            continue
        pair_id = str(row.get("pair_id", "") or "")
        if not pair_id:
            continue
        out.add(pair_id)
        counts[view] = counts.get(view, 0) + 1
    return out


def _asset_pairs(data: dict, views: set[str]) -> set[str]:
    pairs = set()
    for group in data.get("action_groups", []):
        if not isinstance(group, dict):
            continue
        if views and _image_view(group.get("image_name", "")) not in views:
            continue
        pair_id = str(group.get("pair_id", "") or "")
        if pair_id:
            pairs.add(pair_id)
    return pairs


def _extend_unique(dst: list, src: list, pair_ids: set[str], seen_keys: set[tuple[str, str]]) -> None:
    for item in src:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id", "") or "")
        if pair_id not in pair_ids:
            continue
        key = (pair_id, str(item.get("component_key", item.get("key", "")) or ""))
        if key in seen_keys:
            continue
        dst.append(item)
        seen_keys.add(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge v371 portfolio with validated v373b c22 add-ons.")
    parser.add_argument("--base-json", required=True, type=Path)
    parser.add_argument("--addon-json", required=True, type=Path)
    parser.add_argument("--addon-validation-tsv", required=True, type=Path)
    parser.add_argument("--addon-views", default="c22")
    parser.add_argument("--max-addon-per-view", type=int, default=4)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--summary-tsv", required=True, type=Path)
    args = parser.parse_args()

    base = json.loads(args.base_json.read_text(encoding="utf-8"))
    addon = json.loads(args.addon_json.read_text(encoding="utf-8"))
    views = {item.strip() for item in args.addon_views.split(",") if item.strip()}
    base_pairs = _asset_pairs(base, set())
    addon_asset_pairs = _asset_pairs(addon, views)
    validation_pairs = _selected_pairs_from_validation(args.addon_validation_tsv, views, args.max_addon_per_view)
    # v374 is a portfolio merge, so the source of truth is the v373b selected
    # asset. Validation rows are retained for provenance, but may include kept
    # alternatives that were not selected into the final v373b asset.
    addon_pairs = set(sorted(addon_asset_pairs))
    if args.max_addon_per_view > 0:
        addon_pairs = set(sorted(addon_pairs)[: args.max_addon_per_view])
    addon_pairs = {pair for pair in addon_pairs if pair not in base_pairs}
    selected_pairs = set(base_pairs) | addon_pairs

    groups = []
    children = []
    actions = []
    seen_group: set[tuple[str, str]] = set()
    seen_child: set[tuple[str, str]] = set()
    seen_action: set[tuple[str, str]] = set()
    _extend_unique(groups, base.get("action_groups", []), base_pairs, seen_group)
    _extend_unique(children, base.get("children", []), base_pairs, seen_child)
    _extend_unique(actions, base.get("actions", []), base_pairs, seen_action)
    _extend_unique(groups, addon.get("action_groups", []), addon_pairs, seen_group)
    _extend_unique(children, addon.get("children", []), addon_pairs, seen_child)
    _extend_unique(actions, addon.get("actions", []), addon_pairs, seen_action)

    payload = {
        **{key: value for key, value in base.items() if key not in ("action_groups", "children", "actions")},
        "version": "v374_portfolio_merge_grouped_actuator_asset",
        "policy": "v371 selected portfolio plus raw-validated v373b add-on groups for held-out c22 coverage.",
        "group_count": len(groups),
        "child_count": len(children),
        "action_count": len(actions),
        "action_groups": groups,
        "children": children,
        "actions": actions,
        "portfolio_merge": {
            "base_json": str(args.base_json),
            "addon_json": str(args.addon_json),
            "addon_validation_tsv": str(args.addon_validation_tsv),
            "base_pair_count": len(base_pairs),
            "addon_pair_count": len(addon_pairs),
            "addon_asset_pair_count": len(addon_asset_pairs),
            "addon_validation_pair_count": len(validation_pairs),
            "addon_pair_ids": sorted(addon_pairs),
            "selected_pair_count": len(selected_pairs),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload["source"] = {**(payload.get("source") or {}), "v374_base": str(args.base_json)}
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["base_pairs", "addon_pairs", "selected_pairs", "children", "actions", "out_json"])
        writer.writerow([len(base_pairs), len(addon_pairs), len(selected_pairs), len(children), len(actions), str(args.out_json)])
    print(json.dumps(payload["portfolio_merge"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
