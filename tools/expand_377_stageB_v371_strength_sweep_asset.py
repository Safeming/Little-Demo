#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


def _safe_token(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def _scale_covariance(child: dict, radius_mult: float) -> None:
    scale = float(radius_mult) ** 2
    cov = child.get("canonical_covariance", None)
    if isinstance(cov, list):
        try:
            child["canonical_covariance"] = [
                [float(value) * scale for value in row] for row in cov
            ]
        except Exception:
            pass
    cov6 = child.get("canonical_covariance_6", None)
    if isinstance(cov6, list):
        try:
            child["canonical_covariance_6"] = [float(value) * scale for value in cov6]
        except Exception:
            pass
    try:
        child["canonical_radius"] = float(child.get("canonical_radius", 0.0) or 0.0) * float(radius_mult)
    except Exception:
        pass


def _parse_variants(text: str) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for raw in str(text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        if token in ("base", "normal"):
            variants.append({
                "name": "base",
                "opacity_mult": 1.0,
                "radius_mult": 1.0,
                "self_protect": True,
            })
            continue
        match = re.fullmatch(r"co(?P<co>[0-9.]+)_cr(?P<cr>[0-9.]+)_(?P<protect>np|sp)", token)
        if match is None:
            raise ValueError(
                f"bad variant {token!r}; expected base or co<opacity>_cr<radius>_(np|sp)"
            )
        variants.append({
            "name": token.replace(".", "p"),
            "opacity_mult": float(match.group("co")),
            "radius_mult": float(match.group("cr")),
            "self_protect": match.group("protect") == "sp",
        })
    if not variants:
        raise ValueError("no variants requested")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand v369/v371 seed asset with strength-sweep variants.")
    parser.add_argument("--in-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--variants", default="base,co4_cr1.5_np,co3_cr1.5_np")
    args = parser.parse_args()

    data = json.loads(args.in_json.read_text(encoding="utf-8"))
    variants = _parse_variants(args.variants)
    children_by_pair: dict[str, list[dict]] = {}
    actions_by_pair: dict[str, list[dict]] = {}
    for child in data.get("children", []):
        children_by_pair.setdefault(str(child.get("pair_id", "")), []).append(child)
    for action in data.get("actions", []):
        actions_by_pair.setdefault(str(action.get("pair_id", "")), []).append(action)

    groups = []
    children = []
    actions = []
    for group in data.get("action_groups", []):
        pair_id = str(group.get("pair_id", ""))
        if not pair_id:
            continue
        base_children = children_by_pair.get(pair_id, [])
        base_actions = actions_by_pair.get(pair_id, [])
        if not base_children or not base_actions:
            continue
        for variant in variants:
            name = str(variant["name"])
            if name == "base":
                new_pair_id = pair_id
            else:
                new_pair_id = f"{pair_id}:v371_{name}"
            group_children = []
            group_actions = []
            for child in base_children:
                item = deepcopy(child)
                old_key = str(item.get("component_key", ""))
                item["pair_id"] = new_pair_id
                item["action_group_key"] = new_pair_id
                if name != "base":
                    item["component_key"] = f"{old_key}:v371_{name}"
                item["child_opacity"] = max(
                    0.0,
                    min(float(item.get("child_opacity", 0.0) or 0.0) * float(variant["opacity_mult"]), 1.0),
                )
                item["child_self_protect_enable"] = bool(variant["self_protect"])
                _scale_covariance(item, float(variant["radius_mult"]))
                item["reason"] = "v371_strength_sweep_residual_grouped_inner_micro_child"
                item["v371_strength_variant"] = name
                item["v371_child_opacity_mult"] = float(variant["opacity_mult"])
                item["v371_child_radius_mult"] = float(variant["radius_mult"])
                item["v371_self_protect_enable"] = bool(variant["self_protect"])
                group_children.append(item)
            for action in base_actions:
                item = deepcopy(action)
                old_key = str(item.get("component_key", ""))
                item["pair_id"] = new_pair_id
                if name != "base":
                    item["component_key"] = f"{old_key}:v371_{name}"
                item["v371_strength_variant"] = name
                group_actions.append(item)
            new_group = deepcopy(group)
            new_group["pair_id"] = new_pair_id
            new_group["child_component_keys"] = [str(item.get("component_key", "")) for item in group_children]
            new_group["outer_action_keys"] = [str(item.get("component_key", "")) for item in group_actions]
            new_group["strength_variant"] = name
            new_group["child_opacity_mult"] = float(variant["opacity_mult"])
            new_group["child_radius_mult"] = float(variant["radius_mult"])
            new_group["child_self_protect_enable"] = bool(variant["self_protect"])
            groups.append(new_group)
            children.extend(group_children)
            actions.extend(group_actions)

    payload = {
        **{key: value for key, value in data.items() if key not in ("action_groups", "children", "actions")},
        "version": "v371_strength_sweep_grouped_actuator_asset",
        "policy": (
            "v371 expands each residual grouped actuator into action-level strength variants; "
            "raw group validation must select safe variants before raw gate/training."
        ),
        "group_count": len(groups),
        "child_count": len(children),
        "action_count": len(actions),
        "action_groups": groups,
        "children": children,
        "actions": actions,
        "v371_strength_variants": variants,
    }
    thresholds = dict(payload.get("thresholds", {}))
    thresholds["v371_strength_variants"] = variants
    payload["thresholds"] = thresholds
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.out_json} groups={len(groups)} children={len(children)} "
        f"actions={len(actions)} variants={len(variants)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
