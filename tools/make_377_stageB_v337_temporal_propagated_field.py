#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path


IMAGE_RE = re.compile(r"^c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _parse_views(text: str) -> list[int]:
    text = str(text or "").strip().strip("[]")
    return [int(float(tok.strip())) for tok in text.split(",") if tok.strip()]


def _parse_frame_spec(text: str) -> tuple[int, int, int]:
    vals = _parse_views(text)
    if len(vals) == 1:
        return vals[0], vals[0] + 1, 1
    if len(vals) == 2:
        return vals[0], vals[1], 1
    if len(vals) >= 3:
        return vals[0], vals[1], vals[2]
    return 0, 570, 1


def _split_image_name(name: str) -> tuple[int, int] | None:
    match = IMAGE_RE.match(str(name or ""))
    if not match:
        return None
    return int(match.group("cam")), int(match.group("frame"))


def _target_names(views: list[int], frames: tuple[int, int, int]) -> list[str]:
    start, stop, step = frames
    return [f"c{cam:02d}_f{frame:06d}" for cam in views for frame in range(start, stop, step)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Propagate sparse v336 by_image signed fields along time.")
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--views", default="[21,22,23]")
    parser.add_argument("--frames", default="[0,570,1]")
    parser.add_argument("--radius", default="30", help="Frame radius; use all/nearest/-1 for full nearest propagation.")
    parser.add_argument("--drop-images", default="", help="Comma-separated image names to leave empty.")
    parser.add_argument("--grow-only-images", default="", help="Comma-separated image names where shrink is disabled.")
    args = parser.parse_args()

    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    by_image = source.get("by_image", {}) if isinstance(source, dict) else {}
    keyed: dict[int, list[tuple[int, str, dict]]] = {}
    for image_name, payload in by_image.items():
        parsed = _split_image_name(image_name)
        if parsed is None:
            continue
        cam, frame = parsed
        keyed.setdefault(cam, []).append((frame, image_name, payload))
    for records in keyed.values():
        records.sort(key=lambda item: item[0])

    radius_text = str(args.radius or "").strip().lower()
    radius = None if radius_text in {"all", "nearest", "nearest_all", "-1"} else int(float(radius_text))
    drop_images = {tok.strip() for tok in str(args.drop_images or "").replace(";", ",").split(",") if tok.strip()}
    grow_only_images = {tok.strip() for tok in str(args.grow_only_images or "").replace(";", ",").split(",") if tok.strip()}

    propagated = {}
    source_hits = {}
    assigned = 0
    skipped_radius = 0
    missing_camera = 0
    for target in _target_names(_parse_views(args.views), _parse_frame_spec(args.frames)):
        if target in drop_images:
            continue
        cam_frame = _split_image_name(target)
        if cam_frame is None:
            continue
        cam, frame = cam_frame
        candidates = keyed.get(cam, [])
        if not candidates:
            missing_camera += 1
            continue
        nearest = min(candidates, key=lambda item: (abs(item[0] - frame), item[0]))
        distance = abs(nearest[0] - frame)
        if radius is not None and distance > radius:
            skipped_radius += 1
            continue
        payload = deepcopy(nearest[2])
        payload["source_image_name"] = nearest[1]
        payload["temporal_distance"] = int(distance)
        if target in grow_only_images:
            payload["shrink_point_ids"] = []
            payload["shrink_records"] = []
            payload["temporal_mode"] = "grow_only"
        else:
            payload["temporal_mode"] = "nearest"
        propagated[target] = payload
        source_hits[nearest[1]] = source_hits.get(nearest[1], 0) + 1
        assigned += 1

    out = deepcopy(source)
    out["type"] = "temporal_propagated_group_paired_signed_boundary_field"
    out["source_json"] = str(args.source_json)
    out["by_image"] = propagated
    out["shrink_point_ids"] = []
    out["grow_point_ids"] = []
    out["temporal_propagation"] = {
        "views": _parse_views(args.views),
        "frames": list(_parse_frame_spec(args.frames)),
        "radius": "all" if radius is None else radius,
        "drop_images": sorted(drop_images),
        "grow_only_images": sorted(grow_only_images),
        "source_image_count": len(by_image),
        "assigned_image_count": assigned,
        "skipped_radius_count": skipped_radius,
        "missing_camera_count": missing_camera,
        "source_hits": source_hits,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["temporal_propagation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
