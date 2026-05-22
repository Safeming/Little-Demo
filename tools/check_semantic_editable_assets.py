#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def _count_files(root, pattern):
    return sum(1 for _ in root.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Validate exported semantic editable asset structure.")
    parser.add_argument("--asset-root", required=True, help="Path to semantic_editable_assets directory.")
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--require-compact", action="store_true")
    parser.add_argument("--require-parser-preview", action="store_true")
    args = parser.parse_args()

    root = Path(args.asset_root)
    errors = []
    if not root.exists():
        errors.append(f"asset root missing: {root}")
    if not root.is_dir():
        errors.append(f"asset root is not a directory: {root}")

    meta_path = root / "meta.json"
    views_path = root / "view_records.json"
    appearance_path = root / "appearance_bank.json"
    motion_bank_path = root / "motion_bank.npz"
    for path in (meta_path, views_path, appearance_path, motion_bank_path):
        if not path.exists():
            errors.append(f"required file missing: {path}")

    meta = {}
    views = []
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid meta.json: {exc}")
    if views_path.exists():
        try:
            views = json.loads(views_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid view_records.json: {exc}")

    if not isinstance(views, list):
        errors.append("view_records.json must be a list")
        views = []
    if len(views) < args.min_views:
        errors.append(f"view count {len(views)} < min_views {args.min_views}")
    if meta and int(meta.get("num_views", -1)) != len(views):
        errors.append(f"meta num_views {meta.get('num_views')} != view_records {len(views)}")

    required_record_keys = {"image_name", "frame_id", "cam_id", "motion_file", "mask_files", "appearance"}
    for idx, record in enumerate(views):
        missing = sorted(required_record_keys - set(record.keys()))
        if missing:
            errors.append(f"view_records[{idx}] missing keys: {missing}")
            continue
        motion_file = record.get("motion_file")
        if motion_file and not (root / motion_file).exists():
            errors.append(f"view_records[{idx}] motion file missing: {motion_file}")

    source_count = _count_files(root, "source_rgb/*.png")
    preview_count = _count_files(root, "preview/*.png")
    motion_count = _count_files(root, "motions/*.npz")
    compact_count = _count_files(root, "compact_head_masks/*/*.png")
    parser_preview_count = _count_files(root, "parser_preview/*.png")

    if source_count < len(views):
        errors.append(f"source_rgb png count {source_count} < view count {len(views)}")
    if preview_count < len(views):
        errors.append(f"preview png count {preview_count} < view count {len(views)}")
    if motion_count < len(views):
        errors.append(f"motion npz count {motion_count} < view count {len(views)}")
    if args.require_compact and compact_count <= 0:
        errors.append("no compact_head_masks exported")
    if args.require_parser_preview and parser_preview_count < len(views):
        errors.append(f"parser_preview png count {parser_preview_count} < view count {len(views)}")

    summary = {
        "asset_root": str(root),
        "num_views": len(views),
        "source_rgb_png": source_count,
        "preview_png": preview_count,
        "parser_preview_png": parser_preview_count,
        "motion_npz": motion_count,
        "compact_head_mask_png": compact_count,
        "mask_export_mode": meta.get("mask_export_mode"),
        "status": "pass" if not errors else "fail",
    }
    print(json.dumps(summary, indent=2))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
