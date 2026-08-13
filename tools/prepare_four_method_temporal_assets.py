#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.four_method_paper_evidence import (
    build_temporal_windows,
    resolve_frozen_operating_point,
)
from utils.semantic_eval_protocol import file_fingerprint


SUBJECTS = ("377", "386", "394")
METHODS = ("saga", "gaussian_grouping", "sggs", "a5")


def build_record_names(windows) -> list[str]:
    names = [
        f"c{int(window['camera']):02d}_f{int(frame):06d}"
        for window in windows
        for frame in window["frames"]
    ]
    if len(names) != len(set(names)):
        raise ValueError("temporal record names are not unique")
    return names


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_temporal_source_coverage(
    *,
    parser_root: Path | str,
    subject: str,
    cameras=(21, 22, 23),
    frames=None,
) -> dict:
    parser_root = Path(parser_root)
    subject = str(subject)
    if frames is None:
        frames = sorted(
            {
                int(frame)
                for window in build_temporal_windows(cameras=cameras)
                for frame in window["frames"]
            }
        )
    missing = []
    paths = []
    for camera in cameras:
        for frame in frames:
            path = (
                parser_root
                / f"CoreView_{subject}"
                / "mask_cihp"
                / f"Camera_B{int(camera)}"
                / f"{int(frame):06d}.png"
            )
            if path.is_file():
                paths.append(path)
            else:
                missing.append(path)
    if missing:
        raise ValueError(
            "missing parser masks: " + ", ".join(str(path) for path in missing[:8])
        )
    return {
        "subject": subject,
        "camera_count": len(tuple(cameras)),
        "frame_count_per_camera": len(tuple(frames)),
        "parser_mask_count": len(paths),
        "first_path": str(paths[0]),
        "last_path": str(paths[-1]),
    }


def _nested_file_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _nested_file_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_file_values(child)


def _record_referenced_paths(root: Path, record: dict) -> list[Path]:
    paths = []
    for value in _nested_file_values(record):
        candidate = root / value
        if candidate.is_file():
            paths.append(Path(value))
    return sorted(set(paths), key=str)


def _load_records(root: Path) -> list[dict]:
    path = root / "view_records.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"view_records.json must contain a list: {path}")
    return payload


def merge_asset_roots(
    *,
    segment_roots,
    output_root: Path | str,
    expected_names,
) -> dict:
    segment_roots = [Path(path).resolve() for path in segment_roots]
    output_root = Path(output_root)
    expected_names = [str(value) for value in expected_names]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("expected record names are not unique")
    records_by_name = {}
    sources_by_name = {}
    for root in segment_roots:
        for record in _load_records(root):
            name = str(record.get("image_name", ""))
            if name in records_by_name:
                raise ValueError(f"duplicate record key: {name}")
            records_by_name[name] = record
            sources_by_name[name] = root
    missing = [name for name in expected_names if name not in records_by_name]
    extra = sorted(set(records_by_name) - set(expected_names))
    if missing or extra:
        raise ValueError(
            f"asset records do not match expected names: missing={missing[:8]} extra={extra[:8]}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    output_records = []
    files = []
    for name in expected_names:
        record = records_by_name[name]
        source_root = sources_by_name[name]
        for relative in _record_referenced_paths(source_root, record):
            source = source_root / relative
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.is_file() or _file_sha256(destination) != _file_sha256(source):
                    raise ValueError(f"existing merged asset differs: {destination}")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    destination.symlink_to(source)
            files.append(
                {
                    "record": name,
                    "relative_path": str(relative),
                    "source": str(source),
                    "sha256": _file_sha256(source),
                }
            )
        output_records.append(record)
    records_path = output_root / "view_records.json"
    serialized = json.dumps(output_records, indent=2, sort_keys=True)
    if records_path.exists() and records_path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"existing merged record manifest differs: {records_path}")
    records_path.write_text(serialized, encoding="utf-8")
    manifest = {
        "record_count": len(output_records),
        "referenced_file_count": len(files),
        "record_names": expected_names,
        "segment_roots": [str(path) for path in segment_roots],
        "view_records_sha256": _file_sha256(records_path),
        "files": files,
    }
    (output_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def verify_asset_root(output_root: Path | str, *, expected_names) -> dict:
    output_root = Path(output_root)
    records = _load_records(output_root)
    names = [str(row.get("image_name", "")) for row in records]
    expected_names = [str(value) for value in expected_names]
    if names != expected_names:
        raise ValueError("asset root record order/content differs from expected names")
    missing = []
    paths = []
    for record in records:
        referenced = _record_referenced_paths(output_root, record)
        for value in _nested_file_values(record):
            if "/" in value and not (output_root / value).is_file():
                missing.append(output_root / value)
        paths.extend(referenced)
    if missing:
        raise ValueError(
            "missing referenced asset files: " + ", ".join(str(path) for path in missing[:8])
        )
    return {
        "record_count": len(records),
        "referenced_file_count": len(paths),
        "record_names": names,
    }


def derive_aligned_frozen_config(
    *,
    source_config: Path | str,
    checkpoint: Path | str,
    output: Path | str,
) -> dict:
    source_config = Path(source_config)
    checkpoint = Path(checkpoint)
    output = Path(output)
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    old_fingerprint = str(payload.get("checkpoint_fingerprint", ""))
    payload["checkpoint_fingerprint"] = file_fingerprint(checkpoint)
    payload["alignment"] = {
        "mode": "shared_40k_checkpoint_reprojection",
        "source_frozen_config": str(source_config.resolve()),
        "source_checkpoint_fingerprint": old_fingerprint,
        "aligned_checkpoint": str(checkpoint.resolve()),
        "selected_parameters_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def write_operating_point(
    *,
    curve_path: Path | str,
    method: str,
    subject: str,
    output: Path | str,
) -> dict:
    curve_path = Path(curve_path)
    output = Path(output)
    with curve_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    point = resolve_frozen_operating_point(rows, method=method, subject=subject)
    point["reference_baseline"] = "B1"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(point, indent=2, sort_keys=True), encoding="utf-8")
    return point


def write_frozen_protocol(*, output_root: Path | str, output: Path | str) -> dict:
    output_root = Path(output_root).resolve()
    output = Path(output)
    paths = [output_root / "protocol/temporal_record_list.json"]
    for subject in SUBJECTS:
        paths.append(output_root / f"protocol/CoreView_{subject}_a5_shared40k_frozen.json")
        paths.extend(
            output_root / f"protocol/CoreView_{subject}_{method}_operating_point.json"
            for method in METHODS
        )
        paths.append(
            output_root / f"temporal_assets/CoreView_{subject}/merged/asset_manifest.json"
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen protocol input: " + ", ".join(map(str, missing)))
    inputs = [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.resolve().relative_to(output_root)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    result = {
        "schema_version": 1,
        "subjects": list(SUBJECTS),
        "methods": list(METHODS),
        "strict_views": [
            f"c{camera:02d}_f{frame:06d}"
            for camera in (21, 22, 23)
            for frame in (180, 420, 540)
        ],
        "temporal_windows": build_temporal_windows(),
        "target_retention": 0.6,
        "gg_377_retention": 0.4,
        "statistics": {"bootstrap_iterations": 20_000, "seed": 20260813},
        "qualitative": {
            "fixed_view": "c22_f000420",
            "contact_sheet_subject": "386",
            "contact_sheet_camera": 22,
            "contact_sheet_anchor": 420,
        },
        "inputs": inputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare frozen continuous temporal assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-source")
    audit.add_argument("--parser-root", required=True, type=Path)
    audit.add_argument("--subject", required=True)
    record_list = subparsers.add_parser("write-record-list")
    record_list.add_argument("--output", required=True, type=Path)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--segment-root", action="append", required=True, type=Path)
    merge.add_argument("--output-root", required=True, type=Path)
    merge.add_argument("--record-list", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--asset-root", required=True, type=Path)
    verify.add_argument("--record-list", required=True, type=Path)
    align = subparsers.add_parser("align-frozen-config")
    align.add_argument("--source-config", required=True, type=Path)
    align.add_argument("--checkpoint", required=True, type=Path)
    align.add_argument("--output", required=True, type=Path)
    operating_point = subparsers.add_parser("write-operating-point")
    operating_point.add_argument("--curve", required=True, type=Path)
    operating_point.add_argument("--method", required=True)
    operating_point.add_argument("--subject", required=True)
    operating_point.add_argument("--output", required=True, type=Path)
    freeze = subparsers.add_parser("freeze-protocol")
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("record list must contain a JSON list")
    return [str(value) for value in payload]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "audit-source":
        result = validate_temporal_source_coverage(
            parser_root=args.parser_root,
            subject=args.subject,
        )
    elif args.command == "write-record-list":
        names = build_record_names(build_temporal_windows())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(names, indent=2),
            encoding="utf-8",
        )
        result = {"record_count": len(names), "output": str(args.output)}
    elif args.command == "merge":
        result = merge_asset_roots(
            segment_roots=args.segment_root,
            output_root=args.output_root,
            expected_names=_load_names(args.record_list),
        )
    elif args.command == "verify":
        result = verify_asset_root(
            args.asset_root,
            expected_names=_load_names(args.record_list),
        )
    elif args.command == "align-frozen-config":
        result = derive_aligned_frozen_config(
            source_config=args.source_config,
            checkpoint=args.checkpoint,
            output=args.output,
        )
    elif args.command == "write-operating-point":
        result = write_operating_point(
            curve_path=args.curve,
            method=args.method,
            subject=args.subject,
            output=args.output,
        )
    elif args.command == "freeze-protocol":
        result = write_frozen_protocol(output_root=args.output_root, output=args.output)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
