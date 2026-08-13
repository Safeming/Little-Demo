#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.summarize_four_method_paper_evidence import rank_objective_views


PARTS = ("hair", "shoes")
EDIT_METHODS = ("saga", "gaussian_grouping", "sggs", "a5")
FIXED_VIEW = "c22_f000420"


def write_method_strengths(operating_points, *, output: Path | str) -> dict:
    strengths = {}
    for method in EDIT_METHODS:
        payload = json.loads(Path(operating_points[method]).read_text(encoding="utf-8"))
        if str(payload.get("method", "")) != method:
            raise ValueError(f"operating point method mismatch: {method}")
        value = float(payload["edit_strength"])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid edit strength for {method}: {value}")
        strengths[method] = {part: value for part in PARTS}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(strengths, indent=2, sort_keys=True), encoding="utf-8")
    return strengths


def rank_subject_views(rows, *, subject: str) -> list[dict]:
    ranking = rank_objective_views(
        [row for row in rows if str(row["subject"]) == str(subject)],
        parts=PARTS,
        methods=EDIT_METHODS,
    )
    return [{"subject": str(subject), **row} for row in ranking]


def build_qualitative_manifest(*, subjects, methods, render_roots, rankings) -> dict:
    subjects = [str(value) for value in subjects]
    methods = [str(value) for value in methods]
    sets = {"fixed_main": {}, "objectively_selected": {}}
    for set_name in sets:
        for part in PARTS:
            sets[set_name][part] = {}
            for subject in subjects:
                selected_view = (
                    FIXED_VIEW
                    if set_name == "fixed_main"
                    else str(rankings[subject][0]["view"])
                )
                frames = Path(render_roots[subject]) / "frames"
                sets[set_name][part][subject] = {}
                for method in methods:
                    if method == "input":
                        path = frames / f"{selected_view}_rgb.png"
                    else:
                        path = frames / "recolor" / method / f"{selected_view}_{part}.png"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    sets[set_name][part][subject][method] = str(path.resolve())
    return {
        "subjects": subjects,
        "methods": methods,
        "fixed_view": FIXED_VIEW,
        "objective_ranking_key": [
            "mean_actionable_leakage",
            "negative_mean_iou",
            "view",
        ],
        "sets": sets,
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare frozen four-method qualitative inputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    strengths = subparsers.add_parser("strengths")
    strengths.add_argument("--operating-point", action="append", required=True)
    strengths.add_argument("--output", required=True, type=Path)
    rank = subparsers.add_parser("rank")
    rank.add_argument("--input", required=True, type=Path)
    rank.add_argument("--subject", required=True)
    rank.add_argument("--output", required=True, type=Path)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--render-root", action="append", required=True)
    manifest.add_argument("--ranking", action="append", required=True)
    manifest.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _mapping(values) -> dict:
    mapping = {}
    for text in values:
        key, value = str(text).split("=", 1)
        if key in mapping:
            raise ValueError(f"duplicate mapping key: {key}")
        mapping[key] = Path(value)
    return mapping


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "strengths":
        result = write_method_strengths(_mapping(args.operating_point), output=args.output)
    elif args.command == "rank":
        result = rank_subject_views(_read_csv(args.input), subject=args.subject)
        if not result:
            raise ValueError(f"no complete objective view for CoreView_{args.subject}")
        _write_csv(args.output, result)
    elif args.command == "manifest":
        render_roots = _mapping(args.render_root)
        ranking_paths = _mapping(args.ranking)
        rankings = {key: _read_csv(path) for key, path in ranking_paths.items()}
        result = build_qualitative_manifest(
            subjects=("377", "386", "394"),
            methods=("input", *EDIT_METHODS),
            render_roots=render_roots,
            rankings=rankings,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
