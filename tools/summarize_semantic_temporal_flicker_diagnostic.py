#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_semantic_real_editing_paper_suite import paired_bootstrap
from tools.summarize_semantic_temporal_matched_retention import (
    DEFAULT_PARTS,
    DEFAULT_RETENTIONS,
    DEFAULT_SUBJECTS,
    METHODS,
    build_coverage_rows,
    consecutive_supported_flicker,
    match_record_pair,
    pair_method_rows,
    read_metrics,
    select_eligible_parts,
)


FLICKER_METRICS = ("outer_flicker", "boundary_flicker")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare fixed-strength and adaptive matched-retention temporal flicker."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS))
    parser.add_argument("--retentions", nargs="+", type=float, default=list(DEFAULT_RETENTIONS))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args(argv)


def mode_flicker(frame_values: list[tuple[int, float, float]]) -> dict:
    fixed = consecutive_supported_flicker([(frame, fixed_value) for frame, fixed_value, _ in frame_values])
    adaptive = consecutive_supported_flicker(
        [(frame, adaptive_value) for frame, _, adaptive_value in frame_values]
    )
    return {
        "fixed_pair_count": int(fixed["pair_count"]),
        "fixed_flicker": float(fixed["flicker"]),
        "adaptive_pair_count": int(adaptive["pair_count"]),
        "adaptive_flicker": float(adaptive["flicker"]),
        "compensation_penalty": float(adaptive["flicker"] - fixed["flicker"]),
    }


def _strength_stats(rows: list[dict]) -> dict:
    values = np.asarray([float(row["adaptive_strength"]) for row in rows], dtype=np.float64)
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    flicker = consecutive_supported_flicker(
        [(int(row["frame"]), float(row["adaptive_strength"])) for row in rows]
    )
    return {
        "adaptive_strength_mean": mean,
        "adaptive_strength_std": std,
        "adaptive_strength_cv": std / mean if abs(mean) > 1.0e-12 else 0.0,
        "adaptive_strength_flicker": float(flicker["flicker"]),
    }


def build_mode_records(
    pairs: list[tuple[dict, dict]], matches_by_retention: dict[float, list[dict]]
) -> list[dict]:
    raw_lookup = {}
    for voting, a5 in pairs:
        key = (str(voting["subject"]), str(voting["part"]), int(voting["frame"]))
        raw_lookup[key] = {"voting": voting, "a5": a5}
    outputs = []
    for retention, matches in sorted(matches_by_retention.items()):
        for match in matches:
            if not match["reachable"]:
                continue
            key = (str(match["subject"]), str(match["part"]), int(match["frame"]))
            for method in METHODS:
                raw = raw_lookup[key][method]
                adaptive = match[method]
                outputs.append(
                    {
                        "retention": float(retention),
                        "subject": key[0],
                        "part": key[1],
                        "frame": key[2],
                        "method": method,
                        "adaptive_strength": float(adaptive["strength"]),
                        "fixed_outer": float(raw["edit_outer_delta_mean"]),
                        "adaptive_outer": float(adaptive["edit_outer_delta_mean"]),
                        "fixed_boundary": float(raw["edit_boundary_outer_delta_mean"]),
                        "adaptive_boundary": float(adaptive["edit_boundary_outer_delta_mean"]),
                    }
                )
    return outputs


def build_sequence_rows(
    mode_records: list[dict], eligible_parts: dict[float, list[str]]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in mode_records:
        grouped[(float(row["retention"]), row["subject"], row["part"], row["method"])].append(row)
    outputs = []
    for (retention, subject, part, method), members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: int(row["frame"]))
        outer = mode_flicker(
            [(row["frame"], row["fixed_outer"], row["adaptive_outer"]) for row in members]
        )
        boundary = mode_flicker(
            [(row["frame"], row["fixed_boundary"], row["adaptive_boundary"]) for row in members]
        )
        outputs.append(
            {
                "retention": retention,
                "subject": subject,
                "part": part,
                "method": method,
                "formal_part_eligible": part in eligible_parts[retention],
                "supported_frame_count": len(members),
                "fixed_outer_flicker": outer["fixed_flicker"],
                "adaptive_outer_flicker": outer["adaptive_flicker"],
                "outer_compensation_penalty": outer["compensation_penalty"],
                "outer_pair_count": outer["fixed_pair_count"],
                "fixed_boundary_flicker": boundary["fixed_flicker"],
                "adaptive_boundary_flicker": boundary["adaptive_flicker"],
                "boundary_compensation_penalty": boundary["compensation_penalty"],
                "boundary_pair_count": boundary["fixed_pair_count"],
                **_strength_stats(members),
            }
        )
    return outputs


def build_subject_rows(
    sequence_rows: list[dict],
    eligible_parts: dict[float, list[str]],
    *,
    subjects: tuple[str, ...],
    retentions: tuple[float, ...],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in sequence_rows:
        grouped[(float(row["retention"]), row["subject"], row["method"])].append(row)
    outputs = []
    for retention in retentions:
        parts = eligible_parts[float(retention)]
        if not parts:
            raise ValueError(f"no eligible temporal parts at retention {retention}")
        for subject in subjects:
            for method in METHODS:
                members = [
                    row
                    for row in grouped[(float(retention), subject, method)]
                    if row["part"] in parts
                ]
                if len(members) != len(parts):
                    raise ValueError(f"missing temporal sequence: {retention}, {subject}, {method}")
                for mode in ("fixed", "adaptive"):
                    outputs.append(
                        {
                            "retention": float(retention),
                            "subject": subject,
                            "method": method,
                            "mode": mode,
                            "eligible_parts": ";".join(parts),
                            "eligible_part_count": len(parts),
                            "outer_flicker": float(
                                np.mean([float(row[f"{mode}_outer_flicker"]) for row in members])
                            ),
                            "boundary_flicker": float(
                                np.mean([float(row[f"{mode}_boundary_flicker"]) for row in members])
                            ),
                            "adaptive_strength_cv": float(
                                np.mean([float(row["adaptive_strength_cv"]) for row in members])
                            ),
                            "adaptive_strength_flicker": float(
                                np.mean([float(row["adaptive_strength_flicker"]) for row in members])
                            ),
                        }
                    )
    return outputs


def build_formal_rows(subject_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[(float(row["retention"]), row["method"], row["mode"])].append(row)
    outputs = []
    for (retention, method, mode), members in sorted(grouped.items()):
        item = {
            "retention": retention,
            "method": method,
            "mode": mode,
            "subject_count": len(members),
            "eligible_parts": members[0]["eligible_parts"],
        }
        for metric in FLICKER_METRICS + ("adaptive_strength_cv", "adaptive_strength_flicker"):
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_subject_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        outputs.append(item)
    return outputs


def build_paired_rows(subject_rows: list[dict], *, repetitions: int, seed: int) -> list[dict]:
    lookup = {
        (float(row["retention"]), row["subject"], row["method"], row["mode"]): row
        for row in subject_rows
    }
    subjects = sorted({row["subject"] for row in subject_rows})
    retentions = sorted({float(row["retention"]) for row in subject_rows})
    outputs = []
    for retention in retentions:
        for mode in ("fixed", "adaptive"):
            for metric_index, metric in enumerate(FLICKER_METRICS):
                deltas = [
                    float(lookup[(retention, subject, "a5", mode)][metric])
                    - float(lookup[(retention, subject, "voting", mode)][metric])
                    for subject in subjects
                ]
                outputs.append(
                    {
                        "retention": retention,
                        "comparison": "a5-voting",
                        "mode": mode,
                        "metric": metric,
                        "lower_is_better": True,
                        **paired_bootstrap(
                            deltas,
                            seed=int(seed) + metric_index,
                            repetitions=int(repetitions),
                            lower_is_better=True,
                        ),
                    }
                )
        for metric_index, metric in enumerate(FLICKER_METRICS):
            deltas = [
                float(lookup[(retention, subject, "a5", "adaptive")][metric])
                - float(lookup[(retention, subject, "a5", "fixed")][metric])
                for subject in subjects
            ]
            outputs.append(
                {
                    "retention": retention,
                    "comparison": "a5-adaptive-minus-fixed",
                    "mode": "compensation",
                    "metric": metric,
                    "lower_is_better": True,
                    **paired_bootstrap(
                        deltas,
                        seed=int(seed) + 10 + metric_index,
                        repetitions=int(repetitions),
                        lower_is_better=True,
                    ),
                }
            )
    return outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_finite(rows: list[dict], name: str) -> None:
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite {name}.{field} at row {row_index}")


def summarize(
    *,
    input_csv: Path,
    output_dir: Path,
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    retentions: tuple[float, ...] = DEFAULT_RETENTIONS,
    coverage_threshold: float = 0.80,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260727,
) -> dict:
    input_csv = Path(input_csv).resolve()
    output_dir = Path(output_dir).resolve()
    subjects = tuple(str(value) for value in subjects)
    parts = tuple(str(value) for value in parts)
    retentions = tuple(float(value) for value in retentions)
    source_rows = read_metrics(input_csv)
    filtered = [
        row
        for row in source_rows
        if row["subject"] in subjects and row["part"] in parts and row["method"] in METHODS
    ]
    pairs = pair_method_rows(filtered)
    matches_by_retention = {
        retention: [match_record_pair(voting, a5, retention=retention) for voting, a5 in pairs]
        for retention in retentions
    }
    coverage_rows = build_coverage_rows(
        matches_by_retention,
        subjects=subjects,
        parts=parts,
        threshold=float(coverage_threshold),
    )
    eligible_parts = select_eligible_parts(
        coverage_rows,
        subjects=subjects,
        parts=parts,
        retentions=retentions,
        threshold=float(coverage_threshold),
    )
    mode_records = build_mode_records(pairs, matches_by_retention)
    sequence_rows = build_sequence_rows(mode_records, eligible_parts)
    subject_rows = build_subject_rows(
        sequence_rows,
        eligible_parts,
        subjects=subjects,
        retentions=retentions,
    )
    formal_rows = build_formal_rows(subject_rows)
    paired_rows = build_paired_rows(
        subject_rows,
        repetitions=int(bootstrap_repetitions),
        seed=int(bootstrap_seed),
    )
    for row in subject_rows:
        if row["method"] == "voting" and row["mode"] == "fixed":
            adaptive = next(
                candidate
                for candidate in subject_rows
                if candidate["retention"] == row["retention"]
                and candidate["subject"] == row["subject"]
                and candidate["method"] == "voting"
                and candidate["mode"] == "adaptive"
            )
            for metric in FLICKER_METRICS:
                if abs(float(row[metric]) - float(adaptive[metric])) > 1.0e-10:
                    raise ValueError(f"Voting constant scaling changed normalized {metric}")
    for name, table in (
        ("sequence", sequence_rows),
        ("subject", subject_rows),
        ("formal", formal_rows),
        ("paired", paired_rows),
    ):
        _assert_finite(table, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "sequence_table.csv", sequence_rows)
    _write_csv(output_dir / "subject_table.csv", subject_rows)
    _write_csv(output_dir / "formal_table.csv", formal_rows)
    _write_csv(output_dir / "paired_statistics.csv", paired_rows)
    digest = hashlib.sha256(input_csv.read_bytes()).hexdigest()
    summary = {
        "source_csv": str(input_csv),
        "source_sha256": digest,
        "source_row_count": len(source_rows),
        "subjects": list(subjects),
        "parts": list(parts),
        "retentions": list(retentions),
        "coverage_threshold": float(coverage_threshold),
        "eligible_parts": {str(retention): eligible_parts[retention] for retention in retentions},
        "fixed_rule": "Full-strength values evaluated on the same common reachable frames as adaptive matching.",
        "adaptive_rule": "Per-frame A5 strength matches Voting target retention; Voting uses a constant retention multiplier.",
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(bootstrap_seed),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            summarize(
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                subjects=tuple(args.subjects),
                parts=tuple(args.parts),
                retentions=tuple(args.retentions),
                coverage_threshold=float(args.coverage_threshold),
                bootstrap_repetitions=int(args.bootstrap_repetitions),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
