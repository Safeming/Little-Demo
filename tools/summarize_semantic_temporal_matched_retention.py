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


DEFAULT_SUBJECTS = ("377", "386", "387", "393", "394")
DEFAULT_PARTS = ("hair", "face", "upper", "lower", "shoes", "skin")
DEFAULT_RETENTIONS = (0.25, 0.50)
METHODS = ("voting", "a5")
SUM_FIELDS = (
    "edit_target_delta_sum",
    "edit_outer_delta_sum",
    "edit_boundary_outer_delta_sum",
    "inside_selection_mass",
    "outside_selection_mass",
)
MEAN_FIELDS = (
    "edit_target_delta_mean",
    "edit_outer_delta_mean",
    "edit_boundary_outer_delta_mean",
)
NUMERIC_FIELDS = {
    "frame",
    "target_pixel_count",
    *SUM_FIELDS,
    *MEAN_FIELDS,
}
FORMAL_METRICS = (
    "pooled_outer_burden",
    "pooled_boundary_burden",
    "pooled_selection_leakage",
    "matched_outer_flicker",
    "matched_boundary_flicker",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate pooled, coverage-constrained matched-retention temporal tables."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS))
    parser.add_argument("--retentions", nargs="+", type=float, default=list(DEFAULT_RETENTIONS))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args(argv)


def _safe_ratio(numerator: float, denominator: float, *, epsilon: float = 1.0e-12) -> float:
    if abs(float(denominator)) <= epsilon:
        return 0.0
    return float(numerator) / float(denominator)


def read_metrics(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty source CSV: {path}")
    required = {
        "subject",
        "part",
        "frame",
        "method",
        "target_pixel_count",
        *SUM_FIELDS,
        *MEAN_FIELDS,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing required source fields: {sorted(missing)}")
    converted = []
    for row_index, row in enumerate(rows):
        item = dict(row)
        item["subject"] = str(item["subject"])
        item["part"] = str(item["part"])
        item["method"] = str(item["method"]).lower()
        for field in NUMERIC_FIELDS:
            value = float(item[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} at source row {row_index}")
            item[field] = value
        item["frame"] = int(float(item["frame"]))
        converted.append(item)
    return converted


def pair_method_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        method = str(row["method"]).lower()
        if method not in METHODS:
            continue
        key = (str(row["subject"]), str(row["part"]), int(float(row["frame"])))
        if method in grouped[key]:
            raise ValueError(f"duplicate {method} temporal row for {key}")
        grouped[key][method] = row
    pairs = []
    for key, methods in sorted(grouped.items()):
        missing = set(METHODS) - set(methods)
        if missing:
            raise ValueError(f"missing methods {sorted(missing)} for temporal row {key}")
        pairs.append((methods["voting"], methods["a5"]))
    if not pairs:
        raise ValueError("source contains no Voting/A5 temporal pairs")
    return pairs


def _scale_method_row(row: dict, strength: float) -> dict:
    output = {"strength": float(strength)}
    for field in SUM_FIELDS + MEAN_FIELDS:
        output[field] = float(row[field]) * float(strength)
    return output


def match_record_pair(
    voting_row: dict,
    a5_row: dict,
    *,
    retention: float,
    epsilon: float = 1.0e-8,
) -> dict:
    if not 0.0 < float(retention) <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    voting_target = float(voting_row["edit_target_delta_sum"])
    a5_target = float(a5_row["edit_target_delta_sum"])
    reference_supported = float(voting_row["target_pixel_count"]) > 0.0 and voting_target > epsilon
    max_retention = _safe_ratio(a5_target, voting_target, epsilon=epsilon) if reference_supported else 0.0
    reachable = bool(reference_supported and a5_target > epsilon and max_retention + 1.0e-12 >= retention)
    result = {
        "subject": str(voting_row["subject"]),
        "part": str(voting_row["part"]),
        "frame": int(float(voting_row["frame"])),
        "retention": float(retention),
        "reference_supported": reference_supported,
        "reachable": reachable,
        "max_retention": float(max_retention),
    }
    if reachable:
        voting_strength = float(retention)
        a5_strength = float(retention) / float(max_retention)
        result["voting"] = _scale_method_row(voting_row, voting_strength)
        result["a5"] = _scale_method_row(a5_row, a5_strength)
    return result


def build_coverage_rows(
    matches_by_retention: dict[float, list[dict]],
    *,
    subjects: tuple[str, ...],
    parts: tuple[str, ...],
    threshold: float,
) -> list[dict]:
    outputs = []
    for retention, matches in sorted(matches_by_retention.items()):
        grouped = defaultdict(list)
        for match in matches:
            grouped[(str(match["subject"]), str(match["part"]))].append(match)
        for subject in subjects:
            for part in parts:
                members = grouped.get((str(subject), str(part)), [])
                references = [item for item in members if item["reference_supported"]]
                supported = [item for item in references if item["reachable"]]
                max_retentions = [float(item["max_retention"]) for item in references]
                reference_count = len(references)
                supported_count = len(supported)
                coverage_rate = supported_count / reference_count if reference_count else 0.0
                outputs.append(
                    {
                        "retention": float(retention),
                        "subject": str(subject),
                        "part": str(part),
                        "reference_count": reference_count,
                        "supported_count": supported_count,
                        "coverage_rate": float(coverage_rate),
                        "qualified_cell": bool(reference_count > 0 and coverage_rate >= threshold),
                        "median_max_retention": float(np.median(max_retentions)) if max_retentions else 0.0,
                    }
                )
    return outputs


def select_eligible_parts(
    coverage_rows: list[dict],
    *,
    subjects: tuple[str, ...],
    parts: tuple[str, ...],
    retentions: tuple[float, ...],
    threshold: float,
) -> dict[float, list[str]]:
    lookup = {
        (float(row["retention"]), str(row["subject"]), str(row["part"])): row
        for row in coverage_rows
    }
    selected = {}
    for retention in retentions:
        eligible = []
        for part in parts:
            cells = [lookup.get((float(retention), str(subject), str(part))) for subject in subjects]
            if all(
                cell is not None
                and int(cell["reference_count"]) > 0
                and float(cell["coverage_rate"]) >= threshold
                for cell in cells
            ):
                eligible.append(str(part))
        selected[float(retention)] = eligible
    return selected


def consecutive_supported_flicker(
    frame_values: list[tuple[int, float]], *, epsilon: float = 1.0e-12
) -> dict:
    if not frame_values:
        return {"pair_count": 0, "flicker": 0.0}
    ordered = sorted((int(frame), float(value)) for frame, value in frame_values)
    differences = [
        abs(current_value - previous_value)
        for (previous_frame, previous_value), (current_frame, current_value) in zip(ordered, ordered[1:])
        if current_frame == previous_frame + 1
    ]
    if not differences:
        return {"pair_count": 0, "flicker": 0.0}
    sequence_mean = float(np.mean([abs(value) for _, value in ordered]))
    flicker = _safe_ratio(float(np.mean(differences)), sequence_mean, epsilon=epsilon)
    return {"pair_count": len(differences), "flicker": float(flicker)}


def _pooled_metrics(rows: list[dict]) -> dict:
    target_sum = float(sum(float(row["edit_target_delta_sum"]) for row in rows))
    outer_sum = float(sum(float(row["edit_outer_delta_sum"]) for row in rows))
    boundary_sum = float(sum(float(row["edit_boundary_outer_delta_sum"]) for row in rows))
    inside_mass = float(sum(float(row["inside_selection_mass"]) for row in rows))
    outside_mass = float(sum(float(row["outside_selection_mass"]) for row in rows))
    return {
        "matched_target_delta_sum": target_sum,
        "matched_outer_delta_sum": outer_sum,
        "matched_boundary_delta_sum": boundary_sum,
        "matched_inside_selection_mass": inside_mass,
        "matched_outside_selection_mass": outside_mass,
        "pooled_outer_burden": _safe_ratio(outer_sum, target_sum),
        "pooled_boundary_burden": _safe_ratio(boundary_sum, target_sum),
        "pooled_selection_leakage": _safe_ratio(outside_mass, inside_mass),
    }


def build_matched_records(matches_by_retention: dict[float, list[dict]]) -> list[dict]:
    outputs = []
    for retention, matches in sorted(matches_by_retention.items()):
        for match in matches:
            if not match["reachable"]:
                continue
            for method in METHODS:
                outputs.append(
                    {
                        "retention": float(retention),
                        "subject": str(match["subject"]),
                        "part": str(match["part"]),
                        "frame": int(match["frame"]),
                        "method": method,
                        **match[method],
                    }
                )
    return outputs


def build_part_rows(
    matched_records: list[dict],
    coverage_rows: list[dict],
    eligible_parts: dict[float, list[str]],
) -> list[dict]:
    matched_grouped = defaultdict(list)
    for row in matched_records:
        matched_grouped[(float(row["retention"]), str(row["subject"]), str(row["part"]), str(row["method"]))].append(row)
    outputs = []
    for coverage in coverage_rows:
        retention = float(coverage["retention"])
        for method in METHODS:
            key = (retention, str(coverage["subject"]), str(coverage["part"]), method)
            members = matched_grouped.get(key, [])
            outer_flicker = consecutive_supported_flicker(
                [(int(row["frame"]), float(row["edit_outer_delta_mean"])) for row in members]
            )
            boundary_flicker = consecutive_supported_flicker(
                [(int(row["frame"]), float(row["edit_boundary_outer_delta_mean"])) for row in members]
            )
            outputs.append(
                {
                    "retention": retention,
                    "subject": str(coverage["subject"]),
                    "part": str(coverage["part"]),
                    "method": method,
                    "reference_count": int(coverage["reference_count"]),
                    "supported_count": int(coverage["supported_count"]),
                    "coverage_rate": float(coverage["coverage_rate"]),
                    "qualified_cell": bool(coverage["qualified_cell"]),
                    "formal_part_eligible": str(coverage["part"]) in eligible_parts[retention],
                    "median_max_retention": float(coverage["median_max_retention"]),
                    **_pooled_metrics(members),
                    "matched_outer_flicker": float(outer_flicker["flicker"]),
                    "matched_outer_pair_count": int(outer_flicker["pair_count"]),
                    "matched_boundary_flicker": float(boundary_flicker["flicker"]),
                    "matched_boundary_pair_count": int(boundary_flicker["pair_count"]),
                }
            )
    return outputs


def build_subject_rows(
    matched_records: list[dict],
    part_rows: list[dict],
    eligible_parts: dict[float, list[str]],
    *,
    subjects: tuple[str, ...],
    retentions: tuple[float, ...],
) -> list[dict]:
    matched_grouped = defaultdict(list)
    for row in matched_records:
        matched_grouped[(float(row["retention"]), str(row["subject"]), str(row["method"]))].append(row)
    part_lookup = defaultdict(list)
    for row in part_rows:
        part_lookup[(float(row["retention"]), str(row["subject"]), str(row["method"]))].append(row)
    outputs = []
    for retention in retentions:
        eligible = eligible_parts[float(retention)]
        if not eligible:
            raise ValueError(f"no part satisfies coverage rule at retention {retention}")
        for subject in subjects:
            for method in METHODS:
                records = [
                    row
                    for row in matched_grouped[(float(retention), str(subject), method)]
                    if str(row["part"]) in eligible
                ]
                sequence_rows = [
                    row
                    for row in part_lookup[(float(retention), str(subject), method)]
                    if str(row["part"]) in eligible
                ]
                if len(sequence_rows) != len(eligible):
                    raise ValueError(f"missing eligible part summaries for {retention}, {subject}, {method}")
                outputs.append(
                    {
                        "retention": float(retention),
                        "subject": str(subject),
                        "method": method,
                        "eligible_parts": ";".join(eligible),
                        "eligible_part_count": len(eligible),
                        "matched_record_count": len(records),
                        **_pooled_metrics(records),
                        "matched_outer_flicker": float(
                            np.mean([float(row["matched_outer_flicker"]) for row in sequence_rows])
                        ),
                        "matched_outer_pair_count": int(
                            sum(int(row["matched_outer_pair_count"]) for row in sequence_rows)
                        ),
                        "matched_boundary_flicker": float(
                            np.mean([float(row["matched_boundary_flicker"]) for row in sequence_rows])
                        ),
                        "matched_boundary_pair_count": int(
                            sum(int(row["matched_boundary_pair_count"]) for row in sequence_rows)
                        ),
                    }
                )
    return outputs


def build_formal_rows(
    subject_rows: list[dict], eligible_parts: dict[float, list[str]]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[(float(row["retention"]), str(row["method"]))].append(row)
    outputs = []
    for (retention, method), members in sorted(grouped.items()):
        item = {
            "retention": retention,
            "method": method,
            "subject_count": len(members),
            "eligible_parts": ";".join(eligible_parts[retention]),
            "eligible_part_count": len(eligible_parts[retention]),
        }
        for metric in FORMAL_METRICS:
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_subject_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        outputs.append(item)
    return outputs


def build_paired_rows(
    subject_rows: list[dict],
    *,
    retentions: tuple[float, ...],
    repetitions: int,
    seed: int,
) -> list[dict]:
    lookup = {
        (float(row["retention"]), str(row["subject"]), str(row["method"])): row
        for row in subject_rows
    }
    subjects = sorted({str(row["subject"]) for row in subject_rows})
    outputs = []
    for retention in retentions:
        for metric_index, metric in enumerate(FORMAL_METRICS):
            a5_values = [float(lookup[(float(retention), subject, "a5")][metric]) for subject in subjects]
            voting_values = [
                float(lookup[(float(retention), subject, "voting")][metric]) for subject in subjects
            ]
            deltas = [a5 - voting for a5, voting in zip(a5_values, voting_values)]
            stats = paired_bootstrap(
                deltas,
                seed=int(seed) + metric_index,
                repetitions=int(repetitions),
                lower_is_better=True,
            )
            voting_mean = float(np.mean(voting_values))
            a5_mean = float(np.mean(a5_values))
            outputs.append(
                {
                    "retention": float(retention),
                    "comparison": "a5-voting",
                    "metric": metric,
                    "lower_is_better": True,
                    "subjects": ";".join(subjects),
                    "voting_mean": voting_mean,
                    "a5_mean": a5_mean,
                    "relative_reduction_percent": 100.0 * _safe_ratio(voting_mean - a5_mean, voting_mean),
                    **stats,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_finite_rows(rows: list[dict], table_name: str) -> None:
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite {table_name}.{field} at row {row_index}")


def summarize(
    *,
    input_csv: Path,
    output_dir: Path,
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    retentions: tuple[float, ...] = DEFAULT_RETENTIONS,
    coverage_threshold: float = 0.80,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260724,
) -> dict:
    input_csv = Path(input_csv).resolve()
    output_dir = Path(output_dir).resolve()
    subjects = tuple(str(value) for value in subjects)
    parts = tuple(str(value) for value in parts)
    retentions = tuple(float(value) for value in retentions)
    if not 0.0 <= float(coverage_threshold) <= 1.0:
        raise ValueError("coverage threshold must be in [0, 1]")

    source_rows = read_metrics(input_csv)
    filtered_rows = [
        row
        for row in source_rows
        if str(row["subject"]) in subjects and str(row["part"]) in parts and str(row["method"]) in METHODS
    ]
    pairs = pair_method_rows(filtered_rows)
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
    matched_records = build_matched_records(matches_by_retention)
    part_rows = build_part_rows(matched_records, coverage_rows, eligible_parts)
    subject_rows = build_subject_rows(
        matched_records,
        part_rows,
        eligible_parts,
        subjects=subjects,
        retentions=retentions,
    )
    formal_rows = build_formal_rows(subject_rows, eligible_parts)
    paired_rows = build_paired_rows(
        subject_rows,
        retentions=retentions,
        repetitions=int(bootstrap_repetitions),
        seed=int(bootstrap_seed),
    )

    for table_name, rows in (
        ("coverage", coverage_rows),
        ("part", part_rows),
        ("subject", subject_rows),
        ("formal", formal_rows),
        ("paired", paired_rows),
    ):
        _assert_finite_rows(rows, table_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "coverage_table.csv", coverage_rows)
    _write_csv(output_dir / "part_table.csv", part_rows)
    _write_csv(output_dir / "subject_table.csv", subject_rows)
    _write_csv(output_dir / "formal_table.csv", formal_rows)
    _write_csv(output_dir / "paired_statistics.csv", paired_rows)

    eligible_json = {str(retention): eligible_parts[retention] for retention in retentions}
    summary = {
        "source_csv": str(input_csv),
        "source_sha256": _sha256(input_csv),
        "source_row_count": len(source_rows),
        "paired_frame_record_count": len(pairs),
        "subjects": list(subjects),
        "parts": list(parts),
        "retentions": list(retentions),
        "coverage_threshold": float(coverage_threshold),
        "eligible_parts": eligible_json,
        "excluded_parts": {
            str(retention): [part for part in parts if part not in eligible_parts[retention]]
            for retention in retentions
        },
        "exact_matching_assumption": (
            "Geometry and opacity are frozen; recolor RGB deltas and effective selection masses scale "
            "linearly with one global edit-strength multiplier."
        ),
        "matching_rule": (
            "Voting strength equals target retention; A5 strength equals retention divided by its "
            "full-strength target retention relative to Voting."
        ),
        "coverage_rule": (
            "A part enters the formal table only when A5 reaches the target retention on at least "
            f"{float(coverage_threshold):.0%} of Voting-supported frames in every subject."
        ),
        "burden_pooling": "Sum leakage numerator and target denominator within each subject before division.",
        "temporal_rule": (
            "Compute normalized absolute flicker only for consecutive commonly supported frames within "
            "each subject-part sequence, then average eligible part sequences within each subject."
        ),
        "formal_pooling": "Subjects are equal statistical units in the formal table and paired bootstrap.",
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(bootstrap_seed),
        "output_files": [
            "coverage_table.csv",
            "part_table.csv",
            "subject_table.csv",
            "formal_table.csv",
            "paired_statistics.csv",
            "summary.json",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = summarize(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        subjects=tuple(args.subjects),
        parts=tuple(args.parts),
        retentions=tuple(args.retentions),
        coverage_threshold=float(args.coverage_threshold),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
