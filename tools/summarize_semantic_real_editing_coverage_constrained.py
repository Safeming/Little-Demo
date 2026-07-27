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
from utils.semantic_matched_strength import build_matched_strength_curves, match_curves_at_retention


DEFAULT_SUBJECTS = ("377", "386", "387", "393", "394")
DEFAULT_TASKS = ("recolor", "removal", "texture")
DEFAULT_PARTS = ("hair", "face", "upper", "lower", "shoes", "skin")
DEFAULT_RETENTIONS = (0.25, 0.50)
METHODS = ("voting", "a5")
METRICS = ("pooled_outer_burden", "pooled_boundary_burden")
NUMERIC_FIELDS = {
    "edit_strength",
    "target_pixel_count",
    "target_delta_sum",
    "outer_delta_sum",
    "boundary_outer_delta_sum",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate pooled coverage-constrained real-editing tables."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS))
    parser.add_argument("--retentions", nargs="+", type=float, default=list(DEFAULT_RETENTIONS))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args(argv)


def _safe_ratio(numerator: float, denominator: float, epsilon: float = 1.0e-12) -> float:
    if abs(float(denominator)) <= epsilon:
        return 0.0
    return float(numerator) / float(denominator)


def recover_matched_sums(*, reference_target: float, matched_point: dict) -> dict:
    target = float(reference_target) * float(matched_point["retention"])
    return {
        "target_delta_sum": target,
        "outer_delta_sum": float(reference_target) * float(matched_point["outer_burden"]),
        "boundary_delta_sum": float(reference_target) * float(matched_point["boundary_burden"]),
    }


def read_metrics(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty real-editing metrics: {path}")
    outputs = []
    for row_index, row in enumerate(rows):
        item = dict(row)
        for field in NUMERIC_FIELDS:
            value = float(item[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} at row {row_index}: {path}")
            item[field] = value
        item["subject"] = str(item["subject"])
        item["method"] = str(item["method"]).lower()
        outputs.append(item)
    return outputs


def build_coverage_rows(
    *,
    reference_keys: set,
    common_by_retention: dict[float, set],
    subjects: tuple[str, ...],
    tasks: tuple[str, ...],
    parts: tuple[str, ...],
    threshold: float,
) -> list[dict]:
    outputs = []
    for retention, common in sorted(common_by_retention.items()):
        for task in tasks:
            for subject in subjects:
                for part in parts:
                    references = {
                        key
                        for key in reference_keys
                        if str(key[0]) == subject and str(key[2]) == part and str(key[3]) == task
                    }
                    supported = references & common
                    reference_count = len(references)
                    supported_count = len(supported)
                    coverage_rate = supported_count / reference_count if reference_count else 0.0
                    outputs.append(
                        {
                            "task": task,
                            "retention": float(retention),
                            "subject": subject,
                            "part": part,
                            "reference_count": reference_count,
                            "supported_count": supported_count,
                            "coverage_rate": float(coverage_rate),
                            "qualified_cell": bool(reference_count > 0 and coverage_rate >= threshold),
                        }
                    )
    return outputs


def select_eligible_parts(
    coverage_rows: list[dict],
    *,
    tasks: tuple[str, ...],
    retentions: tuple[float, ...],
    subjects: tuple[str, ...],
    parts: tuple[str, ...],
    threshold: float,
) -> dict[tuple[str, float], list[str]]:
    lookup = {
        (str(row["task"]), float(row["retention"]), str(row["subject"]), str(row["part"])): row
        for row in coverage_rows
    }
    outputs = {}
    for task in tasks:
        for retention in retentions:
            eligible = []
            for part in parts:
                cells = [lookup.get((task, float(retention), subject, part)) for subject in subjects]
                if all(
                    cell is not None
                    and int(cell["reference_count"]) > 0
                    and float(cell["coverage_rate"]) >= threshold
                    for cell in cells
                ):
                    eligible.append(part)
            outputs[(task, float(retention))] = eligible
    return outputs


def _pooled(rows: list[dict]) -> dict:
    target = float(sum(float(row["target_delta_sum"]) for row in rows))
    outer = float(sum(float(row["outer_delta_sum"]) for row in rows))
    boundary = float(sum(float(row["boundary_delta_sum"]) for row in rows))
    return {
        "matched_target_delta_sum": target,
        "matched_outer_delta_sum": outer,
        "matched_boundary_delta_sum": boundary,
        "pooled_outer_burden": _safe_ratio(outer, target),
        "pooled_boundary_burden": _safe_ratio(boundary, target),
    }


def build_matched_records(curve_result: dict, retentions: tuple[float, ...]) -> tuple[list[dict], dict]:
    outputs = []
    common_by_retention = {}
    for retention in retentions:
        matched_result = match_curves_at_retention(
            curve_result["curves"],
            reference_keys=set(curve_result["reference_supported"]),
            retention=float(retention),
        )
        common = matched_result["common_coverage"].get(("a5", "voting"), set())
        common_by_retention[float(retention)] = set(common)
        for key in sorted(common):
            subject, view, part, task = key
            reference_target = float(curve_result["references"][key]["target_delta_sum"])
            for method in METHODS:
                outputs.append(
                    {
                        "subject": str(subject),
                        "view": str(view),
                        "part": str(part),
                        "task": str(task),
                        "method": method,
                        "retention": float(retention),
                        **recover_matched_sums(
                            reference_target=reference_target,
                            matched_point=matched_result["matched"][(key, method)],
                        ),
                    }
                )
    return outputs, common_by_retention


def build_part_rows(
    matched_records: list[dict],
    coverage_rows: list[dict],
    eligible: dict[tuple[str, float], list[str]],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in matched_records:
        grouped[(row["task"], float(row["retention"]), row["subject"], row["part"], row["method"])].append(row)
    outputs = []
    for coverage in coverage_rows:
        task = str(coverage["task"])
        retention = float(coverage["retention"])
        for method in METHODS:
            members = grouped.get(
                (task, retention, str(coverage["subject"]), str(coverage["part"]), method), []
            )
            outputs.append(
                {
                    **coverage,
                    "method": method,
                    "formal_part_eligible": str(coverage["part"]) in eligible[(task, retention)],
                    **_pooled(members),
                }
            )
    return outputs


def build_subject_rows(
    matched_records: list[dict],
    eligible: dict[tuple[str, float], list[str]],
    *,
    subjects: tuple[str, ...],
    tasks: tuple[str, ...],
    retentions: tuple[float, ...],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in matched_records:
        grouped[(row["task"], float(row["retention"]), row["subject"], row["method"])].append(row)
    outputs = []
    for task in tasks:
        for retention in retentions:
            parts = eligible[(task, float(retention))]
            if not parts:
                continue
            for subject in subjects:
                for method in METHODS:
                    members = [
                        row
                        for row in grouped[(task, float(retention), subject, method)]
                        if row["part"] in parts
                    ]
                    if not members:
                        raise ValueError(f"missing formal matched records: {task}, {retention}, {subject}, {method}")
                    outputs.append(
                        {
                            "task": task,
                            "retention": float(retention),
                            "subject": subject,
                            "method": method,
                            "eligible_parts": ";".join(parts),
                            "eligible_part_count": len(parts),
                            "matched_record_count": len(members),
                            **_pooled(members),
                        }
                    )
    return outputs


def build_formal_rows(subject_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[(row["task"], float(row["retention"]), row["method"])].append(row)
    outputs = []
    for (task, retention, method), members in sorted(grouped.items()):
        item = {
            "task": task,
            "retention": retention,
            "method": method,
            "subject_count": len(members),
            "eligible_parts": members[0]["eligible_parts"],
            "eligible_part_count": members[0]["eligible_part_count"],
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_subject_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        outputs.append(item)
    return outputs


def build_paired_rows(subject_rows: list[dict], *, repetitions: int, seed: int) -> list[dict]:
    lookup = {
        (row["task"], float(row["retention"]), row["subject"], row["method"]): row
        for row in subject_rows
    }
    groups = sorted({(row["task"], float(row["retention"])) for row in subject_rows})
    subjects = sorted({row["subject"] for row in subject_rows})
    outputs = []
    for task, retention in groups:
        used_subjects = [
            subject
            for subject in subjects
            if (task, retention, subject, "a5") in lookup
            and (task, retention, subject, "voting") in lookup
        ]
        for metric_index, metric in enumerate(METRICS):
            a5_values = [float(lookup[(task, retention, subject, "a5")][metric]) for subject in used_subjects]
            voting_values = [
                float(lookup[(task, retention, subject, "voting")][metric]) for subject in used_subjects
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
                    "task": task,
                    "retention": retention,
                    "comparison": "a5-voting",
                    "metric": metric,
                    "lower_is_better": True,
                    "subjects": ";".join(used_subjects),
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


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_finite(rows: list[dict], name: str) -> None:
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError(f"non-finite {name}.{field} at row {row_index}")


def summarize(
    *,
    input_root: Path,
    output_dir: Path,
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    retentions: tuple[float, ...] = DEFAULT_RETENTIONS,
    coverage_threshold: float = 0.80,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260727,
) -> dict:
    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    subjects = tuple(str(value) for value in subjects)
    tasks = tuple(str(value) for value in tasks)
    parts = tuple(str(value) for value in parts)
    retentions = tuple(float(value) for value in retentions)
    source_paths = [input_root / f"CoreView_{subject}" / "metrics.csv" for subject in subjects]
    rows = [row for path in source_paths for row in read_metrics(path)]
    rows = [
        row
        for row in rows
        if row["subject"] in subjects
        and row["task"] in tasks
        and row["part"] in parts
        and row["method"] in METHODS
    ]
    curve_result = build_matched_strength_curves(rows)
    matched_records, common_by_retention = build_matched_records(curve_result, retentions)
    coverage_rows = build_coverage_rows(
        reference_keys=set(curve_result["reference_supported"]),
        common_by_retention=common_by_retention,
        subjects=subjects,
        tasks=tasks,
        parts=parts,
        threshold=float(coverage_threshold),
    )
    eligible = select_eligible_parts(
        coverage_rows,
        tasks=tasks,
        retentions=retentions,
        subjects=subjects,
        parts=parts,
        threshold=float(coverage_threshold),
    )
    part_rows = build_part_rows(matched_records, coverage_rows, eligible)
    subject_rows = build_subject_rows(
        matched_records,
        eligible,
        subjects=subjects,
        tasks=tasks,
        retentions=retentions,
    )
    formal_rows = build_formal_rows(subject_rows)
    paired_rows = build_paired_rows(
        subject_rows,
        repetitions=int(bootstrap_repetitions),
        seed=int(bootstrap_seed),
    )
    for name, table in (
        ("coverage", coverage_rows),
        ("part", part_rows),
        ("subject", subject_rows),
        ("formal", formal_rows),
        ("paired", paired_rows),
    ):
        _assert_finite(table, name)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "coverage_table.csv", coverage_rows)
    _write_csv(output_dir / "part_table.csv", part_rows)
    _write_csv(output_dir / "subject_table.csv", subject_rows)
    _write_csv(output_dir / "formal_table.csv", formal_rows)
    _write_csv(output_dir / "paired_statistics.csv", paired_rows)
    eligible_json = {f"{task}@{retention}": eligible[(task, retention)] for task in tasks for retention in retentions}
    summary = {
        "source_paths": [str(path) for path in source_paths],
        "source_fingerprint": _fingerprint(source_paths),
        "source_row_count": len(rows),
        "subjects": list(subjects),
        "tasks": list(tasks),
        "parts": list(parts),
        "retentions": list(retentions),
        "coverage_threshold": float(coverage_threshold),
        "eligible_parts": eligible_json,
        "excluded_parts": {
            f"{task}@{retention}": [part for part in parts if part not in eligible[(task, retention)]]
            for task in tasks
            for retention in retentions
        },
        "pooling_rule": "Pool matched leakage and target sums within each subject before division.",
        "eligibility_rule": "A part must reach the retention on at least 80% of Voting-supported records in every subject for each task independently.",
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
                input_root=args.input_root,
                output_dir=args.output_dir,
                subjects=tuple(args.subjects),
                tasks=tuple(args.tasks),
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
