#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_semantic_real_editing_paper_suite import paired_bootstrap
from utils.semantic_matched_strength import build_matched_strength_curves, match_curves_at_retention


METHODS = ("raw_hard", "voting", "a5")
COMPARISONS = (("a5", "voting"), ("a5", "raw_hard"))
METRICS = ("outer_burden", "boundary_burden")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize matched-strength real semantic editing curves.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=["377", "386", "387", "393", "394"])
    parser.add_argument("--retentions", nargs="+", type=float, default=[0.25, 0.50])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty matched-strength table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics(path: Path) -> list[dict]:
    numeric = {
        "edit_strength",
        "target_pixel_count",
        "target_delta_sum",
        "outer_delta_sum",
        "boundary_outer_delta_sum",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def _pair_token(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def _coverage_rows(curve_result: dict, matched_result: dict) -> list[dict]:
    reference_keys = set(curve_result["reference_supported"])
    rows = []
    tasks = sorted({key[3] for key in reference_keys})
    for task in tasks:
        task_keys = {key for key in reference_keys if key[3] == task}
        parts = sorted({key[2] for key in task_keys})
        scopes = [("task", "ALL", task_keys)] + [
            ("part", part, {key for key in task_keys if key[2] == part}) for part in parts
        ]
        for scope, part, scope_keys in scopes:
            for method in METHODS:
                covered = scope_keys & matched_result["covered_keys"].get(method, set())
                rows.append(
                    {
                        "scope": scope,
                        "task": task,
                        "part": part,
                        "retention": float(matched_result["retention"]),
                        "method_or_comparison": method,
                        "reference_count": len(scope_keys),
                        "covered_count": len(covered),
                        "coverage_rate": len(covered) / max(len(scope_keys), 1),
                    }
                )
            for first, second in COMPARISONS:
                pair = _pair_token(first, second)
                common = scope_keys & matched_result["common_coverage"].get(pair, set())
                rows.append(
                    {
                        "scope": scope,
                        "task": task,
                        "part": part,
                        "retention": float(matched_result["retention"]),
                        "method_or_comparison": f"{first}-{second}",
                        "reference_count": len(scope_keys),
                        "covered_count": len(common),
                        "coverage_rate": len(common) / max(len(scope_keys), 1),
                    }
                )
    return rows


def summarize_matched_rows(
    rows: list[dict],
    *,
    retentions,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict:
    curve_result = build_matched_strength_curves(rows)
    curve_rows = []
    for (key, method), points in sorted(curve_result["curves"].items()):
        subject, view, part, task = key
        for point in points:
            curve_rows.append(
                {
                    "subject": subject,
                    "view": view,
                    "part": part,
                    "task": task,
                    "method": method,
                    **point,
                }
            )

    matched_rows = []
    coverage_rows = []
    subject_rows = []
    for retention in [float(value) for value in retentions]:
        matched_result = match_curves_at_retention(
            curve_result["curves"],
            reference_keys=set(curve_result["reference_supported"]),
            retention=retention,
        )
        coverage_rows.extend(_coverage_rows(curve_result, matched_result))
        for (key, method), point in sorted(matched_result["matched"].items()):
            subject, view, part, task = key
            matched_rows.append(
                {
                    "subject": subject,
                    "view": view,
                    "part": part,
                    "task": task,
                    "method": method,
                    **point,
                }
            )

        tasks = sorted({key[3] for key in curve_result["reference_supported"]})
        subjects = sorted({key[0] for key in curve_result["reference_supported"]})
        for first, second in COMPARISONS:
            pair = _pair_token(first, second)
            common = matched_result["common_coverage"].get(pair, set())
            for task in tasks:
                for subject in subjects:
                    keys = sorted(key for key in common if key[0] == subject and key[3] == task)
                    if not keys:
                        continue
                    for metric in METRICS:
                        first_mean = float(np.mean([matched_result["matched"][(key, first)][metric] for key in keys]))
                        second_mean = float(np.mean([matched_result["matched"][(key, second)][metric] for key in keys]))
                        subject_rows.append(
                            {
                                "subject": subject,
                                "task": task,
                                "retention": retention,
                                "comparison": f"{first}-{second}",
                                "metric": metric,
                                "common_record_count": len(keys),
                                "a5_mean": first_mean,
                                "baseline_mean": second_mean,
                                "delta": first_mean - second_mean,
                            }
                        )

    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[(row["task"], row["retention"], row["comparison"], row["metric"])].append(row)
    aggregate_rows = []
    paired_rows = []
    for (task, retention, comparison, metric), members in sorted(grouped.items()):
        a5_values = np.asarray([float(row["a5_mean"]) for row in members], dtype=np.float64)
        baseline_values = np.asarray([float(row["baseline_mean"]) for row in members], dtype=np.float64)
        deltas = np.asarray([float(row["delta"]) for row in members], dtype=np.float64)
        aggregate_rows.append(
            {
                "task": task,
                "retention": retention,
                "comparison": comparison,
                "metric": metric,
                "subject_count": len(members),
                "a5_mean": float(np.mean(a5_values)),
                "baseline_mean": float(np.mean(baseline_values)),
                "mean_delta": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            }
        )
        paired_rows.append(
            {
                "task": task,
                "retention": retention,
                "comparison": comparison,
                "metric": metric,
                "lower_is_better": True,
                **paired_bootstrap(
                    deltas,
                    seed=int(bootstrap_seed),
                    repetitions=int(bootstrap_repetitions),
                    lower_is_better=True,
                ),
            }
        )
    return {
        "curve_rows": curve_rows,
        "matched_rows": matched_rows,
        "coverage_rows": coverage_rows,
        "subject_rows": subject_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "reference_supported_count": len(curve_result["reference_supported"]),
        "unsupported_reference_count": int(curve_result["unsupported_reference_count"]),
    }


def summarize(args) -> dict:
    output_root = args.output_root.resolve()
    rows = []
    for subject in args.subjects:
        path = output_root / f"CoreView_{subject}" / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing matched-strength subject metrics: {path}")
        rows.extend(_read_metrics(path))
    result = summarize_matched_rows(
        rows,
        retentions=args.retentions,
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    aggregate_dir = output_root / "aggregate"
    _write_csv(aggregate_dir / "curve_table.csv", result["curve_rows"])
    _write_csv(aggregate_dir / "matched_table.csv", result["matched_rows"])
    _write_csv(aggregate_dir / "coverage_table.csv", result["coverage_rows"])
    _write_csv(aggregate_dir / "subject_table.csv", result["subject_rows"])
    _write_csv(aggregate_dir / "aggregate_table.csv", result["aggregate_rows"])
    _write_csv(aggregate_dir / "paired_statistics.csv", result["paired_rows"])
    summary = {
        "subjects": [str(value) for value in args.subjects],
        "strengths": [0.2, 0.4, 0.6, 0.8, 1.0],
        "retentions": [float(value) for value in args.retentions],
        "input_row_count": len(rows),
        "reference_supported_count": result["reference_supported_count"],
        "unsupported_reference_count": result["unsupported_reference_count"],
        "matched_row_count": len(result["matched_rows"]),
        "uses_test_parser_for_edit_selection": False,
        "uses_test_masks_for_evaluation_matching": True,
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(summarize(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
