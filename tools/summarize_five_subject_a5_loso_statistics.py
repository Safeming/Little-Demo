#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_paper_baseline_loso_ablation import _write_csv


SUBJECTS = ("377", "386", "387", "393", "394")
RETENTION_TARGETS = (0.5, 0.6)
MAIN_METHODS = {"B0", "B1", "B2", "B3", "B4", "A5"}
SUMMARY_METRICS = (
    "macro_miou",
    "mean_boundary_f1",
    "mean_boundary_iou",
    "mean_soft_iou",
    "micro_iou",
)


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_main_methods(methods) -> None:
    values = {str(value) for value in methods}
    if "A6" in values:
        raise ValueError("A6 is ablation-only and cannot appear in the main table")
    if values != MAIN_METHODS:
        raise ValueError(f"formal main table methods must be {sorted(MAIN_METHODS)}, got {sorted(values)}")


def paired_statistics(
    deltas: list[float],
    *,
    repetitions: int = 10000,
    seed: int = 20260723,
    higher_is_better: bool = True,
) -> dict:
    values = [float(value) for value in deltas]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("paired deltas must be non-empty and finite")
    rng = random.Random(int(seed))
    bootstrap = []
    for _ in range(int(repetitions)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()
    low_index = max(0, min(len(bootstrap) - 1, int(0.025 * (len(bootstrap) - 1))))
    high_index = max(0, min(len(bootstrap) - 1, int(0.975 * (len(bootstrap) - 1))))
    wins = sum(value > 1.0e-12 for value in values) if higher_is_better else sum(value < -1.0e-12 for value in values)
    losses = sum(value < -1.0e-12 for value in values) if higher_is_better else sum(value > 1.0e-12 for value in values)
    return {
        "subject_count": len(values),
        "mean_delta": sum(values) / len(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_ci95_low": bootstrap[low_index],
        "bootstrap_ci95_high": bootstrap[high_index],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "wins": wins,
        "ties": len(values) - wins - losses,
        "losses": losses,
    }


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_main(root: Path) -> list[dict]:
    subject_rows = []
    for subject in SUBJECTS:
        source = _read_csv(root / f"CoreView_{subject}" / "main" / "baseline_summary.csv")
        validate_main_methods(row["baseline"] for row in source)
        for row in source:
            subject_rows.append({"subject": subject, **row})
    aggregate_rows = []
    for label, reducer in (("MEAN", statistics.mean), ("STD", statistics.stdev)):
        for method in sorted(MAIN_METHODS):
            selected = [row for row in subject_rows if row["baseline"] == method]
            aggregate = {
                "subject": label,
                "baseline": method,
                "display_label": "Ours (A5)" if method == "A5" else method,
                "name": selected[0].get("name", ""),
            }
            for key in selected[0]:
                if key in aggregate or key in ("subject", "baseline", "name", "oracle", "persistent_asset"):
                    continue
                values = [_numeric(row.get(key)) for row in selected]
                if all(value is not None for value in values):
                    aggregate[key] = reducer(values) if label == "MEAN" or len(values) > 1 else 0.0
            aggregate_rows.append(aggregate)
    return subject_rows + aggregate_rows


def _row_at_retention(rows: list[dict], *, method: str, retention: float) -> dict:
    matches = [
        row
        for row in rows
        if str(row.get("baseline")) == method
        and abs(float(row.get("retention", -1.0)) - retention) <= 1.0e-6
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {method} row at retention {retention}, found {len(matches)}")
    return matches[0]


def _matched_rows(root: Path) -> list[dict]:
    subject_rows = []
    aggregate_rows = []
    for subject in SUBJECTS:
        subject_root = root / f"CoreView_{subject}" / "main"
        curve = _read_csv(subject_root / "leakage_retention_curve.csv")
        matched = _read_csv(subject_root / "matched_retention.csv")
        for retention in RETENTION_TARGETS:
            for method, source in (("B1", curve), ("A5", matched)):
                row = _row_at_retention(source, method=method, retention=retention)
                subject_rows.append(
                    {
                        "subject": subject,
                        "baseline": method,
                        "retention": retention,
                        "actionable_leakage": float(row["actionable_leakage"]),
                        "actionable_leakage_ratio": float(row["actionable_leakage_ratio"]),
                        "raw_leakage": float(row["raw_leakage"]),
                        "raw_leakage_ratio": float(row["raw_leakage_ratio"]),
                    }
                )
    for label, reducer in (("MEAN", statistics.mean), ("STD", statistics.stdev)):
        for retention in RETENTION_TARGETS:
            for method in ("B1", "A5"):
                selected = [
                    row for row in subject_rows
                    if row["baseline"] == method and row["retention"] == retention
                ]
                aggregate_rows.append(
                    {
                        "subject": label,
                        "baseline": method,
                        "retention": retention,
                        **{
                            key: reducer([float(row[key]) for row in selected])
                            for key in (
                                "actionable_leakage",
                                "actionable_leakage_ratio",
                                "raw_leakage",
                                "raw_leakage_ratio",
                            )
                        },
                    }
                )
    return subject_rows + aggregate_rows


def _paired_subject_rows(main_rows: list[dict], matched_rows: list[dict]) -> list[dict]:
    rows = []
    for subject in SUBJECTS:
        by_method = {
            row["baseline"]: row
            for row in main_rows
            if row["subject"] == subject and row["baseline"] in ("B1", "A5")
        }
        row = {"subject": subject}
        for metric in SUMMARY_METRICS:
            b1 = float(by_method["B1"][metric])
            a5 = float(by_method["A5"][metric])
            row[f"b1_{metric}"] = b1
            row[f"a5_{metric}"] = a5
            row[f"delta_{metric}"] = a5 - b1
        for retention in RETENTION_TARGETS:
            token = str(retention).replace(".", "p")
            selected = {
                item["baseline"]: item
                for item in matched_rows
                if item["subject"] == subject and item["retention"] == retention
            }
            for metric in ("actionable_leakage", "raw_leakage"):
                b1 = float(selected["B1"][metric])
                a5 = float(selected["A5"][metric])
                row[f"b1_{metric}_r{token}"] = b1
                row[f"a5_{metric}_r{token}"] = a5
                row[f"delta_{metric}_r{token}"] = a5 - b1
        rows.append(row)
    return rows


def _paired_statistics_rows(subject_rows: list[dict], *, repetitions: int, seed: int) -> list[dict]:
    rows = []
    metric_specs = [(metric, True) for metric in SUMMARY_METRICS]
    metric_specs.extend(
        (f"actionable_leakage_r{str(retention).replace('.', 'p')}", False)
        for retention in RETENTION_TARGETS
    )
    metric_specs.extend(
        (f"raw_leakage_r{str(retention).replace('.', 'p')}", False)
        for retention in RETENTION_TARGETS
    )
    for metric, higher_is_better in metric_specs:
        deltas = [float(row[f"delta_{metric}"]) for row in subject_rows]
        rows.append(
            {
                "metric": metric,
                "delta_definition": "A5-B1",
                "higher_is_better": higher_is_better,
                **paired_statistics(
                    deltas,
                    repetitions=repetitions,
                    seed=seed,
                    higher_is_better=higher_is_better,
                ),
            }
        )
    return rows


def _paired_detail_rows(root: Path, filename: str, keys: tuple[str, ...]) -> list[dict]:
    rows = []
    metrics = ("iou", "boundary_f1", "boundary_iou", "soft_iou", "target", "predicted")
    for subject in SUBJECTS:
        source = _read_csv(root / f"CoreView_{subject}" / "main" / filename)
        grouped = {}
        for row in source:
            method = str(row.get("baseline"))
            if method not in ("B1", "A5"):
                continue
            key = tuple(str(row.get(value, "")) for value in keys)
            grouped.setdefault(key, {})[method] = row
        for key, methods in sorted(grouped.items()):
            if set(methods) != {"B1", "A5"}:
                raise ValueError(f"missing paired B1/A5 detail rows for {subject} {key}")
            output = {"subject": subject, **dict(zip(keys, key))}
            for metric in metrics:
                if metric not in methods["B1"] or metric not in methods["A5"]:
                    continue
                b1 = float(methods["B1"][metric])
                a5 = float(methods["A5"][metric])
                output[f"b1_{metric}"] = b1
                output[f"a5_{metric}"] = a5
                output[f"delta_{metric}"] = a5 - b1
            rows.append(output)
    return rows


def summarize(
    output_root: Path,
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260723,
) -> dict:
    root = Path(output_root)
    for subject in SUBJECTS:
        config = json.loads(
            (root / f"CoreView_{subject}" / "loso_frozen_config.json").read_text(
                encoding="utf-8"
            )
        )
        if config.get("selection_mode") != "leave_one_subject_out_a5":
            raise ValueError(f"CoreView_{subject} is not an A5 LOSO config")
        donors = {str(value) for value in config.get("donor_subjects", [])}
        if len(donors) != 4 or subject in donors:
            raise ValueError(f"CoreView_{subject} must have four non-self donors")

    main_rows = _aggregate_main(root)
    matched_rows = _matched_rows(root)
    subject_rows = _paired_subject_rows(main_rows, matched_rows)
    stats_rows = _paired_statistics_rows(
        subject_rows,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    part_rows = _paired_detail_rows(root, "per_part_metrics.csv", ("part",))
    view_rows = _paired_detail_rows(root, "per_view_metrics.csv", ("view", "part"))
    aggregate = root / "aggregate"
    _write_csv(aggregate / "main_table.csv", main_rows)
    _write_csv(aggregate / "matched_retention_table.csv", matched_rows)
    _write_csv(aggregate / "paired_subject_deltas.csv", subject_rows)
    _write_csv(aggregate / "paired_statistics.csv", stats_rows)
    _write_csv(aggregate / "per_part_deltas.csv", part_rows)
    _write_csv(aggregate / "per_view_deltas.csv", view_rows)
    configs = {
        subject: json.loads(
            (root / f"CoreView_{subject}" / "loso_frozen_config.json").read_text(
                encoding="utf-8"
            )
        )
        for subject in SUBJECTS
    }
    summary = {
        "subjects": list(SUBJECTS),
        "primary_method": "A5",
        "comparison_method": "B1",
        "selection_mode": "leave_one_subject_out_a5",
        "main_methods": sorted(MAIN_METHODS),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(bootstrap_seed),
        "method_freeze_fingerprints": sorted(
            {str(config["method_freeze_fingerprint"]) for config in configs.values()}
        ),
        "selected_thresholds": {
            subject: float(config["selected"]["soft_threshold"])
            for subject, config in configs.items()
        },
        "paired_statistics": stats_rows,
    }
    aggregate.mkdir(parents=True, exist_ok=True)
    (aggregate / "statistics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(aggregate / "statistics_summary.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize five-subject frozen A5 LOSO statistics.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summarize(
        args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
