#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_five_subject_a5_loso_statistics import paired_statistics
from tools.summarize_paper_baseline_loso_ablation import (
    _write_csv,
    aggregate_method_reports,
)


SUBJECTS = ("377", "386", "387", "393", "394")
COMPONENT_METHODS = {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}
RETENTION_TARGETS = (0.5, 0.6)
VARIANTS = ("A4", "center_only", "no_outer", "full")
SEGMENTATION_METRICS = (
    "macro_miou",
    "mean_boundary_f1",
    "mean_boundary_iou",
    "mean_soft_iou",
    "micro_iou",
)


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def variant_label(variant: str) -> str:
    labels = {
        "A4": "Voting posterior (A4)",
        "center_only": "A5 center-only evidence",
        "no_outer": "A5 without outer penalty",
        "full": "Ours (A5 full footprint)",
    }
    try:
        return labels[str(variant)]
    except KeyError as error:
        raise ValueError(f"unknown A5 micro-ablation variant: {variant}") from error


def validate_component_methods(methods) -> None:
    values = {str(value) for value in methods}
    if values != COMPONENT_METHODS:
        raise ValueError(
            f"component ablation requires the complete A0-A6 chain, got {sorted(values)}"
        )


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_mean_std(rows: list[dict], *, group_key: str) -> list[dict]:
    subject_rows = list(rows)
    aggregate_rows = []
    for label, reducer in (("MEAN", statistics.mean), ("STD", statistics.stdev)):
        for group in sorted({str(row[group_key]) for row in subject_rows}):
            selected = [row for row in subject_rows if str(row[group_key]) == group]
            aggregate = {
                "subject": label,
                group_key: group,
                "display_label": str(selected[0].get("display_label", group)),
            }
            for key in sorted({key for row in selected for key in row}):
                if key in aggregate or key in ("subject", group_key, "display_label", "name", "oracle", "persistent_asset"):
                    continue
                values = [_numeric(row.get(key)) for row in selected]
                if all(value is not None for value in values):
                    aggregate[key] = reducer(values)
            aggregate_rows.append(aggregate)
    return subject_rows + aggregate_rows


def _component_rows(root: Path) -> list[dict]:
    reports = {
        subject: root / f"CoreView_{subject}" / "component" / "baseline_summary.csv"
        for subject in SUBJECTS
    }
    rows = aggregate_method_reports(reports)
    validate_component_methods(row["baseline"] for row in rows if row["subject"] != "MEAN")
    decorated = []
    for row in rows:
        method = str(row["baseline"])
        decorated.append(
            {
                **row,
                "display_label": (
                    "Ours (A5)" if method == "A5" else
                    "Target/Support extension (A6)" if method == "A6" else method
                ),
                "role": "primary" if method == "A5" else (
                    "extension_ablation" if method == "A6" else "component"
                ),
            }
        )
    return decorated


def _find_baseline_row(path: Path, method: str) -> dict:
    rows = [row for row in _read_csv(path) if str(row.get("baseline")) == method]
    if len(rows) != 1:
        raise ValueError(f"expected one {method} row in {path}, found {len(rows)}")
    return rows[0]


def _micro_rows(root: Path) -> list[dict]:
    specs = {
        "A4": ("component", "A4"),
        "center_only": ("center_only", "A5"),
        "no_outer": ("no_outer", "A5"),
        "full": ("component", "A5"),
    }
    rows = []
    for subject in SUBJECTS:
        for variant, (stage, method) in specs.items():
            source = _find_baseline_row(
                root / f"CoreView_{subject}" / stage / "baseline_summary.csv",
                method,
            )
            row = {
                "subject": subject,
                "variant": variant,
                "display_label": variant_label(variant),
                "source_method": method,
            }
            for metric in SEGMENTATION_METRICS:
                row[metric] = float(source[metric])
            rows.append(row)
    return _append_mean_std(rows, group_key="variant")


def _row_at_retention(path: Path, *, method: str, retention: float) -> dict:
    matches = [
        row
        for row in _read_csv(path)
        if str(row.get("baseline")) == method
        and abs(float(row.get("retention", -1.0)) - float(retention)) <= 1.0e-6
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} row at retention {retention} in {path}, found {len(matches)}"
        )
    return matches[0]


def _matched_rows(root: Path) -> list[dict]:
    specs = {
        "A4": ("component", "A4"),
        "center_only": ("center_only", "A5"),
        "no_outer": ("no_outer", "A5"),
        "full": ("component", "A5"),
    }
    rows = []
    for subject in SUBJECTS:
        for retention in RETENTION_TARGETS:
            for variant, (stage, method) in specs.items():
                source = _row_at_retention(
                    root / f"CoreView_{subject}" / stage / "matched_retention.csv",
                    method=method,
                    retention=retention,
                )
                rows.append(
                    {
                        "subject": subject,
                        "variant": variant,
                        "display_label": variant_label(variant),
                        "retention": retention,
                        "actionable_leakage": float(source["actionable_leakage"]),
                        "actionable_leakage_ratio": float(source["actionable_leakage_ratio"]),
                        "raw_leakage": float(source["raw_leakage"]),
                        "raw_leakage_ratio": float(source["raw_leakage_ratio"]),
                    }
                )
    subject_rows = list(rows)
    aggregate_rows = []
    for label, reducer in (("MEAN", statistics.mean), ("STD", statistics.stdev)):
        for retention in RETENTION_TARGETS:
            for variant in VARIANTS:
                selected = [
                    row for row in subject_rows
                    if row["variant"] == variant and row["retention"] == retention
                ]
                aggregate_rows.append(
                    {
                        "subject": label,
                        "variant": variant,
                        "display_label": variant_label(variant),
                        "retention": retention,
                        **{
                            metric: reducer([float(row[metric]) for row in selected])
                            for metric in (
                                "actionable_leakage",
                                "actionable_leakage_ratio",
                                "raw_leakage",
                                "raw_leakage_ratio",
                            )
                        },
                    }
                )
    return subject_rows + aggregate_rows


def _paired_statistics_rows(
    micro_rows: list[dict],
    matched_rows: list[dict],
    *,
    repetitions: int,
    seed: int,
) -> list[dict]:
    rows = []
    for variant in ("center_only", "no_outer", "full"):
        for metric in SEGMENTATION_METRICS:
            deltas = []
            for subject in SUBJECTS:
                selected = {
                    row["variant"]: row
                    for row in micro_rows
                    if row["subject"] == subject and row["variant"] in ("A4", variant)
                }
                deltas.append(float(selected[variant][metric]) - float(selected["A4"][metric]))
            rows.append(
                {
                    "variant": variant,
                    "display_label": variant_label(variant),
                    "metric": metric,
                    "delta_definition": f"{variant}-A4",
                    "higher_is_better": True,
                    **paired_statistics(
                        deltas,
                        repetitions=repetitions,
                        seed=seed,
                        higher_is_better=True,
                    ),
                }
            )
        for retention in RETENTION_TARGETS:
            token = str(retention).replace(".", "p")
            for metric in ("actionable_leakage", "raw_leakage"):
                deltas = []
                for subject in SUBJECTS:
                    selected = {
                        row["variant"]: row
                        for row in matched_rows
                        if row["subject"] == subject
                        and row["retention"] == retention
                        and row["variant"] in ("A4", variant)
                    }
                    deltas.append(float(selected[variant][metric]) - float(selected["A4"][metric]))
                rows.append(
                    {
                        "variant": variant,
                        "display_label": variant_label(variant),
                        "metric": f"{metric}_r{token}",
                        "delta_definition": f"{variant}-A4",
                        "higher_is_better": False,
                        **paired_statistics(
                            deltas,
                            repetitions=repetitions,
                            seed=seed,
                            higher_is_better=False,
                        ),
                    }
                )
    return rows


def summarize(
    output_root: Path,
    *,
    bootstrap_repetitions: int = 10000,
    bootstrap_seed: int = 20260723,
) -> dict:
    root = Path(output_root)
    thresholds = {}
    for subject in SUBJECTS:
        config = json.loads(
            (root / f"CoreView_{subject}" / "loso_frozen_config.json").read_text(
                encoding="utf-8"
            )
        )
        if config.get("selection_mode") != "leave_one_subject_out_a5":
            raise ValueError(f"CoreView_{subject} does not use frozen A5 LOSO")
        thresholds[subject] = float(config["selected"]["soft_threshold"])

    component_rows = _component_rows(root)
    micro_rows = _micro_rows(root)
    matched_rows = _matched_rows(root)
    stats_rows = _paired_statistics_rows(
        micro_rows,
        matched_rows,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    aggregate = root / "aggregate"
    _write_csv(aggregate / "component_table.csv", component_rows)
    _write_csv(aggregate / "a5_micro_ablation_table.csv", micro_rows)
    _write_csv(aggregate / "matched_retention_table.csv", matched_rows)
    _write_csv(aggregate / "paired_statistics.csv", stats_rows)
    summary = {
        "subjects": list(SUBJECTS),
        "selection_mode": "leave_one_subject_out_a5",
        "selected_thresholds": thresholds,
        "component_methods": sorted(COMPONENT_METHODS),
        "micro_variants": list(VARIANTS),
        "primary_method": "A5",
        "extension_methods": ["A6"],
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(bootstrap_seed),
        "paired_statistics": stats_rows,
    }
    aggregate.mkdir(parents=True, exist_ok=True)
    (aggregate / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(aggregate / "summary.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize unified five-subject A5 paper ablations.")
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
