#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tools.make_semantic_edit_render_preview import compose_preview_sheet


GROUP_KEYS = ("method", "task")
SUBJECT_METRICS = (
    "target_delta_mean",
    "outer_delta_mean",
    "boundary_outer_delta_mean",
    "outer_to_target_delta_ratio",
    "target_retention_vs_voting",
)
FIXED_SHEET_ROWS = (
    ("377", "c21_f000180", "upper"),
    ("377", "c23_f000540", "shoes"),
    ("393", "c22_f000420", "face"),
    ("394", "c21_f000180", "skin"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the five-subject real semantic editing paper suite.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=["377", "386", "387", "393", "394"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args(argv)


def _record_key(row: dict) -> tuple:
    return tuple(row[key] for key in ("subject", "view", "part", "task", "edit_strength"))


def add_voting_target_retention(rows: list[dict]) -> list[dict]:
    reference = {
        _record_key(row): float(row["target_delta_sum"])
        for row in rows
        if str(row["method"]) == "voting"
    }
    enriched = []
    for row in rows:
        key = _record_key(row)
        if key not in reference:
            raise ValueError(f"missing Voting target reference for {key}")
        item = dict(row)
        item["target_retention_vs_voting"] = float(row["target_delta_sum"]) / max(reference[key], 1.0e-8)
        enriched.append(item)
    return enriched


def subject_equal_aggregate(rows: list[dict], *, metrics=SUBJECT_METRICS) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject"]), str(row["method"]), str(row["task"]))].append(row)
    subject_rows = []
    for (subject, method, task), members in sorted(grouped.items()):
        item = {"subject": subject, "method": method, "task": task, "row_count": len(members)}
        for metric in metrics:
            item[metric] = float(np.mean([float(row[metric]) for row in members]))
        subject_rows.append(item)

    aggregate_grouped = defaultdict(list)
    for row in subject_rows:
        aggregate_grouped[(row["method"], row["task"])].append(row)
    aggregate_rows = []
    for (method, task), members in sorted(aggregate_grouped.items()):
        item = {"method": method, "task": task, "subject_count": len(members)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        aggregate_rows.append(item)
    return subject_rows, aggregate_rows


def paired_bootstrap(deltas, *, seed: int, repetitions: int, lower_is_better: bool) -> dict:
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = np.random.default_rng(int(seed))
    sampled = values[rng.integers(0, values.size, size=(int(repetitions), values.size))].mean(axis=1)
    if lower_is_better:
        wins = int(np.sum(values < 0.0))
        losses = int(np.sum(values > 0.0))
    else:
        wins = int(np.sum(values > 0.0))
        losses = int(np.sum(values < 0.0))
    return {
        "subject_count": int(values.size),
        "mean_delta": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "bootstrap_ci95_low": float(np.quantile(sampled, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(values == 0.0)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics(path: Path) -> list[dict]:
    numeric = {
        "edit_strength",
        "soft_threshold",
        "selected_gaussian_count",
        "edit_weight_sum",
        "target_pixel_count",
        "outer_pixel_count",
        "boundary_outer_pixel_count",
        "target_delta_sum",
        "outer_delta_sum",
        "boundary_outer_delta_sum",
        "target_delta_mean",
        "outer_delta_mean",
        "boundary_outer_delta_mean",
        "outer_to_target_delta_ratio",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def _paired_rows(subject_rows: list[dict], *, repetitions: int, seed: int) -> list[dict]:
    lookup = {(row["subject"], row["method"], row["task"]): row for row in subject_rows}
    metrics = {
        "outer_delta_mean": True,
        "boundary_outer_delta_mean": True,
        "outer_to_target_delta_ratio": True,
        "target_delta_mean": False,
        "target_retention_vs_voting": False,
    }
    out = []
    tasks = sorted({row["task"] for row in subject_rows})
    subjects = sorted({row["subject"] for row in subject_rows})
    for task in tasks:
        for baseline in ("raw_hard", "voting"):
            for metric, lower_is_better in metrics.items():
                deltas = [
                    float(lookup[(subject, "a5", task)][metric]) - float(lookup[(subject, baseline, task)][metric])
                    for subject in subjects
                ]
                stats = paired_bootstrap(
                    deltas,
                    seed=int(seed),
                    repetitions=int(repetitions),
                    lower_is_better=lower_is_better,
                )
                out.append(
                    {
                        "task": task,
                        "comparison": f"a5-{baseline}",
                        "metric": metric,
                        "lower_is_better": lower_is_better,
                        **stats,
                    }
                )
    return out


def _make_fixed_sheets(output_root: Path) -> list[str]:
    outputs = []
    for task in ("recolor", "removal", "texture"):
        panels = []
        for subject, view, part in FIXED_SHEET_ROWS:
            subject_root = output_root / f"CoreView_{subject}"
            paths = [
                ("RGB", subject_root / "frames" / f"{view}_rgb.png"),
                ("Raw Hard", subject_root / "frames" / task / "raw_hard" / f"{view}_{part}.png"),
                ("Voting", subject_root / "frames" / task / "voting" / f"{view}_{part}.png"),
                ("Ours (A5)", subject_root / "frames" / task / "a5" / f"{view}_{part}.png"),
            ]
            if not all(path.exists() for _, path in paths):
                continue
            panels.append(
                {
                    "view": f"CoreView_{subject}/{view}",
                    "part": part,
                    "images": [(label, Image.open(path).convert("RGB")) for label, path in paths],
                }
            )
        if panels:
            output = output_root / "aggregate" / f"paper_sheet_{task}.png"
            compose_preview_sheet(panels, output, thumb_size=256)
            outputs.append(str(output))
    return outputs


def summarize(args) -> dict:
    output_root = args.output_root.resolve()
    rows = []
    for subject in args.subjects:
        metrics_path = output_root / f"CoreView_{subject}" / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing subject metrics: {metrics_path}")
        rows.extend(_read_metrics(metrics_path))
    rows = add_voting_target_retention(rows)
    subject_rows, aggregate_rows = subject_equal_aggregate(rows)
    paired_rows = _paired_rows(
        subject_rows,
        repetitions=int(args.bootstrap_repetitions),
        seed=int(args.bootstrap_seed),
    )
    aggregate_dir = output_root / "aggregate"
    _write_csv(aggregate_dir / "all_metrics.csv", rows)
    _write_csv(aggregate_dir / "subject_means.csv", subject_rows)
    _write_csv(aggregate_dir / "aggregate_means.csv", aggregate_rows)
    _write_csv(aggregate_dir / "paired_statistics.csv", paired_rows)
    sheets = _make_fixed_sheets(output_root)
    summary = {
        "subjects": [str(value) for value in args.subjects],
        "row_count": len(rows),
        "subject_mean_count": len(subject_rows),
        "aggregate_count": len(aggregate_rows),
        "paired_statistic_count": len(paired_rows),
        "paper_sheets": sheets,
        "uses_test_parser_for_edit_selection": False,
        "uses_test_masks_for_metrics": True,
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(summarize(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
