#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from utils.semantic_temporal_stability import summarize_temporal_signal


SEQUENCE_METRICS = (
    "screen_soft_iou",
    "screen_hard_iou",
    "screen_precision",
    "screen_recall",
    "selection_leakage_ratio",
    "edit_target_delta_mean",
    "edit_outer_delta_mean",
    "edit_boundary_outer_delta_mean",
    "edit_outer_to_target_delta_ratio",
)
DEFAULT_PAIRED_DIRECTIONS = {
    "screen_soft_iou_mean": True,
    "screen_hard_iou_mean": True,
    "selection_leakage_ratio_mean": False,
    "edit_outer_to_target_delta_ratio_mean": False,
    "screen_soft_iou_cv": False,
    "edit_target_delta_mean_adjacent_flicker": False,
    "selection_leakage_ratio_adjacent_flicker": False,
}
NUMERIC_INPUT_FIELDS = {
    "camera",
    "frame",
    "soft_threshold",
    "screen_threshold",
    "selected_gaussian_count",
    "edit_weight_sum",
    "target_pixel_count",
    "valid_pixel_count",
    "inside_selection_mass",
    "outside_selection_mass",
    *SEQUENCE_METRICS,
    "edit_target_pixel_count",
    "edit_outer_pixel_count",
    "edit_boundary_outer_pixel_count",
    "edit_target_delta_sum",
    "edit_outer_delta_sum",
    "edit_boundary_outer_delta_sum",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize five-subject semantic temporal stability results.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", default=["377", "386", "387", "393", "394"])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args(argv)


def read_metrics(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty temporal metric CSV: {path}")
    for row_index, row in enumerate(rows):
        for key in NUMERIC_INPUT_FIELDS:
            if key not in row:
                continue
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite temporal metric {key} at row {row_index}: {path}")
            row[key] = value
    return rows


def summarize_sequences(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject"]), str(row["method"]), str(row["part"]))].append(row)
    outputs = []
    for (subject, method, part), members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: int(float(item["frame"])))
        supported = [row for row in members if float(row["target_pixel_count"]) > 0.0]
        item = {
            "subject": subject,
            "method": method,
            "part": part,
            "frame_count": len(members),
            "supported_frame_count": len(supported),
            "target_present_rate": len(supported) / max(len(members), 1),
        }
        for metric in SEQUENCE_METRICS:
            summary = (
                summarize_temporal_signal([float(row[metric]) for row in supported])
                if supported
                else {"mean": 0.0, "std": 0.0, "cv": 0.0, "adjacent_flicker": 0.0}
            )
            for statistic, value in summary.items():
                item[f"{metric}_{statistic}"] = value
        outputs.append(item)
    return outputs


def _sequence_value_fields(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    excluded = {"subject", "method", "part", "frame_count", "supported_frame_count"}
    return [key for key in rows[0] if key not in excluded]


def _mean_rows(rows: list[dict], *, keys: tuple[str, ...], value_fields: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    outputs = []
    for group_key, members in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["subject_count"] = len({str(member["subject"]) for member in members})
        for field in value_fields:
            values = np.asarray([float(member[field]) for member in members], dtype=np.float64)
            item[field] = float(np.mean(values))
            item[f"{field}_subject_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        outputs.append(item)
    return outputs


def subject_overall_rows(sequence_rows: list[dict]) -> list[dict]:
    value_fields = _sequence_value_fields(sequence_rows)
    grouped = defaultdict(list)
    for row in sequence_rows:
        grouped[(str(row["subject"]), str(row["method"]))].append(row)
    outputs = []
    for (subject, method), members in sorted(grouped.items()):
        item = {"subject": subject, "method": method, "part_count": len(members)}
        for field in value_fields:
            item[field] = float(np.mean([float(member[field]) for member in members]))
        outputs.append(item)
    return outputs


def aggregate_sequences(sequence_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    value_fields = _sequence_value_fields(sequence_rows)
    part_rows = _mean_rows(sequence_rows, keys=("method", "part"), value_fields=value_fields)
    subject_rows = subject_overall_rows(sequence_rows)
    overall_rows = _mean_rows(subject_rows, keys=("method",), value_fields=value_fields)
    return part_rows, overall_rows


def paired_statistics(
    subject_rows: list[dict],
    *,
    metric_directions: dict[str, bool] = DEFAULT_PAIRED_DIRECTIONS,
    repetitions: int,
    seed: int,
) -> list[dict]:
    lookup = {(str(row["subject"]), str(row["method"])): row for row in subject_rows}
    subjects = sorted({str(row["subject"]) for row in subject_rows})
    outputs = []
    for metric, higher_is_better in metric_directions.items():
        deltas = []
        used_subjects = []
        for subject in subjects:
            a5 = lookup.get((subject, "a5"))
            voting = lookup.get((subject, "voting"))
            if a5 is None or voting is None or metric not in a5 or metric not in voting:
                continue
            deltas.append(float(a5[metric]) - float(voting[metric]))
            used_subjects.append(subject)
        if not deltas:
            continue
        stats = paired_bootstrap(
            deltas,
            seed=int(seed),
            repetitions=int(repetitions),
            lower_is_better=not bool(higher_is_better),
        )
        outputs.append(
            {
                "comparison": "a5-voting",
                "metric": metric,
                "higher_is_better": bool(higher_is_better),
                "subjects": ";".join(used_subjects),
                **stats,
            }
        )
    return outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty temporal summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args) -> dict:
    output_root = args.output_root.resolve()
    all_rows = []
    subject_summaries = []
    for subject in args.subjects:
        subject_root = output_root / f"CoreView_{subject}"
        all_rows.extend(read_metrics(subject_root / "metrics.csv"))
        summary_path = subject_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing temporal subject summary: {summary_path}")
        subject_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    sequence_rows = summarize_sequences(all_rows)
    part_rows, overall_rows = aggregate_sequences(sequence_rows)
    subject_rows = subject_overall_rows(sequence_rows)
    paired_rows = paired_statistics(
        subject_rows,
        repetitions=int(args.bootstrap_repetitions),
        seed=int(args.bootstrap_seed),
    )
    aggregate_dir = output_root / "aggregate"
    _write_csv(aggregate_dir / "all_metrics.csv", all_rows)
    _write_csv(aggregate_dir / "per_sequence.csv", sequence_rows)
    _write_csv(aggregate_dir / "subject_overall.csv", subject_rows)
    _write_csv(aggregate_dir / "part_aggregate.csv", part_rows)
    _write_csv(aggregate_dir / "overall_aggregate.csv", overall_rows)
    _write_csv(aggregate_dir / "paired_statistics.csv", paired_rows)
    summary = {
        "subjects": [str(value) for value in args.subjects],
        "input_row_count": len(all_rows),
        "sequence_count": len(sequence_rows),
        "subject_overall_count": len(subject_rows),
        "part_aggregate_count": len(part_rows),
        "paired_statistic_count": len(paired_rows),
        "videos": [video for item in subject_summaries for video in item.get("videos", [])],
        "uses_test_parser_for_edit_selection": False,
        "uses_test_masks_for_metrics": True,
        "canonical_selection_fixed_across_frames": True,
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(summarize(parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
