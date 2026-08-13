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

from utils.four_method_paper_evidence import (
    aggregate_frame,
    build_temporal_windows,
    exact_block_sign_flip,
    hierarchical_bootstrap_paired,
    holm_adjust,
    summarize_temporal_window,
)


METHOD_ORDER = ("saga", "gaussian_grouping", "sggs", "a5")
METHOD_LABELS = {
    "input": "Input",
    "saga": "SAGA",
    "gaussian_grouping": "Gaussian Grouping",
    "sggs": "SG-GS",
    "a5": "Ours",
}
METHOD_COLORS = {
    "a5": "#176f48",
    "saga": "#356aa0",
    "gaussian_grouping": "#b26a1b",
    "sggs": "#8a4f8f",
}
METRICS = (
    "actionable_leakage",
    "raw_leakage",
    "macro_miou",
    "mean_boundary_f1",
)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_method_frames(rows) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            str(row["subject"]),
            str(row["method"]),
            int(row["camera"]),
            int(row["frame"]),
        )
        grouped[key].append(row)
    outputs = []
    for (subject, method, camera, frame), members in sorted(grouped.items()):
        summary = aggregate_frame(members)
        outputs.append(
            {
                "subject": subject,
                "method": method,
                "camera": camera,
                "frame": frame,
                "retention": float(members[0].get("retention", 0.6)),
                "target_retention_feasible": _as_bool(
                    members[0].get("target_retention_feasible", True)
                ),
                **summary,
            }
        )
    return outputs


def summarize_significance(
    rows,
    *,
    iterations: int = 20_000,
    seed: int = 20260813,
) -> dict:
    frames = aggregate_method_frames(rows)
    by_method_key = {
        (row["method"], row["subject"], row["camera"], row["frame"]): row
        for row in frames
    }
    comparisons = []
    per_subject = []
    raw_p_values = defaultdict(dict)
    for comparator in METHOD_ORDER:
        if comparator == "a5":
            continue
        comparator_rows = [
            row
            for row in frames
            if row["method"] == comparator and row["target_retention_feasible"]
        ]
        paired = []
        for external in comparator_rows:
            ours = by_method_key.get(
                ("a5", external["subject"], external["camera"], external["frame"])
            )
            if ours is None or not ours["target_retention_feasible"]:
                continue
            paired.append((ours, external))
        if not paired:
            raise ValueError(f"no paired rows for a5 vs {comparator}")
        paired_keys = {
            (row[0]["subject"], row[0]["camera"], row[0]["frame"])
            for row in paired
        }
        if len(paired_keys) != len(paired):
            raise ValueError(f"duplicate paired keys for a5 vs {comparator}")
        for metric in METRICS:
            difference_rows = [
                {
                    "subject": ours["subject"],
                    "camera": ours["camera"],
                    "frame": ours["frame"],
                    "difference": float(ours[metric]) - float(external[metric]),
                }
                for ours, external in paired
            ]
            bootstrap = hierarchical_bootstrap_paired(
                difference_rows,
                value_key="difference",
                iterations=iterations,
                seed=seed,
            )
            permutation = exact_block_sign_flip(
                difference_rows,
                value_key="difference",
            )
            ours_estimate = float(np.mean([float(pair[0][metric]) for pair in paired]))
            comparator_estimate = float(
                np.mean([float(pair[1][metric]) for pair in paired])
            )
            difference = ours_estimate - comparator_estimate
            relative = (
                (comparator_estimate - ours_estimate) / comparator_estimate
                if comparator_estimate != 0.0
                else 0.0
            )
            result = {
                "method": "a5",
                "comparison_method": comparator,
                "metric": metric,
                "ours_estimate": ours_estimate,
                "comparison_estimate": comparator_estimate,
                "absolute_difference": difference,
                "paired_median_difference": float(
                    np.median([row["difference"] for row in difference_rows])
                ),
                "relative_reduction": float(relative),
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "p_value_raw": permutation["p_value"],
                "subject_count": len({pair[0]["subject"] for pair in paired}),
                "camera_frame_count": len(paired),
                "block_count": permutation["block_count"],
                "permutation_count": permutation["permutation_count"],
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(seed),
            }
            comparisons.append(result)
            raw_p_values[metric][comparator] = permutation["p_value"]
            for subject in sorted({pair[0]["subject"] for pair in paired}):
                subject_pairs = [pair for pair in paired if pair[0]["subject"] == subject]
                ours_subject = float(
                    np.mean([float(pair[0][metric]) for pair in subject_pairs])
                )
                external_subject = float(
                    np.mean([float(pair[1][metric]) for pair in subject_pairs])
                )
                per_subject.append(
                    {
                        "subject": subject,
                        "method": "a5",
                        "comparison_method": comparator,
                        "metric": metric,
                        "ours_estimate": ours_subject,
                        "comparison_estimate": external_subject,
                        "absolute_difference": ours_subject - external_subject,
                        "camera_frame_count": len(subject_pairs),
                    }
                )
    adjusted = {
        metric: holm_adjust(values) for metric, values in raw_p_values.items()
    }
    for row in comparisons:
        row["p_value_holm"] = adjusted[row["metric"]][row["comparison_method"]]
    comparisons.sort(key=lambda row: (METRICS.index(row["metric"]), METHOD_ORDER.index(row["comparison_method"])))
    per_subject.sort(
        key=lambda row: (
            METRICS.index(row["metric"]),
            METHOD_ORDER.index(row["comparison_method"]),
            row["subject"],
        )
    )
    return {
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
        "comparisons": comparisons,
        "per_subject": per_subject,
    }


def summarize_temporal(rows) -> dict:
    frames = aggregate_method_frames(rows)
    windows = build_temporal_windows()
    window_lookup = {
        (int(window["camera"]), int(frame)): window
        for window in windows
        for frame in window["frames"]
    }
    grouped = defaultdict(list)
    for row in frames:
        window = window_lookup.get((int(row["camera"]), int(row["frame"])))
        if window is None:
            raise ValueError(
                f"frame outside frozen temporal windows: c{row['camera']} f{row['frame']}"
            )
        grouped[(row["subject"], row["method"], window["window"])].append(row)
    window_rows = []
    for (subject, method, window_name), members in sorted(grouped.items()):
        window = next(item for item in windows if item["window"] == window_name)
        summary = summarize_temporal_window(members, metric_names=METRICS)
        window_rows.append(
            {
                "subject": subject,
                "method": method,
                "camera": int(window["camera"]),
                "anchor": int(window["anchor"]),
                "window": window_name,
                "retention": float(members[0].get("retention", 0.6)),
                "target_retention_feasible": _as_bool(
                    members[0].get("target_retention_feasible", True)
                ),
                **summary,
            }
        )
    method_rows = []
    by_subject_method = defaultdict(list)
    for row in window_rows:
        by_subject_method[(row["subject"], row["method"])].append(row)
    for (subject, method), members in sorted(by_subject_method.items()):
        output = {
            "subject": subject,
            "method": method,
            "window_count": len(members),
            "camera_frame_count": int(sum(int(row["frame_count"]) for row in members)),
        }
        for metric in METRICS:
            for suffix in ("mean", "std", "mean_abs_delta", "p95_abs_delta"):
                key = f"{metric}_{suffix}"
                output[key] = float(np.mean([float(row[key]) for row in members]))
        method_rows.append(output)
    main_table = []
    by_method = defaultdict(list)
    for row in window_rows:
        if row["target_retention_feasible"]:
            by_method[row["method"]].append(row)
    for method in METHOD_ORDER:
        members = by_method.get(method, [])
        if not members:
            continue
        output = {
            "method": method,
            "subject_count": len({row["subject"] for row in members}),
            "window_count": len(members),
            "camera_frame_count": int(sum(int(row["frame_count"]) for row in members)),
        }
        for metric in METRICS:
            for suffix in ("mean", "std", "mean_abs_delta", "p95_abs_delta"):
                key = f"{metric}_{suffix}"
                values = np.asarray([float(row[key]) for row in members], dtype=np.float64)
                output[key] = float(np.mean(values))
                output[f"{key}_across_window_std"] = float(np.std(values))
        main_table.append(output)
    return {
        "per_frame": frames,
        "windows": window_rows,
        "methods": method_rows,
        "main_table": main_table,
    }


def compose_part_layout(
    *,
    subjects,
    methods,
    part: str,
    frame_paths,
    output_dir: Path | str,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    subjects = [str(value) for value in subjects]
    methods = [str(value) for value in methods]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(subjects),
        len(methods),
        figsize=(2.35 * len(methods), 3.05 * len(subjects)),
        squeeze=False,
        facecolor="white",
    )
    for row_index, subject in enumerate(subjects):
        for column_index, method in enumerate(methods):
            path = Path(frame_paths[(subject, method)])
            with Image.open(path) as image:
                axes[row_index][column_index].imshow(image.convert("RGB"))
            axes[row_index][column_index].axis("off")
            if row_index == 0:
                axes[row_index][column_index].set_title(
                    METHOD_LABELS[method], fontsize=12, pad=7
                )
            if column_index == 0:
                axes[row_index][column_index].set_ylabel(
                    subject,
                    fontsize=12,
                    rotation=0,
                    labelpad=24,
                    va="center",
                )
            if method == "gaussian_grouping" and subject == "377":
                axes[row_index][column_index].text(
                    0.04,
                    0.96,
                    "GG\N{DAGGER}",
                    transform=axes[row_index][column_index].transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="black",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.01, wspace=0.03, hspace=0.05)
    stem = f"{part}_three_subject_five_method"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {
        "part": str(part),
        "columns": [METHOD_LABELS[method] for method in methods],
        "subjects": subjects,
        "gg_377_label": "GG\N{DAGGER}",
        "png": str(png),
        "pdf": str(pdf),
    }


def _format_significance_markdown(rows: list[dict]) -> str:
    lines = [
        "| Metric | Comparison | Ours | External | Difference | 95% CI | p (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {comparison_method} | {ours_estimate:.6f} | "
            "{comparison_estimate:.6f} | {absolute_difference:.6f} | "
            "[{ci_low:.6f}, {ci_high:.6f}] | {p_value_holm:.6g} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _format_significance_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"Metric & Comparison & Ours & External & $\Delta$ & Holm $p$ \\",
        r"\hline",
    ]
    for row in rows:
        metric = str(row["metric"]).replace("_", r"\_")
        comparison = METHOD_LABELS[str(row["comparison_method"])].replace("_", r"\_")
        lines.append(
            f"{metric} & {comparison} & {row['ours_estimate']:.6f} & "
            f"{row['comparison_estimate']:.6f} & {row['absolute_difference']:.6f} & "
            f"{row['p_value_holm']:.6g} " + r"\\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def write_significance(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "significance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "comparisons.csv", result["comparisons"])
    _write_csv(output / "per_subject.csv", result["per_subject"])
    (output / "significance.md").write_text(
        _format_significance_markdown(result["comparisons"]), encoding="utf-8"
    )
    (output / "significance.tex").write_text(
        _format_significance_latex(result["comparisons"]), encoding="utf-8"
    )


def write_temporal(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "per_frame.csv", result["per_frame"])
    _write_csv(output / "per_window.csv", result["windows"])
    _write_csv(output / "temporal_table.csv", result["methods"])
    _write_csv(output / "main_table.csv", result["main_table"])
    (output / "main_table.md").write_text(
        _format_temporal_markdown(result["main_table"]), encoding="utf-8"
    )
    (output / "main_table.tex").write_text(
        _format_temporal_latex(result["main_table"]), encoding="utf-8"
    )
    _plot_temporal_curves(output / "curves", result["per_frame"])


def _format_temporal_markdown(rows: list[dict]) -> str:
    lines = [
        "| Method | Subjects | Windows | Actionable leakage | Actionable delta | mIoU | mIoU delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {subject_count} | {window_count} | {leak:.6f} | "
            "{leak_delta:.6f} | {miou:.6f} | {miou_delta:.6f} |".format(
                label=METHOD_LABELS[row["method"]],
                subject_count=row["subject_count"],
                window_count=row["window_count"],
                leak=row["actionable_leakage_mean"],
                leak_delta=row["actionable_leakage_mean_abs_delta"],
                miou=row["macro_miou_mean"],
                miou_delta=row["macro_miou_mean_abs_delta"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_temporal_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"Method & Subjects & Windows & Act. leak & Act. $\Delta$ & mIoU & mIoU $\Delta$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{METHOD_LABELS[row['method']]} & {row['subject_count']} & {row['window_count']} & "
            f"{row['actionable_leakage_mean']:.6f} & "
            f"{row['actionable_leakage_mean_abs_delta']:.6f} & "
            f"{row['macro_miou_mean']:.6f} & "
            f"{row['macro_miou_mean_abs_delta']:.6f} " + r"\\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _plot_temporal_curves(output: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    subjects = sorted({str(row["subject"]) for row in rows})
    cameras = sorted({int(row["camera"]) for row in rows})
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    for metric in METRICS:
        fig, axes = plt.subplots(
            len(subjects),
            len(cameras),
            figsize=(4.0 * len(cameras), 2.65 * len(subjects)),
            squeeze=False,
            facecolor="white",
            sharex=True,
        )
        for row_index, subject in enumerate(subjects):
            for column_index, camera in enumerate(cameras):
                axis = axes[row_index][column_index]
                for method in methods:
                    selected = sorted(
                        (
                            row
                            for row in rows
                            if row["subject"] == subject
                            and int(row["camera"]) == camera
                            and row["method"] == method
                        ),
                        key=lambda row: int(row["frame"]),
                    )
                    if selected:
                        axis.plot(
                            [int(row["frame"]) for row in selected],
                            [float(row[metric]) for row in selected],
                            color=METHOD_COLORS[method],
                            linewidth=1.25,
                            label=METHOD_LABELS[method],
                        )
                axis.set_title(f"CoreView {subject}, c{camera}", fontsize=9)
                axis.grid(True, color="#dddddd", linewidth=0.5)
                if column_index == 0:
                    axis.set_ylabel(metric.replace("_", " "), fontsize=8)
                if row_index == len(subjects) - 1:
                    axis.set_xlabel("Frame", fontsize=8)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
        fig.subplots_adjust(top=0.91, left=0.08, right=0.99, bottom=0.09, wspace=0.22, hspace=0.33)
        fig.savefig(output / f"{metric}.png", dpi=220, facecolor="white")
        fig.savefig(output / f"{metric}.pdf", facecolor="white")
        plt.close(fig)


def concat_csv(inputs: list[Path], output: Path) -> dict:
    fieldnames = None
    rows = []
    for path in inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise ValueError(f"CSV headers differ: {path}")
            rows.extend(reader)
    if not fieldnames:
        raise ValueError("input CSVs have no header")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"input_count": len(inputs), "row_count": len(rows), "output": str(output)}


def write_qualitative(manifest: Path, output: Path) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    subjects = payload["subjects"]
    methods = payload["methods"]
    layouts = []
    for set_name, part_payload in payload["sets"].items():
        for part, subject_payload in part_payload.items():
            paths = {
                (str(subject), str(method)): Path(subject_payload[str(subject)][str(method)])
                for subject in subjects
                for method in methods
            }
            layouts.append(
                {
                    "set": set_name,
                    **compose_part_layout(
                        subjects=subjects,
                        methods=methods,
                        part=part,
                        frame_paths=paths,
                        output_dir=output / set_name,
                    ),
                }
            )
    result = {"manifest": str(manifest), "layouts": layouts}
    output.mkdir(parents=True, exist_ok=True)
    (output / "qualitative_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize frozen four-method paper evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    significance = subparsers.add_parser("significance")
    significance.add_argument("--input", required=True, type=Path)
    significance.add_argument("--output", required=True, type=Path)
    significance.add_argument("--iterations", type=int, default=20_000)
    significance.add_argument("--seed", type=int, default=20260813)
    temporal = subparsers.add_parser("temporal")
    temporal.add_argument("--input", required=True, type=Path)
    temporal.add_argument("--output", required=True, type=Path)
    concat = subparsers.add_parser("concat")
    concat.add_argument("--input", action="append", required=True, type=Path)
    concat.add_argument("--output", required=True, type=Path)
    qualitative = subparsers.add_parser("qualitative")
    qualitative.add_argument("--manifest", required=True, type=Path)
    qualitative.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "significance":
        rows = _load_csv(args.input)
        write_significance(
            args.output,
            summarize_significance(rows, iterations=args.iterations, seed=args.seed),
        )
    elif args.command == "temporal":
        rows = _load_csv(args.input)
        write_temporal(args.output, summarize_temporal(rows))
    elif args.command == "concat":
        concat_csv(args.input, args.output)
    elif args.command == "qualitative":
        write_qualitative(args.manifest, args.output)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
