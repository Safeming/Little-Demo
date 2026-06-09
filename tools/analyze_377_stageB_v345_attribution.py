#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")
SAMPLE_FILES = (
    ("contours/contour_samples.csv", {
        "fg": "fg_l1",
        "boundary": "boundary_l1",
        "edge": "edge_symmetric_dist_px",
    }),
    ("boundary_residuals/boundary_residual_samples.csv", {
        "inner": "inner_missing_pixels",
        "outer": "outer_leak_pixels",
        "hard": "hard_residual_score",
    }),
    ("opacity_footprint/opacity_footprint_samples.csv", {
        "opacity_inner": "primary_opacity_inner_missing_pixels",
        "opacity_outer": "primary_opacity_outer_leak_pixels",
    }),
)


def image_name_from_row(row: dict) -> str:
    cam = int(float(row.get("cam", 0)))
    frame = int(float(row.get("frame", 0)))
    return f"c{cam:02d}_f{frame:06d}"


def load_samples(render_exp: Path) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    diag = render_exp / "diagnostics"
    for rel_path, mapping in SAMPLE_FILES:
        path = diag / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_name = image_name_from_row(row)
                rec = records.setdefault(image_name, {})
                for metric, column in mapping.items():
                    rec[metric] = float(row[column])
    return records


def load_summary(render_exp: Path) -> dict[str, float]:
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads(
        (render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8")
    )
    opacity = json.loads(
        (render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8")
    )
    return {
        "samples": int(contour.get("n_samples", residual.get("n_samples", opacity.get("n_samples", 0)))),
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    }


def load_component_coverage(path: Path) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_name = str(row.get("image_name", "")).strip()
            direction = str(row.get("direction", "")).strip().lower()
            if not image_name or direction not in ("inner", "outer"):
                continue
            item = coverage.setdefault(image_name, {"inner_component_rows": 0, "outer_component_rows": 0})
            item[f"{direction}_component_rows"] += 1
    return coverage


def load_signed_point_stats(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image = data.get("by_image", {}) if isinstance(data, dict) else {}
    stats: dict[str, dict[str, object]] = {}
    for image_name, item in by_image.items():
        if not isinstance(item, dict):
            continue
        shrink = item.get("shrink_point_ids", []) or []
        grow = item.get("grow_point_ids", []) or []
        stats[image_name] = {
            "signed_shrink_points": len(shrink),
            "signed_grow_points": len(grow),
            "temporal_source": item.get("source_image_name", ""),
            "temporal_distance": item.get("temporal_distance", ""),
            "temporal_mode": item.get("temporal_mode", ""),
        }
    return stats


def status_for(delta: dict[str, float]) -> str:
    if all(delta[key] <= 0.0 for key in METRICS):
        return "strict_pass"
    probe = (
        delta["fg"] <= 0.00002
        and delta["boundary"] <= 0.00002
        and delta["edge"] <= 0.01
        and delta["inner"] <= 2.0
        and delta["outer"] <= 2.0
        and delta["hard"] <= 0.00025
        and delta["opacity_inner"] <= 2.0
        and delta["opacity_outer"] <= 2.0
    )
    return "probe_pass" if probe else "regresses"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize v345 worst-frame attribution ablations.")
    parser.add_argument("--exp-root", required=True, type=Path)
    parser.add_argument("--component-csv", required=True, type=Path)
    parser.add_argument("--signed-point-json", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--attribution-out", required=True, type=Path)
    parser.add_argument("--baseline-variant", default="baseline_no_preset")
    parser.add_argument("--current-variant", default="formal_v338_current")
    args = parser.parse_args()

    component_coverage = load_component_coverage(args.component_csv)
    signed_stats = load_signed_point_stats(args.signed_point_json)

    window_dirs = sorted(path for path in args.exp_root.iterdir() if path.is_dir())
    if not window_dirs:
        raise FileNotFoundError(f"no window dirs under {args.exp_root}")

    totals: dict[str, dict[str, float]] = {}
    by_window_variant: dict[tuple[str, str], dict[str, float]] = {}
    samples_by_window_variant: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    variants: list[str] = []

    for window_dir in window_dirs:
        variant_dirs = sorted(path for path in window_dir.iterdir() if path.is_dir())
        for variant_dir in variant_dirs:
            variant = variant_dir.name
            if not (variant_dir / "diagnostics/contours/contour_summary.json").exists():
                continue
            if variant not in variants:
                variants.append(variant)
            summary = load_summary(variant_dir)
            by_window_variant[(window_dir.name, variant)] = summary
            samples_by_window_variant[(window_dir.name, variant)] = load_samples(variant_dir)
            total = totals.setdefault(variant, {"samples": 0.0, **{metric: 0.0 for metric in METRICS}})
            total["samples"] += summary["samples"]
            for metric in METRICS:
                total[metric] += summary[metric] * summary["samples"]

    if args.baseline_variant not in totals:
        raise KeyError(f"missing baseline variant: {args.baseline_variant}")
    base_total = totals[args.baseline_variant]
    base_samples = max(base_total["samples"], 1.0)
    base_mean = {metric: base_total[metric] / base_samples for metric in METRICS}

    summary_rows = []
    for variant in variants:
        total = totals[variant]
        samples = max(total["samples"], 1.0)
        metrics = {metric: total[metric] / samples for metric in METRICS}
        delta = {metric: metrics[metric] - base_mean[metric] for metric in METRICS}
        summary_rows.append({
            "variant": variant,
            "samples": int(total["samples"]),
            **metrics,
            **{f"{metric}_delta_base": delta[metric] for metric in METRICS},
            "status": "baseline" if variant == args.baseline_variant else status_for(delta),
        })

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    attribution_rows = []
    for window_dir in window_dirs:
        window = window_dir.name
        baseline = samples_by_window_variant.get((window, args.baseline_variant), {})
        current = samples_by_window_variant.get((window, args.current_variant), {})
        if not baseline or not current:
            continue
        for image_name, cur in current.items():
            base = baseline.get(image_name)
            if not base:
                continue
            current_delta = {metric: cur.get(metric, 0.0) - base.get(metric, 0.0) for metric in METRICS}
            if max(current_delta.values()) <= 0.0:
                continue
            cov = component_coverage.get(image_name, {"inner_component_rows": 0, "outer_component_rows": 0})
            sstat = signed_stats.get(image_name, {
                "signed_shrink_points": 0,
                "signed_grow_points": 0,
                "temporal_source": "",
                "temporal_distance": "",
                "temporal_mode": "",
            })
            row = {
                "window": window,
                "image_name": image_name,
                "current_max_positive_delta": max(current_delta.values()),
                **{f"current_{metric}_delta_base": current_delta[metric] for metric in METRICS},
                **cov,
                **sstat,
            }
            for variant in variants:
                if variant in (args.baseline_variant, args.current_variant):
                    continue
                candidate = samples_by_window_variant.get((window, variant), {}).get(image_name)
                if not candidate:
                    continue
                variant_delta = {metric: candidate.get(metric, 0.0) - base.get(metric, 0.0) for metric in METRICS}
                row[f"{variant}_opacity_outer_delta_base"] = variant_delta["opacity_outer"]
                row[f"{variant}_hard_delta_base"] = variant_delta["hard"]
                row[f"{variant}_outer_delta_base"] = variant_delta["outer"]
                row[f"{variant}_max_positive_delta"] = max(variant_delta.values())
                row[f"{variant}_fixes_opacity_outer"] = (
                    current_delta["opacity_outer"] > 0.0 and variant_delta["opacity_outer"] <= 0.0
                )
            attribution_rows.append(row)

    attribution_rows.sort(
        key=lambda item: (float(item.get("current_max_positive_delta", 0.0)), str(item.get("image_name", ""))),
        reverse=True,
    )
    fieldnames = []
    for row in attribution_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with args.attribution_out.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(attribution_rows)
        else:
            handle.write("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
