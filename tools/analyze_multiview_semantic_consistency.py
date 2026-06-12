#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_projected_soft_edit_leakage import analyze_scene as analyze_projected_scene
from utils.part_label_bank import PART_NAMES


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _stats(rows: list[dict], key: str) -> tuple[float, float, float]:
    values = np.array([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std())
    return mean, std, _safe_ratio(std, abs(mean))


def summarize_multiview_rows(rows: list[dict], *, soft_threshold: float) -> dict:
    rows = list(rows)
    parts = sorted({str(row["part"]) for row in rows})
    views = sorted({str(row["view"]) for row in rows})
    per_part = []
    for part in parts:
        part_rows = [row for row in rows if str(row["part"]) == part]
        out = {"part": part, "view_count": int(len({str(row["view"]) for row in part_rows}))}
        for mode in ("hard", "soft"):
            mode_rows = [row for row in part_rows if str(row["mode"]) == mode]
            for key in ("target_activation", "leakage_ratio", "boundary_leakage_ratio", "target_coverage"):
                mean, std, cv = _stats(mode_rows, key)
                out[f"{mode}_{key}_mean"] = mean
                out[f"{mode}_{key}_std"] = std
                out[f"{mode}_{key}_cv"] = cv
        out["leakage_mean_delta_soft_minus_hard"] = out["soft_leakage_ratio_mean"] - out["hard_leakage_ratio_mean"]
        out["leakage_std_delta_soft_minus_hard"] = out["soft_leakage_ratio_std"] - out["hard_leakage_ratio_std"]
        out["boundary_leakage_mean_delta_soft_minus_hard"] = (
            out["soft_boundary_leakage_ratio_mean"] - out["hard_boundary_leakage_ratio_mean"]
        )
        out["boundary_leakage_std_delta_soft_minus_hard"] = (
            out["soft_boundary_leakage_ratio_std"] - out["hard_boundary_leakage_ratio_std"]
        )
        out["target_activation_cv_delta_soft_minus_hard"] = (
            out["soft_target_activation_cv"] - out["hard_target_activation_cv"]
        )
        per_part.append(out)

    leakage_std_delta = np.array(
        [row["leakage_std_delta_soft_minus_hard"] for row in per_part], dtype=np.float64
    )
    boundary_std_delta = np.array(
        [row["boundary_leakage_std_delta_soft_minus_hard"] for row in per_part], dtype=np.float64
    )
    target_cv_delta = np.array(
        [row["target_activation_cv_delta_soft_minus_hard"] for row in per_part], dtype=np.float64
    )
    summary = {
        "part_count": int(len(per_part)),
        "view_count": int(len(views)),
        "view_row_count": int(len(rows)),
        "soft_threshold": float(soft_threshold),
        "mean_leakage_std_delta_soft_minus_hard": (
            float(leakage_std_delta.mean()) if leakage_std_delta.size else 0.0
        ),
        "mean_boundary_leakage_std_delta_soft_minus_hard": (
            float(boundary_std_delta.mean()) if boundary_std_delta.size else 0.0
        ),
        "mean_target_activation_cv_delta_soft_minus_hard": (
            float(target_cv_delta.mean()) if target_cv_delta.size else 0.0
        ),
    }
    return {"summary": summary, "per_part": per_part, "per_view": rows}


def _write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir: Path | str, result: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        output_dir / "per_part.csv",
        list(result["per_part"]),
        [
            "part",
            "view_count",
            "leakage_std_delta_soft_minus_hard",
            "boundary_leakage_std_delta_soft_minus_hard",
            "target_activation_cv_delta_soft_minus_hard",
            "soft_leakage_ratio_mean",
            "hard_leakage_ratio_mean",
            "soft_boundary_leakage_ratio_mean",
            "hard_boundary_leakage_ratio_mean",
        ],
    )
    _write_csv(
        output_dir / "per_view.csv",
        list(result["per_view"]),
        ["part", "mode", "view", "leakage_ratio", "boundary_leakage_ratio", "target_activation"],
    )


def ensure_projected_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(args, "soft_boundary_radius"):
        args.soft_boundary_radius = 0
    if not hasattr(args, "soft_boundary_min_value"):
        args.soft_boundary_min_value = 0.25
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure hard-vs-soft semantic edit consistency across projected views."
    )
    parser.add_argument("--part-label-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", default=list(PART_NAMES), choices=list(PART_NAMES))
    parser.add_argument("--soft-threshold", type=float, default=0.20)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--soft-boundary-radius", type=int, default=0)
    parser.add_argument("--soft-boundary-min-value", type=float, default=0.25)
    parser.add_argument("--depth-margin", type=float, default=0.02)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    return parser.parse_args()


def analyze_scene(args: argparse.Namespace) -> dict:
    ensure_projected_arg_defaults(args)
    projected = analyze_projected_scene(args)
    result = summarize_multiview_rows(projected["per_view"], soft_threshold=float(args.soft_threshold))
    result["summary"]["part_label_bank"] = str(args.part_label_bank)
    result["summary"]["checkpoint"] = str(args.checkpoint)
    result["summary"]["asset_root"] = str(args.asset_root)
    result["summary"]["processed_views"] = int(projected["summary"].get("processed_views", 0))
    result["summary"]["boundary_radius"] = int(args.boundary_radius)
    result["summary"]["soft_boundary_radius"] = int(args.soft_boundary_radius)
    result["summary"]["soft_boundary_min_value"] = float(args.soft_boundary_min_value)
    return result


def main() -> int:
    args = parse_args()
    result = analyze_scene(args)
    write_reports(args.output_dir, result)
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'per_part.csv'}")
    print(f"wrote {args.output_dir / 'per_view.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
