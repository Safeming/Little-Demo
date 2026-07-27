#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_COVERAGE_FAILURE_PARTS = ("face", "skin", "shoes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the formal semantic-editing failure report.")
    parser.add_argument("--real-dir", required=True, type=Path)
    parser.add_argument("--temporal-part-csv", required=True, type=Path)
    parser.add_argument("--flicker-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown-path", required=True, type=Path)
    parser.add_argument("--real-asset-dir", required=True, type=Path)
    parser.add_argument("--temporal-video-root", required=True, type=Path)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("failure report cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite failure metric: {value}")
    return result


def _coverage_failures(real_dir: Path) -> list[dict]:
    coverage = _read_csv(real_dir / "coverage_table.csv")
    summary = json.loads((real_dir / "summary.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in coverage:
        part = str(row["part"])
        if part in REQUIRED_COVERAGE_FAILURE_PARTS:
            grouped[(str(row["task"]), float(row["retention"]), part)].append(row)
    outputs = []
    for (task, retention, part), members in sorted(grouped.items()):
        eligible = summary["eligible_parts"].get(f"{task}@{retention}", [])
        if part in eligible:
            continue
        supported_references = [row for row in members if int(float(row["reference_count"])) > 0]
        candidates = supported_references or members
        worst = min(
            candidates,
            key=lambda row: (_number(row["coverage_rate"]), int(float(row["reference_count"]))),
        )
        rate = _number(worst["coverage_rate"])
        outputs.append(
            {
                "category": "coverage_failure",
                "label": f"{task} {part} coverage",
                "task": task,
                "subject": str(worst["subject"]),
                "part": part,
                "retention": retention,
                "metric": "coverage_rate",
                "voting_value": 1.0,
                "a5_value": rate,
                "delta": rate - 1.0,
                "direction": "worse",
                "interpretation": "A5 fails the every-subject 80% retention coverage rule.",
                "artifact_path": str(real_dir / "coverage_table.csv"),
            }
        )
    return outputs


def _lower_failure(temporal_part_csv: Path) -> list[dict]:
    rows = _read_csv(temporal_part_csv)
    lookup = {
        (float(row["retention"]), str(row["subject"]), str(row["part"]), str(row["method"])): row
        for row in rows
    }
    key = (0.25, "377", "lower")
    voting = lookup.get((*key, "voting"))
    a5 = lookup.get((*key, "a5"))
    if voting is None or a5 is None:
        return []
    voting_value = _number(voting["pooled_outer_burden"])
    a5_value = _number(a5["pooled_outer_burden"])
    return [
        {
            "category": "spatial_failure",
            "label": "CoreView_377 lower",
            "task": "temporal_recolor",
            "subject": "377",
            "part": "lower",
            "retention": 0.25,
            "metric": "pooled_outer_burden",
            "voting_value": voting_value,
            "a5_value": a5_value,
            "delta": a5_value - voting_value,
            "direction": "worse" if a5_value > voting_value else "better",
            "interpretation": "CoreView_377 lower dominates the retention-0.25 pooled spatial failure.",
            "artifact_path": str(temporal_part_csv),
        }
    ]


def _flicker_failures(flicker_dir: Path) -> list[dict]:
    formal = _read_csv(flicker_dir / "formal_table.csv")
    paired = _read_csv(flicker_dir / "paired_statistics.csv")
    lookup = {
        (float(row["retention"]), str(row["method"]), str(row["mode"])): row for row in formal
    }
    outputs = []
    for row in paired:
        if str(row["comparison"]) != "a5-voting" or str(row["mode"]) != "adaptive":
            continue
        delta = _number(row["mean_delta"])
        if delta <= 0.0:
            continue
        retention = float(row["retention"])
        metric = str(row["metric"])
        voting = _number(lookup[(retention, "voting", "adaptive")][f"{metric}_mean"])
        a5 = _number(lookup[(retention, "a5", "adaptive")][f"{metric}_mean"])
        readable = "outer" if metric == "outer_flicker" else "boundary"
        outputs.append(
            {
                "category": "temporal_failure",
                "label": f"temporal {readable} flicker",
                "task": "temporal_recolor",
                "subject": "ALL",
                "part": "formal_eligible",
                "retention": retention,
                "metric": metric,
                "voting_value": voting,
                "a5_value": a5,
                "delta": delta,
                "direction": "worse",
                "interpretation": (
                    "Adaptive matched-retention A5 flicker exceeds Voting; fixed-mode comparison "
                    "is required to attribute the regression."
                ),
                "artifact_path": str(flicker_dir / "paired_statistics.csv"),
            }
        )
    return outputs


def _markdown(rows: list[dict], real_asset_dir: Path, temporal_video_root: Path) -> str:
    lines = [
        "# 正式论文失败案例与时序诊断",
        "",
        "日期：2026-07-27",
        "",
        "本报告自动汇总覆盖率不足、空间 burden 退化和 temporal flicker 回归。失败项不参与方法调参，也不从论文结果中隐藏。",
        "",
        "## 正式失败表",
        "",
        "| Failure | Task | Subject | Part | Retention | Metric | Voting | A5 | Delta |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {task} | {subject} | {part} | {retention:.2f} | {metric} | "
            "{voting_value:.6f} | {a5_value:.6f} | {delta:+.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- face、skin、shoes 进入 coverage failure 表，不能宣称六部件完整覆盖。",
            "- CoreView_377 lower 是明确空间失败，必须与 hair/upper 正结果同时展示。",
            "- temporal outer flicker 和 temporal boundary flicker 若为正 delta，不能宣称 A5 时序更稳定。",
            "- fixed 与 adaptive 的差异用于判断逐帧 strength compensation 是否造成额外波动。",
            "",
            "## 可复用定性资产",
            "",
            f"真实编辑论文图目录：`{real_asset_dir}`",
            "",
            f"连续动画视频目录：`{temporal_video_root}`",
            "",
            "建议主文放置一个覆盖失败小部件、CoreView_377 lower 和一个 temporal flicker 片段；其余放 supplementary。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    real_dir: Path,
    temporal_part_csv: Path,
    flicker_dir: Path,
    output_dir: Path,
    markdown_path: Path,
    real_asset_dir: Path,
    temporal_video_root: Path,
) -> dict:
    real_dir = Path(real_dir).resolve()
    temporal_part_csv = Path(temporal_part_csv).resolve()
    flicker_dir = Path(flicker_dir).resolve()
    output_dir = Path(output_dir).resolve()
    markdown_path = Path(markdown_path).resolve()
    rows = [
        *_coverage_failures(real_dir),
        *_lower_failure(temporal_part_csv),
        *_flicker_failures(flicker_dir),
    ]
    _write_csv(output_dir / "failure_cases.csv", rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _markdown(rows, Path(real_asset_dir).resolve(), Path(temporal_video_root).resolve()),
        encoding="utf-8",
    )
    summary = {
        "failure_count": len(rows),
        "coverage_failure_count": sum(row["category"] == "coverage_failure" for row in rows),
        "spatial_failure_count": sum(row["category"] == "spatial_failure" for row in rows),
        "temporal_failure_count": sum(row["category"] == "temporal_failure" for row in rows),
        "markdown_path": str(markdown_path),
        "real_asset_dir": str(Path(real_asset_dir).resolve()),
        "temporal_video_root": str(Path(temporal_video_root).resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            build_report(
                real_dir=args.real_dir,
                temporal_part_csv=args.temporal_part_csv,
                flicker_dir=args.flicker_dir,
                output_dir=args.output_dir,
                markdown_path=args.markdown_path,
                real_asset_dir=args.real_asset_dir,
                temporal_video_root=args.temporal_video_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
