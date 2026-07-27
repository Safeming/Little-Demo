import csv
import json
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_failure_report_contains_required_categories_and_assets(tmp_path: Path):
    from tools.build_semantic_paper_failure_report import build_report

    real_dir = tmp_path / "real"
    real_coverage = []
    for part, rate in (("face", 0.0), ("skin", 0.5), ("shoes", 0.7)):
        real_coverage.append(
            {
                "task": "recolor",
                "retention": 0.5,
                "subject": "377",
                "part": part,
                "reference_count": 10,
                "supported_count": int(rate * 10),
                "coverage_rate": rate,
                "qualified_cell": False,
            }
        )
    real_coverage.append(
        {
            "task": "recolor",
            "retention": 0.5,
            "subject": "386",
            "part": "face",
            "reference_count": 6,
            "supported_count": 0,
            "coverage_rate": 0.0,
            "qualified_cell": False,
        }
    )
    real_coverage[0]["reference_count"] = 0
    _write_csv(real_dir / "coverage_table.csv", real_coverage)
    (real_dir / "summary.json").write_text(
        json.dumps({"eligible_parts": {"recolor@0.5": ["upper"]}}), encoding="utf-8"
    )

    temporal_part = tmp_path / "temporal_part.csv"
    _write_csv(
        temporal_part,
        [
            {"retention": 0.25, "subject": "377", "part": "lower", "method": "voting", "pooled_outer_burden": 1.0},
            {"retention": 0.25, "subject": "377", "part": "lower", "method": "a5", "pooled_outer_burden": 1.5},
        ],
    )

    flicker_dir = tmp_path / "flicker"
    _write_csv(
        flicker_dir / "formal_table.csv",
        [
            {"retention": 0.5, "method": "voting", "mode": "adaptive", "outer_flicker_mean": 0.03, "boundary_flicker_mean": 0.08},
            {"retention": 0.5, "method": "a5", "mode": "adaptive", "outer_flicker_mean": 0.04, "boundary_flicker_mean": 0.09},
        ],
    )
    _write_csv(
        flicker_dir / "paired_statistics.csv",
        [
            {"retention": 0.5, "comparison": "a5-voting", "mode": "adaptive", "metric": "outer_flicker", "mean_delta": 0.01, "bootstrap_ci95_low": 0.005, "bootstrap_ci95_high": 0.015},
            {"retention": 0.5, "comparison": "a5-voting", "mode": "adaptive", "metric": "boundary_flicker", "mean_delta": 0.01, "bootstrap_ci95_low": 0.002, "bootstrap_ci95_high": 0.018},
        ],
    )

    output_dir = tmp_path / "failures"
    markdown = tmp_path / "failure_report.md"
    summary = build_report(
        real_dir=real_dir,
        temporal_part_csv=temporal_part,
        flicker_dir=flicker_dir,
        output_dir=output_dir,
        markdown_path=markdown,
        real_asset_dir=tmp_path / "paper_assets",
        temporal_video_root=tmp_path / "videos",
    )

    assert summary["failure_count"] == 6
    text = markdown.read_text(encoding="utf-8")
    for token in ("face", "skin", "shoes", "CoreView_377 lower", "temporal outer flicker", "temporal boundary flicker"):
        assert token in text
    assert "paper_assets" in text
    assert "videos" in text
    assert (output_dir / "failure_cases.csv").exists()
    assert (output_dir / "summary.json").exists()
    with (output_dir / "failure_cases.csv").open("r", encoding="utf-8", newline="") as handle:
        failures = list(csv.DictReader(handle))
    face = next(row for row in failures if row["label"] == "recolor face coverage")
    assert face["subject"] == "386"
