import csv
import json
from pathlib import Path

import pytest
from PIL import Image


def _rows():
    rows = []
    for subject in ("377", "386", "394"):
        for camera in (21, 22, 23):
            for frame in (180, 181, 182):
                for method, leakage, iou in (
                    ("a5", 0.10, 0.80),
                    ("saga", 0.20, 0.70),
                    ("gaussian_grouping", 0.30, 0.65),
                    ("sggs", 0.40, 0.60),
                ):
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "camera": camera,
                            "frame": frame,
                            "part": "hair",
                            "target_activation": 6.0,
                            "reference_target_activation": 10.0,
                            "outer_activation": leakage * 10.0,
                            "actionable_outer_activation": leakage * 10.0,
                            "iou": iou,
                            "boundary_f1": iou - 0.1,
                            "target_empty": False,
                            "retention": 0.4 if method == "gaussian_grouping" and subject == "377" else 0.6,
                            "target_retention_feasible": not (
                                method == "gaussian_grouping" and subject == "377"
                            ),
                        }
                    )
    return rows


def test_significance_uses_common_subjects_and_is_reproducible():
    from tools.summarize_four_method_paper_evidence import summarize_significance

    first = summarize_significance(_rows(), iterations=200, seed=11)
    second = summarize_significance(_rows(), iterations=200, seed=11)

    assert first == second
    by_method = {
        row["comparison_method"]: row
        for row in first["comparisons"]
        if row["metric"] == "actionable_leakage"
    }
    assert by_method["saga"]["subject_count"] == 3
    assert by_method["sggs"]["subject_count"] == 3
    assert by_method["gaussian_grouping"]["subject_count"] == 2
    assert by_method["saga"]["absolute_difference"] == pytest.approx(-0.10)
    assert by_method["saga"]["relative_reduction"] == pytest.approx(0.5)
    assert "ci_low" in by_method["saga"]
    assert "p_value_raw" in by_method["saga"]
    assert "p_value_holm" in by_method["saga"]
    assert len(first["per_subject"]) > 0


def test_temporal_summary_reports_nine_windows_per_subject_method():
    from tools.summarize_four_method_paper_evidence import summarize_temporal

    rows = []
    for method in ("a5", "saga"):
        for camera in (21, 22, 23):
            for anchor in (180, 420, 540):
                for frame in range(anchor - 10, anchor + 11):
                    rows.append(
                        {
                            "subject": "377",
                            "method": method,
                            "camera": camera,
                            "frame": frame,
                            "part": "hair",
                            "target_activation": 6.0,
                            "reference_target_activation": 10.0,
                            "outer_activation": frame / 10000.0,
                            "actionable_outer_activation": frame / 20000.0,
                            "iou": 0.8,
                            "boundary_f1": 0.7,
                            "target_empty": False,
                        }
                    )

    result = summarize_temporal(rows)

    assert len(result["per_frame"]) == 378
    assert len(result["windows"]) == 18
    assert all(row["frame_count"] == 21 for row in result["windows"])
    assert all(row["window_count"] == 9 for row in result["methods"])
    assert all(row["camera_frame_count"] == 189 for row in result["methods"])


def test_compose_part_layout_uses_confirmed_three_by_five_order(tmp_path):
    from tools.summarize_four_method_paper_evidence import compose_part_layout

    frame_paths = {}
    for subject in ("377", "386", "394"):
        for method in ("input", "saga", "gaussian_grouping", "sggs", "a5"):
            path = tmp_path / f"{subject}_{method}.png"
            Image.new("RGB", (64, 96), (int(subject) % 255, len(method) * 10, 80)).save(path)
            frame_paths[(subject, method)] = path

    result = compose_part_layout(
        subjects=("377", "386", "394"),
        methods=("input", "saga", "gaussian_grouping", "sggs", "a5"),
        part="hair",
        frame_paths=frame_paths,
        output_dir=tmp_path,
    )

    assert result["columns"] == ["Input", "SAGA", "Gaussian Grouping", "SG-GS", "Ours"]
    assert result["subjects"] == ["377", "386", "394"]
    assert result["gg_377_label"] == "GG\N{DAGGER}"
    assert Path(result["png"]).stat().st_size > 0
    assert Path(result["pdf"]).stat().st_size > 0


def test_significance_cli_writes_csv_json_markdown_and_latex(tmp_path):
    from tools.summarize_four_method_paper_evidence import main

    input_path = tmp_path / "rows.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_rows()[0]))
        writer.writeheader()
        writer.writerows(_rows())

    output = tmp_path / "out"
    assert main(
        [
            "significance",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--iterations",
            "100",
            "--seed",
            "5",
        ]
    ) == 0

    for filename in (
        "significance.json",
        "comparisons.csv",
        "per_subject.csv",
        "significance.md",
        "significance.tex",
    ):
        assert (output / filename).stat().st_size > 0
    payload = json.loads((output / "significance.json").read_text(encoding="utf-8"))
    assert payload["bootstrap_iterations"] == 100
    assert payload["bootstrap_seed"] == 5
