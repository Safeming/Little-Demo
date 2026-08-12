import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def test_select_matched_strength_uses_voting_full_response():
    from tools.make_saga_a5_same_view_paper_visual import select_matched_strength

    rows = []
    for view, scale in (("v1", 1.0), ("v2", 2.0)):
        rows.append({"method": "voting", "part": "hair", "edit_strength": 1.0, "view": view, "target_delta_sum": 100.0 * scale})
        for strength, response in ((0.4, 35.0), (0.6, 55.0), (0.8, 75.0)):
            rows.append({"method": "saga", "part": "hair", "edit_strength": strength, "view": view, "target_delta_sum": response * scale})

    selected = select_matched_strength(rows, method="saga", part="hair", reference_method="voting", retention=0.6)

    assert selected["selected_strength"] == 0.6
    assert selected["reachable"] is True
    assert np.isclose(selected["actual_retention"], 0.55)


def test_select_matched_strength_deduplicates_overlapping_scan_rows():
    from tools.make_saga_a5_same_view_paper_visual import select_matched_strength

    rows = [
        {"method": "voting", "part": "hair", "edit_strength": 1.0, "view": "v1", "target_delta_sum": 100.0},
        {"method": "saga", "part": "hair", "edit_strength": 0.6, "view": "v1", "target_delta_sum": 60.0},
    ]

    selected = select_matched_strength(rows + rows, method="saga", part="hair", retention=0.6)

    assert selected["reference_target_response"] == 100.0
    assert selected["selected_target_response"] == 60.0


def test_select_matched_strength_rejects_unreachable_target():
    import pytest
    from tools.make_saga_a5_same_view_paper_visual import select_matched_strength

    rows = [
        {"method": "voting", "part": "hair", "edit_strength": 1.0, "view": "v1", "target_delta_sum": 100.0},
        {"method": "saga", "part": "hair", "edit_strength": 1.0, "view": "v1", "target_delta_sum": 50.0},
    ]

    with pytest.raises(ValueError, match="cannot reach"):
        select_matched_strength(rows, method="saga", part="hair", reference_method="voting", retention=0.6)


def test_rank_objective_views_uses_leakage_iou_then_name():
    from tools.make_saga_a5_same_view_paper_visual import rank_objective_views

    rows = []
    values = {
        "v1": (0.20, 0.80),
        "v2": (0.10, 0.60),
        "v3": (0.10, 0.75),
    }
    for view, (leakage, iou) in values.items():
        for method in ("saga", "a5"):
            for part in ("hair", "shoes"):
                rows.append(
                    {
                        "view": view,
                        "method": method,
                        "part": part,
                        "target_delta_sum": 10.0,
                        "outer_to_target_delta_ratio": leakage,
                        "edit_response_iou": iou,
                    }
                )

    ranked = rank_objective_views(rows, parts=("hair", "shoes"), methods=("saga", "a5"))

    assert [row["view"] for row in ranked] == ["v3", "v2", "v1"]
    assert ranked[0]["rank"] == 1


def test_rank_objective_views_excludes_incomplete_or_zero_response_views():
    from tools.make_saga_a5_same_view_paper_visual import rank_objective_views

    complete = [
        {"view": "good", "method": method, "part": part, "target_delta_sum": 1.0, "outer_to_target_delta_ratio": 0.2, "edit_response_iou": 0.5}
        for method in ("saga", "a5")
        for part in ("hair", "shoes")
    ]
    incomplete = complete[:-1]
    for row in incomplete:
        row = row.copy()
        row["view"] = "missing"
    zero = [dict(row, view="zero", target_delta_sum=0.0) for row in complete]

    ranked = rank_objective_views(complete + incomplete + zero, parts=("hair", "shoes"), methods=("saga", "a5"))

    assert [row["view"] for row in ranked] == ["good"]


def test_compose_b_layout_writes_three_by_five_png_and_pdf(tmp_path: Path):
    from tools.make_saga_a5_same_view_paper_visual import compose_b_layout

    rows = []
    for subject_index, subject in enumerate(("377", "386", "394")):
        for column, (method, part) in enumerate((("input", "input"), ("saga", "hair"), ("a5", "hair"), ("saga", "shoes"), ("a5", "shoes"))):
            image = np.full((60, 40, 3), 25 + subject_index * 40 + column * 5, dtype=np.uint8)
            path = tmp_path / f"{subject}_{method}_{part}.png"
            Image.fromarray(image).save(path)
            rows.append({"subject": subject, "view": "c22_f000420", "method": method, "part": part, "frame": str(path)})
    png = tmp_path / "sheet.png"
    pdf = tmp_path / "sheet.pdf"

    metadata = compose_b_layout(rows, png, pdf, fixed_view="c22_f000420")

    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
    assert metadata["subjects"] == ["377", "386", "394"]
    assert metadata["columns"] == ["Input", "Hair / SAGA", "Hair / Ours", "Shoes / SAGA", "Shoes / Ours"]
    with Image.open(png) as sheet:
        assert sheet.width > sheet.height


def test_compose_b_layout_accepts_one_objectively_selected_view_per_subject(tmp_path: Path):
    from tools.make_saga_a5_same_view_paper_visual import compose_b_layout

    selected_views = {"377": "c21_f000180", "386": "c22_f000420", "394": "c23_f000540"}
    rows = []
    for subject, view in selected_views.items():
        for column, (method, part) in enumerate((("input", "input"), ("saga", "hair"), ("a5", "hair"), ("saga", "shoes"), ("a5", "shoes"))):
            path = tmp_path / f"{subject}_{view}_{method}_{part}.png"
            Image.fromarray(np.full((60, 40, 3), 30 + column, dtype=np.uint8)).save(path)
            rows.append({"subject": subject, "view": view, "method": method, "part": part, "frame": str(path)})

    metadata = compose_b_layout(
        rows,
        tmp_path / "selected.png",
        tmp_path / "selected.pdf",
        view_by_subject=selected_views,
    )

    assert metadata["views"] == selected_views


def test_build_subject_specs_uses_joint_test_checkpoint_and_fixed_protocol(tmp_path: Path):
    from tools.make_saga_a5_same_view_paper_visual import TEST_VIEWS, build_subject_specs

    specs = build_subject_specs(tmp_path, tmp_path / "out")

    assert [spec["subject"] for spec in specs] == ["377", "386", "394"]
    assert list(TEST_VIEWS) == [
        "c21_f000180", "c21_f000420", "c21_f000540",
        "c22_f000180", "c22_f000420", "c22_f000540",
        "c23_f000180", "c23_f000420", "c23_f000540",
    ]
    assert all(str(spec["checkpoint"]).endswith("base_train_40k/ckpt40000.pth") for spec in specs)
    assert all(str(spec["saga_bank"]).endswith("train_30k/part_label_bank.npz") for spec in specs)


def test_build_render_command_freezes_scan_protocol(tmp_path: Path):
    from tools.make_saga_a5_same_view_paper_visual import TEST_VIEWS, build_render_command, build_subject_specs

    spec = build_subject_specs(tmp_path, tmp_path / "out")[0]
    command = build_render_command(
        spec,
        output_dir=tmp_path / "scan",
        python_bin=Path("/env/python"),
        methods=("voting", "saga", "a5"),
        strengths=(0.1, 0.2, 1.0),
        metrics_only=True,
    )
    joined = " ".join(map(str, command))

    assert command[:2] == ["/env/python", "tools/render_semantic_real_editing_paper_suite.py"]
    assert "--saga-threshold 0.5" in joined
    assert "--methods voting saga a5" in joined
    assert "--parts hair shoes" in joined
    assert "--tasks recolor" in joined
    assert "--edit-strengths 0.1 0.2 1.0" in joined
    assert all(view in command for view in TEST_VIEWS)
    assert "--metrics-only" in command


def test_build_dry_run_manifest_lists_three_ordered_subjects_and_stages(tmp_path: Path):
    from tools.make_saga_a5_same_view_paper_visual import build_dry_run_manifest, build_subject_specs

    specs = build_subject_specs(tmp_path, tmp_path / "out")
    manifest = build_dry_run_manifest(specs, python_bin=Path("/env/python"))

    assert [item["subject"] for item in manifest] == ["377", "386", "394"]
    assert all(item["stages"] == ["coarse_scan", "fine_scan", "final_render"] for item in manifest)
    assert all("--metrics-only" in item["coarse_command"] for item in manifest)
    assert all(item["fixed_view"] == "c22_f000420" for item in manifest)


def test_verify_output_rejects_missing_complete_marker(tmp_path: Path):
    import pytest
    from tools.make_saga_a5_same_view_paper_visual import verify_output

    (tmp_path / "summary.json").write_text(json.dumps({"subjects": ["377", "386", "394"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="COMPLETE"):
        verify_output(tmp_path)


def test_retention_within_tolerance_rejects_large_protocol_drift():
    import pytest
    from tools.make_saga_a5_same_view_paper_visual import validate_retention_value

    validate_retention_value(0.579, target=0.6, tolerance=0.03)
    with pytest.raises(ValueError, match="outside tolerance"):
        validate_retention_value(0.55, target=0.6, tolerance=0.03)


def test_run_script_invokes_frozen_visual_orchestrator():
    script = Path("tools/run_saga_a5_same_view_paper_visual.sh").read_text(encoding="utf-8")

    assert "make_saga_a5_same_view_paper_visual.py" in script
    assert "saga_a5_same_view_paper_visual_20260812" in script
    assert "--dry-run" in script
