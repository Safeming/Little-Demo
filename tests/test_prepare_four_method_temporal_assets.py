import json
from pathlib import Path

import pytest


def _segment(root: Path, name: str, camera: int, frame: int):
    root.mkdir(parents=True)
    mask = root / "masks" / "hair" / f"render_{name}.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"png")
    records = [
        {
            "image_name": name,
            "cam_id": camera,
            "frame_id": frame,
            "mask_files": {"hair": str(mask.relative_to(root))},
        }
    ]
    (root / "view_records.json").write_text(json.dumps(records), encoding="utf-8")


def test_build_record_names_has_189_unique_temporal_views():
    from tools.prepare_four_method_temporal_assets import build_record_names
    from utils.four_method_paper_evidence import build_temporal_windows

    names = build_record_names(build_temporal_windows())

    assert len(names) == 189
    assert len(set(names)) == 189
    assert names[0] == "c21_f000170"
    assert names[-1] == "c23_f000550"


def test_validate_temporal_source_coverage_reports_missing_parser_file(tmp_path):
    from tools.prepare_four_method_temporal_assets import validate_temporal_source_coverage

    parser_root = tmp_path / "parser"
    directory = parser_root / "CoreView_377" / "mask_cihp" / "Camera_B21"
    directory.mkdir(parents=True)
    (directory / "000170.png").write_bytes(b"png")

    with pytest.raises(ValueError, match="missing parser masks"):
        validate_temporal_source_coverage(
            parser_root=parser_root,
            subject="377",
            cameras=(21,),
            frames=(170, 171),
        )


def test_merge_asset_roots_links_referenced_files_and_rejects_duplicate_keys(tmp_path):
    from tools.prepare_four_method_temporal_assets import merge_asset_roots, verify_asset_root

    first = tmp_path / "first"
    second = tmp_path / "second"
    _segment(first, "c21_f000170", 21, 170)
    _segment(second, "c21_f000171", 21, 171)
    output = tmp_path / "merged"

    manifest = merge_asset_roots(
        segment_roots=[first, second],
        output_root=output,
        expected_names=["c21_f000170", "c21_f000171"],
    )

    assert manifest["record_count"] == 2
    assert verify_asset_root(output, expected_names=["c21_f000170", "c21_f000171"])[
        "referenced_file_count"
    ] == 2
    duplicate = tmp_path / "duplicate"
    _segment(duplicate, "c21_f000170", 21, 170)
    with pytest.raises(ValueError, match="duplicate record key"):
        merge_asset_roots(
            segment_roots=[first, duplicate],
            output_root=tmp_path / "bad",
            expected_names=["c21_f000170"],
        )
