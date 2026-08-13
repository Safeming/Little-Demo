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


def test_derive_aligned_frozen_config_only_rebinds_checkpoint(tmp_path):
    from tools.prepare_four_method_temporal_assets import derive_aligned_frozen_config
    from utils.semantic_eval_protocol import file_fingerprint

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "checkpoint_fingerprint": "old",
                "bank_fingerprint": "bank",
                "protocol_fingerprint": "protocol",
                "selected": {"soft_threshold": 0.15, "boundary_radius": 6},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_bytes(b"checkpoint")

    result = derive_aligned_frozen_config(
        source_config=source,
        checkpoint=checkpoint,
        output=tmp_path / "aligned.json",
    )

    assert result["checkpoint_fingerprint"] == file_fingerprint(checkpoint)
    assert result["bank_fingerprint"] == "bank"
    assert result["selected"] == {"soft_threshold": 0.15, "boundary_radius": 6}
    assert result["alignment"]["source_checkpoint_fingerprint"] == "old"


def test_write_operating_point_selects_unique_method_retention(tmp_path):
    from tools.prepare_four_method_temporal_assets import write_operating_point

    curve = tmp_path / "matched.csv"
    curve.write_text(
        "baseline,retention,threshold,edit_strength\n"
        "B1,0.6,0.5,0.6\n"
        "A5,0.6,0.15,0.72\n",
        encoding="utf-8",
    )
    result = write_operating_point(
        curve_path=curve,
        method="a5",
        subject="386",
        output=tmp_path / "point.json",
    )

    assert result["baseline"] == "A5"
    assert result["retention"] == 0.6
    assert result["edit_strength"] == 0.72
    assert result["reference_baseline"] == "B1"


def test_write_frozen_protocol_records_inputs_and_fixed_statistics(tmp_path):
    from tools.prepare_four_method_temporal_assets import write_frozen_protocol

    root = tmp_path / "evidence"
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    (protocol / "temporal_record_list.json").write_text(
        json.dumps(["c21_f000170", "c23_f000550"]), encoding="utf-8"
    )
    for subject in ("377", "386", "394"):
        (protocol / f"CoreView_{subject}_a5_shared40k_frozen.json").write_text(
            json.dumps({"subject": subject}), encoding="utf-8"
        )
        for method in ("saga", "gaussian_grouping", "sggs", "a5"):
            (protocol / f"CoreView_{subject}_{method}_operating_point.json").write_text(
                json.dumps(
                    {
                        "subject": subject,
                        "method": method,
                        "retention": (
                            0.4 if subject == "377" and method == "gaussian_grouping" else 0.6
                        ),
                        "target_retention_feasible": not (
                            subject == "377" and method == "gaussian_grouping"
                        ),
                    }
                ),
                encoding="utf-8",
            )
        asset = root / "temporal_assets" / f"CoreView_{subject}" / "merged"
        asset.mkdir(parents=True)
        (asset / "asset_manifest.json").write_text(
            json.dumps({"record_count": 189}), encoding="utf-8"
        )

    output = protocol / "frozen_protocol.json"
    result = write_frozen_protocol(output_root=root, output=output)

    assert result["subjects"] == ["377", "386", "394"]
    assert result["methods"] == ["saga", "gaussian_grouping", "sggs", "a5"]
    assert result["statistics"] == {"bootstrap_iterations": 20000, "seed": 20260813}
    assert len(result["inputs"]) == 19
    assert all(len(row["sha256"]) == 64 for row in result["inputs"])
    assert json.loads(output.read_text(encoding="utf-8")) == result
