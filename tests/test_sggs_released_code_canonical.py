import json
from pathlib import Path

import pytest


def _make_release_tree(root: Path) -> Path:
    repo = root / "SGGS"
    repo.mkdir()
    (repo / "README.md").write_text("# SG-GS\n", encoding="utf-8")
    (repo / "train.py").write_text(
        "\n".join(
            [
                "pcd_path = './body_models/smpl/neutral/smpl_semantic.ply'",
                "gaussians.frozen_labels = labels.cuda()",
                "# lambda_semantic = C(iteration, 0.01)",
                "# loss_consistency = neighborhood_consistency_loss(xyz, objects)",
                "# loss += semantic_loss * lambda_semantic",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "gaussian_renderer").mkdir()
    (repo / "gaussian_renderer" / "__init__.py").write_text(
        "import diff_gaussian_rasterization_obj as dgro\n", encoding="utf-8"
    )
    return repo


def test_scan_release_tree_reports_missing_release_files(tmp_path):
    from utils.sggs_released_code_canonical import scan_release_tree

    repo = _make_release_tree(tmp_path)
    result = scan_release_tree(repo)

    assert result["present"]["README.md"] is True
    assert result["present"]["environment.yml"] is False
    assert result["present"][".gitmodules"] is False
    assert result["present"]["license"] is False
    assert "diff_gaussian_rasterization_obj" in result["declared_missing_local_modules"]


def test_scan_semantic_code_distinguishes_active_initialization_from_commented_losses(tmp_path):
    from utils.sggs_released_code_canonical import scan_semantic_code

    repo = _make_release_tree(tmp_path)
    result = scan_semantic_code(repo / "train.py")

    assert result["active_smpl_label_initialization"] is True
    assert result["active_semantic_loss"] is False
    assert result["commented_semantic_loss"] is True
    assert result["active_neighborhood_consistency"] is False
    assert result["commented_neighborhood_consistency"] is True
    assert result["evidence"]["smpl_label_initialization"] == [2]


def test_probe_modules_records_success_and_failure(tmp_path):
    from utils.sggs_released_code_canonical import probe_modules

    result = probe_modules(
        Path("/opt/miniconda3/envs/ictrl/bin/python"), ["json", "module_that_does_not_exist_sggs"]
    )

    assert result["json"]["available"] is True
    assert result["module_that_does_not_exist_sggs"]["available"] is False
    assert "ModuleNotFoundError" in result["module_that_does_not_exist_sggs"]["error"]


def test_audit_cli_writes_required_schema_even_when_native_launch_is_blocked(tmp_path):
    from tools.audit_sggs_released_code import main

    repo = _make_release_tree(tmp_path)
    output = tmp_path / "audit.json"
    exit_code = main(
        [
            "--repo",
            str(repo),
            "--dataset",
            str(tmp_path / "dataset"),
            "--body-models",
            str(tmp_path / "body_models"),
            "--python",
            "/opt/miniconda3/envs/ictrl/bin/python",
            "--output",
            str(output),
            "--skip-network",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) >= {
        "official_identity",
        "release_completeness",
        "semantic_code_state",
        "dependency_probe",
        "native_launch",
        "record_sha256",
    }
    assert report["native_launch"]["status"] == "blocked"
    assert report["native_launch"]["first_blocker"]
    assert len(report["record_sha256"]) == 64
