import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _make_release_tree(root: Path) -> Path:
    repo = root / "SGGS"
    repo.mkdir()
    (repo / "README.md").write_text("# SG-GS\n", encoding="utf-8")
    (repo / "train.py").write_text(
        "\n".join(
            [
                "from utils.loss_utils import neighborhood_consistency_loss",
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
    assert result["evidence"]["smpl_label_initialization"] == [3]


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


def test_native_smpl_labels_follow_released_joint_to_part_mapping():
    from utils.sggs_released_code_canonical import native_smpl_labels

    weights = torch.zeros((5, 24), dtype=torch.float32)
    joints = torch.tensor([0, 1, 12, 13, 15])
    weights[torch.arange(5), joints] = 1.0

    labels = native_smpl_labels(weights)

    torch.testing.assert_close(labels, torch.tensor([4, 1, 3, 2, 3]))


def test_interpolate_smpl_prior_returns_normalized_topology_and_native_probabilities():
    from utils.sggs_released_code_canonical import interpolate_smpl_prior

    smpl_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    skinning = torch.zeros((4, 24), dtype=torch.float32)
    skinning[:, 0] = torch.tensor([1.0, 0.8, 0.2, 0.0])
    skinning[:, 1] = 1.0 - skinning[:, 0]
    gaussian_xyz = torch.tensor([[0.05, 0.05, 0.0], [0.95, 0.95, 0.0]])

    result = interpolate_smpl_prior(gaussian_xyz, smpl_xyz, skinning, k=2)

    assert result["skinning_weights"].shape == (2, 24)
    assert result["native_semantic_probs"].shape == (2, 5)
    assert result["knn_indices"].shape == (2, 2)
    torch.testing.assert_close(result["skinning_weights"].sum(dim=1), torch.ones(2))
    torch.testing.assert_close(result["native_semantic_probs"].sum(dim=1), torch.ones(2))
    assert torch.isfinite(result["knn_distances"]).all()


def test_build_topology_geometric_features_has_documented_32_channel_layout():
    from utils.sggs_released_code_canonical import build_topology_geometric_features

    skinning = torch.zeros((2, 24), dtype=torch.float32)
    skinning[:, 0] = 1.0
    semantics = torch.zeros((2, 5), dtype=torch.float32)
    semantics[:, 4] = 1.0
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    smpl_xyz = torch.tensor([[-2.0, -1.0, -1.0], [2.0, 1.0, 1.0]])

    features = build_topology_geometric_features(xyz, smpl_xyz, skinning, semantics)

    assert features.shape == (2, 32)
    torch.testing.assert_close(features[:, :24], skinning)
    torch.testing.assert_close(features[:, 24:29], semantics)
    assert torch.all(features[:, 29:] <= 1.0)
    assert torch.all(features[:, 29:] >= -1.0)
    assert torch.isfinite(features).all()


def test_topology_consistency_is_lower_for_equal_neighbor_predictions():
    from utils.sggs_released_code_canonical import topology_consistency_loss

    neighbors = torch.tensor([[0, 1], [1, 0]])
    weights = torch.full((2, 2), 0.5)
    equal = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    different = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    equal_loss = topology_consistency_loss(equal, neighbors, weights)
    different_loss = topology_consistency_loss(different, neighbors, weights)

    assert equal_loss < different_loss
    assert equal_loss >= 0.0


def test_build_topology_knn_excludes_self_and_normalizes_weights():
    from utils.sggs_released_code_canonical import build_topology_knn

    features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0]], dtype=torch.float32
    )

    result = build_topology_knn(features, k=2)

    assert result["indices"].shape == (4, 2)
    assert not torch.any(result["indices"] == torch.arange(4)[:, None])
    torch.testing.assert_close(result["weights"].sum(dim=1), torch.ones(4))
    assert torch.isfinite(result["distances"]).all()


def _make_prior_export_inputs(tmp_path: Path):
    input_dir = tmp_path / "frozen_views"
    input_dir.mkdir()
    canonical_xyz = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    torch.save(canonical_xyz, input_dir / "canonical_xyz.pt")
    manifest = {
        "schema_version": 1,
        "subject": "CoreView_377",
        "point_count": 2,
        "view_count": 80,
        "views": [f"view_{i:03d}.pt" for i in range(80)],
        "part_names": ["hair", "face", "upper", "lower", "shoes", "skin"],
        "source_checkpoint": "/tmp/checkpoint.pth",
        "source_checkpoint_sha256": "a" * 64,
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    avatar_root = tmp_path / "avatar"
    models = avatar_root / "CoreView_377" / "models"
    models.mkdir(parents=True)
    minimal_shape = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    np.savez(models / "000000.npz", minimal_shape=minimal_shape)

    body_models = tmp_path / "body_models"
    misc = body_models / "misc"
    misc.mkdir(parents=True)
    skinning = np.zeros((4, 24), dtype=np.float32)
    skinning[:, 0] = 1.0
    regressor = np.zeros((24, 4), dtype=np.float32)
    np.savez(misc / "skinning_weights_all.npz", neutral=skinning)
    np.savez(misc / "J_regressors.npz", neutral=regressor)
    return input_dir, avatar_root, body_models


def test_export_prior_cli_writes_32d_features_and_provenance(tmp_path):
    from tools.export_sggs_canonical_prior import main

    input_dir, avatar_root, body_models = _make_prior_export_inputs(tmp_path)
    output = tmp_path / "prior"
    exit_code = main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--avatar-data-root",
            str(avatar_root),
            "--body-models",
            str(body_models),
            "--sggs-repo",
            "/remote-home/ming/SGGS",
            "--knn-k",
            "1",
        ]
    )

    assert exit_code == 0
    features = torch.load(output / "topology_features.pt", map_location="cpu")
    graph = torch.load(output / "topology_knn.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert features.shape == (2, 32)
    assert graph["indices"].shape == (2, 1)
    assert manifest["point_count"] == 2
    assert manifest["feature_layout"] == {"skinning": [0, 24], "native_semantics": [24, 29], "xyz": [29, 32]}
    assert manifest["sggs_head"] == "27b9ed9c9e4c5663deb169247c2339ccafe1c254"
    assert manifest["trainable"] == ["compact6_readout_mlp"]
