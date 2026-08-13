import json
import os
from pathlib import Path
import subprocess

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
    assert report["native_launch"]["attempted"] is True
    assert isinstance(report["native_launch"]["returncode"], int)
    assert report["native_launch"]["stderr"]
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


def test_train_parser_has_frozen_paper_defaults():
    from tools.train_sggs_released_code_canonical import build_parser

    args = build_parser().parse_args(["--input", "/tmp/input", "--prior", "/tmp/prior", "--output", "/tmp/out"])

    assert args.iterations == 30000
    assert args.hidden_dim == 64
    assert args.learning_rate == pytest.approx(1.0e-3)
    assert args.samples_per_class == 512
    assert args.topology_lambda == pytest.approx(0.1)
    assert args.topology_interval == 2
    assert args.seed == 0


def test_compact6_readout_and_predictions_are_normalized():
    from utils.sggs_released_code_canonical import Compact6Readout, compact6_predictions

    readout = Compact6Readout(input_dim=32, hidden_dim=8, class_count=6)
    features = torch.randn((7, 32), generator=torch.Generator().manual_seed(4))
    logits = readout(features)
    result = compact6_predictions(logits)

    assert logits.shape == (7, 6)
    np.testing.assert_allclose(result["semantic_probs"].sum(axis=1), 1.0, atol=1.0e-6)
    np.testing.assert_array_equal(result["part_label"], np.argmax(result["semantic_probs"], axis=1))
    assert np.all(result["confidence"] >= 0.0)
    assert np.all(result["semantic_margin"] >= 0.0)


def _make_train_inputs(tmp_path: Path):
    input_dir, avatar_root, body_models = _make_prior_export_inputs(tmp_path)
    views = input_dir / "views"
    views.mkdir()
    for name in json.loads((input_dir / "manifest.json").read_text())["views"]:
        torch.save({"xyz": torch.zeros((2, 3)), "labels": torch.zeros((2, 2), dtype=torch.int16)}, views / name)
    prior = tmp_path / "prior"
    prior.mkdir()
    torch.save(torch.zeros((2, 32)), prior / "topology_features.pt")
    torch.save(
        {"indices": torch.tensor([[1], [0]]), "distances": torch.ones((2, 1)), "weights": torch.ones((2, 1))},
        prior / "topology_knn.pt",
    )
    manifest = {
        "schema_version": 1,
        "subject": "CoreView_377",
        "point_count": 2,
        "view_count": 80,
        "feature_dim": 32,
        "source_frozen_views": str(input_dir.resolve()),
        "source_checkpoint": "/tmp/checkpoint.pth",
        "source_checkpoint_sha256": "a" * 64,
        "sggs_head": "27b9ed9c9e4c5663deb169247c2339ccafe1c254",
    }
    (prior / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return input_dir, prior


def test_validate_training_inputs_rejects_point_count_and_prior_shape_mismatch(tmp_path):
    from tools.train_sggs_released_code_canonical import validate_training_inputs

    input_dir, prior = _make_train_inputs(tmp_path)
    source, prior_manifest, features, graph, view_paths = validate_training_inputs(input_dir, prior)
    assert features.shape == (2, 32)
    assert graph["indices"].shape == (2, 1)
    assert len(view_paths) == 80

    torch.save(torch.zeros((3, 32)), prior / "topology_features.pt")
    with pytest.raises(ValueError, match="point count"):
        validate_training_inputs(input_dir, prior)


def test_find_sggs_resume_checkpoint_prefers_latest_unfinished(tmp_path):
    from tools.train_sggs_released_code_canonical import find_resume_checkpoint

    torch.save({"iteration": 100}, tmp_path / "checkpoint_000100.pt")
    torch.save({"iteration": 200}, tmp_path / "checkpoint_000200.pt")
    assert find_resume_checkpoint(tmp_path, iterations=300).name == "checkpoint_000200.pt"
    (tmp_path / "COMPLETE").write_text("complete\n")
    assert find_resume_checkpoint(tmp_path, iterations=300) is None


def test_sggs_queue_contract_freezes_release_inputs_order_and_resume():
    script = Path("tools/run_sggs_released_code_canonical_three_subject.sh").read_text(encoding="utf-8")

    assert "/opt/miniconda3/envs/gaussian_splatting/bin/python" in script
    assert "/remote-home/ming/SGGS" in script
    assert "27b9ed9c9e4c5663deb169247c2339ccafe1c254" in script
    assert "/remote-home/ming/dataSet" in script
    assert "/remote-home/ming/3dgs-avatar-release-main/body_models" in script
    assert 'SUBJECTS="${SUBJECTS:-377 386 394}"' in script
    assert 'ITERATIONS="${ITERATIONS:-30000}"' in script
    assert 'CANARY_ITERATIONS="${CANARY_ITERATIONS:-100}"' in script
    assert "queue_state.json" in script
    assert "estimated_completion_bjt" in script
    assert "trap on_exit EXIT" in script
    assert "--resume auto" in script


def test_estimate_sggs_queue_seconds_uses_steady_canary_rate_and_buffer():
    from utils.sggs_released_code_canonical import estimate_queue_seconds

    rows = [
        {"iteration": 1, "elapsed_seconds": 1.0},
        {"iteration": 20, "elapsed_seconds": 5.0},
        {"iteration": 100, "elapsed_seconds": 21.0},
    ]

    estimate = estimate_queue_seconds(
        rows,
        canary_iterations=100,
        formal_iterations=30000,
        subject_count=3,
        buffer_ratio=0.15,
    )

    assert estimate["steady_seconds_per_iteration"] == pytest.approx(0.2)
    assert estimate["estimated_seconds"] == pytest.approx(20700.0)


def test_sggs_queue_dry_run_does_not_write_completion_or_state(tmp_path):
    output = tmp_path / "dry-run"
    completed = subprocess.run(
        ["bash", "tools/run_sggs_released_code_canonical_three_subject.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DRY_RUN": "1", "OUTPUT_ROOT": str(output)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "train_sggs_released_code_canonical.py" in completed.stdout
    assert not (output / "COMPLETE").exists()
    assert not (output / "queue_state.json").exists()


def _validation_candidate(max_retention, leakage, *, miou=0.4, boundary=0.3):
    rows = []
    for retention in (0.4, 0.5, 0.6):
        if retention <= max_retention:
            rows.append(
                {
                    "baseline": "B4",
                    "retention": retention,
                    "actionable_leakage": leakage * retention,
                    "raw_leakage": leakage * retention * 2.0,
                    "actionable_leakage_ratio": leakage,
                }
            )
    return {
        "matched_retention": rows,
        "macro_miou": miou,
        "mean_boundary_f1": boundary,
        "report_dir": "/validation/only",
    }


def test_select_loso_threshold_prefers_feasible_low_leakage_then_quality_and_threshold():
    from utils.sggs_released_code_canonical import select_loso_threshold

    validation = {
        "377": {
            0.05: _validation_candidate(0.6, 0.30),
            0.10: _validation_candidate(0.6, 0.20, miou=0.45),
            0.15: _validation_candidate(0.6, 0.20, miou=0.40),
        },
        "386": {
            0.05: _validation_candidate(0.6, 0.20),
            0.10: _validation_candidate(0.6, 0.10, miou=0.45),
            0.15: _validation_candidate(0.6, 0.10, miou=0.40),
        },
        "394": {
            0.05: _validation_candidate(0.6, 0.50),
            0.10: _validation_candidate(0.6, 0.50),
            0.15: _validation_candidate(0.6, 0.50),
        },
    }

    selected = select_loso_threshold(validation, held_out_subject="394", target_retention=0.6)

    assert selected["soft_threshold"] == pytest.approx(0.10)
    assert selected["validation_target_feasible"] is True
    assert selected["validation_selection_retention"] == pytest.approx(0.6)
    assert selected["donor_subjects"] == ["377", "386"]


def test_select_loso_threshold_falls_back_to_largest_common_reachable_retention():
    from utils.sggs_released_code_canonical import select_loso_threshold

    validation = {
        "377": {0.1: _validation_candidate(0.5, 0.4), 0.2: _validation_candidate(0.4, 0.1)},
        "386": {0.1: _validation_candidate(0.5, 0.2), 0.2: _validation_candidate(0.6, 0.1)},
        "394": {0.1: _validation_candidate(0.6, 0.3), 0.2: _validation_candidate(0.6, 0.3)},
    }

    selected = select_loso_threshold(validation, held_out_subject="394", target_retention=0.6)

    assert selected["soft_threshold"] == pytest.approx(0.1)
    assert selected["validation_target_feasible"] is False
    assert selected["validation_selection_retention"] == pytest.approx(0.5)


def test_build_sggs_comparison_row_marks_missing_60_percent_without_interpolation(tmp_path):
    from tools.evaluate_sggs_released_code_canonical import build_method_row

    report = tmp_path / "report"
    report.mkdir()
    (report / "matched_retention.csv").write_text(
        "baseline,retention,actionable_leakage,raw_leakage,actionable_leakage_ratio,edit_strength\n"
        "B4,0.4,0.2,0.4,0.5,0.9\n",
        encoding="utf-8",
    )
    (report / "baseline_summary.csv").write_text(
        "baseline,macro_miou,mean_boundary_f1\nB4,0.3,0.2\n", encoding="utf-8"
    )

    row = build_method_row("SG-GS", "377", report, target_retention=0.6)

    assert row["retention_0p6_feasible"] is False
    assert row["actionable_leakage_at_0p6"] == ""
    assert row["max_reachable_retention"] == pytest.approx(0.4)
    assert row["actionable_leakage_at_0p4"] == pytest.approx(0.2)


def test_sggs_evaluator_uses_avatar_dataset_root_for_scene_loading():
    source = Path("tools/evaluate_sggs_released_code_canonical.py").read_text(encoding="utf-8")

    assert '"--dataset-root", str(paper_root / "data/ZJUMoCap")' in source
    assert '"--dataset-root", "/remote-home/ming/dataSet"' not in source


def test_materialize_sggs_evaluation_config_adds_only_checkpoint_compatibility_keys(tmp_path):
    from omegaconf import OmegaConf
    from tools.evaluate_sggs_released_code_canonical import materialize_evaluation_config

    source = tmp_path / "source.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {"name": "zjumocap", "root_dir": "/original"},
                "resume": {
                    "allow_partial_converter_load": True,
                    "partial_converter_missing_keys_allow_patterns": ["texture.structured_trunk_"],
                },
            }
        ),
        source,
    )
    output = tmp_path / "compat.yaml"

    materialize_evaluation_config(source, output)

    original = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    result = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
    assert result["dataset"] == original["dataset"]
    assert result["resume"]["allow_partial_converter_load"] is True
    assert result["resume"]["partial_converter_missing_keys_allow_patterns"] == [
        "texture.structured_trunk_",
        "camera_geometry.rot_raw",
        "camera_geometry.trans_raw",
    ]


def test_sggs_render_preset_is_subject_validated():
    from tools.evaluate_sggs_released_code_canonical import render_preset_for_subject

    assert render_preset_for_subject("377") == "v338_temporal_selector_grow_only_guard"
    assert render_preset_for_subject("386") == "none"
    assert render_preset_for_subject("394") == "none"


def test_verify_frozen_external_bank_rejects_post_freeze_mutation(tmp_path):
    from tools.evaluate_sggs_released_code_canonical import _sha256, verify_frozen_external_bank

    bank = tmp_path / "bank.npz"
    bank.write_bytes(b"frozen-bank")
    frozen = {"external_bank_fingerprint": _sha256(bank)}
    verify_frozen_external_bank(frozen, bank)

    bank.write_bytes(b"mutated-bank")
    with pytest.raises(ValueError, match="external SG-GS bank fingerprint mismatch"):
        verify_frozen_external_bank(frozen, bank)


def test_attach_training_efficiency_reads_sggs_summary(tmp_path):
    from tools.evaluate_sggs_released_code_canonical import attach_training_efficiency

    train = tmp_path / "CoreView_377" / "train_30k"
    train.mkdir(parents=True)
    (train / "summary.json").write_text(
        json.dumps({"elapsed_seconds": 12.5, "peak_memory_bytes": 123456}), encoding="utf-8"
    )
    rows = [{"method": "SG-GS", "subject": "377"}, {"method": "A5", "subject": "377"}]

    result = attach_training_efficiency(rows, tmp_path)

    assert result[0]["training_seconds"] == pytest.approx(12.5)
    assert result[0]["peak_memory_bytes"] == 123456
    assert result[1]["training_seconds"] == ""
    assert result[1]["peak_memory_bytes"] == ""
