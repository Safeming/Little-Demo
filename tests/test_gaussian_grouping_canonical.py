import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch


def test_balanced_pixel_indices_ignores_background_and_caps_each_class():
    from utils.gaussian_grouping_canonical import balanced_pixel_indices

    labels = torch.tensor([[-1, 0, 0, 0], [1, 1, 2, 2]])
    first = balanced_pixel_indices(labels, samples_per_class=2, seed=7)
    second = balanced_pixel_indices(labels, samples_per_class=2, seed=7)

    assert torch.equal(first, second)
    sampled = labels.reshape(-1)[first]
    assert -1 not in sampled.tolist()
    assert {label: sampled.tolist().count(label) for label in (0, 1, 2)} == {
        0: 2,
        1: 2,
        2: 2,
    }


def test_grouping_3d_consistency_prefers_matching_neighbor_probabilities():
    from utils.gaussian_grouping_canonical import grouping_3d_consistency_loss

    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]])
    consistent = torch.tensor([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]])
    inconsistent = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])

    good = grouping_3d_consistency_loss(xyz, consistent, k=2, sample_size=4, seed=3)
    bad = grouping_3d_consistency_loss(xyz, inconsistent, k=2, sample_size=4, seed=3)

    assert torch.isfinite(good)
    assert good < bad


def test_identity_predictions_exports_normalized_probabilities_labels_and_margin():
    from utils.gaussian_grouping_canonical import identity_predictions

    encodings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    bias = torch.tensor([0.0, 0.0, -2.0])

    result = identity_predictions(encodings, weight, bias)

    np.testing.assert_allclose(result["semantic_probs"].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(result["part_label"], np.array([0, 1, 0], dtype=np.int16))
    assert np.all(result["semantic_margin"] >= 0.0)


def _manifest(tmp_path: Path, *, view_count=80, point_count=4, part_names=None):
    views = []
    view_dir = tmp_path / "views"
    view_dir.mkdir(parents=True)
    for index in range(view_count):
        name = f"view_{index:03d}.pt"
        (view_dir / name).touch()
        views.append(name)
    torch.save(torch.zeros((point_count, 3)), tmp_path / "canonical_xyz.pt")
    payload = {
        "subject": "CoreView_377",
        "source_checkpoint": "/tmp/ckpt40000.pth",
        "source_checkpoint_sha256": "abc",
        "loaded_iteration": 40000,
        "point_count": point_count,
        "part_names": part_names or ["hair", "face", "upper", "lower", "shoes", "skin"],
        "view_count": view_count,
        "views": views,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_training_defaults_match_frozen_protocol():
    from tools.train_gaussian_grouping_canonical import build_parser

    args = build_parser().parse_args(["--input", "/tmp/in", "--output", "/tmp/out"])

    assert args.identity_dim == 16
    assert args.iterations == 30000
    assert args.identity_lr == pytest.approx(0.0025)
    assert args.classifier_lr == pytest.approx(5e-4)
    assert args.reg3d_interval == 2
    assert args.reg3d_k == 5
    assert args.reg3d_lambda == pytest.approx(2.0)
    assert args.reg3d_sample_size == 1000


@pytest.mark.parametrize(
    ("view_count", "point_count", "part_names", "match"),
    [
        (79, 4, None, "80 frozen training views"),
        (80, 5, None, "canonical point count"),
        (80, 4, ["face", "hair", "upper", "lower", "shoes", "skin"], "part order"),
    ],
)
def test_validate_input_manifest_rejects_protocol_mismatch(tmp_path, view_count, point_count, part_names, match):
    from tools.train_gaussian_grouping_canonical import validate_input_manifest

    _manifest(tmp_path, view_count=view_count, point_count=4, part_names=part_names)
    if point_count != 4:
        payload = json.loads((tmp_path / "manifest.json").read_text())
        payload["point_count"] = point_count
        (tmp_path / "manifest.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        validate_input_manifest(tmp_path)


def test_find_resume_checkpoint_prefers_latest_unfinished_checkpoint(tmp_path):
    from tools.train_gaussian_grouping_canonical import find_resume_checkpoint

    torch.save({"iteration": 10}, tmp_path / "checkpoint_000010.pt")
    torch.save({"iteration": 20}, tmp_path / "checkpoint_000020.pt")
    assert find_resume_checkpoint(tmp_path, iterations=30).name == "checkpoint_000020.pt"
    (tmp_path / "COMPLETE").touch()
    assert find_resume_checkpoint(tmp_path, iterations=30) is None


def test_queue_contract_uses_fixed_environment_subject_order_and_resume():
    script = Path("tools/run_gaussian_grouping_canonical_three_subject.sh").read_text(encoding="utf-8")

    assert "/opt/miniconda3/envs/gaussian_grouping/bin/python" in script
    assert 'SUBJECTS="${SUBJECTS:-377 386 394}"' in script
    assert 'ITERATIONS="${ITERATIONS:-30000}"' in script
    assert 'CANARY_ITERATIONS="${CANARY_ITERATIONS:-100}"' in script
    assert "saga_canonical_five_subject_20260812_120625_bjt" in script
    assert "queue_state.json" in script
    assert "estimated_completion_bjt" in script
    assert "--resume auto" in script


def test_estimate_queue_seconds_uses_steady_canary_rate_and_buffer():
    from utils.gaussian_grouping_canonical import estimate_queue_seconds

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


def test_queue_dry_run_does_not_write_completion_or_state(tmp_path):
    output = tmp_path / "dry-run"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "OUTPUT_ROOT": str(output),
    }

    subprocess.run(
        ["bash", "tools/run_gaussian_grouping_canonical_three_subject.sh"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )

    assert not (output / "COMPLETE").exists()
    assert not (output / "queue_state.json").exists()
