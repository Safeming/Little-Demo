import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_1_transmittance_ray_context_377_v1.json"


def test_contract_freezes_nested_feature_groups_and_scope():
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    groups = payload["feature_groups"]
    assert payload["status"] == "frozen"
    assert payload["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert payload["audit_cameras"] == ["c17", "c18", "c19", "c20"]
    assert payload["forbidden_cameras"] == ["c21", "c22", "c23"]
    assert set(groups["F0"]) < set(groups["F1"])
    assert set(groups["F1"]) < set(groups["F2"])
    assert set(groups["F2"]) < set(groups["F3"])
    assert "alpha_transmittance_mass" in groups["F2"]
    assert payload["paper_test_eligible"] is False


def test_exact_mass_is_color_gradient_and_invisible_mass_is_zero():
    from utils.a7c_ray_context_probe import exact_alpha_transmittance_mass

    colors = torch.ones(2, 3, requires_grad=True)
    weights = torch.tensor([0.3, 0.0])
    rendered = (colors * weights[:, None]).sum(dim=0).reshape(3, 1, 1)
    mass = exact_alpha_transmittance_mass(rendered, colors)

    torch.testing.assert_close(mass, weights)
    assert float(mass[1]) == 0.0


def test_ray_moments_handle_empty_and_nonempty_rays():
    from utils.a7c_ray_context_probe import ray_depth_moments

    alpha = np.array([[0.0, 0.5]])
    depth = np.array([[0.0, 1.0]])
    depth2 = np.array([[0.0, 2.5]])
    mean, variance, available = ray_depth_moments(alpha, depth, depth2)

    np.testing.assert_allclose(mean, [[0.0, 2.0]])
    np.testing.assert_allclose(variance, [[0.0, 1.0]])
    np.testing.assert_array_equal(available, [[0.0, 1.0]])


def test_footprint_sampling_returns_center_mean_and_variance():
    from utils.a7c_ray_context_probe import sample_footprint_context

    buffer = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    result = sample_footprint_context(
        buffer, projected_xy=np.array([[2.0, 2.0]]), radii=np.array([1.0])
    )

    assert result.shape == (1, 3)
    assert result[0, 0] == pytest.approx(12.0)
    assert result[0, 1] == pytest.approx(12.0)
    assert result[0, 2] > 0.0


def test_footprint_sampling_outside_image_is_finite():
    from utils.a7c_ray_context_probe import sample_footprint_context

    result = sample_footprint_context(
        np.ones((1, 3, 3), dtype=np.float32),
        projected_xy=np.array([[-100.0, -100.0]]),
        radii=np.array([1.0]),
    )

    np.testing.assert_allclose(result, [[1.0, 1.0, 0.0]])
