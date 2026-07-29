import numpy as np
import pytest
import torch


def test_extract_renderer_region_contributions_uses_rgb_gradient_channels():
    from utils.renderer_aligned_temporal_evidence import (
        extract_renderer_region_contributions,
    )

    colors = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
    coefficients = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.0]],
            [[0.0, 2.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    rendered = torch.stack(
        [torch.einsum("hwn,n->hw", coefficients, colors[:, channel]) for channel in range(3)]
    )
    result = extract_renderer_region_contributions(
        rendered=rendered,
        attribution_colors=colors,
        target_mask=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        valid_mask=torch.ones((2, 2), dtype=torch.float32),
        boundary_mask=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        edit_sensitivity=torch.tensor([2.0, 0.5], dtype=torch.float32),
    )

    torch.testing.assert_close(result["target"], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(result["outer"], torch.tensor([1.0, 1.5]))
    torch.testing.assert_close(result["boundary"], torch.tensor([0.0, 1.0]))


def test_renderer_contribution_accumulator_exports_raw_and_compatibility_fields():
    from utils.renderer_aligned_temporal_evidence import (
        accumulate_renderer_contribution_frame,
        finalize_renderer_contribution_evidence,
    )

    state = {}
    accumulate_renderer_contribution_frame(
        state,
        frame_index=0,
        target_contribution=np.array([[2.0], [0.0]], dtype=np.float32),
        outer_contribution=np.array([[1.0], [0.0]], dtype=np.float32),
        boundary_contribution=np.array([[0.5], [0.0]], dtype=np.float32),
    )
    accumulate_renderer_contribution_frame(
        state,
        frame_index=5,
        target_contribution=np.array([[1.0], [3.0]], dtype=np.float32),
        outer_contribution=np.array([[1.0], [1.0]], dtype=np.float32),
        boundary_contribution=np.array([[0.25], [0.5]], dtype=np.float32),
    )

    result = finalize_renderer_contribution_evidence(state)

    assert result["temporal_visible_count"].tolist() == [[2], [1]]
    assert result["temporal_consecutive_visible_count"].tolist() == [[1], [0]]
    assert result["renderer_target_contribution_mean_raw"][0, 0] == pytest.approx(1.5)
    assert result["renderer_outer_contribution_mean_raw"][0, 0] == pytest.approx(1.0)
    assert result["renderer_target_contribution_flicker"][0, 0] == pytest.approx(2.0 / 3.0)
    assert np.all(result["temporal_target_ratio_mean"] >= 0.0)
    assert np.all(result["temporal_target_ratio_mean"] <= 1.0)
    assert np.all(result["temporal_outer_ratio_mean"] >= 0.0)
    assert np.all(result["temporal_outer_ratio_mean"] <= 1.0)
    assert np.all(np.isfinite(result["renderer_boundary_contribution_flicker"]))
