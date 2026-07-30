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
    torch.testing.assert_close(result["selection_target"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(result["selection_outer"], torch.tensor([0.5, 3.0]))
    torch.testing.assert_close(result["selection_boundary"], torch.tensor([0.0, 2.0]))


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


def test_renderer_contribution_sequence_exports_float16_samples_and_metadata():
    from utils.renderer_aligned_temporal_evidence import (
        append_renderer_contribution_sequence,
        finalize_renderer_contribution_sequence,
    )

    state = {}
    for camera_index, frame_index, scale in ((0, 0, 1.0), (0, 5, 2.0), (1, 0, 3.0)):
        values = np.full((2, 1), scale, dtype=np.float32)
        append_renderer_contribution_sequence(
            state,
            camera_index=camera_index,
            frame_index=frame_index,
            target_contribution=values,
            outer_contribution=values * 2.0,
            boundary_contribution=values * 3.0,
            selection_target_contribution=values * 4.0,
            selection_outer_contribution=values * 5.0,
            selection_boundary_contribution=values * 6.0,
            target_pixel_count=np.array([10.0 + scale], dtype=np.float32),
        )

    result = finalize_renderer_contribution_sequence(state)

    assert result["renderer_target_contribution_sequence"].shape == (3, 2, 1)
    assert result["renderer_target_contribution_sequence"].dtype == np.float16
    assert result["renderer_outer_contribution_sequence"].dtype == np.float16
    assert result["renderer_boundary_contribution_sequence"].dtype == np.float16
    assert result["renderer_selection_target_contribution_sequence"].dtype == np.float16
    assert result["renderer_selection_outer_contribution_sequence"].dtype == np.float16
    assert result["renderer_selection_boundary_contribution_sequence"].dtype == np.float16
    assert result["renderer_sequence_camera_index"].tolist() == [0, 0, 1]
    assert result["renderer_sequence_frame_index"].tolist() == [0, 5, 0]
    np.testing.assert_allclose(
        result["renderer_sequence_target_pixel_count"],
        np.array([[11.0], [12.0], [13.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result["renderer_boundary_contribution_sequence"][:, 0, 0],
        np.array([3.0, 6.0, 9.0], dtype=np.float16),
    )
