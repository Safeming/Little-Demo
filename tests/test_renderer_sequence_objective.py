import numpy as np
import pytest


def test_renderer_sequence_objective_resets_adjacency_at_camera_boundaries():
    from utils.renderer_sequence_objective import summarize_renderer_sequence_objective

    target = np.array([[[2.0]], [[4.0]], [[20.0]], [[24.0]]], dtype=np.float32)
    outer = np.array([[[1.0]], [[3.0]], [[10.0]], [[14.0]]], dtype=np.float32)
    boundary = np.array([[[2.0]], [[6.0]], [[5.0]], [[15.0]]], dtype=np.float32)
    result = summarize_renderer_sequence_objective(
        a5_weights=np.array([[1.0]], dtype=np.float32),
        candidate_weights=np.array([[0.5]], dtype=np.float32),
        target_sequence=target,
        outer_sequence=outer,
        boundary_sequence=boundary,
        camera_index=np.array([0, 0, 1, 1], dtype=np.int16),
        frame_index=np.array([0, 5, 0, 5], dtype=np.int32),
        processed_part_indices=(0,),
    )

    a5 = result["methods"]["a5"]["0"]
    assert a5["outer_mean_response"] == pytest.approx(7.0)
    assert a5["outer_adjacent_absolute_change"] == pytest.approx(3.0)
    assert a5["outer_normalized_flicker"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert result["per_part_ratios"]["0"]["outer_mean_response"] == pytest.approx(0.5)
    assert result["per_part_ratios"]["0"]["outer_adjacent_absolute_change"] == pytest.approx(0.5)
    assert result["per_part_ratios"]["0"]["outer_normalized_flicker"] == pytest.approx(1.0)


def test_renderer_sequence_objective_aggregates_processed_parts_equally():
    from utils.renderer_sequence_objective import summarize_renderer_sequence_objective

    base = np.ones((2, 2, 2), dtype=np.float32)
    base[1, :, 0] = 2.0
    base[1, :, 1] = 3.0
    result = summarize_renderer_sequence_objective(
        a5_weights=np.ones((2, 2), dtype=np.float32),
        candidate_weights=np.array([[0.5, 1.0], [0.5, 0.5]], dtype=np.float32),
        target_sequence=base,
        outer_sequence=base,
        boundary_sequence=base,
        camera_index=np.array([0, 0], dtype=np.int16),
        frame_index=np.array([0, 5], dtype=np.int32),
        processed_part_indices=(0, 1),
    )

    expected = np.mean(
        [
            result["per_part_ratios"]["0"]["target_mean_response"],
            result["per_part_ratios"]["1"]["target_mean_response"],
        ]
    )
    assert result["aggregate_ratios"]["target_mean_response"] == pytest.approx(expected)
    assert result["processed_part_indices"] == [0, 1]


def test_renderer_sequence_objective_rejects_unsorted_frames_within_camera():
    from utils.renderer_sequence_objective import summarize_renderer_sequence_objective

    sequence = np.ones((2, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_renderer_sequence_objective(
            a5_weights=np.ones((1, 1), dtype=np.float32),
            candidate_weights=np.ones((1, 1), dtype=np.float32),
            target_sequence=sequence,
            outer_sequence=sequence,
            boundary_sequence=sequence,
            camera_index=np.array([0, 0], dtype=np.int16),
            frame_index=np.array([5, 0], dtype=np.int32),
            processed_part_indices=(0,),
        )
