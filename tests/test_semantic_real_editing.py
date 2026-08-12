import numpy as np


def _banks():
    raw = {"part_label": np.array([0, 1, 0, -1], dtype=np.int16)}
    voting = {"editable_label": np.array([1, 1, 0, -1], dtype=np.int16)}
    a5_weights = np.zeros((4, 6), dtype=np.float32)
    a5_weights[:, 0] = [0.8, 0.3, 0.19, 0.0]
    a5 = {
        "part_label": np.array([0, 1, 0, -1], dtype=np.int16),
        "soft_edit_weights": a5_weights,
    }
    return raw, voting, a5


def test_resolve_method_weights_routes_frozen_banks():
    from utils.semantic_real_editing import resolve_method_weights

    raw, voting, a5 = _banks()

    assert np.allclose(resolve_method_weights(raw, voting, a5, method="raw_hard", part="hair", threshold=0.2), [1, 0, 1, 0])
    assert np.allclose(resolve_method_weights(raw, voting, a5, method="voting", part="hair", threshold=0.2), [0, 0, 1, 0])
    assert np.allclose(resolve_method_weights(raw, voting, a5, method="a5", part="hair", threshold=0.2), [0.8, 0.3, 0, 0])


def test_resolve_method_weights_rejects_missing_a5_soft_weights():
    import pytest
    from utils.semantic_real_editing import resolve_method_weights

    raw, voting, _ = _banks()
    with pytest.raises(ValueError, match="soft_edit_weights"):
        resolve_method_weights(raw, voting, {"part_label": raw["part_label"]}, method="a5", part="hair", threshold=0.2)


def test_resolve_method_weights_recomputes_saga_b4_weights():
    from utils.semantic_real_editing import resolve_method_weights

    raw, voting, a5 = _banks()
    probabilities = np.zeros((4, 6), dtype=np.float32)
    probabilities[:, 0] = [0.9, 0.8, 0.7, 0.6]
    saga = {
        "part_label": np.array([0, 0, 0, 0], dtype=np.int16),
        "semantic_probs": probabilities,
        "confidence": np.array([1.0, 0.8, 0.9, 0.5], dtype=np.float32),
        "semantic_margin": np.array([1.0, 0.9, 0.5, 1.0], dtype=np.float32),
    }

    weights = resolve_method_weights(
        raw,
        voting,
        a5,
        saga_bank=saga,
        method="saga",
        part="hair",
        threshold=0.5,
    )

    assert np.allclose(weights, [0.9, 0.576, 0.0, 0.0])


def test_resolve_method_weights_rejects_incomplete_saga_bank():
    import pytest
    from utils.semantic_real_editing import resolve_method_weights

    raw, voting, a5 = _banks()
    with pytest.raises(ValueError, match="SAGA bank must contain"):
        resolve_method_weights(
            raw,
            voting,
            a5,
            saga_bank={"part_label": raw["part_label"]},
            method="saga",
            part="hair",
            threshold=0.5,
        )


def test_canonical_stripe_colors_are_deterministic_and_point_aligned():
    from utils.semantic_real_editing import canonical_stripe_colors

    xyz = np.array([[0, 0, 0], [0, 0.2, 0], [0, 0.4, 0], [0, 0.6, 0]], dtype=np.float32)
    first = canonical_stripe_colors(xyz, primary_rgb=(1, 0, 0), secondary_rgb=(0, 0, 1), frequency=4.0)
    second = canonical_stripe_colors(xyz, primary_rgb=(1, 0, 0), secondary_rgb=(0, 0, 1), frequency=4.0)

    assert first.shape == (4, 3)
    assert np.array_equal(first, second)
    assert set(map(tuple, first.tolist())) == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}


def test_build_edit_overrides_handles_recolor_removal_and_texture():
    from utils.semantic_real_editing import build_edit_overrides

    colors = np.array([[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]], dtype=np.float32)
    opacity = np.array([[0.8], [0.4]], dtype=np.float32)
    weights = np.array([1.0, 0.25], dtype=np.float32)
    pattern = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)

    recolor = build_edit_overrides(colors, opacity, weights, task="recolor", strength=0.5, target_rgb=(1, 0, 0), texture_colors=pattern)
    assert np.allclose(recolor["colors"][0], [0.6, 0.2, 0.3])
    assert np.allclose(recolor["opacities"], opacity)

    removal = build_edit_overrides(colors, opacity, weights, task="removal", strength=1.0, target_rgb=(1, 0, 0), texture_colors=pattern)
    assert np.allclose(removal["colors"], colors)
    assert np.allclose(removal["opacities"].reshape(-1), [0.0, 0.3])

    texture = build_edit_overrides(colors, opacity, weights, task="texture", strength=1.0, target_rgb=(1, 0, 0), texture_colors=pattern)
    assert np.allclose(texture["colors"][0], pattern[0])
    assert np.allclose(texture["colors"][1], colors[1] * 0.75 + pattern[1] * 0.25)


def test_compute_edit_delta_metrics_separates_target_outer_and_boundary():
    from utils.semantic_real_editing import compute_edit_delta_metrics

    base = np.zeros((5, 5, 3), dtype=np.float32)
    edited = np.zeros_like(base)
    edited[2, 2] = 1.0
    edited[2, 3] = 0.5
    edited[0, 0] = 0.25
    target = np.zeros((5, 5), dtype=np.float32)
    target[2, 2] = 1.0
    valid = np.ones((5, 5), dtype=np.float32)

    metrics = compute_edit_delta_metrics(base, edited, target, valid, boundary_radius=1)

    assert metrics["target_pixel_count"] == 1
    assert metrics["outer_pixel_count"] == 24
    assert metrics["target_delta_sum"] == 3.0
    assert metrics["outer_delta_sum"] == 2.25
    assert metrics["boundary_outer_delta_sum"] == 1.5
    assert metrics["outer_to_target_delta_ratio"] == 0.75
    assert metrics["edit_response_intersection"] == 1
    assert metrics["edit_response_union"] == 3
    assert np.isclose(metrics["edit_response_iou"], 1.0 / 3.0)
