import torch
from omegaconf import OmegaConf

from utils.boundary_support_bank import (
    boundary_bad_frame_attribution_stats,
    boundary_residual_support_stats,
    boundary_support_overlap_stats,
    initialize_boundary_support_bank_state,
    materialize_effective_boundary_tags,
    promote_boundary_candidate_support,
    update_boundary_candidate_support_bank,
)
from scene.gaussian_model import GaussianModel


def _minimal_gaussian_model(point_count=4):
    model = GaussianModel(OmegaConf.create({
        "use_sh": True,
        "sh_degree": 0,
        "feature_dim": 3,
        "directional_boundary_residual_enable": True,
        "directional_boundary_residual_conflict_mode": "freeze",
    }))
    model._xyz = torch.zeros((point_count, 3))
    model._boundary_under_tag = torch.zeros((point_count,), dtype=torch.float32)
    model._boundary_over_tag = torch.zeros((point_count,), dtype=torch.float32)
    return model


def _populate_capture_fields(model, point_count):
    model._features_dc = torch.zeros((point_count, 1, 3))
    model._features_rest = torch.zeros((point_count, 0, 3))
    model._scaling = torch.zeros((point_count, 3))
    model._rotation = torch.zeros((point_count, 4))
    model._opacity = torch.zeros((point_count, 1))
    model._boundary_tag = torch.zeros((point_count,), dtype=torch.float32)
    model._boundary_opacity_residual = torch.zeros((point_count, 1))
    model._boundary_scaling_residual = torch.zeros((point_count, 3))
    model._boundary_grow_opacity_residual = torch.zeros((point_count, 1))
    model._boundary_shrink_opacity_residual = torch.zeros((point_count, 1))
    model._boundary_grow_scaling_residual = torch.zeros((point_count, 3))
    model._boundary_shrink_scaling_residual = torch.zeros((point_count, 3))
    model._boundary_cov_residual = torch.zeros((point_count, 1))
    model._binding_layer_logits_residual = torch.zeros((point_count, 3))
    model._semantic_region_logits_residual = torch.zeros((point_count, 3))
    model._semantic_compact_logits_residual = torch.zeros((point_count, 6))
    model._semantic_asset_region_logits_residual = torch.zeros((point_count, 3))
    model._semantic_asset_compact_logits_residual = torch.zeros((point_count, 6))
    model.max_radii2D = torch.zeros((point_count,))
    model.xyz_gradient_accum = torch.zeros((point_count, 1))
    model.denom = torch.zeros((point_count, 1))


def test_adopted_support_is_materialized_and_frozen():
    under = torch.tensor([1.0, 0.0, 1.0, 0.0])
    over = torch.tensor([0.0, 1.0, 0.0, 0.0])

    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=under,
        adopted_over=over,
        source_iteration=140160,
    )

    effective_under, effective_over = materialize_effective_boundary_tags(state)

    assert torch.equal(effective_under, under)
    assert torch.equal(effective_over, over)
    assert torch.equal(state["boundary_adopted_under_frozen"], under > 0)
    assert torch.equal(state["boundary_adopted_over_frozen"], over > 0)
    assert int(state["boundary_adopted_source_iteration"].item()) == 140160


def test_candidate_promotion_only_adds_new_support_without_erasing_adopted():
    state = initialize_boundary_support_bank_state(
        point_count=5,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        source_iteration=140160,
    )

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.0, 0.92, 0.90, 0.0, 0.0]),
        over_score=torch.tensor([0.0, 0.0, 0.0, 0.91, 0.95]),
        valid_mask=torch.tensor([True, True, True, True, True]),
        key="c01_f000000",
        iteration=140200,
        ema=0.0,
        score_threshold=0.80,
        bad_frame=False,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=1,
        min_frame_bits=1,
        score_threshold=0.80,
        dominance_margin=1.10,
        iteration=140240,
    )

    effective_under, effective_over = materialize_effective_boundary_tags(state)

    assert torch.equal(effective_under > 0, torch.tensor([True, True, True, False, False]))
    assert torch.equal(effective_over > 0, torch.tensor([False, False, False, True, True]))
    assert bool(effective_under[0].item()) is True
    assert bool(effective_over[3].item()) is True


def test_overlap_stats_report_low_jaccard_and_lost_adopted_support():
    adopted = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    effective = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])

    stats = boundary_support_overlap_stats(adopted, effective)

    assert stats["adopted_count"] == 2
    assert stats["effective_count"] == 2
    assert stats["intersection"] == 1
    assert stats["union"] == 3
    assert abs(stats["jaccard"] - (1.0 / 3.0)) < 1e-6
    assert stats["adopted_lost"] == 1
    assert stats["new_only"] == 1


def test_residual_support_stats_are_directional():
    under = torch.tensor([1.0, 0.0, 1.0, 0.0])
    over = torch.tensor([0.0, 1.0, 0.0, 1.0])
    grow = torch.tensor([[0.10], [0.20], [0.30], [0.40]])
    shrink = torch.tensor([[-0.10], [-0.20], [-0.30], [-0.40]])

    stats = boundary_residual_support_stats(under, over, grow, shrink)

    assert stats["grow_count"] == 2
    assert stats["shrink_count"] == 2
    assert abs(stats["grow_mean"] - 0.20) < 1e-6
    assert abs(stats["grow_abs_mean"] - 0.20) < 1e-6
    assert abs(stats["shrink_mean"] - (-0.30)) < 1e-6
    assert abs(stats["shrink_abs_mean"] - 0.30) < 1e-6


def test_bad_frame_attribution_counts_new_only_support():
    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        source_iteration=140160,
    )
    state["boundary_candidate_bad_frame_hits"] = torch.tensor([0, 3, 0, 5])
    effective_under = torch.tensor([1.0, 1.0, 0.0, 0.0])
    effective_over = torch.tensor([0.0, 0.0, 1.0, 1.0])

    stats = boundary_bad_frame_attribution_stats(state, effective_under, effective_over)

    assert stats["under_new_only_bad_hits"] == 3
    assert stats["over_new_only_bad_hits"] == 5
    assert stats["total_new_only_bad_hits"] == 8


def test_gaussian_model_initializes_support_bank_from_current_tags():
    model = _minimal_gaussian_model(4)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 1.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0, 0.0])

    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)

    state = model.get_boundary_support_bank_state()
    assert torch.equal(state["boundary_adopted_under_tag"], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(state["boundary_adopted_over_tag"], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert bool(state["boundary_adopted_initialized"].item()) is True
    assert model.get_binding_state()["anchor_refresh_mask"].shape[0] == 4


def test_gaussian_model_applies_effective_support_without_losing_adopted_tags():
    model = _minimal_gaussian_model(4)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 0.0, 1.0, 0.0])
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][1] = 1.0
    state["boundary_persistent_over_tag"][3] = 1.0
    model.set_boundary_support_bank_state(state)

    model.apply_boundary_support_bank_effective_tags(conflict_mode="freeze")

    assert torch.equal(model._boundary_under_tag > 0, torch.tensor([True, True, False, False]))
    assert torch.equal(model._boundary_over_tag > 0, torch.tensor([False, False, True, True]))


def test_candidate_requires_multiple_view_bits_before_promotion():
    state = initialize_boundary_support_bank_state(
        point_count=3,
        device=torch.device("cpu"),
        adopted_under=torch.zeros(3),
        adopted_over=torch.zeros(3),
        source_iteration=0,
    )

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.9, 0.0, 0.0]),
        over_score=torch.zeros(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        key="c01_f000000",
        iteration=1,
        ema=0.0,
        score_threshold=0.8,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=2,
        min_frame_bits=1,
        score_threshold=0.8,
        dominance_margin=1.1,
        iteration=2,
    )
    under, _ = materialize_effective_boundary_tags(state)
    assert int((under > 0).sum().item()) == 0

    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.95, 0.0, 0.0]),
        over_score=torch.zeros(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        key="c02_f000000",
        iteration=3,
        ema=0.0,
        score_threshold=0.8,
    )
    promote_boundary_candidate_support(
        state,
        min_hits=2,
        min_view_bits=2,
        min_frame_bits=1,
        score_threshold=0.8,
        dominance_margin=1.1,
        iteration=4,
    )
    under, _ = materialize_effective_boundary_tags(state)
    assert torch.equal(under > 0, torch.tensor([True, False, False]))


def test_gaussian_model_support_bank_diagnostics_include_overlap_and_residual_keys():
    model = _minimal_gaussian_model(3)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0])
    model._boundary_grow_opacity_residual = torch.tensor([[0.1], [0.0], [0.2]])
    model._boundary_shrink_opacity_residual = torch.tensor([[0.0], [-0.3], [-0.4]])
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][2] = 1.0
    model.set_boundary_support_bank_state(state)
    model.apply_boundary_support_bank_effective_tags(conflict_mode="freeze")

    diag = model.get_boundary_support_bank_diagnostics()

    assert "under_jaccard" in diag
    assert "over_jaccard" in diag
    assert "under_adopted_lost" in diag
    assert "under_new_only" in diag
    assert "grow_abs_mean" in diag
    assert "shrink_abs_mean" in diag


def test_gaussian_model_support_bank_diagnostics_include_cap_counters():
    model = _minimal_gaussian_model(4)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model._boundary_over_tag = torch.zeros(4)
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_support_under_last_promote_blocked"] = torch.tensor([3], dtype=torch.long)
    state["boundary_support_over_last_promote_blocked"] = torch.tensor([2], dtype=torch.long)
    state["boundary_support_under_cap_blocked_total"] = torch.tensor([7], dtype=torch.long)
    state["boundary_support_over_cap_blocked_total"] = torch.tensor([5], dtype=torch.long)
    model.set_boundary_support_bank_state(state)

    diag = model.get_boundary_support_bank_diagnostics()

    assert diag["support_under_last_promote_blocked"] == 3
    assert diag["support_over_last_promote_blocked"] == 2
    assert diag["support_under_cap_blocked_total"] == 7
    assert diag["support_over_cap_blocked_total"] == 5


def test_support_bank_survives_capture_restore_through_binding_state():
    model = _minimal_gaussian_model(3)
    _populate_capture_fields(model, 3)
    model._boundary_under_tag = torch.tensor([1.0, 0.0, 0.0])
    model._boundary_over_tag = torch.tensor([0.0, 1.0, 0.0])
    model.ensure_boundary_state_matches_points(verbose=False)
    model.initialize_boundary_support_bank_from_current_tags(source_iteration=140160)
    state = model.get_boundary_support_bank_state()
    state["boundary_persistent_under_tag"][2] = 1.0
    model.set_boundary_support_bank_state(state)

    captured = model.capture()
    assert len(captured) == 28

    restored = GaussianModel(model.cfg)
    restored.restore(captured, training_args={}, resume_cfg={})

    restored_state = restored.get_boundary_support_bank_state()
    assert restored_state is not None
    assert torch.equal(restored_state["boundary_adopted_under_tag"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(restored_state["boundary_persistent_under_tag"], torch.tensor([0.0, 0.0, 1.0]))


def test_promotion_respects_directional_growth_caps_without_dropping_adopted():
    state = initialize_boundary_support_bank_state(
        point_count=10,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.tensor([0.0] * 10),
        source_iteration=140160,
    )
    update_boundary_candidate_support_bank(
        state,
        under_score=torch.tensor([0.0, 0.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]),
        over_score=torch.zeros(10),
        valid_mask=torch.ones(10, dtype=torch.bool),
        key="c01_f000000",
        iteration=140200,
        ema=0.0,
        score_threshold=0.80,
    )

    new_under, new_over = promote_boundary_candidate_support(
        state,
        min_hits=1,
        min_view_bits=1,
        min_frame_bits=1,
        score_threshold=0.80,
        dominance_margin=1.10,
        iteration=140240,
        under_max_effective_ratio=0.40,
        under_max_new_only_ratio=0.20,
        over_max_effective_ratio=1.0,
        over_max_new_only_ratio=1.0,
    )
    effective_under, _ = materialize_effective_boundary_tags(state)

    assert int((effective_under > 0).sum().item()) == 4
    assert int(new_under.sum().item()) == 2
    assert int(new_over.sum().item()) == 0
    assert torch.equal(effective_under[:2] > 0, torch.tensor([True, True]))
    assert int(state["boundary_support_under_last_promote_blocked"].item()) == 6
    assert int(state["boundary_support_under_last_promote_allowed"].item()) == 2


def test_candidate_stats_continue_accumulating_after_cap_blocks_promotion():
    state = initialize_boundary_support_bank_state(
        point_count=4,
        device=torch.device("cpu"),
        adopted_under=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        adopted_over=torch.zeros(4),
        source_iteration=140160,
    )
    scores = torch.tensor([0.0, 0.95, 0.94, 0.93])
    for idx, key in enumerate(["c01_f000000", "c02_f000000"]):
        update_boundary_candidate_support_bank(
            state,
            under_score=scores,
            over_score=torch.zeros(4),
            valid_mask=torch.ones(4, dtype=torch.bool),
            key=key,
            iteration=140200 + idx,
            ema=0.0,
            score_threshold=0.80,
        )
        promote_boundary_candidate_support(
            state,
            min_hits=1,
            min_view_bits=1,
            min_frame_bits=1,
            score_threshold=0.80,
            dominance_margin=1.10,
            iteration=140240 + idx,
            under_max_effective_ratio=0.25,
            under_max_new_only_ratio=0.0,
            over_max_effective_ratio=1.0,
            over_max_new_only_ratio=1.0,
        )

    effective_under, _ = materialize_effective_boundary_tags(state)
    assert int((effective_under > 0).sum().item()) == 1
    assert int(state["boundary_candidate_under_hits"][1].item()) == 2
    assert int(state["boundary_support_under_cap_blocked_total"].item()) >= 3
