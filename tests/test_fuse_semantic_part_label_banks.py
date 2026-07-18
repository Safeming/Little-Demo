import numpy as np
import pytest


def test_fuse_semantic_posteriors_blends_and_normalizes():
    from utils.semantic_posterior_fusion import fuse_semantic_posteriors

    trained = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    voting = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)

    fused = fuse_semantic_posteriors(trained, voting, voting_alpha=0.75)

    np.testing.assert_allclose(fused.sum(axis=1), 1.0)
    np.testing.assert_allclose(fused[0], [0.35, 0.65], atol=1.0e-6)


def test_fuse_semantic_posteriors_rejects_shape_mismatch():
    from utils.semantic_posterior_fusion import fuse_semantic_posteriors

    with pytest.raises(ValueError, match="same shape"):
        fuse_semantic_posteriors(
            np.ones((2, 6), dtype=np.float32),
            np.ones((3, 6), dtype=np.float32),
            voting_alpha=0.5,
        )


def test_fuse_semantic_posteriors_uses_uniform_fallback_for_zero_rows():
    from utils.semantic_posterior_fusion import fuse_semantic_posteriors

    fused = fuse_semantic_posteriors(
        np.zeros((1, 6), dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
        voting_alpha=0.5,
    )

    np.testing.assert_allclose(fused[0], np.full((6,), 1.0 / 6.0, dtype=np.float32))


def test_fusion_cli_roundtrip_exports_required_fields(tmp_path):
    from tools.fuse_semantic_part_label_banks import main
    from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_part_label_bank

    point_count = 3
    common = {
        "part_label": np.array([0, 1, 2], dtype=np.int16),
        "confidence": np.ones((point_count,), dtype=np.float32),
        "vote_count": np.ones((point_count,), dtype=np.int16),
        "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        "visible_vote_count": np.ones((point_count,), dtype=np.int16),
        "conflict_count": np.zeros((point_count,), dtype=np.int16),
        "source_checkpoint": "/tmp/checkpoint.pth",
        "source_asset_root": "/tmp/assets",
        "source_iteration": 12,
    }
    trained_path = tmp_path / "trained.npz"
    voting_path = tmp_path / "voting.npz"
    output_path = tmp_path / "fused.npz"
    trained_probs = np.full((point_count, len(PART_NAMES)), 0.05, dtype=np.float32)
    trained_probs[:, 0] = 0.75
    voting_probs = np.full((point_count, len(PART_NAMES)), 0.04, dtype=np.float32)
    voting_probs[:, 1] = 0.80
    save_part_label_bank(trained_path, semantic_probs=trained_probs, **common)
    save_part_label_bank(voting_path, semantic_probs=voting_probs, **common)

    status = main(
        [
            "--trained-bank", str(trained_path),
            "--voting-bank", str(voting_path),
            "--voting-alpha", "0.75",
            "--output", str(output_path),
        ]
    )

    assert status == 0
    bank = load_part_label_bank(output_path)
    for key in (
        "semantic_probs",
        "confidence",
        "semantic_margin",
        "reliable_mask",
        "editable_label",
        "soft_edit_weights",
        "trained_bank_fingerprint",
        "voting_bank_fingerprint",
        "fusion_alpha",
    ):
        assert key in bank
    assert bank["source_type"].item() == "fused_trained_voting_semantic_probs"
    assert bank["fusion_alpha"].item() == pytest.approx(0.75)
