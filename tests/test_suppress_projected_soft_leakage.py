import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.suppress_projected_soft_leakage import (
    apply_soft_weight_suppression,
    build_view_leakage_record,
    classify_suppression_mask,
    compute_point_leakage_stats,
    save_suppressed_bank,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_part_label_bank


class SuppressProjectedSoftLeakageTests(unittest.TestCase):
    def test_compute_point_leakage_stats_accumulates_target_outer_and_boundary_hits(self):
        point_count = 3
        lower = PART_NAMES.index("lower")
        weights = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
        weights[:, lower] = np.array([0.8, 0.6, 0.4], dtype=np.float32)
        records = [
            {
                "observed": np.array([True, True, False]),
                "target": np.array([False, True, False]),
                "outer": np.array([True, False, False]),
                "boundary": np.array([True, False, False]),
            },
            {
                "observed": np.array([True, True, True]),
                "target": np.array([False, True, True]),
                "outer": np.array([True, False, False]),
                "boundary": np.array([False, True, False]),
            },
        ]

        stats = compute_point_leakage_stats(records, weights, part_index=lower)

        self.assertEqual(stats["observed_view_count"].tolist(), [2, 2, 1])
        self.assertEqual(stats["target_hit_count"].tolist(), [0, 2, 1])
        self.assertEqual(stats["outer_hit_count"].tolist(), [2, 0, 0])
        self.assertEqual(stats["boundary_hit_count"].tolist(), [1, 1, 0])
        self.assertTrue(np.allclose(stats["outer_weight_sum"], [1.6, 0.0, 0.0]))
        self.assertTrue(np.allclose(stats["target_hit_ratio"], [0.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(stats["stable_leak_score"], [1.0, -1.0, -1.0]))

    def test_classify_suppression_mask_only_marks_stable_outer_low_target_points(self):
        weights = np.array([0.8, 0.7, 0.9, 0.4], dtype=np.float32)
        stats = {
            "observed_view_count": np.array([6, 6, 4, 6], dtype=np.int32),
            "target_hit_count": np.array([1, 4, 0, 0], dtype=np.int32),
            "outer_hit_count": np.array([5, 5, 4, 5], dtype=np.int32),
            "boundary_hit_count": np.array([1, 1, 1, 6], dtype=np.int32),
            "target_hit_ratio": np.array([1 / 6, 4 / 6, 0.0, 0.0], dtype=np.float32),
            "outer_hit_ratio": np.array([5 / 6, 5 / 6, 1.0, 5 / 6], dtype=np.float32),
        }

        severe, boundary = classify_suppression_mask(
            stats,
            weights,
            soft_threshold=0.5,
            min_observed_views=5,
            min_outer_views=3,
            max_target_hit_ratio=0.35,
            min_outer_hit_ratio=0.55,
        )

        self.assertEqual(severe.tolist(), [True, False, False, False])
        self.assertEqual(boundary.tolist(), [False, False, False, True])

    def test_apply_soft_weight_suppression_only_changes_requested_part_channel(self):
        weights = np.arange(4 * len(PART_NAMES), dtype=np.float32).reshape(4, len(PART_NAMES)) / 10.0
        lower = PART_NAMES.index("lower")
        severe = np.array([True, False, False, False])
        boundary = np.array([False, True, False, False])

        updated, summary = apply_soft_weight_suppression(
            weights,
            part_name="lower",
            severe_mask=severe,
            boundary_mask=boundary,
            suppress_factor=0.25,
            boundary_cap=0.30,
        )

        expected = weights.copy()
        expected[0, lower] *= 0.25
        expected[1, lower] = min(expected[1, lower], 0.30)
        self.assertTrue(np.allclose(updated, expected))
        self.assertTrue(np.allclose(updated[:, :lower], weights[:, :lower]))
        self.assertTrue(np.allclose(updated[:, lower + 1 :], weights[:, lower + 1 :]))
        self.assertEqual(summary["severe_suppressed_count"], 1)
        self.assertEqual(summary["boundary_capped_count"], 1)

    def test_save_suppressed_bank_preserves_labels_and_non_target_soft_channels(self):
        point_count = 3
        weights = np.arange(point_count * len(PART_NAMES), dtype=np.float32).reshape(point_count, len(PART_NAMES)) / 10.0
        bank = {
            "part_label": np.array([0, 3, 4], dtype=np.int16),
            "editable_label": np.array([0, -1, 4], dtype=np.int16),
            "confidence": np.linspace(0.1, 0.3, point_count, dtype=np.float32),
            "vote_count": np.arange(point_count, dtype=np.int16),
            "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
            "visible_vote_count": np.ones((point_count,), dtype=np.int16),
            "conflict_count": np.zeros((point_count,), dtype=np.int16),
            "semantic_probs": np.full((point_count, len(PART_NAMES)), 1 / len(PART_NAMES), dtype=np.float32),
            "soft_edit_weights": weights,
            "source_checkpoint": np.array("/tmp/source.pth"),
            "source_asset_root": np.array("/tmp/assets"),
            "source_iteration": np.array(123, dtype=np.int64),
            "source_type": np.array("base"),
        }
        updated_weights = weights.copy()
        updated_weights[:, PART_NAMES.index("lower")] *= 0.5

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bank.npz"
            save_suppressed_bank(output, bank, updated_weights, source_type="leak_suppressed_lower")
            loaded = load_part_label_bank(output)

        self.assertEqual(loaded["part_label"].tolist(), bank["part_label"].tolist())
        self.assertEqual(loaded["editable_label"].tolist(), bank["editable_label"].tolist())
        self.assertTrue(np.allclose(loaded["semantic_probs"], bank["semantic_probs"]))
        self.assertTrue(np.allclose(loaded["soft_edit_weights"], updated_weights))
        self.assertTrue(np.allclose(loaded["soft_edit_weights"][:, PART_NAMES.index("hair")], weights[:, PART_NAMES.index("hair")]))
        self.assertEqual(str(loaded["source_type"]), "leak_suppressed_lower")

    def test_save_suppressed_bank_rejects_point_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.npz"
            save_part_label_bank(
                path,
                part_label=np.array([0, 1], dtype=np.int16),
                confidence=np.ones((2,), dtype=np.float32),
                vote_count=np.zeros((2,), dtype=np.int16),
                per_part_votes=np.zeros((2, len(PART_NAMES)), dtype=np.int16),
                visible_vote_count=np.zeros((2,), dtype=np.int16),
                conflict_count=np.zeros((2,), dtype=np.int16),
                source_checkpoint="/tmp/source.pth",
                source_asset_root="/tmp/assets",
                source_iteration=1,
                soft_edit_weights=np.zeros((2, len(PART_NAMES)), dtype=np.float32),
            )
            bank = load_part_label_bank(path)
            bad_weights = np.zeros((3, len(PART_NAMES)), dtype=np.float32)
            with self.assertRaises(ValueError):
                save_suppressed_bank(Path(tmp) / "out.npz", bank, bad_weights)

    def test_build_view_leakage_record_ignores_renderer_tail_points(self):
        xy = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        proj_valid = np.array([True, True])
        visibility_filter = np.array([True, True, True])
        radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        part_masks = {name: np.zeros((4, 4), dtype=np.float32) for name in PART_NAMES}
        part_masks["lower"][1, 1] = 1.0
        foreground = np.ones((4, 4), dtype=np.float32)
        valid = np.ones((4, 4), dtype=np.float32)

        record = build_view_leakage_record(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility_filter,
            radii=radii,
            image_size=(4, 4),
            part_masks=part_masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="lower",
            mask_threshold=0.5,
            boundary_radius=1,
        )

        self.assertEqual(record["observed"].tolist(), [True, True])
        self.assertEqual(record["target"].tolist(), [True, False])
        self.assertEqual(record["outer"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
