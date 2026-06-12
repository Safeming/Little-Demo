import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.calibrate_evidence_soft_edit_weights import (
    accumulate_center_consistency_evidence,
    accumulate_footprint_evidence,
    apply_evidence_calibration_to_bank,
    build_part_candidate_mask,
    build_center_consistency_evidence_record,
    build_footprint_evidence_record,
    build_soft_boundary_target_mask,
    save_footprint_evidence_npz,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank, save_part_label_bank


class EvidenceCalibratedSemanticBankTests(unittest.TestCase):
    def test_build_soft_boundary_target_mask_keeps_inside_one_and_falls_off_outside(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[2, 2] = 1.0

        soft = build_soft_boundary_target_mask(mask, radius=2, threshold=0.5, min_boundary_value=0.25)

        self.assertAlmostEqual(float(soft[2, 2]), 1.0)
        self.assertAlmostEqual(float(soft[2, 3]), 0.625)
        self.assertAlmostEqual(float(soft[2, 4]), 0.25)
        self.assertAlmostEqual(float(soft[0, 0]), 0.0)

    def test_build_footprint_evidence_record_measures_target_outer_and_conflict_ratios(self):
        xy = np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32)
        proj_valid = np.array([True, True])
        visibility = np.array([True, True])
        radii = np.array([1.0, 1.0], dtype=np.float32)
        masks = {name: np.zeros((7, 7), dtype=np.float32) for name in PART_NAMES}
        masks["lower"][1:4, 1:4] = 1.0
        masks["upper"][2, 2] = 1.0
        foreground = np.ones((7, 7), dtype=np.float32)
        valid = np.ones((7, 7), dtype=np.float32)

        record = build_footprint_evidence_record(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility,
            radii=radii,
            image_size=(7, 7),
            part_masks=masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="lower",
            mask_threshold=0.5,
            footprint_radius_scale=1.0,
            min_footprint_radius=1,
            max_footprint_radius=2,
        )

        self.assertEqual(record["observed"].tolist(), [True, True])
        self.assertGreater(record["target_ratio"][0], 0.9)
        self.assertLess(record["outer_ratio"][0], 0.1)
        self.assertGreater(record["conflict_ratio"][0], 0.0)
        self.assertEqual(float(record["target_ratio"][1]), 0.0)
        self.assertEqual(float(record["outer_ratio"][1]), 1.0)

    def test_build_footprint_evidence_record_uses_soft_boundary_target_values(self):
        xy = np.array([[1.0, 2.0]], dtype=np.float32)
        proj_valid = np.array([True])
        visibility = np.array([True])
        radii = np.array([1.0], dtype=np.float32)
        masks = {name: np.zeros((5, 5), dtype=np.float32) for name in PART_NAMES}
        masks["shoes"][2, 2] = 1.0
        foreground = np.ones((5, 5), dtype=np.float32)
        valid = np.ones((5, 5), dtype=np.float32)

        hard = build_footprint_evidence_record(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility,
            radii=radii,
            image_size=(5, 5),
            part_masks=masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="shoes",
            min_footprint_radius=1,
            max_footprint_radius=1,
        )
        soft = build_footprint_evidence_record(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility,
            radii=radii,
            image_size=(5, 5),
            part_masks=masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="shoes",
            min_footprint_radius=1,
            max_footprint_radius=1,
            soft_boundary_radius=1,
            soft_boundary_min_value=0.5,
        )

        self.assertAlmostEqual(float(hard["target_ratio"][0]), 0.2)
        self.assertAlmostEqual(float(soft["target_ratio"][0]), 0.3)
        self.assertAlmostEqual(float(soft["outer_ratio"][0]), 0.7)

    def test_build_footprint_evidence_record_skips_non_candidate_points(self):
        xy = np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32)
        proj_valid = np.array([True, True])
        visibility = np.array([True, True])
        radii = np.array([1.0, 1.0], dtype=np.float32)
        masks = {name: np.zeros((7, 7), dtype=np.float32) for name in PART_NAMES}
        masks["lower"][1:4, 1:4] = 1.0
        foreground = np.ones((7, 7), dtype=np.float32)
        valid = np.ones((7, 7), dtype=np.float32)

        record = build_footprint_evidence_record(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility,
            radii=radii,
            image_size=(7, 7),
            part_masks=masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="lower",
            candidate_mask=np.array([True, False]),
        )

        self.assertEqual(record["observed"].tolist(), [True, False])
        self.assertGreater(record["target_ratio"][0], 0.0)
        self.assertEqual(float(record["target_ratio"][1]), 0.0)
        self.assertEqual(float(record["outer_ratio"][1]), 0.0)
        self.assertEqual(float(record["conflict_ratio"][1]), 0.0)

    def test_build_part_candidate_mask_uses_soft_weight_or_editable_label(self):
        weights = np.zeros((4, len(PART_NAMES)), dtype=np.float32)
        lower = PART_NAMES.index("lower")
        shoes = PART_NAMES.index("shoes")
        weights[:, lower] = [0.01, 0.05, 0.30, 0.0]
        editable = np.array([shoes, shoes, shoes, lower], dtype=np.int16)

        mask = build_part_candidate_mask(
            soft_edit_weights=weights,
            editable_label=editable,
            part_name="lower",
            soft_min_weight=0.05,
        )

        self.assertEqual(mask.tolist(), [False, True, True, True])

    def test_accumulate_footprint_evidence_averages_observed_views(self):
        records = [
            {
                "observed": np.array([True, True, False]),
                "target_ratio": np.array([0.8, 0.1, 0.0], dtype=np.float32),
                "outer_ratio": np.array([0.2, 0.9, 0.0], dtype=np.float32),
                "conflict_ratio": np.array([0.0, 0.5, 0.0], dtype=np.float32),
            },
            {
                "observed": np.array([True, False, True]),
                "target_ratio": np.array([1.0, 0.0, 0.3], dtype=np.float32),
                "outer_ratio": np.array([0.0, 0.0, 0.7], dtype=np.float32),
                "conflict_ratio": np.array([0.2, 0.0, 0.0], dtype=np.float32),
            },
        ]

        stats = accumulate_footprint_evidence(records, point_count=3)

        self.assertEqual(stats["view_support_count"].tolist(), [2, 1, 1])
        self.assertTrue(np.allclose(stats["footprint_target_ratio"], [0.9, 0.1, 0.3]))
        self.assertTrue(np.allclose(stats["footprint_outer_ratio"], [0.1, 0.9, 0.7]))
        self.assertTrue(np.allclose(stats["conflict_ratio"], [0.1, 0.5, 0.0]))

    def test_center_consistency_evidence_counts_target_outer_and_invalid_centers(self):
        xy = np.array([[2.0, 2.0], [5.0, 5.0], [20.0, 20.0]], dtype=np.float32)
        proj_valid = np.array([True, True, True])
        candidate = np.array([True, True, True])
        masks = {name: np.zeros((7, 7), dtype=np.float32) for name in PART_NAMES}
        masks["shoes"][1:4, 1:4] = 1.0
        foreground = np.ones((7, 7), dtype=np.float32)
        valid = np.ones((7, 7), dtype=np.float32)

        record = build_center_consistency_evidence_record(
            xy=xy,
            proj_valid=proj_valid,
            image_size=(7, 7),
            part_masks=masks,
            foreground_mask=foreground,
            valid_mask=valid,
            part_name="shoes",
            candidate_mask=candidate,
        )

        self.assertEqual(record["valid_center"].tolist(), [True, True, False])
        self.assertEqual(record["target_center"].tolist(), [True, False, False])
        self.assertEqual(record["outer_center"].tolist(), [False, True, False])

    def test_accumulate_center_consistency_evidence_reports_outer_ratio(self):
        records = [
            {
                "valid_center": np.array([True, True, False]),
                "target_center": np.array([True, False, False]),
                "outer_center": np.array([False, True, False]),
            },
            {
                "valid_center": np.array([True, True, True]),
                "target_center": np.array([False, False, True]),
                "outer_center": np.array([True, True, False]),
            },
        ]

        stats = accumulate_center_consistency_evidence(records, point_count=3)

        self.assertEqual(stats["center_valid_count"].tolist(), [2, 2, 1])
        self.assertEqual(stats["center_target_hit_count"].tolist(), [1, 0, 1])
        self.assertEqual(stats["center_outer_hit_count"].tolist(), [1, 2, 0])
        self.assertTrue(np.allclose(stats["center_outer_ratio"], [0.5, 1.0, 0.0]))

    def test_apply_evidence_calibration_to_bank_preserves_labels_and_non_target_channels(self):
        point_count = 3
        weights = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
        lower = PART_NAMES.index("lower")
        shoes = PART_NAMES.index("shoes")
        weights[:, lower] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        weights[:, shoes] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        bank = {
            "part_label": np.array([lower, lower, shoes], dtype=np.int16),
            "editable_label": np.array([lower, lower, shoes], dtype=np.int16),
            "confidence": np.ones((point_count,), dtype=np.float32),
            "vote_count": np.ones((point_count,), dtype=np.int16),
            "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
            "visible_vote_count": np.ones((point_count,), dtype=np.int16),
            "conflict_count": np.zeros((point_count,), dtype=np.int16),
            "semantic_probs": np.full((point_count, len(PART_NAMES)), 1.0 / len(PART_NAMES), dtype=np.float32),
            "soft_edit_weights": weights,
            "source_checkpoint": np.array("/tmp/ckpt.pth"),
            "source_asset_root": np.array("/tmp/assets"),
            "source_iteration": np.array(7, dtype=np.int64),
        }
        evidence = {
            "footprint_target_ratio": np.zeros_like(weights),
            "footprint_outer_ratio": np.zeros_like(weights),
            "view_support_count": np.zeros_like(weights, dtype=np.int16),
            "conflict_ratio": np.zeros_like(weights),
        }
        evidence["footprint_target_ratio"][:, lower] = [0.9, 0.1, 0.9]
        evidence["footprint_outer_ratio"][:, lower] = [0.1, 0.9, 0.1]
        evidence["view_support_count"][:, lower] = [6, 6, 0]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bank.npz"
            summary = apply_evidence_calibration_to_bank(
                bank,
                evidence,
                output=output,
                parts=("lower",),
                min_support_views=5,
                target_retention_floor=0.6,
            )
            loaded = load_part_label_bank(output)

        self.assertEqual(loaded["part_label"].tolist(), bank["part_label"].tolist())
        self.assertTrue(np.allclose(loaded["soft_edit_weights"][:, shoes], weights[:, shoes]))
        self.assertGreaterEqual(loaded["soft_edit_weights"][0, lower], weights[0, lower] * 0.6)
        self.assertLess(loaded["soft_edit_weights"][1, lower], weights[1, lower] * 0.2)
        self.assertEqual(summary["parts"]["lower"]["calibrated_count"], 2)

    def test_apply_evidence_calibration_uses_center_outer_ratio_with_footprint_retention_guard(self):
        point_count = 3
        weights = np.zeros((point_count, len(PART_NAMES)), dtype=np.float32)
        shoes = PART_NAMES.index("shoes")
        lower = PART_NAMES.index("lower")
        weights[:, shoes] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        weights[:, lower] = 0.4
        bank = {
            "part_label": np.array([shoes, shoes, shoes], dtype=np.int16),
            "editable_label": np.array([shoes, shoes, shoes], dtype=np.int16),
            "confidence": np.ones((point_count,), dtype=np.float32),
            "vote_count": np.ones((point_count,), dtype=np.int16),
            "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
            "visible_vote_count": np.ones((point_count,), dtype=np.int16),
            "conflict_count": np.zeros((point_count,), dtype=np.int16),
            "semantic_probs": np.full((point_count, len(PART_NAMES)), 1.0 / len(PART_NAMES), dtype=np.float32),
            "soft_edit_weights": weights,
        }
        evidence = {
            "footprint_target_ratio": np.zeros_like(weights),
            "footprint_outer_ratio": np.zeros_like(weights),
            "view_support_count": np.zeros_like(weights, dtype=np.int16),
            "conflict_ratio": np.zeros_like(weights),
            "center_outer_ratio": np.zeros_like(weights),
            "center_valid_count": np.zeros_like(weights, dtype=np.int16),
            "center_outer_hit_count": np.zeros_like(weights, dtype=np.int16),
            "center_target_hit_count": np.zeros_like(weights, dtype=np.int16),
        }
        evidence["view_support_count"][:, shoes] = 6
        evidence["footprint_target_ratio"][:, shoes] = [0.95, 0.70, 0.20]
        evidence["footprint_outer_ratio"][:, shoes] = [0.05, 0.30, 0.80]
        evidence["center_valid_count"][:, shoes] = [8, 8, 8]
        evidence["center_outer_ratio"][:, shoes] = [1.0, 1.0, 1.0]
        evidence["center_outer_hit_count"][:, shoes] = [8, 8, 8]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bank.npz"
            summary = apply_evidence_calibration_to_bank(
                bank,
                evidence,
                output=output,
                parts=("shoes",),
                min_support_views=5,
                min_center_views=3,
                target_retention_floor=0.6,
                center_penalty_power=1.0,
                center_target_retention_floor=0.75,
            )
            loaded = load_part_label_bank(output)

        self.assertGreaterEqual(loaded["soft_edit_weights"][0, shoes], weights[0, shoes] * 0.75)
        self.assertLess(loaded["soft_edit_weights"][1, shoes], weights[1, shoes] * 0.1)
        self.assertLess(loaded["soft_edit_weights"][2, shoes], weights[2, shoes] * 0.1)
        self.assertTrue(np.allclose(loaded["soft_edit_weights"][:, lower], weights[:, lower]))
        self.assertIn("center_penalized_count", summary["parts"]["shoes"])

    def test_save_footprint_evidence_npz_writes_reproducible_sidecar(self):
        point_count = 2
        evidence = {
            "footprint_target_ratio": np.full((point_count, len(PART_NAMES)), 0.25, dtype=np.float32),
            "footprint_outer_ratio": np.full((point_count, len(PART_NAMES)), 0.75, dtype=np.float32),
            "view_support_count": np.ones((point_count, len(PART_NAMES)), dtype=np.int16),
            "conflict_ratio": np.zeros((point_count, len(PART_NAMES)), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "footprint_evidence.npz"
            save_footprint_evidence_npz(output, evidence, part_names=("lower", "shoes"))
            loaded = np.load(output, allow_pickle=False)

            self.assertIn("footprint_target_ratio", loaded.files)
            self.assertIn("footprint_outer_ratio", loaded.files)
            self.assertIn("view_support_count", loaded.files)
            self.assertIn("conflict_ratio", loaded.files)
            self.assertIn("part_names", loaded.files)
            self.assertEqual(loaded["footprint_target_ratio"].shape, (point_count, len(PART_NAMES)))
            self.assertEqual(loaded["part_names"].tolist(), ["lower", "shoes"])


if __name__ == "__main__":
    unittest.main()
