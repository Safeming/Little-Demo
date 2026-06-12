import json
import unittest

import numpy as np
import torch

from utils.part_label_bank import (
    PART_NAMES,
    apply_face_label_guard,
    apply_neighbor_reliable_fill,
    apply_reliable_label_mask,
    compute_evidence_calibrated_soft_edit_weights,
    compute_semantic_margin,
    compute_semantic_reliable_mask,
    compute_soft_edit_weights,
    apply_lower_label_guard,
    finalize_votes,
    finalize_trained_semantic_probs,
    load_part_label_bank,
    save_part_label_bank,
    summarize_part_label_bank,
    validate_part_label_bank_arrays,
    write_preview_ply,
    write_summary_json,
)
from tools.semantic_viewer.build_part_label_bank import accumulate_projected_votes
from tools.semantic_viewer.build_part_label_bank import build_output_manifest


class PartLabelBankTests(unittest.TestCase):
    def test_finalize_votes_labels_confidence_and_unknowns(self):
        per_part_votes = np.array(
            [
                [2, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        visible_vote_count = np.array([3, 3, 2, 2], dtype=np.int32)
        conflict_count = np.array([0, 0, 0, 1], dtype=np.int32)

        bank = finalize_votes(per_part_votes, visible_vote_count, conflict_count)

        self.assertEqual(bank["part_label"].dtype, np.int16)
        self.assertEqual(bank["confidence"].dtype, np.float32)
        self.assertEqual(bank["vote_count"].dtype, np.int16)
        self.assertEqual(bank["per_part_votes"].dtype, np.int16)
        self.assertEqual(bank["visible_vote_count"].dtype, np.int16)
        self.assertEqual(bank["conflict_count"].dtype, np.int16)
        self.assertEqual(bank["part_label"].tolist(), [0, 1, -1, 1])
        self.assertTrue(np.allclose(bank["confidence"], [1.0, 2.0 / 3.0, 0.0, 0.5]))
        self.assertEqual(bank["vote_count"].tolist(), [2, 3, 0, 2])
        self.assertEqual(bank["conflict_count"].tolist(), [0, 0, 0, 1])

    def test_finalize_votes_exports_vote_normalized_semantic_probs(self):
        per_part_votes = np.array(
            [
                [3, 1, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        bank = finalize_votes(
            per_part_votes,
            np.array([4, 4, 1], dtype=np.int32),
            np.array([0, 1, 0], dtype=np.int32),
        )

        self.assertEqual(bank["semantic_probs"].dtype, np.float32)
        self.assertTrue(np.allclose(bank["semantic_probs"][0], [0.75, 0.25, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(bank["semantic_probs"][1], [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]))
        self.assertTrue(np.allclose(bank["semantic_probs"][2], np.zeros((len(PART_NAMES),), dtype=np.float32)))

    def test_schema_validation_rejects_wrong_part_vote_shape(self):
        arrays = {
            "schema_version": np.array(1, dtype=np.int32),
            "point_count": np.array(2, dtype=np.int64),
            "part_names": np.asarray(PART_NAMES, dtype="U16"),
            "part_label": np.array([0, -1], dtype=np.int16),
            "confidence": np.array([1.0, 0.0], dtype=np.float32),
            "vote_count": np.array([1, 0], dtype=np.int16),
            "per_part_votes": np.zeros((2, 5), dtype=np.int16),
            "visible_vote_count": np.array([1, 0], dtype=np.int16),
            "conflict_count": np.array([0, 0], dtype=np.int16),
            "source_checkpoint": np.array("/tmp/ckpt.pth"),
            "source_asset_root": np.array("/tmp/assets"),
            "source_iteration": np.array(1, dtype=np.int64),
        }

        with self.assertRaisesRegex(ValueError, "per_part_votes"):
            validate_part_label_bank_arrays(arrays)

    def test_summary_and_preview_outputs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            finalized = finalize_votes(
                np.array(
                    [
                        [2, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int32,
                ),
                np.array([2, 1, 1], dtype=np.int32),
                np.array([0, 1, 0], dtype=np.int32),
            )
            summary = summarize_part_label_bank(finalized)
            summary_path = tmp_path / "summary.json"
            ply_path = tmp_path / "preview.ply"

            write_summary_json(summary_path, summary)
            write_preview_ply(
                ply_path,
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                finalized["part_label"],
            )

            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["hair_count"], 1)
            self.assertEqual(loaded_summary["face_count"], 1)
            self.assertEqual(loaded_summary["unknown_count"], 1)
            self.assertEqual(loaded_summary["conflict_stats"]["conflicted_point_count"], 1)
            text = ply_path.read_text(encoding="utf-8")
            self.assertIn("element vertex 3", text)
            self.assertIn("property uchar red", text)

    def test_accumulate_projected_votes_uses_visibility_radii_foreground_and_conflicts(self):
        xy = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
                [4.0, 1.0],
                [5.0, 1.0],
                [8.0, 8.0],
            ],
            dtype=torch.float32,
        )
        proj_valid = torch.tensor([True, True, True, True, True, False])
        visibility_filter = torch.tensor([True, True, False, True, True, True])
        radii = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
        foreground = np.ones((4, 6), dtype=np.float32)
        valid_mask = np.ones((4, 6), dtype=np.float32)
        valid_mask[1, 4] = 0.0
        part_masks = {name: np.zeros((4, 6), dtype=np.float32) for name in PART_NAMES}
        part_masks["hair"][1, 1] = 0.8
        part_masks["face"][1, 2] = 0.7
        part_masks["hair"][1, 5] = 0.6
        part_masks["face"][1, 5] = 0.9

        per_part_votes = np.zeros((6, len(PART_NAMES)), dtype=np.int32)
        visible_vote_count = np.zeros((6,), dtype=np.int32)
        conflict_count = np.zeros((6,), dtype=np.int32)

        stats = accumulate_projected_votes(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility_filter,
            radii=radii,
            image_size=(6, 4),
            part_masks=part_masks,
            foreground_mask=foreground,
            valid_mask=valid_mask,
            per_part_votes=per_part_votes,
            visible_vote_count=visible_vote_count,
            conflict_count=conflict_count,
            mask_threshold=0.5,
        )

        self.assertEqual(per_part_votes[:, 0].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(per_part_votes[:, 1].tolist(), [0, 1, 0, 0, 1, 0])
        self.assertEqual(visible_vote_count.tolist(), [1, 1, 0, 0, 1, 0])
        self.assertEqual(conflict_count.tolist(), [0, 0, 0, 0, 1, 0])
        self.assertEqual(stats["visible_projected_count"], 4)
        self.assertEqual(stats["valid_mask_count"], 3)
        self.assertEqual(stats["part_vote_count"], 3)
        self.assertEqual(stats["conflict_count"], 1)

    def test_accumulate_projected_votes_can_use_footprint_hit_ratio(self):
        xy = torch.tensor([[3.0, 3.0], [1.0, 1.0]], dtype=torch.float32)
        proj_valid = torch.tensor([True, True])
        visibility_filter = torch.tensor([True, True])
        radii = torch.tensor([2.0, 2.0])
        foreground = np.ones((7, 7), dtype=np.float32)
        valid_mask = np.ones((7, 7), dtype=np.float32)
        part_masks = {name: np.zeros((7, 7), dtype=np.float32) for name in PART_NAMES}
        part_masks["lower"][2:5, 2:5] = 1.0
        part_masks["shoes"][0:3, 0:3] = 1.0
        per_part_votes = np.zeros((2, len(PART_NAMES)), dtype=np.int32)
        visible_vote_count = np.zeros((2,), dtype=np.int32)
        conflict_count = np.zeros((2,), dtype=np.int32)

        stats = accumulate_projected_votes(
            xy=xy,
            proj_valid=proj_valid,
            visibility_filter=visibility_filter,
            radii=radii,
            image_size=(7, 7),
            part_masks=part_masks,
            foreground_mask=foreground,
            valid_mask=valid_mask,
            per_part_votes=per_part_votes,
            visible_vote_count=visible_vote_count,
            conflict_count=conflict_count,
            mask_threshold=0.5,
            footprint_mode="footprint",
            footprint_radius_scale=1.0,
            min_footprint_radius=1,
            max_footprint_radius=3,
            min_footprint_hit_ratio=0.50,
        )

        self.assertEqual(per_part_votes[:, PART_NAMES.index("lower")].tolist(), [1, 0])
        self.assertEqual(per_part_votes[:, PART_NAMES.index("shoes")].tolist(), [0, 1])
        self.assertEqual(visible_vote_count.tolist(), [1, 1])
        self.assertEqual(stats["part_vote_count"], 2)
        self.assertEqual(stats["footprint_vote_count"], 2)
        self.assertGreater(stats["mean_winning_footprint_hit_ratio"], 0.5)

    def test_accumulate_projected_votes_can_filter_occluded_points_by_depth(self):
        xy = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        depth = torch.tensor([1.0, 2.0, 1.4], dtype=torch.float32)
        proj_valid = torch.tensor([True, True, True])
        visibility_filter = torch.tensor([True, True, True])
        radii = torch.tensor([1.0, 1.0, 1.0])
        foreground = np.ones((3, 4), dtype=np.float32)
        valid_mask = np.ones((3, 4), dtype=np.float32)
        part_masks = {name: np.zeros((3, 4), dtype=np.float32) for name in PART_NAMES}
        part_masks["hair"][1, 1] = 1.0
        part_masks["face"][1, 2] = 1.0
        per_part_votes = np.zeros((3, len(PART_NAMES)), dtype=np.int32)
        visible_vote_count = np.zeros((3,), dtype=np.int32)
        conflict_count = np.zeros((3,), dtype=np.int32)

        stats = accumulate_projected_votes(
            xy=xy,
            depth=depth,
            depth_margin=0.05,
            proj_valid=proj_valid,
            visibility_filter=visibility_filter,
            radii=radii,
            image_size=(4, 3),
            part_masks=part_masks,
            foreground_mask=foreground,
            valid_mask=valid_mask,
            per_part_votes=per_part_votes,
            visible_vote_count=visible_vote_count,
            conflict_count=conflict_count,
            mask_threshold=0.5,
        )

        self.assertEqual(visible_vote_count.tolist(), [1, 0, 1])
        self.assertEqual(per_part_votes[:, 0].tolist(), [1, 0, 0])
        self.assertEqual(per_part_votes[:, 1].tolist(), [0, 0, 1])
        self.assertEqual(stats["depth_visible_count"], 2)
        self.assertEqual(stats["part_vote_count"], 2)

    def test_accumulate_projected_votes_fails_alignment_gate_for_low_part_hit_ratio(self):
        xy = torch.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=torch.float32)
        proj_valid = torch.tensor([True, True])
        visibility_filter = torch.tensor([True, True])
        radii = torch.tensor([1.0, 1.0])
        foreground = np.ones((3, 4), dtype=np.float32)
        valid_mask = np.ones((3, 4), dtype=np.float32)
        part_masks = {name: np.zeros((3, 4), dtype=np.float32) for name in PART_NAMES}
        per_part_votes = np.zeros((2, len(PART_NAMES)), dtype=np.int32)
        visible_vote_count = np.zeros((2,), dtype=np.int32)
        conflict_count = np.zeros((2,), dtype=np.int32)

        with self.assertRaisesRegex(RuntimeError, "part hit ratio"):
            accumulate_projected_votes(
                xy=xy,
                proj_valid=proj_valid,
                visibility_filter=visibility_filter,
                radii=radii,
                image_size=(4, 3),
                part_masks=part_masks,
                foreground_mask=foreground,
                valid_mask=valid_mask,
                per_part_votes=per_part_votes,
                visible_vote_count=visible_vote_count,
                conflict_count=conflict_count,
                mask_threshold=0.5,
                min_part_hit_ratio=0.5,
                view_name="fake_view",
            )

    def test_part_label_bank_save_load_validates_schema_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part_label_bank.npz"
            finalized = finalize_votes(
                np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.int32),
                np.array([1, 0], dtype=np.int32),
                np.array([0, 0], dtype=np.int32),
            )
            save_part_label_bank(
                out,
                **finalized,
                source_checkpoint="/tmp/ckpt.pth",
                source_asset_root="/tmp/assets",
                source_iteration=141160,
            )
            loaded = load_part_label_bank(out)

            validate_part_label_bank_arrays(loaded)
            self.assertEqual(int(loaded["schema_version"]), 1)
            self.assertEqual(int(loaded["point_count"]), 2)
            self.assertEqual(loaded["part_names"].tolist(), list(PART_NAMES))
            self.assertEqual(str(loaded["source_checkpoint"]), "/tmp/ckpt.pth")
            self.assertEqual(str(loaded["source_asset_root"]), "/tmp/assets")
            self.assertEqual(int(loaded["source_iteration"]), 141160)

    def test_finalize_trained_semantic_probs_remaps_training_order_without_votes(self):
        training_names = ("hair", "face", "skin", "upper", "lower", "shoes")
        probs = np.array(
            [
                [0.02, 0.03, 0.90, 0.03, 0.01, 0.01],
                [0.01, 0.02, 0.03, 0.88, 0.04, 0.02],
                [0.10, 0.11, 0.12, 0.13, 0.40, 0.14],
            ],
            dtype=np.float32,
        )

        bank = finalize_trained_semantic_probs(probs, training_names)

        self.assertEqual(bank["source_type"], "trained_semantic_asset_probs")
        self.assertEqual(bank["semantic_probs"].dtype, np.float32)
        self.assertEqual(bank["semantic_probs"].shape, (3, len(PART_NAMES)))
        self.assertTrue(np.allclose(bank["semantic_probs"][0], [0.02, 0.03, 0.03, 0.01, 0.01, 0.90]))
        self.assertEqual(bank["part_label"].tolist(), [5, 2, 3])
        self.assertTrue(np.allclose(bank["confidence"], [0.90, 0.88, 0.40]))
        self.assertEqual(bank["vote_count"].tolist(), [0, 0, 0])
        self.assertEqual(bank["visible_vote_count"].tolist(), [0, 0, 0])
        self.assertEqual(bank["conflict_count"].tolist(), [0, 0, 0])
        self.assertTrue(np.all(bank["per_part_votes"] == 0))

    def test_finalize_trained_semantic_probs_can_mark_invisible_points_unknown(self):
        training_names = ("hair", "face", "skin", "upper", "lower", "shoes")
        probs = np.array(
            [
                [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.95, 0.01, 0.01, 0.01, 0.02, 0.01],
                [0.02, 0.03, 0.90, 0.03, 0.01, 0.01],
            ],
            dtype=np.float32,
        )

        bank = finalize_trained_semantic_probs(probs, training_names, valid_mask=np.array([True, False, True]))

        self.assertEqual(bank["part_label"].tolist(), [0, -1, 5])
        self.assertTrue(np.allclose(bank["confidence"], [0.90, 0.0, 0.90]))
        self.assertTrue(np.all(bank["semantic_probs"][1] == 0.0))
        self.assertEqual(summarize_part_label_bank(bank)["unknown_count"], 1)

    def test_compute_semantic_margin_returns_top1_minus_top2(self):
        probs = np.array(
            [
                [0.70, 0.20, 0.10, 0.00, 0.00, 0.00],
                [0.32, 0.31, 0.25, 0.12, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            ],
            dtype=np.float32,
        )

        margin = compute_semantic_margin(probs)

        self.assertEqual(margin.dtype, np.float32)
        self.assertTrue(np.allclose(margin, [0.50, 0.01, 0.00]))

    def test_compute_semantic_reliable_mask_uses_confidence_margin_opacity_and_labels(self):
        part_label = np.array([0, 1, 2, 3, -1], dtype=np.int16)
        confidence = np.array([0.90, 0.64, 0.80, 0.75, 0.99], dtype=np.float32)
        semantic_margin = np.array([0.30, 0.30, 0.19, 0.25, 0.50], dtype=np.float32)
        opacity = np.array([0.010, 0.010, 0.010, 0.004, 0.010], dtype=np.float32)

        reliable = compute_semantic_reliable_mask(
            part_label=part_label,
            confidence=confidence,
            semantic_margin=semantic_margin,
            opacity=opacity,
            min_confidence=0.65,
            min_margin=0.20,
            min_opacity=0.005,
        )

        self.assertEqual(reliable.dtype, np.uint8)
        self.assertEqual(reliable.tolist(), [1, 0, 0, 0, 0])

    def test_apply_reliable_label_mask_adds_optional_export_fields_without_changing_part_label(self):
        probs = np.array(
            [
                [0.80, 0.10, 0.10, 0.00, 0.00, 0.00],
                [0.34, 0.33, 0.33, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.76, 0.24, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        bank = finalize_trained_semantic_probs(probs, PART_NAMES)

        stats = apply_reliable_label_mask(
            bank,
            opacity=np.array([0.010, 0.010, 0.004], dtype=np.float32),
            min_confidence=0.65,
            min_margin=0.20,
            min_opacity=0.005,
        )

        self.assertEqual(bank["part_label"].tolist(), [0, 0, 2])
        self.assertEqual(bank["editable_label"].tolist(), [0, -1, -1])
        self.assertEqual(bank["reliable_mask"].dtype, np.uint8)
        self.assertEqual(bank["reliable_mask"].tolist(), [1, 0, 0])
        self.assertTrue(np.allclose(bank["semantic_margin"], [0.70, 0.01, 0.52]))
        self.assertEqual(stats["reliable_count"], 1)
        self.assertEqual(stats["unreliable_count"], 2)
        self.assertEqual(stats["low_margin_count"], 1)
        self.assertEqual(stats["low_opacity_count"], 1)

    def test_compute_soft_edit_weights_combines_probs_confidence_margin_and_reliability(self):
        semantic_probs = np.array(
            [
                [0.80, 0.20, 0.00, 0.00, 0.00, 0.00],
                [0.30, 0.70, 0.00, 0.00, 0.00, 0.00],
            ],
            dtype=np.float32,
        )
        confidence = np.array([0.90, 0.50], dtype=np.float32)
        semantic_margin = np.array([0.60, 0.20], dtype=np.float32)
        reliable_mask = np.array([1, 0], dtype=np.uint8)

        weights = compute_soft_edit_weights(
            semantic_probs=semantic_probs,
            confidence=confidence,
            semantic_margin=semantic_margin,
            reliable_mask=reliable_mask,
            reliable_floor=0.25,
            confidence_power=2.0,
            margin_power=1.0,
        )

        expected = semantic_probs.copy()
        expected[0] *= 0.90**2 * 0.60
        expected[1] *= 0.50**2 * 0.20 * 0.25
        self.assertEqual(weights.dtype, np.float32)
        self.assertEqual(weights.shape, (2, len(PART_NAMES)))
        self.assertTrue(np.allclose(weights, expected))

    def test_evidence_calibrated_soft_weights_penalize_stable_outer_without_deleting_target(self):
        lower = PART_NAMES.index("lower")
        weights = np.zeros((4, len(PART_NAMES)), dtype=np.float32)
        weights[:, lower] = np.array([0.80, 0.80, 0.80, 0.10], dtype=np.float32)
        target_ratio = np.zeros_like(weights)
        outer_ratio = np.zeros_like(weights)
        support = np.zeros_like(weights, dtype=np.int16)
        conflict_ratio = np.zeros_like(weights)
        target_ratio[:, lower] = np.array([0.90, 0.15, 0.80, 0.05], dtype=np.float32)
        outer_ratio[:, lower] = np.array([0.10, 0.85, 0.20, 0.95], dtype=np.float32)
        support[:, lower] = np.array([12, 12, 2, 12], dtype=np.int16)
        conflict_ratio[:, lower] = np.array([0.00, 0.00, 0.00, 0.50], dtype=np.float32)

        calibrated, stats = compute_evidence_calibrated_soft_edit_weights(
            soft_edit_weights=weights,
            footprint_target_ratio=target_ratio,
            footprint_outer_ratio=outer_ratio,
            view_support_count=support,
            conflict_ratio=conflict_ratio,
            parts=("lower",),
            min_support_views=5,
            target_retention_floor=0.60,
            outer_penalty_power=1.0,
            conflict_penalty_power=1.0,
        )

        self.assertGreaterEqual(calibrated[0, lower], 0.80 * 0.60)
        self.assertLess(calibrated[1, lower], weights[1, lower] * 0.40)
        self.assertAlmostEqual(float(calibrated[2, lower]), float(weights[2, lower]))
        self.assertLess(calibrated[3, lower], weights[3, lower])
        self.assertEqual(stats["parts"]["lower"]["calibrated_count"], 3)

    def test_part_label_bank_save_load_roundtrips_reliability_fields(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part_label_bank.npz"
            probs = np.array(
                [
                    [0.80, 0.10, 0.10, 0.00, 0.00, 0.00],
                    [0.34, 0.33, 0.33, 0.00, 0.00, 0.00],
                ],
                dtype=np.float32,
            )
            bank = finalize_trained_semantic_probs(probs, PART_NAMES)
            apply_reliable_label_mask(
                bank,
                opacity=np.array([0.010, 0.010], dtype=np.float32),
                min_confidence=0.65,
                min_margin=0.20,
                min_opacity=0.005,
            )

            save_part_label_bank(
                out,
                **bank,
                source_checkpoint="/tmp/ckpt.pth",
                source_asset_root="/tmp/assets",
                source_iteration=141160,
            )
            loaded = load_part_label_bank(out)

            self.assertEqual(loaded["semantic_margin"].dtype, np.float32)
            self.assertEqual(loaded["reliable_mask"].dtype, np.uint8)
            self.assertEqual(loaded["editable_label"].dtype, np.int16)
            self.assertEqual(loaded["editable_label"].tolist(), [0, -1])

    def test_part_label_bank_save_load_roundtrips_soft_edit_weights(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part_label_bank.npz"
            probs = np.array(
                [
                    [0.80, 0.10, 0.10, 0.00, 0.00, 0.00],
                    [0.34, 0.33, 0.33, 0.00, 0.00, 0.00],
                ],
                dtype=np.float32,
            )
            bank = finalize_trained_semantic_probs(probs, PART_NAMES)
            apply_reliable_label_mask(
                bank,
                opacity=np.array([0.010, 0.010], dtype=np.float32),
                min_confidence=0.65,
                min_margin=0.20,
                min_opacity=0.005,
            )
            bank["soft_edit_weights"] = compute_soft_edit_weights(
                semantic_probs=bank["semantic_probs"],
                confidence=bank["confidence"],
                semantic_margin=bank["semantic_margin"],
                reliable_mask=bank["reliable_mask"],
            )

            save_part_label_bank(
                out,
                **bank,
                source_checkpoint="/tmp/ckpt.pth",
                source_asset_root="/tmp/assets",
                source_iteration=141160,
            )
            loaded = load_part_label_bank(out)

            self.assertEqual(loaded["soft_edit_weights"].dtype, np.float32)
            self.assertEqual(loaded["soft_edit_weights"].shape, (2, len(PART_NAMES)))
            self.assertTrue(np.allclose(loaded["soft_edit_weights"], bank["soft_edit_weights"]))

    def test_apply_neighbor_reliable_fill_only_fills_unknowns_with_same_label_majority(self):
        bank = {
            "part_label": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16),
            "editable_label": np.array([0, 0, 0, -1, 1, 1, 1, -1], dtype=np.int16),
            "reliable_mask": np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.uint8),
        }
        xyz = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.00, 0.10, 0.00],
                [0.05, 0.05, 0.00],
                [1.00, 0.00, 0.00],
                [1.10, 0.00, 0.00],
                [1.00, 0.10, 0.00],
                [0.06, 0.04, 0.00],
            ],
            dtype=np.float32,
        )

        stats = apply_neighbor_reliable_fill(
            bank,
            xyz=xyz,
            k=3,
            min_reliable_neighbors=3,
            majority_ratio=0.70,
            min_candidate_confidence=0.0,
            confidence=np.ones((8,), dtype=np.float32),
        )

        self.assertEqual(bank["part_label"].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(bank["editable_label"].tolist(), [0, 0, 0, 0, 1, 1, 1, -1])
        self.assertEqual(bank["neighbor_fill_mask"].dtype, np.uint8)
        self.assertEqual(bank["neighbor_fill_mask"].tolist(), [0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(stats["filled_count"], 1)
        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(stats["rejected_label_mismatch_count"], 1)

    def test_part_label_bank_save_load_roundtrips_neighbor_fill_mask(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "part_label_bank.npz"
            probs = np.array(
                [
                    [0.80, 0.10, 0.10, 0.00, 0.00, 0.00],
                    [0.34, 0.33, 0.33, 0.00, 0.00, 0.00],
                ],
                dtype=np.float32,
            )
            bank = finalize_trained_semantic_probs(probs, PART_NAMES)
            apply_reliable_label_mask(
                bank,
                opacity=np.array([0.010, 0.010], dtype=np.float32),
                min_confidence=0.65,
                min_margin=0.20,
                min_opacity=0.005,
            )
            bank["neighbor_fill_mask"] = np.array([0, 1], dtype=np.uint8)

            save_part_label_bank(
                out,
                **bank,
                source_checkpoint="/tmp/ckpt.pth",
                source_asset_root="/tmp/assets",
                source_iteration=141160,
            )
            loaded = load_part_label_bank(out)

            self.assertEqual(loaded["neighbor_fill_mask"].dtype, np.uint8)
            self.assertEqual(loaded["neighbor_fill_mask"].tolist(), [0, 1])

    def test_apply_face_label_guard_reassigns_weak_or_oversized_face_points(self):
        probs = np.array(
            [
                [0.10, 0.80, 0.00, 0.00, 0.00, 0.10],
                [0.38, 0.62, 0.00, 0.00, 0.00, 0.00],
                [0.10, 0.75, 0.00, 0.00, 0.00, 0.15],
                [0.15, 0.83, 0.00, 0.00, 0.00, 0.02],
                [0.00, 0.86, 0.00, 0.00, 0.00, 0.14],
            ],
            dtype=np.float32,
        )
        bank = finalize_trained_semantic_probs(probs, PART_NAMES)
        scale_max = np.array([0.02, 0.02, 0.02, 0.18, 0.19], dtype=np.float32)

        stats = apply_face_label_guard(
            bank,
            min_prob=0.70,
            min_margin=0.15,
            max_scale=0.12,
            scale_max=scale_max,
            oversized_action="second",
        )

        self.assertEqual(bank["part_label"].tolist(), [1, 0, 1, 0, 5])
        self.assertTrue(np.allclose(bank["confidence"], [0.80, 0.38, 0.75, 0.15, 0.14]))
        self.assertEqual(stats["face_initial_count"], 5)
        self.assertEqual(stats["face_final_count"], 2)
        self.assertEqual(stats["reassigned_to_hair_count"], 2)
        self.assertEqual(stats["reassigned_to_skin_count"], 1)
        self.assertEqual(stats["low_prob_count"], 1)
        self.assertEqual(stats["low_margin_count"], 0)
        self.assertEqual(stats["oversized_count"], 2)

    def test_apply_lower_label_guard_reassigns_high_upper_body_lower_points(self):
        probs = np.array(
            [
                [0.0, 0.0, 0.24, 0.75, 0.0, 0.01],
                [0.0, 0.0, 0.30, 0.68, 0.0, 0.02],
                [0.0, 0.0, 0.24, 0.75, 0.0, 0.01],
                [0.0, 0.0, 0.05, 0.92, 0.0, 0.03],
            ],
            dtype=np.float32,
        )
        bank = finalize_trained_semantic_probs(probs, PART_NAMES)
        xyz = np.array(
            [
                [0.02, 0.33, 0.01],
                [0.01, 0.31, 0.02],
                [0.00, 0.20, 0.01],
                [0.50, 0.34, 0.01],
            ],
            dtype=np.float32,
        )

        stats = apply_lower_label_guard(
            bank,
            xyz=xyz,
            high_y_threshold=0.30,
            max_abs_x=0.35,
            max_abs_z=0.18,
            target_second_name="upper",
        )

        self.assertEqual(bank["part_label"].tolist(), [2, 2, 3, 3])
        self.assertTrue(np.allclose(bank["confidence"], [0.24, 0.30, 0.75, 0.92]))
        self.assertEqual(stats["lower_initial_count"], 4)
        self.assertEqual(stats["reassigned_to_upper_count"], 2)
        self.assertEqual(stats["lower_final_count"], 2)

    def test_build_output_manifest_records_source_and_depth_settings(self):
        manifest = build_output_manifest(
            checkpoint="/tmp/live/ckpt141160.pth",
            config="/tmp/live/config.yaml",
            asset_root="/tmp/assets/semantic_editable_assets",
            output="/tmp/out/part_label_bank.npz",
            summary_json="/tmp/out/summary.json",
            preview_ply="/tmp/out/preview.ply",
            point_count=46801,
            source_iteration=141160,
            processed_views=3,
            depth_margin=0.02,
            min_part_hit_ratio=0.25,
        )

        self.assertEqual(manifest["source_checkpoint"], "/tmp/live/ckpt141160.pth")
        self.assertEqual(manifest["source_config"], "/tmp/live/config.yaml")
        self.assertEqual(manifest["source_asset_root"], "/tmp/assets/semantic_editable_assets")
        self.assertEqual(manifest["point_count"], 46801)
        self.assertEqual(manifest["source_iteration"], 141160)
        self.assertEqual(manifest["processed_views"], 3)
        self.assertEqual(manifest["depth_margin"], 0.02)
        self.assertEqual(manifest["min_part_hit_ratio"], 0.25)

    def test_build_output_manifest_can_record_soft_edit_field(self):
        manifest = build_output_manifest(
            checkpoint="/tmp/live/ckpt141160.pth",
            config="/tmp/live/config.yaml",
            asset_root="/tmp/assets/semantic_editable_assets",
            output="/tmp/out/part_label_bank.npz",
            summary_json="/tmp/out/summary.json",
            preview_ply="/tmp/out/preview.ply",
            point_count=46801,
            source_iteration=141160,
            processed_views=3,
            depth_margin=0.02,
            min_part_hit_ratio=0.25,
            soft_edit_weight_field="soft_edit_weights",
        )

        self.assertEqual(manifest["soft_edit_weight_field"], "soft_edit_weights")
        self.assertEqual(manifest["soft_edit_part_names"], list(PART_NAMES))

    def test_parse_args_accepts_voting_and_soft_edit_modes(self):
        import sys
        from unittest import mock

        from tools.semantic_viewer import build_part_label_bank

        argv = [
            "build_part_label_bank.py",
            "--checkpoint",
            "/tmp/ckpt.pth",
            "--asset-root",
            "/tmp/assets",
            "--output",
            "/tmp/out.npz",
            "--label-bank-source",
            "projected-2d-voting",
            "--vote-footprint-mode",
            "footprint",
            "--vote-use-render-radii",
            "--export-soft-edit-weights",
            "--soft-edit-reliable-floor",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = build_part_label_bank.parse_args()

        self.assertEqual(args.label_bank_source, "projected-2d-voting")
        self.assertEqual(args.vote_footprint_mode, "footprint")
        self.assertTrue(args.vote_use_render_radii)
        self.assertTrue(args.export_soft_edit_weights)
        self.assertEqual(args.soft_edit_reliable_floor, 0.25)


if __name__ == "__main__":
    unittest.main()
