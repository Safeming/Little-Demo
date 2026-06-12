import unittest

import numpy as np

from tools.analyze_projected_soft_edit_leakage import (
    compute_footprint_leakage_for_selection,
    compute_projected_leakage_for_selection,
    make_boundary_band,
    resolve_soft_edit_weights,
)


class ProjectedSoftEditLeakageTests(unittest.TestCase):
    def test_make_boundary_band_marks_target_edges_without_filling_interior(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[1:4, 1:4] = 1.0

        band = make_boundary_band(mask, radius=1)

        self.assertEqual(band.dtype, np.bool_)
        self.assertTrue(band[1, 1])
        self.assertTrue(band[0, 1])
        self.assertFalse(band[2, 2])
        self.assertFalse(band[0, 0])

    def test_compute_projected_leakage_for_selection_counts_target_outer_and_boundary_activation(self):
        target_mask = np.zeros((5, 5), dtype=np.float32)
        target_mask[1:4, 1:4] = 1.0
        valid_mask = np.ones((5, 5), dtype=np.float32)
        px = np.array([2, 4, 1, 0], dtype=np.int64)
        py = np.array([2, 2, 1, 0], dtype=np.int64)
        selected = np.array([True, True, False, True])
        weights = np.array([1.0, 0.8, 0.5, 0.4], dtype=np.float32)

        row = compute_projected_leakage_for_selection(
            part="face",
            mode="soft",
            view_name="c00_f000001",
            px=px,
            py=py,
            selected=selected,
            weights=weights,
            target_mask=target_mask,
            valid_mask=valid_mask,
            boundary_radius=1,
            mask_threshold=0.5,
        )

        self.assertEqual(row["selected_count"], 3)
        self.assertAlmostEqual(row["target_activation"], 1.0)
        self.assertAlmostEqual(row["outer_activation"], 1.2)
        self.assertAlmostEqual(row["leakage_ratio"], 1.2)
        self.assertAlmostEqual(row["target_coverage"], 1.0 / 9.0)
        self.assertGreater(row["boundary_activation"], 0.0)

    def test_compute_footprint_leakage_for_selection_uses_disk_overlap_not_center_only(self):
        xy = np.array([[1.0, 2.0]], dtype=np.float32)
        selected = np.array([True])
        weights = np.array([1.0], dtype=np.float32)
        radii = np.array([1.0], dtype=np.float32)
        target_mask = np.zeros((5, 5), dtype=np.float32)
        target_mask[2, 2] = 1.0
        valid_mask = np.ones((5, 5), dtype=np.float32)

        row = compute_footprint_leakage_for_selection(
            part="shoes",
            mode="soft_footprint",
            view_name="view0",
            xy=xy,
            selected=selected,
            weights=weights,
            radii=radii,
            target_mask=target_mask,
            valid_mask=valid_mask,
            mask_threshold=0.5,
            footprint_radius_scale=1.0,
            min_footprint_radius=1,
            max_footprint_radius=1,
        )

        self.assertEqual(row["selected_count"], 1)
        self.assertEqual(row["observed_footprint_count"], 1)
        self.assertAlmostEqual(row["target_activation"], 0.2)
        self.assertAlmostEqual(row["outer_activation"], 0.8)
        self.assertAlmostEqual(row["leakage_ratio"], 4.0)

    def test_compute_footprint_leakage_for_selection_accepts_soft_target_mask(self):
        xy = np.array([[1.0, 2.0]], dtype=np.float32)
        selected = np.array([True])
        weights = np.array([1.0], dtype=np.float32)
        radii = np.array([1.0], dtype=np.float32)
        target_mask = np.zeros((5, 5), dtype=np.float32)
        target_mask[2, 2] = 1.0
        target_mask[2, 1] = 0.5
        target_mask[1, 1] = 0.5
        target_mask[3, 1] = 0.5
        valid_mask = np.ones((5, 5), dtype=np.float32)

        row = compute_footprint_leakage_for_selection(
            part="shoes",
            mode="soft_footprint",
            view_name="view0",
            xy=xy,
            selected=selected,
            weights=weights,
            radii=radii,
            target_mask=target_mask,
            valid_mask=valid_mask,
            mask_threshold=0.5,
            footprint_radius_scale=1.0,
            min_footprint_radius=1,
            max_footprint_radius=1,
            use_soft_target=True,
        )

        self.assertAlmostEqual(row["target_activation"], 0.5)
        self.assertAlmostEqual(row["outer_activation"], 0.5)
        self.assertAlmostEqual(row["leakage_ratio"], 1.0)

    def test_summarize_rows_aggregates_per_part_and_summary(self):
        from tools.analyze_projected_soft_edit_leakage import summarize_rows

        rows = [
            {
                "part": "face",
                "mode": "hard",
                "view": "v0",
                "target_activation": 2.0,
                "outer_activation": 1.0,
                "boundary_activation": 0.5,
                "selected_count": 3,
            },
            {
                "part": "face",
                "mode": "soft",
                "view": "v0",
                "target_activation": 2.0,
                "outer_activation": 0.2,
                "boundary_activation": 0.1,
                "selected_count": 2,
            },
        ]

        result = summarize_rows(rows, soft_threshold=0.25)

        self.assertAlmostEqual(result["summary"]["mean_hard_leakage_ratio"], 0.5)
        self.assertAlmostEqual(result["summary"]["mean_soft_leakage_ratio"], 0.1)
        self.assertEqual(result["per_part"][0]["part"], "face")
        self.assertAlmostEqual(result["per_part"][0]["hard_leakage_ratio"], 0.5)
        self.assertAlmostEqual(result["per_part"][0]["soft_leakage_ratio"], 0.1)

    def test_summarize_rows_includes_footprint_modes_when_present(self):
        from tools.analyze_projected_soft_edit_leakage import summarize_rows

        rows = [
            {"part": "shoes", "mode": "hard", "view": "v0", "target_activation": 2.0, "outer_activation": 1.0, "boundary_activation": 0.0, "selected_count": 2},
            {"part": "shoes", "mode": "soft", "view": "v0", "target_activation": 2.0, "outer_activation": 0.5, "boundary_activation": 0.0, "selected_count": 2},
            {"part": "shoes", "mode": "hard_footprint", "view": "v0", "target_activation": 4.0, "outer_activation": 1.0, "boundary_activation": 0.0, "selected_count": 2},
            {"part": "shoes", "mode": "soft_footprint", "view": "v0", "target_activation": 4.0, "outer_activation": 0.4, "boundary_activation": 0.0, "selected_count": 2},
        ]

        result = summarize_rows(rows, soft_threshold=0.2)

        self.assertAlmostEqual(result["per_part"][0]["soft_footprint_leakage_ratio"], 0.1)
        self.assertAlmostEqual(result["per_part"][0]["hard_footprint_leakage_ratio"], 0.25)
        self.assertAlmostEqual(result["summary"]["mean_soft_footprint_leakage_ratio"], 0.1)

    def test_resolve_soft_edit_weights_falls_back_to_one_hot_labels(self):
        bank = {
            "part_label": np.array([0, 3, -1], dtype=np.int16),
            "editable_label": np.array([1, 3, -1], dtype=np.int16),
        }

        weights, source = resolve_soft_edit_weights(bank, point_count=3)

        self.assertEqual(source, "editable_label_one_hot_fallback")
        self.assertEqual(weights.shape, (3, 6))
        self.assertAlmostEqual(float(weights[0, 1]), 1.0)
        self.assertAlmostEqual(float(weights[1, 3]), 1.0)
        self.assertAlmostEqual(float(weights[2].sum()), 0.0)

    def test_write_reports_creates_summary_per_part_and_per_view(self):
        import json
        import tempfile
        from pathlib import Path

        from tools.analyze_projected_soft_edit_leakage import write_reports

        result = {
            "summary": {"part_count": 1, "mean_soft_leakage_ratio": 0.1},
            "per_part": [{"part": "face", "soft_leakage_ratio": 0.1}],
            "per_view": [{"part": "face", "mode": "soft", "view": "v0", "leakage_ratio": 0.1}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_reports(out, result)
            self.assertEqual(json.loads((out / "summary.json").read_text())["part_count"], 1)
            self.assertIn("part,soft_leakage_ratio", (out / "per_part.csv").read_text())
            self.assertIn("part,mode,view,leakage_ratio", (out / "per_view.csv").read_text())


if __name__ == "__main__":
    unittest.main()
