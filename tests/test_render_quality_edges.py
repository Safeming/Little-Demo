import unittest

import numpy as np

from tools.analyze_render_quality_edges import analyze_one, summarize_records


class RenderQualityEdgesTest(unittest.TestCase):
    def test_color_bias_and_boundary_error_are_reported(self):
        h, w = 32, 32
        gt = np.zeros((h, w, 3), dtype=np.uint8)
        render = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        mask[8:24, 8:24] = 255
        gt[8:24, 8:24] = [120, 140, 160]
        render[8:24, 8:24] = [100, 120, 140]
        render[7:25, 7:25] = np.maximum(render[7:25, 7:25], [20, 20, 20])

        record = analyze_one(render, gt, mask, band_width=2)

        self.assertGreater(record["foreground_l1"], 0.0)
        self.assertGreater(record["boundary_l1"], 0.0)
        self.assertLess(record["render_mean_luma_fg"], record["gt_mean_luma_fg"])
        self.assertGreaterEqual(record["halo_luma_outside"], 0.0)

    def test_summary_contains_stable_keys(self):
        records = [
            {
                "foreground_l1": 0.1,
                "boundary_l1": 0.2,
                "interior_l1": 0.05,
                "edge_symmetric_dist_px": 1.0,
                "render_minus_gt_luma_fg": -0.02,
                "halo_luma_outside": 0.03,
                "hard_score": 0.25,
            },
            {
                "foreground_l1": 0.2,
                "boundary_l1": 0.3,
                "interior_l1": 0.10,
                "edge_symmetric_dist_px": 2.0,
                "render_minus_gt_luma_fg": -0.03,
                "halo_luma_outside": 0.04,
                "hard_score": 0.40,
            },
        ]

        summary = summarize_records(records, topk=1)

        self.assertEqual(summary["n_samples"], 2)
        self.assertAlmostEqual(summary["mean_foreground_l1"], 0.15)
        self.assertEqual(len(summary["top_hard_samples"]), 1)


if __name__ == "__main__":
    unittest.main()
