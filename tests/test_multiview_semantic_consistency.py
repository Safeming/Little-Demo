import unittest


class MultiviewSemanticConsistencyTests(unittest.TestCase):
    def test_ensure_projected_args_adds_soft_boundary_defaults(self):
        import argparse

        from tools.analyze_multiview_semantic_consistency import ensure_projected_arg_defaults

        args = argparse.Namespace(mask_threshold=0.5)

        ensure_projected_arg_defaults(args)

        self.assertEqual(args.soft_boundary_radius, 0)
        self.assertAlmostEqual(args.soft_boundary_min_value, 0.25)

    def test_summarize_multiview_rows_reports_mean_std_cv_and_soft_hard_deltas(self):
        from tools.analyze_multiview_semantic_consistency import summarize_multiview_rows

        rows = [
            {
                "part": "hair",
                "mode": "hard",
                "view": "v0",
                "target_activation": 10.0,
                "leakage_ratio": 0.4,
                "boundary_leakage_ratio": 0.2,
                "target_coverage": 0.5,
            },
            {
                "part": "hair",
                "mode": "hard",
                "view": "v1",
                "target_activation": 20.0,
                "leakage_ratio": 0.6,
                "boundary_leakage_ratio": 0.4,
                "target_coverage": 0.7,
            },
            {
                "part": "hair",
                "mode": "soft",
                "view": "v0",
                "target_activation": 8.0,
                "leakage_ratio": 0.3,
                "boundary_leakage_ratio": 0.1,
                "target_coverage": 0.4,
            },
            {
                "part": "hair",
                "mode": "soft",
                "view": "v1",
                "target_activation": 12.0,
                "leakage_ratio": 0.5,
                "boundary_leakage_ratio": 0.2,
                "target_coverage": 0.5,
            },
        ]

        result = summarize_multiview_rows(rows, soft_threshold=0.2)

        summary = result["summary"]
        per_part = result["per_part"][0]
        self.assertEqual(summary["part_count"], 1)
        self.assertEqual(summary["view_count"], 2)
        self.assertAlmostEqual(summary["mean_leakage_std_delta_soft_minus_hard"], 0.0)
        self.assertEqual(per_part["part"], "hair")
        self.assertAlmostEqual(per_part["hard_target_activation_mean"], 15.0)
        self.assertAlmostEqual(per_part["hard_target_activation_std"], 5.0)
        self.assertAlmostEqual(per_part["hard_target_activation_cv"], 5.0 / 15.0)
        self.assertAlmostEqual(per_part["soft_target_activation_mean"], 10.0)
        self.assertAlmostEqual(per_part["soft_target_activation_std"], 2.0)
        self.assertAlmostEqual(per_part["soft_target_activation_cv"], 0.2)
        self.assertAlmostEqual(per_part["hard_leakage_ratio_mean"], 0.5)
        self.assertAlmostEqual(per_part["soft_leakage_ratio_mean"], 0.4)
        self.assertAlmostEqual(per_part["leakage_mean_delta_soft_minus_hard"], -0.1)
        self.assertAlmostEqual(per_part["boundary_leakage_std_delta_soft_minus_hard"], -0.05)

    def test_write_reports_creates_summary_per_part_and_per_view(self):
        import json
        import tempfile
        from pathlib import Path

        from tools.analyze_multiview_semantic_consistency import write_reports

        result = {
            "summary": {"part_count": 1, "view_count": 2},
            "per_part": [{"part": "hair", "soft_leakage_ratio_mean": 0.4}],
            "per_view": [{"part": "hair", "mode": "soft", "view": "v0", "leakage_ratio": 0.3}],
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_reports(out, result)

            self.assertEqual(json.loads((out / "summary.json").read_text())["view_count"], 2)
            self.assertIn("part,soft_leakage_ratio_mean", (out / "per_part.csv").read_text())
            self.assertIn("part,mode,view,leakage_ratio", (out / "per_view.csv").read_text())


if __name__ == "__main__":
    unittest.main()
