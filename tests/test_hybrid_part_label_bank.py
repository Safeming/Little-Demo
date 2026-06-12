import unittest

import numpy as np

from tools.build_hybrid_part_label_bank import build_hybrid_bank
from utils.part_label_bank import PART_NAMES


class HybridPartLabelBankTests(unittest.TestCase):
    def test_build_hybrid_bank_replaces_only_requested_soft_channels(self):
        point_count = 4
        labels = np.array([0, 1, 3, 4], dtype=np.int16)
        base_weights = np.arange(point_count * len(PART_NAMES), dtype=np.float32).reshape(point_count, len(PART_NAMES))
        override_weights = base_weights + 100.0
        base = {
            "part_label": labels,
            "editable_label": np.array([0, 1, -1, 4], dtype=np.int16),
            "confidence": np.linspace(0.1, 0.4, point_count, dtype=np.float32),
            "vote_count": np.arange(point_count, dtype=np.int16),
            "per_part_votes": np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
            "visible_vote_count": np.ones((point_count,), dtype=np.int16),
            "conflict_count": np.zeros((point_count,), dtype=np.int16),
            "semantic_probs": np.full((point_count, len(PART_NAMES)), 0.1, dtype=np.float32),
            "semantic_margin": np.linspace(0.2, 0.5, point_count, dtype=np.float32),
            "reliable_mask": np.array([1, 1, 0, 1], dtype=np.uint8),
            "soft_edit_weights": base_weights,
            "source_checkpoint": np.array("/tmp/base.pth"),
            "source_asset_root": np.array("/tmp/base_assets"),
            "source_iteration": np.array(123, dtype=np.int64),
            "source_type": np.array("base_source"),
        }
        override = {
            "part_label": np.array([5, 5, 5, 5], dtype=np.int16),
            "editable_label": np.array([5, 5, 5, 5], dtype=np.int16),
            "confidence": np.ones((point_count,), dtype=np.float32),
            "vote_count": np.ones((point_count,), dtype=np.int16),
            "per_part_votes": np.ones((point_count, len(PART_NAMES)), dtype=np.int16),
            "visible_vote_count": np.ones((point_count,), dtype=np.int16),
            "conflict_count": np.ones((point_count,), dtype=np.int16),
            "semantic_probs": np.full((point_count, len(PART_NAMES)), 0.9, dtype=np.float32),
            "soft_edit_weights": override_weights,
            "source_checkpoint": np.array("/tmp/override.pth"),
            "source_asset_root": np.array("/tmp/override_assets"),
            "source_iteration": np.array(456, dtype=np.int64),
            "source_type": np.array("override_source"),
        }

        hybrid, summary = build_hybrid_bank(base, override, parts=("lower", "shoes"))

        expected = base_weights.copy()
        for part in ("lower", "shoes"):
            expected[:, PART_NAMES.index(part)] = override_weights[:, PART_NAMES.index(part)]
        self.assertTrue(np.allclose(hybrid["soft_edit_weights"], expected))
        self.assertEqual(hybrid["part_label"].tolist(), base["part_label"].tolist())
        self.assertEqual(hybrid["editable_label"].tolist(), base["editable_label"].tolist())
        self.assertTrue(np.allclose(hybrid["semantic_probs"], base["semantic_probs"]))
        self.assertEqual(str(hybrid["source_checkpoint"]), "/tmp/base.pth")
        self.assertIn("hybrid_soft_channels", str(hybrid["source_type"]))
        self.assertEqual(summary["override_parts"], ["lower", "shoes"])
        self.assertEqual(summary["unchanged_parts"], ["hair", "face", "upper", "skin"])

    def test_build_hybrid_bank_rejects_point_count_mismatch(self):
        base = {"soft_edit_weights": np.zeros((2, len(PART_NAMES)), dtype=np.float32)}
        override = {"soft_edit_weights": np.zeros((3, len(PART_NAMES)), dtype=np.float32)}

        with self.assertRaises(ValueError):
            build_hybrid_bank(base, override, parts=("lower",))


if __name__ == "__main__":
    unittest.main()
