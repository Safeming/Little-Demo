import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class SemanticEditRenderPreviewTests(unittest.TestCase):
    def test_resolve_part_weights_uses_hard_and_soft_modes(self):
        from tools.make_semantic_edit_render_preview import resolve_part_weights

        labels = np.array([0, 3, -1, 3], dtype=np.int16)
        soft = np.zeros((4, 6), dtype=np.float32)
        soft[:, 3] = np.array([0.1, 0.8, 0.3, 0.19], dtype=np.float32)

        hard_weights = resolve_part_weights(labels=labels, soft_weights=soft, part_index=3, mode="hard", threshold=0.2)
        soft_weights = resolve_part_weights(labels=labels, soft_weights=soft, part_index=3, mode="soft", threshold=0.2)

        self.assertTrue(np.allclose(hard_weights, [0.0, 1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(soft_weights, [0.0, 0.8, 0.3, 0.0]))

    def test_blend_edit_colors_mixes_selected_points_only(self):
        from tools.make_semantic_edit_render_preview import blend_edit_colors

        base = np.array(
            [
                [0.2, 0.4, 0.6],
                [0.8, 0.6, 0.4],
            ],
            dtype=np.float32,
        )
        weights = np.array([1.0, 0.25], dtype=np.float32)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        edited = blend_edit_colors(base, weights, target, alpha=0.5)

        self.assertTrue(np.allclose(edited[0], [0.6, 0.2, 0.3]))
        self.assertTrue(np.allclose(edited[1], [0.825, 0.525, 0.35]))

    def test_compose_preview_sheet_writes_image(self):
        from tools.make_semantic_edit_render_preview import compose_preview_sheet

        panels = [
            {
                "view": "c00_f000001",
                "part": "lower",
                "images": [
                    ("RGB", Image.new("RGB", (8, 8), (10, 20, 30))),
                    ("Hard", Image.new("RGB", (8, 8), (200, 30, 30))),
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.png"
            compose_preview_sheet(panels, out, thumb_size=32)
            with Image.open(out) as image:
                self.assertGreater(image.width, 64)
                self.assertGreater(image.height, 32)


if __name__ == "__main__":
    unittest.main()
