import imageio.v2 as imageio
import numpy as np


def test_load_record_masks_treats_missing_compact_class_as_empty(tmp_path):
    from tools.semantic_viewer.build_part_label_bank import _load_record_masks

    hair = tmp_path / "compact_head_masks" / "hair" / "render_view.png"
    foreground = tmp_path / "coarse_masks" / "foreground" / "render_view.png"
    valid = tmp_path / "coarse_masks" / "valid" / "render_view.png"
    for path in (hair, foreground, valid):
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.full((4, 5), 255, dtype=np.uint8))
    masks, loaded_foreground, loaded_valid = _load_record_masks(
        tmp_path,
        {
            "image_name": "view",
            "compact_head_mask_files": {"hair": str(hair.relative_to(tmp_path))},
            "coarse_mask_files": {
                "foreground": str(foreground.relative_to(tmp_path)),
                "valid": str(valid.relative_to(tmp_path)),
            },
        },
    )

    assert masks["hair"].sum() == 20
    assert masks["face"].shape == (4, 5)
    assert masks["face"].sum() == 0
    assert loaded_foreground.sum() == 20
    assert loaded_valid.sum() == 20
