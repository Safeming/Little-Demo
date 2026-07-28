from pathlib import Path

import numpy as np


def _required_args():
    return [
        "--subject",
        "377",
        "--voting-bank",
        "voting.npz",
        "--a5-bank",
        "a5.npz",
        "--loso-config",
        "loso.json",
        "--method-freeze",
        "freeze.json",
        "--checkpoint",
        "ckpt.pth",
        "--config",
        "config.yaml",
        "--output-dir",
        "out",
    ]


def test_temporal_renderer_defaults_match_frozen_protocol():
    from tools.render_semantic_temporal_stability import parse_args

    args = parse_args(_required_args())

    assert args.camera == 21
    assert (args.frame_start, args.frame_end, args.frame_step) == (0, 570, 1)
    assert args.methods == ["voting", "a5"]
    assert args.parts == ["hair", "face", "upper", "lower", "shoes", "skin"]
    assert args.video_parts == ["upper", "hair", "shoes"]
    assert args.video_fps == 25
    assert args.screen_threshold == 0.2


def test_temporal_renderer_accepts_optional_static_a7_method():
    from tools.render_semantic_temporal_stability import parse_args

    args = parse_args(
        _required_args()
        + [
            "--a7-bank",
            "a7.npz",
            "--a7-contract",
            "a7.json",
            "--methods",
            "a5",
            "a7",
        ]
    )

    assert args.a7_bank == Path("a7.npz")
    assert args.a7_contract == Path("a7.json")
    assert args.methods == ["a5", "a7"]


def test_expected_metric_row_count_matches_full_sequence():
    from tools.render_semantic_temporal_stability import expected_metric_row_count

    assert expected_metric_row_count(0, 570, 1, part_count=6, method_count=2) == 6840
    assert expected_metric_row_count(0, 10, 2, part_count=2, method_count=2) == 20


def test_extract_compact_masks_routes_named_six_part_parser_channels():
    from tools.render_semantic_temporal_stability import extract_compact_masks

    compact = np.stack([np.full((2, 3), value / 5.0, dtype=np.float32) for value in range(6)])
    names = ("hair", "face", "skin", "upper", "lower", "shoes")
    valid = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32)

    masks = extract_compact_masks(compact, names, valid)

    assert set(masks) == {"hair", "face", "skin", "upper", "lower", "shoes"}
    assert np.array_equal(masks["upper"], compact[3] * valid)


def test_extract_compact_masks_rejects_missing_part():
    import pytest

    from tools.render_semantic_temporal_stability import extract_compact_masks

    with pytest.raises(ValueError, match="missing compact parser parts"):
        extract_compact_masks(np.zeros((2, 2, 2)), ("hair", "face"), np.ones((2, 2)))


def test_compose_video_panel_has_fixed_1920_width():
    from tools.render_semantic_temporal_stability import compose_video_panel

    image = np.zeros((64, 80, 3), dtype=np.uint8)
    panel = compose_video_panel(image, image, image, image, image)

    assert panel.dtype == np.uint8
    assert panel.shape == (384, 1920, 3)


def test_system_ffmpeg_video_writer_encodes_mp4(tmp_path: Path):
    from tools.render_semantic_temporal_stability import FFmpegVideoWriter

    output = tmp_path / "sample.mp4"
    with FFmpegVideoWriter(output, fps=5) as writer:
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))

    assert output.exists()
    assert output.stat().st_size > 0


def test_temporal_renderer_help_is_available():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "tools/render_semantic_temporal_stability.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "continuous-frame" in result.stdout.lower()
