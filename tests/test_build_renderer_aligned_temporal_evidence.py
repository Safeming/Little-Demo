def test_renderer_aligned_evidence_parser_exposes_frozen_inputs(tmp_path):
    from tools.build_renderer_aligned_temporal_evidence import parse_args

    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--a5-bank",
            "a5.npz",
            "--method-freeze",
            "a5.json",
            "--a7-contract",
            "a7-v2.json",
            "--output",
            str(tmp_path / "evidence.npz"),
            "--dry-run",
        ]
    )

    assert args.dry_run is True
    assert args.output == tmp_path / "evidence.npz"
