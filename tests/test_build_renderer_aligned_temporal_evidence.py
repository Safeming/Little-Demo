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


def test_v5_3_evidence_dry_run_declares_912_samples(tmp_path, capsys):
    import json

    from tools.build_renderer_aligned_temporal_evidence import main

    assert main(
        [
            "--config", "config.yaml",
            "--checkpoint", "checkpoint.pth",
            "--a5-bank", "a5.npz",
            "--method-freeze", "configs/semantic/frozen_a5_main_method_v1.json",
            "--a7-contract", "configs/semantic/frozen_a7_dual_evidence_v5_3_evidence_377.json",
            "--output", str(tmp_path / "evidence.npz"),
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 912
    assert payload["backward_calls_per_sample"] == 6
