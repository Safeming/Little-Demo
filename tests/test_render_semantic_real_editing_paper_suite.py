import json
from pathlib import Path

import pytest


def test_parse_args_defaults_to_frozen_paper_matrix():
    from tools.render_semantic_real_editing_paper_suite import parse_args

    args = parse_args(
        [
            "--subject", "377",
            "--raw-bank", "raw.npz",
            "--voting-bank", "voting.npz",
            "--a5-bank", "a5.npz",
            "--loso-config", "loso.json",
            "--method-freeze", "freeze.json",
            "--checkpoint", "ckpt.pth",
            "--asset-root", "assets",
            "--output-dir", "out",
        ]
    )

    assert args.methods == ["raw_hard", "voting", "a5"]
    assert args.tasks == ["recolor", "removal", "texture"]
    assert args.parts == ["hair", "face", "upper", "lower", "shoes", "skin"]


def test_build_experiment_matrix_contains_every_method_task_part():
    from tools.render_semantic_real_editing_paper_suite import build_experiment_matrix

    rows = build_experiment_matrix(
        methods=["raw_hard", "voting", "a5"],
        tasks=["recolor", "removal", "texture"],
        parts=["hair", "face"],
    )

    assert len(rows) == 18
    assert {tuple(row[key] for key in ("method", "task", "part")) for row in rows} == {
        (method, task, part)
        for method in ("raw_hard", "voting", "a5")
        for task in ("recolor", "removal", "texture")
        for part in ("hair", "face")
    }


def test_load_frozen_run_config_reads_loso_threshold_and_checks_subject(tmp_path: Path):
    from utils.frozen_semantic_method import frozen_method_fingerprint
    from tools.render_semantic_real_editing_paper_suite import load_frozen_run_config

    freeze = {
        "schema_version": 1,
        "freeze_id": "a5_main_method_v1_20260723",
        "status": "frozen",
        "primary_method": "A5",
        "extension_methods": ["A6"],
        "components": {
            "footprint_evidence_calibration": {"mode": "evidence-calibrated", "output_field": "soft_edit_weights"},
            "target_support_extension": {"method": "A6", "target_field": "edit_target_weights", "support_field": "edit_support_weights"},
        },
        "reporting": {"main_table_method": "A5", "ablation_only": ["A6"]},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    loso = {
        "held_out_subject": "377",
        "method_freeze_fingerprint": frozen_method_fingerprint(freeze),
        "selected": {"soft_threshold": 0.25, "boundary_radius": 6},
    }
    loso_path = tmp_path / "loso.json"
    loso_path.write_text(json.dumps(loso), encoding="utf-8")

    run = load_frozen_run_config(subject="377", loso_config=loso_path, method_freeze=freeze_path)

    assert run["soft_threshold"] == 0.25
    assert run["boundary_radius"] == 6
    assert run["method_freeze_id"] == "a5_main_method_v1_20260723"

    with pytest.raises(ValueError, match="held-out subject"):
        load_frozen_run_config(subject="386", loso_config=loso_path, method_freeze=freeze_path)


def test_formal_provenance_marks_test_masks_as_evaluation_only():
    from tools.render_semantic_real_editing_paper_suite import formal_provenance

    provenance = formal_provenance()

    assert provenance["uses_test_parser_for_edit_selection"] is False
    assert provenance["uses_test_masks_for_metrics"] is True
    assert provenance["shared_rasterizer_across_methods"] is True


def test_normalize_dataset_subject_converts_numeric_omegaconf_value_to_string():
    from omegaconf import OmegaConf
    from tools.render_semantic_real_editing_paper_suite import normalize_dataset_subject

    config = OmegaConf.create({"dataset": {"subject": 377}})
    normalize_dataset_subject(config, "377")

    assert config.dataset.subject == "CoreView_377"
    assert isinstance(config.dataset.subject, str)


def test_parse_args_accepts_multi_strength_metrics_only_mode():
    from tools.render_semantic_real_editing_paper_suite import parse_args, resolve_edit_strengths

    args = parse_args(
        [
            "--subject", "377",
            "--raw-bank", "raw.npz",
            "--voting-bank", "voting.npz",
            "--a5-bank", "a5.npz",
            "--loso-config", "loso.json",
            "--method-freeze", "freeze.json",
            "--checkpoint", "ckpt.pth",
            "--asset-root", "assets",
            "--output-dir", "out",
            "--edit-strengths", "0.2", "0.4", "0.6", "0.8", "1.0",
            "--metrics-only",
        ]
    )

    assert resolve_edit_strengths(args) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert args.metrics_only is True


def test_resolve_edit_strengths_preserves_single_strength_default():
    from argparse import Namespace
    from tools.render_semantic_real_editing_paper_suite import resolve_edit_strengths

    args = Namespace(edit_strength=0.75, edit_strengths=None)

    assert resolve_edit_strengths(args) == [0.75]


@pytest.mark.parametrize(
    "values",
    [
        [0.4, 0.2],
        [0.2, 0.2],
        [0.0, 0.2],
        [0.2, 1.1],
    ],
)
def test_resolve_edit_strengths_rejects_invalid_grids(values):
    from argparse import Namespace
    from tools.render_semantic_real_editing_paper_suite import resolve_edit_strengths

    with pytest.raises(ValueError, match="strength"):
        resolve_edit_strengths(Namespace(edit_strength=1.0, edit_strengths=values))
