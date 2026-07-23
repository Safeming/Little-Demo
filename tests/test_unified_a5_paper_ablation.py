from pathlib import Path

import pytest


def test_queue_runs_unified_component_and_a5_micro_ablations_without_training():
    script = Path("tools/run_unified_a5_paper_ablation.sh").read_text(encoding="utf-8")

    for subject in ("377", "386", "387", "393", "394"):
        assert subject in script
    assert "frozen_a5_five_subject_loso_stats_20260723" in script
    assert "--baselines A0 A1 A2 A3 A4 A5 A6" in script
    assert script.count("--baselines A0 A4 A5") >= 2
    assert "--footprint-radius-scale 0" in script
    assert "--min-footprint-radius 0" in script
    assert "--max-footprint-radius 0" in script
    assert "--outer-penalty-power 0.2" in script
    assert "--outer-penalty-power 0" in script
    assert script.count("--retention-reference-baseline A0") >= 3
    assert "summarize_unified_a5_paper_ablation.py" in script
    assert "train.py" not in script
    assert "semantic-train" not in script


def test_component_methods_require_complete_a0_a6_chain():
    from tools.summarize_unified_a5_paper_ablation import validate_component_methods

    validate_component_methods(["A0", "A1", "A2", "A3", "A4", "A5", "A6"])
    with pytest.raises(ValueError, match="complete A0-A6"):
        validate_component_methods(["A0", "A1", "A4", "A5"])


def test_micro_ablation_variant_labels_are_paper_explicit():
    from tools.summarize_unified_a5_paper_ablation import variant_label

    assert variant_label("A4") == "Voting posterior (A4)"
    assert variant_label("center_only") == "A5 center-only evidence"
    assert variant_label("no_outer") == "A5 without outer penalty"
    assert variant_label("full") == "Ours (A5 full footprint)"
