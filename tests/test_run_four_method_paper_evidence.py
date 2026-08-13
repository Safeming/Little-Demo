from pathlib import Path


def test_orchestrator_declares_confirmed_subject_order_and_gg_exception():
    path = Path(__file__).parents[1] / "tools" / "run_four_method_paper_evidence.sh"
    text = path.read_text(encoding="utf-8")

    assert "SUBJECTS=(377 386 394)" in text
    assert "GG_377_RETENTION=0.40" in text
    assert "TARGET_RETENTION=0.60" in text
    assert "canary" in text
    assert "significance" in text
    assert "temporal" in text
    assert "qualitative" in text


def test_orchestrator_dry_run_has_canary_before_formal_stages():
    path = Path(__file__).parents[1] / "tools" / "run_four_method_paper_evidence.sh"
    text = path.read_text(encoding="utf-8")

    assert text.index("canary)") < text.index("significance)")
    assert "DRY_RUN" in text
