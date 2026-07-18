import pytest


def _baseline_rows(*, b1_miou=0.62, b5_miou=0.61):
    return [
        {"baseline": "B1", "macro_miou": str(b1_miou)},
        {"baseline": "B5", "macro_miou": str(b5_miou)},
    ]


def _curve_rows(*, b1=(0.030, 0.025), b5=(0.029, 0.024), include_six=True):
    rows = []
    for baseline, values in (("B1", b1), ("B5", b5)):
        rows.append(
            {
                "baseline": baseline,
                "retention": "0.5",
                "actionable_leakage": str(values[0]),
            }
        )
        if include_six:
            rows.append(
                {
                    "baseline": baseline,
                    "retention": "0.6",
                    "actionable_leakage": str(values[1]),
                }
            )
    return rows


def test_candidate_passes_all_validation_gates():
    from tools.assess_voting_posterior_candidate import assess_candidate

    result = assess_candidate(_baseline_rows(), _curve_rows())

    assert result["passed"] is True
    assert result["miou_gap"] == pytest.approx(0.01)
    assert [row["retention"] for row in result["retention_checks"]] == [0.5, 0.6]


def test_candidate_fails_when_required_retention_would_extrapolate():
    from tools.assess_voting_posterior_candidate import assess_candidate

    result = assess_candidate(_baseline_rows(), _curve_rows(include_six=False))

    assert result["passed"] is False
    assert any("outside observed retention range" in reason for reason in result["failure_reasons"])


def test_candidate_fails_when_b5_leakage_exceeds_b1():
    from tools.assess_voting_posterior_candidate import assess_candidate

    result = assess_candidate(_baseline_rows(), _curve_rows(b5=(0.031, 0.024)))

    assert result["passed"] is False
    assert any("leakage" in reason for reason in result["failure_reasons"])


def test_candidate_fails_when_miou_gap_is_too_large():
    from tools.assess_voting_posterior_candidate import assess_candidate

    result = assess_candidate(_baseline_rows(b5_miou=0.59), _curve_rows())

    assert result["passed"] is False
    assert any("mIoU gap" in reason for reason in result["failure_reasons"])
