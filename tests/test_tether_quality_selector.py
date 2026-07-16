import pytest

from tools.select_tether_quality_candidate import (
    evaluate_final_candidate,
    select_continuation,
)


def test_continuation_selection_prefers_raw_pareto_candidate():
    candidates = [
        {
            "label": "lpips_only",
            "lpips_fg": 0.1290,
            "psnr_fg": 21.80,
            "edge_px": 3.20,
            "boundary_l1": 0.0690,
        },
        {
            "label": "balanced",
            "lpips_fg": 0.1294,
            "psnr_fg": 21.90,
            "edge_px": 3.00,
            "boundary_l1": 0.0675,
        },
        {
            "label": "too_soft",
            "lpips_fg": 0.1310,
            "psnr_fg": 22.00,
            "edge_px": 2.80,
            "boundary_l1": 0.0660,
        },
    ]

    selected = select_continuation(candidates, lpips_tolerance=0.001)

    assert selected["label"] == "balanced"


def test_continuation_selection_requires_candidates():
    with pytest.raises(ValueError, match="candidate"):
        select_continuation([])


def test_final_gate_accepts_raw_quality_without_legacy_calibrated_psnr():
    result = evaluate_final_candidate(
        same30={"lpips_fg": 0.1260, "psnr_fg": 21.97},
        original57={"lpips_fg": 0.1287, "psnr_fg": 21.80},
        contour={"edge_px": 2.88, "boundary_l1": 0.0670},
    )

    assert result["accepted"] is True
    assert result["gates"] == {
        "same30_lpips": True,
        "same30_psnr": True,
        "original57_lpips": True,
        "original57_psnr": True,
        "contour_edge": True,
        "boundary_l1": True,
    }


def test_final_gate_rejects_contour_regression_even_when_image_metrics_pass():
    result = evaluate_final_candidate(
        same30={"lpips_fg": 0.1260, "psnr_fg": 21.97},
        original57={"lpips_fg": 0.1287, "psnr_fg": 21.80},
        contour={"edge_px": 3.01, "boundary_l1": 0.0670},
    )

    assert result["accepted"] is False
    assert result["gates"]["contour_edge"] is False
