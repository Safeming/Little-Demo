import math


def _row(method, strength, target, outer, boundary, *, target_pixels=10):
    return {
        "subject": "377",
        "view": "c21_f000180",
        "part": "upper",
        "task": "recolor",
        "method": method,
        "edit_strength": strength,
        "target_pixel_count": target_pixels,
        "target_delta_sum": target,
        "outer_delta_sum": outer,
        "boundary_outer_delta_sum": boundary,
    }


def test_build_curves_uses_voting_full_reference_and_inserts_origin():
    from utils.semantic_matched_strength import build_matched_strength_curves

    rows = [
        _row("voting", 0.5, 5.0, 1.0, 0.5),
        _row("voting", 1.0, 10.0, 2.0, 1.0),
        _row("a5", 0.5, 4.0, 0.4, 0.2),
        _row("a5", 1.0, 8.0, 0.8, 0.4),
    ]

    result = build_matched_strength_curves(rows)
    key = ("377", "c21_f000180", "upper", "recolor")

    assert result["reference_supported"] == {key}
    assert result["curves"][(key, "a5")][0]["retention"] == 0.0
    assert result["curves"][(key, "a5")][-1]["retention"] == 0.8
    assert result["curves"][(key, "a5")][-1]["outer_burden"] == 0.08


def test_build_curves_excludes_zero_reference_without_infinite_values():
    from utils.semantic_matched_strength import build_matched_strength_curves

    rows = [
        _row("voting", 1.0, 0.0, 1.0, 1.0),
        _row("a5", 1.0, 2.0, 0.2, 0.1),
    ]

    result = build_matched_strength_curves(rows)

    assert result["reference_supported"] == set()
    assert result["unsupported_reference_count"] == 1
    assert result["curves"] == {}


def test_interpolate_curve_matches_requested_retention_and_rejects_unreachable():
    from utils.semantic_matched_strength import interpolate_curve

    curve = [
        {"retention": 0.0, "outer_burden": 0.0, "boundary_burden": 0.0},
        {"retention": 0.4, "outer_burden": 0.08, "boundary_burden": 0.04},
        {"retention": 0.8, "outer_burden": 0.16, "boundary_burden": 0.08},
    ]

    matched = interpolate_curve(curve, 0.5)

    assert math.isclose(matched["outer_burden"], 0.1)
    assert math.isclose(matched["boundary_burden"], 0.05)
    assert interpolate_curve(curve, 0.9) is None


def test_duplicate_retention_points_collapse_to_lowest_burden():
    from utils.semantic_matched_strength import collapse_curve_points

    points = [
        {"retention": 0.0, "outer_burden": 0.0, "boundary_burden": 0.0},
        {"retention": 0.5, "outer_burden": 0.2, "boundary_burden": 0.1},
        {"retention": 0.5, "outer_burden": 0.15, "boundary_burden": 0.08},
    ]

    collapsed = collapse_curve_points(points)

    assert len(collapsed) == 2
    assert collapsed[-1]["outer_burden"] == 0.15
    assert collapsed[-1]["boundary_burden"] == 0.08


def test_match_retention_reports_method_and_common_coverage():
    from utils.semantic_matched_strength import match_curves_at_retention

    key1 = ("377", "v1", "upper", "recolor")
    key2 = ("377", "v2", "upper", "recolor")
    curve_full = [
        {"retention": 0.0, "outer_burden": 0.0, "boundary_burden": 0.0},
        {"retention": 0.6, "outer_burden": 0.1, "boundary_burden": 0.05},
    ]
    curve_short = [
        {"retention": 0.0, "outer_burden": 0.0, "boundary_burden": 0.0},
        {"retention": 0.3, "outer_burden": 0.05, "boundary_burden": 0.02},
    ]
    curves = {
        (key1, "voting"): curve_full,
        (key1, "a5"): curve_full,
        (key2, "voting"): curve_full,
        (key2, "a5"): curve_short,
    }

    matched = match_curves_at_retention(curves, reference_keys={key1, key2}, retention=0.5)

    assert matched["coverage"]["voting"] == 2
    assert matched["coverage"]["a5"] == 1
    assert matched["common_coverage"][("a5", "voting")] == {key1}
