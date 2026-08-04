import numpy as np


def _synthetic_streams():
    base = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    point = base[:, None]
    return {
        "objective": {
            "outer": {"base": base, "point": point},
            "boundary": {"base": base, "point": point},
        },
        "guard": {
            "target": {
                "base": np.ones(5),
                "point": np.zeros((5, 1)),
            },
            "outer": {"base": base, "point": point},
        },
    }


def test_soft_iou_linear_slack_matches_direct_ratio():
    from utils.a7c_feasibility_oracle import soft_iou_linear_slack

    target = np.array([0.98, 0.96])
    outer = np.array([0.02, 0.04])
    base_target = np.ones(2)
    base_outer = np.zeros(2)
    slack = soft_iou_linear_slack(
        target,
        outer,
        base_target,
        base_outer,
        maximum_drop=0.005,
    )
    direct_drop = base_target / (base_target + base_outer) - target / (
        target + outer
    )

    assert np.array_equal(
        slack >= -1.0e-12, direct_drop <= 0.005 + 1.0e-12
    )


def test_fixed_gain_oracle_returns_directly_valid_witness():
    from utils.a7c_feasibility_oracle import solve_fixed_gain_oracle

    result = solve_fixed_gain_oracle(
        runtime_mass=np.zeros((5, 1)),
        a5_weight=np.array([0.8]),
        streams=_synthetic_streams(),
        minimum_gate=0.9,
        maximum_gate=1.0,
        selection_threshold=0.2,
        proxy_target_response=0.995,
        maximum_gate_jump=0.015,
        minimum_target_response=0.99,
        maximum_soft_iou_drop=0.005,
        minimum_outer_gain=0.01,
        minimum_boundary_gain=0.01,
    )

    assert result["metrics"]["outer_gain"] >= 0.01 - 1.0e-7
    assert result["metrics"]["boundary_gain"] >= 0.01 - 1.0e-7
    assert result["metrics"]["minimum_target_response"] >= 0.99 - 1.0e-7
    assert result["certificate"]["maximum_primal_violation"] <= 1.0e-7


def test_capacity_bisection_brackets_true_capacity():
    from utils.a7c_feasibility_oracle import bisect_feasible_gain

    result = bisect_feasible_gain(
        lambda gain: gain <= 0.25,
        lower=-0.01,
        upper=1.00001,
        tolerance=1.0e-5,
    )

    assert result["feasible_lower"] <= 0.25
    assert result["infeasible_upper"] >= 0.25
    assert result["interval_width"] <= 1.0e-5


def _classification_contract():
    return {
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
        "minimum_positive_block_fraction": 0.9,
        "minimum_block_gain_quantile": 0.0,
        "maximum_worst_block_regression": 0.005,
        "r1_1_f1_outer_gain": -0.00012761059760764496,
        "r1_1_f1_boundary_gain": 0.023481874880317264,
    }


def _optimistic_summary(boundary_gain=0.03):
    return {
        "outer_gain": 0.01,
        "boundary_gain": boundary_gain,
        "outer_positive_block_fraction": 1.0,
        "boundary_positive_block_fraction": 1.0,
        "outer_block_gain_quantile": 0.01,
        "boundary_block_gain_quantile": boundary_gain,
        "outer_worst_block_gain": 0.01,
        "boundary_worst_block_gain": boundary_gain,
    }


def test_oracle_classification_has_three_rigorous_verdicts():
    from utils.a7c_feasibility_oracle import classify_oracle

    contract = _classification_contract()
    optimistic = _optimistic_summary()
    assert (
        classify_oracle(
            sufficient_audit_passed=True,
            optimistic_summary=optimistic,
            contract=contract,
        )
        == "CERTIFIED_FEASIBLE"
    )
    assert (
        classify_oracle(
            sufficient_audit_passed=False,
            optimistic_summary=_optimistic_summary(boundary_gain=0.01),
            contract=contract,
        )
        == "CERTIFIED_INFEASIBLE"
    )
    assert (
        classify_oracle(
            sufficient_audit_passed=False,
            optimistic_summary=optimistic,
            contract=contract,
        )
        == "UNRESOLVED"
    )
