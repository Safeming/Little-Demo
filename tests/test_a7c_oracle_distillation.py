import numpy as np
import pytest


def _teacher_problem():
    frames, carriers = 4, 2
    zeros = np.zeros((frames, carriers), dtype=np.float64)
    base = np.array([1.0, 2.0, 1.0, 2.0])
    point = np.stack([base, np.zeros_like(base)], axis=1)
    streams = {
        "objective": {
            "outer": {"base": base, "point": point},
            "boundary": {"base": base, "point": point},
        },
        "guard": {
            "target": {"base": np.ones(frames), "point": zeros},
            "outer": {"base": np.ones(frames), "point": zeros},
        },
    }
    return {
        "runtime_mass": zeros,
        "a5_weight": np.array([0.8, 0.8]),
        "streams": streams,
        "anchor_gates": np.array([[0.97, 0.98]] * frames),
    }


def _teacher_thresholds():
    return {
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_gate_jump": 0.015,
        "minimum_target_response": 0.99,
        "maximum_soft_iou_drop": 0.005,
        "minimum_outer_gain": 0.005,
        "boundary_margin": 2.0e-5,
        "bisection_tolerance": 1.0e-5,
        "lexicographic_tolerance": 1.0e-9,
        "primal_tolerance": 1.0e-9,
        "residual_tolerance": 1.0e-7,
    }


def test_teacher_capacity_and_anchor_are_feasible_and_deterministic():
    from utils.a7c_oracle_distillation import solve_fit_teacher

    first = solve_fit_teacher(**_teacher_problem(), **_teacher_thresholds())
    second = solve_fit_teacher(**_teacher_problem(), **_teacher_thresholds())

    np.testing.assert_array_equal(first["gates"], second["gates"])
    assert first["capacity"] == second["capacity"]
    assert first["request"] == second["request"]
    assert first["certificate"] == second["certificate"]
    assert first["capacity"]["interval_width"] <= 1.0e-5
    assert first["request"]["boundary_gain"] == pytest.approx(
        first["capacity"]["feasible_lower"] - 2.0e-5
    )
    assert first["metrics"]["outer_gain"] >= 0.005 - 1.0e-7
    assert first["metrics"]["boundary_gain"] >= (
        first["request"]["boundary_gain"] - 1.0e-7
    )
    certificate = first["certificate"]
    assert certificate["maximum_primal_violation"] <= 1.0e-7
    assert certificate["stage_one_presolve"] is True
    assert certificate["stage_two_presolve"] is True
    assert certificate["stage_three_presolve"] is False
    assert certificate["stage_one_maximum_deviation"] >= 0.0
    assert certificate["stage_two_total_deviation"] >= 0.0
    assert certificate["stage_three_total_gate_change"] >= 0.0


def test_teacher_solver_rejects_anchor_shape_mismatch():
    from utils.a7c_oracle_distillation import solve_fit_teacher

    problem = _teacher_problem()
    problem["anchor_gates"] = np.ones((4, 1))
    with pytest.raises(ValueError, match="anchor"):
        solve_fit_teacher(**problem, **_teacher_thresholds())


def test_insert_teacher_segment_keeps_every_nonfit_value_nan():
    from utils.a7c_oracle_distillation import insert_teacher_segment

    gates = np.full((12, 2), np.nan)
    teacher_mask = np.zeros(12, dtype=bool)
    selected = np.array([True, True, False, False] * 3)
    solved = {
        "gates": np.full((6, 2), 0.97),
        "certificate": {"maximum_primal_violation": 1.0e-9},
    }
    insert_teacher_segment(
        gates, teacher_mask, selected, solved, residual_tolerance=1.0e-7
    )
    assert np.isfinite(gates[selected]).all()
    assert np.isnan(gates[~selected]).all()
    np.testing.assert_array_equal(teacher_mask, selected)

    with pytest.raises(ValueError, match="overlap"):
        insert_teacher_segment(
            gates,
            teacher_mask,
            selected,
            solved,
            residual_tolerance=1.0e-7,
        )


def test_insert_teacher_segment_rejects_bad_certificate():
    from utils.a7c_oracle_distillation import insert_teacher_segment

    gates = np.full((2, 1), np.nan)
    mask = np.zeros(2, dtype=bool)
    solved = {
        "gates": np.full((2, 1), 0.97),
        "certificate": {"maximum_primal_violation": 2.0e-7},
    }
    with pytest.raises(ValueError, match="residual"):
        insert_teacher_segment(
            gates,
            mask,
            np.ones(2, dtype=bool),
            solved,
            residual_tolerance=1.0e-7,
        )


def test_teacher_artifact_has_exact_fold_local_masks_and_eligibility(tmp_path):
    from tools.build_a7c_r1_4vp_fit_teachers import write_fold_teacher

    camera = np.repeat(np.arange(4), 6)
    frame = np.tile(np.arange(6) * 5, 4)
    block = np.tile(np.arange(6), 4)
    fit_mask = block != 2
    gates = np.full((24, 2), np.nan)
    gates[fit_mask] = 0.97
    certificates = [
        {
            "fold": 2,
            "camera_index": camera,
            "block_id": block_id,
            "maximum_primal_violation": 1.0e-9,
        }
        for camera in range(4)
        for block_id in (0, 1, 3, 4, 5)
    ]
    write_fold_teacher(
        output_dir=tmp_path,
        fold=2,
        gates=gates,
        teacher_mask=fit_mask,
        camera_index=camera,
        frame_index=frame,
        block_ids=block,
        carrier_ids=np.array([3, 7]),
        certificates=certificates,
        source_fingerprints={"probe": "abc"},
    )
    with np.load(tmp_path / "fold_2/teacher.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["teacher_mask"], fit_mask)
        assert np.isnan(saved["teacher_gates"][~fit_mask]).all()
        assert np.isfinite(saved["teacher_gates"][fit_mask]).all()
        assert int(saved["deployment_eligible"]) == 0
        assert int(saved["teacher_eligible"]) == 0
        assert int(saved["paper_test_eligible"]) == 0
    assert (tmp_path / "fold_2/certificates.json").is_file()
