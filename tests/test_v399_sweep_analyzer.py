import json

from tools.analyze_377_v399_diagnostic_sweep import summarize_run


def test_summarize_run_reports_floor_bad_frame_and_cap_status(tmp_path):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    selected = {
        "selected": {
            "iteration": "141160",
            "hard_delta": -0.000231,
            "fg_delta": -0.1,
            "boundary_delta": -0.1,
            "edge_delta": -0.1,
        },
        "v392_floor": {"hard_delta": -0.00023074},
        "bad_frame_diagnostics": {
            "selected_bad_frame_max_outer_delta": 7.0,
            "selected_bad_frame_max_hard_delta": 0.0003,
        },
        "cap_diagnostics": {
            "selected_support_cap_saturated": True,
            "selected_support_over_cap_saturation": 1.0,
        },
    }
    (log_dir / "v399_selected_checkpoint.json").write_text(json.dumps(selected), encoding="utf-8")

    row = summarize_run(log_dir)

    assert row["iteration"] == "141160"
    assert row["hard_floor_pass"] == "1"
    assert row["bad_frame_not_worse_than_v398"] == "1"
    assert row["over_cap_saturation"] == "1.0000"
