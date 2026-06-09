import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_377_explicit_binding_v396_generalized_boundary_controller.sh"


def _selector_python_source():
    text = SCRIPT.read_text(encoding="utf-8")
    invocation = text.index('"$PYTHON_BIN" - "$SELECTOR_SUMMARY" "$SELECTED_JSON"')
    marker = "<<'PY'\n"
    start = text.index(marker, invocation) + len(marker)
    end = text.index("\nPY\n\nEND_BJT", start)
    return text[start:end]


def test_v396_raw_gate_uses_isolated_child_paths():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'gate_log_dir="$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${gate_run_id}"' in text
    assert 'gate_exp_root="$ROOT/exp/formal/377_v338_raw_contour_gate_${gate_run_id}"' in text
    assert 'gate_summary="$gate_log_dir/summary.tsv"' in text

    invocation = text[text.index('RUN_ID="$gate_run_id" \\') : text.index('"$RAW_GATE_SCRIPT" > "$gate_log" 2>&1')]
    assert 'EXP_ROOT="$gate_exp_root" \\' in invocation
    assert 'LOG_DIR="$gate_log_dir" \\' in invocation
    assert 'HYDRA_RUN_ROOT="$gate_log_dir/hydra_runtime" \\' in invocation


def test_support_floor_keeps_jaccard_diagnostic_only():
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index('row["support_floor_pass"] = (')
    end = text.index("\n\nstrict = [", start)
    support_floor_block = text[start:end]

    assert "under_adopted_lost" in support_floor_block
    assert "over_adopted_lost" in support_floor_block
    assert "under_jaccard" not in support_floor_block
    assert "over_jaccard" not in support_floor_block
    v392_floor_start = text.index('"v392_floor": {')
    v392_floor_end = text.index("}", v392_floor_start)
    v392_floor_block = text[v392_floor_start:v392_floor_end]
    assert "under_jaccard" not in v392_floor_block
    assert "over_jaccard" not in v392_floor_block
    assert '"support_diagnostics"' in text


def test_selector_payload_has_v398_schema_counts_and_selector_name():
    text = SCRIPT.read_text(encoding="utf-8")
    wrapper = ROOT / "tools" / "run_377_explicit_binding_v398_stable_generalization.sh"
    wrapper_text = wrapper.read_text(encoding="utf-8")

    assert "SELECTOR_SCHEMA_NAME" in text
    assert '"selector": selector_schema_name' in text
    assert '"counts": {' in text
    assert "v398_stable_generalization" in wrapper_text


def test_selector_records_raw_gate_worst_frames_path():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'gate_worst="$gate_log_dir/worst_frames.tsv"' in text
    assert '"bad_frame_summary"' in text
    assert '"worst_frames_summary"' in text


def test_selector_has_bad_frame_veto_thresholds():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_SELECTOR_ENABLE" in text
    assert "BAD_FRAME_OUTER_VETO" in text
    assert "BAD_FRAME_HARD_VETO" in text
    assert "BAD_FRAME_FG_POSITIVE_MAX" in text
    assert "BAD_FRAME_BOUNDARY_POSITIVE_MAX" in text
    assert "BAD_FRAME_EDGE_POSITIVE_MAX" in text
    assert 'row["bad_frame_gate_pass"]' in text


def test_selector_reports_stable_window_counts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "STABLE_WINDOW_TARGET" in text
    assert "num_stable_window_pass" in text
    assert "stable_window_pass" in text
    assert "fewer_than_target_stable_checkpoints" in text
    assert "ordered_rows = sorted(rows, key=_iteration)" in text
    assert "if prev_row in stable and next_row in stable:" in text


def test_selector_writes_selected_checkpoint_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SELECTED_CHECKPOINT_PATH_TXT" in text
    assert "SELECTED_CHECKPOINT_METRICS_JSON" in text
    assert "SELECTED_CKPT_LINK" in text
    assert "selected_checkpoint_path.txt" in text
    assert "selected_checkpoint_metrics.json" in text
    assert "selected_ckpt.pth" in text
    assert "best_ckpt.pth" not in text[text.index("SELECTED_CHECKPOINT_PATH_TXT"):]


def test_v398_wrapper_enables_stable_generalization_defaults():
    wrapper = ROOT / "tools" / "run_377_explicit_binding_v398_stable_generalization.sh"
    text = wrapper.read_text(encoding="utf-8")

    assert "SUPPORT_BANK_TRAIN_ENABLE" in text
    assert "BAD_FRAME_SELECTOR_ENABLE" in text
    assert "STABLE_WINDOW_TARGET" in text
    assert "boundary_support_bank_under_max_effective_ratio" in text
    assert "boundary_support_bank_over_max_effective_ratio" in text
    assert "run_377_explicit_binding_v396_generalized_boundary_controller.sh" in text


def test_controller_appends_inherited_extra_train_args_after_defaults():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'INHERITED_EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"' in text
    assert 'EXTRA_TRAIN_ARGS_VALUE="$EXTRA_TRAIN_ARGS_VALUE $INHERITED_EXTRA_TRAIN_ARGS"' in text


def test_v399_wrapper_sweeps_only_over_caps_and_keeps_under_caps():
    wrapper = ROOT / "tools" / "run_377_explicit_binding_v399_diagnostic_sweep.sh"
    text = wrapper.read_text(encoding="utf-8")

    assert "V399_OVER_EFFECTIVE_RATIO" in text
    assert "V399_OVER_NEW_ONLY_RATIO" in text
    assert "boundary_support_bank_under_max_effective_ratio=0.30" in text
    assert "boundary_support_bank_under_max_new_only_ratio=0.24" in text
    assert "boundary_support_bank_over_max_effective_ratio=$V399_OVER_EFFECTIVE_RATIO" in text
    assert "boundary_support_bank_over_max_new_only_ratio=$V399_OVER_NEW_ONLY_RATIO" in text
    assert "BAD_FRAME_SELECTOR_MODE" in text
    assert "v399_diagnostic_sweep" in text
    assert "run_377_explicit_binding_v396_generalized_boundary_controller.sh" in text


def test_selector_supports_bad_frame_hard_veto_plus_penalty_mode():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_SELECTOR_MODE" in text
    assert "bad_frame_hard_veto_pass" in text
    assert "bad_frame_penalty" in text
    assert "bad_frame_penalty_reasons" in text
    assert "BAD_FRAME_OUTER_HARD_VETO" in text
    assert "BAD_FRAME_HARD_HARD_VETO" in text
    assert "hard_delta_positive_penalty" in text
    assert "fg_positive_count_penalty" in text
    assert "boundary_positive_count_penalty" in text
    assert "edge_positive_count_penalty" in text


def test_penalty_mode_keeps_rankable_pool_when_all_candidates_have_penalties():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'bad_frame_selector_mode == "penalty"' in text
    assert 'row["bad_frame_gate_pass"] = (' in text
    assert 'row.get("bad_frame_hard_veto_pass", True)' in text
    assert 'row.get("bad_frame_penalty", 0.0)' in text


def test_penalty_mode_selects_lower_penalty_when_all_candidates_have_penalties(tmp_path):
    selector_py = tmp_path / "selector.py"
    selector_py.write_text(_selector_python_source(), encoding="utf-8")

    ckpt_high = tmp_path / "ckpt140410.pth"
    ckpt_low = tmp_path / "ckpt140660.pth"
    ckpt_high.write_text("high", encoding="utf-8")
    ckpt_low.write_text("low", encoding="utf-8")
    worst_high = tmp_path / "worst_high.tsv"
    worst_low = tmp_path / "worst_low.tsv"
    worst_header = (
        "variant\timage\tworsen_score\tfg_delta\tboundary_delta\tedge_delta\t"
        "inner_delta\touter_delta\thard_delta\topacity_inner_delta\topacity_outer_delta\n"
    )
    worst_high.write_text(
        worst_header
        + "candidate_high\tc23_f000000\t4.0\t0.00001\t0.0\t0.0\t0\t3\t0.00040000\t0\t0\n",
        encoding="utf-8",
    )
    worst_low.write_text(
        worst_header
        + "candidate_low\tc23_f000000\t2.0\t0.00001\t0.0\t0.0\t0\t3\t0.00010000\t0\t0\n",
        encoding="utf-8",
    )

    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "checkpoint\titeration\tstatus\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\t"
        "hard_delta\topacity_inner_delta\topacity_outer_delta\traw_gate_summary\tbad_frame_summary\n"
        f"{ckpt_high}\t140410\tstrict_pass\t-0.1\t-0.1\t-0.1\t-6\t-2\t-0.00024\t0\t-30\traw\t{worst_high}\n"
        f"{ckpt_low}\t140660\tstrict_pass\t-0.1\t-0.1\t-0.1\t-6\t-2\t-0.00024\t0\t-30\traw\t{worst_low}\n",
        encoding="utf-8",
    )

    support = tmp_path / "support.tsv"
    support.write_text(
        "base\tcheckpoint\tdirection\thas_bank\tadopted_count\teffective_count\tintersection\tunion\t"
        "jaccard\tadopted_lost\tnew_only\tgrow_count\tshrink_count\tgrow_mean\tgrow_abs_mean\tshrink_mean\tshrink_abs_mean\n"
        f"base\t{ckpt_high}\tunder\t1\t10\t20\t10\t20\t0.5\t0\t10\t20\t20\t0\t0\t0\t0\n"
        f"base\t{ckpt_high}\tover\t1\t10\t20\t10\t20\t0.5\t0\t10\t20\t20\t0\t0\t0\t0\n"
        f"base\t{ckpt_low}\tunder\t1\t10\t20\t10\t20\t0.5\t0\t10\t20\t20\t0\t0\t0\t0\n"
        f"base\t{ckpt_low}\tover\t1\t10\t20\t10\t20\t0.5\t0\t10\t20\t20\t0\t0\t0\t0\n",
        encoding="utf-8",
    )

    selected_json = tmp_path / "selected.json"
    selected_path = tmp_path / "selected_checkpoint_path.txt"
    selected_metrics = tmp_path / "selected_checkpoint_metrics.json"
    selected_link = tmp_path / "selected_ckpt.pth"
    image_summary = tmp_path / "bad_frame_image_summary.tsv"

    subprocess.run(
        [
            sys.executable,
            str(selector_py),
            str(summary),
            str(selected_json),
            "-5.4333",
            "-1.6333",
            "-0.00023074",
            "-26.8667",
            "true",
            str(support),
            "0.95",
            "0.95",
            "0",
            "true",
            "5.0",
            "0.0",
            "0.00005",
            "0",
            "0",
            "0",
            "3",
            str(selected_path),
            str(selected_metrics),
            str(selected_link),
            "test_selector",
            "penalty",
            "8.0",
            "0.0005",
            str(image_summary),
            "0.30",
            "0.24",
            "0.50",
            "0.46",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(selected_json.read_text(encoding="utf-8"))
    assert payload["selected"]["checkpoint"] == str(ckpt_low)
    assert payload["selected"]["bad_frame_gate_pass"] is True
    assert payload["selected"]["bad_frame_penalty"] < 2.0


def test_selector_reports_cap_saturation_and_failure_reason_counts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "V399_OVER_EFFECTIVE_RATIO" in text
    assert "V399_OVER_NEW_ONLY_RATIO" in text
    assert "support_cap_saturated" in text
    assert "support_under_cap_saturation" in text
    assert "support_over_cap_saturation" in text
    assert "failure_reason_counts" in text
    assert "v392_floor_miss" in text
    assert "bad_frame_veto" in text
    assert "cap_saturation" in text
    assert "fg_boundary_edge_regression" in text


def test_selector_writes_bad_frame_image_summary():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_IMAGE_SUMMARY" in text
    assert "bad_frame_image_summary.tsv" in text
    assert "bad_frame_image_summary" in text
    assert "image_failure_aggregate" in text
    assert "candidate_bad_frame_rows" in text
