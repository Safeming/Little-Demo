import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json"
)
RUNNER = ROOT / "tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh"


def test_r4b0_contract_freezes_projection_aware_training_only():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["experiment_id"] == (
        "a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1"
    )
    assert contract["status"] == "frozen"
    assert contract["projection_training_mode"] == (
        "exact_highs_straight_through"
    )
    assert contract["straight_through_forward"] == (
        "raw_plus_stop_gradient_exact_minus_raw"
    )
    assert contract["solver"] == "highs"
    assert contract["batch_unit"] == "complete_camera_block_segment"
    assert contract["scale_scope"] == (
        "global_median_over_fit_segments_at_initialization"
    )
    assert contract["scale_component_names"] == [
        "trajectory_outer",
        "trajectory_boundary",
        "gain_outer",
        "gain_boundary",
        "target",
        "gate",
        "action",
    ]
    assert contract["projection_consistency_scale"] == 0.0002
    assert contract["gain_huber_delta"] == 0.005
    assert contract["renderer_trajectory_huber_delta"] == 0.005
    assert contract["target_response_huber_delta"] == 0.005
    assert contract["gate_huber_delta"] == 0.01
    assert contract["temporal_huber_delta"] == 0.005
    assert contract["temporal_loss_weight"] == 0.25
    assert contract["action_cosine_epsilon"] == 1e-12

    assert contract["training_epochs"] == 400
    assert contract["random_seed"] == 20260805
    assert contract["optimizer"] == "AdamW"
    assert contract["learning_rate"] == 0.001
    assert contract["weight_decay"] == 0.0001
    assert contract["gradient_clip_norm"] == 1.0
    assert contract["expected_parameter_count"] == 9073
    assert contract["attention"] is False
    assert contract["carrier_embedding"] is False

    assert contract["minimum_observability_gradient_norm"] == 1e-12
    assert contract["observability_step_count"] == 1
    assert contract["maximum_fit_projected_teacher_mae"] == 0.0065
    assert contract["minimum_fit_outer_recovery"] == 0.75
    assert contract["minimum_fit_boundary_recovery"] == 0.75
    assert contract["minimum_fit_positive_segment_fraction"] == 0.95
    assert contract["minimum_fit_action_cosine"] == 0.90
    assert contract["minimum_fit_top_k_overlap"] == 0.45
    assert contract["maximum_fit_missed_suppression_fraction"] == 0.55
    assert contract["maximum_fit_raw_to_exact_mae"] == 0.0002
    assert contract["maximum_fit_projection_changed_fraction"] == 0.05
    assert contract["projection_changed_threshold"] == 1e-12

    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17", "c18", "c19", "c20", "c21", "c22", "c23"
    ]
    assert contract["open_held_after_fit_fold_count"] == 6
    assert contract["observability_negative_status"] == (
        "FEATURE_OBSERVABILITY_NEGATIVE"
    )
    assert contract["fit_negative_status"] == "FIT_PROJECTED_ENTRY_NEGATIVE"
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False

    assert contract["source_design_sha256"] == (
        "3e71d92496e5d21d3ec2235c683857e260e162ba6e0923c64eb8c96aa907f704"
    )
    assert contract["source_r4a_contract_sha256"] == (
        "d397642d86013eae446c5af1484cfb3f4d537dc79f417c3657fd1e9fc9ddd9e7"
    )
    assert contract["source_r4a_policy_sha256"] == (
        "74fa9528b7683364b1b4dd5a767be16f26c66f80d5d534ec3103bcf63b35bf7f"
    )
    assert contract["source_r4a_trainer_sha256"] == (
        "f627c5147dd912344307568fc74e3b8f403877b8bcd838c2b354065d99082ff3"
    )
    assert contract["source_r4a_auditor_sha256"] == (
        "d8ef2bc7a93bcee8abb3b2bfdc4e808e9af4fb7c2954a5c1848801f21d64d69b"
    )
    assert contract["source_r4a_runner_sha256"] == (
        "ab4d06f4401d50f8c1546fabd9cc033b338516cf141d77cc3bc4734e5292a9ff"
    )
