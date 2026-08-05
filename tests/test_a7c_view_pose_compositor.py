import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/semantic/a7c_r1_4vp_oracle_distilled_view_pose_377_v1.json"
)


def test_r1_4vp_contract_freezes_model_isolation_and_promotion():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert (
        contract["experiment_id"]
        == "a7c_r1_4vp_oracle_distilled_view_pose_377_v1"
    )
    assert contract["status"] == "frozen"
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17",
        "c18",
        "c19",
        "c20",
        "c21",
        "c22",
        "c23",
    ]
    assert contract["temporal_block_count"] == 6
    assert contract["fit_teacher_segment_count"] == 120
    assert contract["held_audit_record_count"] == 24
    assert contract["processed_parts"] == ["lower"]
    assert contract["frozen_parts"] == [
        "hair",
        "face",
        "upper",
        "shoes",
        "skin",
    ]
    assert contract["min_pair_support"] == 8
    assert contract["minimum_evidence_support_coverage"] == 0.8
    assert contract["pose_body_joint_indices"] == [0, 1, 3, 4, 6, 7]
    assert contract["pose_dimension"] == 36
    assert contract["view_feature_group"] == "F3"
    assert contract["view_embedding_dimension"] == 16
    assert contract["pose_embedding_dimension"] == 16
    assert contract["gru_hidden_dimension"] == 16
    assert contract["maximum_parameter_count"] == 50_000
    assert contract["residual_gate_scale"] == 0.1
    assert contract["training_epochs"] == 400
    assert contract["random_seed"] == 20260805
    assert contract["nearest_neighbor_k"] == 4
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["minimum_boundary_gain"] == 0.005
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["maximum_selection_soft_iou_drop"] == 0.005
    assert contract["maximum_projection_gate_jump"] == 0.015
    assert contract["maximum_adjacent_gate_change"] == 0.02
    assert contract["r1_2b_outer_gain"] == 0.005196372744170267
    assert contract["r1_2b_boundary_gain"] == 0.002866365549963367
    assert contract["source_pose_manifest_sha256"] == (
        "5d138f7f06ffaccb6b9a59d538028f0f298f0c538ea6845a49e9e6c2eda6f116"
    )
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False
