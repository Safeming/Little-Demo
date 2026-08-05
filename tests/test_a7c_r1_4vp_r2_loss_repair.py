import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_4vp_r2_loss_scale_repair_377_v1.json"


def test_r2_contract_changes_only_registered_loss_and_integrity_behavior():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "a7c_r1_4vp_r2_loss_scale_repair_377_v1"
    assert contract["status"] == "frozen"
    assert contract["residual_loss_weight"] == 0.00001
    assert contract["training_epochs"] == 400
    assert contract["maximum_fit_teacher_mae"] == 0.007
    assert contract["require_final_fit_loss_improvement"] is True
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["temporal_block_count"] == 6
    assert contract["view_feature_group"] == "F3"
    assert contract["maximum_parameter_count"] == 50_000
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["minimum_boundary_gain"] == 0.005
    assert contract["maximum_visibility_response_ratio"] == 1.0
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False


def test_registered_loss_can_move_off_anchor_without_canceling_huber_gradient():
    from utils.a7c_r1_4vp_r2 import r2_distillation_loss

    residual = torch.zeros((2, 1), requires_grad=True)
    base = torch.full((2, 1), 0.97)
    teacher = torch.full((2, 1), 0.95)
    prediction = base + 0.1 * torch.tanh(residual)
    components = r2_distillation_loss(
        prediction,
        teacher,
        residual,
        gate_delta=0.01,
        temporal_delta=0.005,
        temporal_weight=0.25,
        residual_weight=0.00001,
    )
    components["loss"].backward()
    assert residual.grad.mean().item() > 0.0
    assert abs(residual.grad.mean().item()) >= 0.00049


def test_fit_integrity_requires_loss_improvement_and_mae_limit():
    from utils.a7c_r1_4vp_r2 import require_fit_integrity

    require_fit_integrity(1.0e-4, 5.0e-5, 0.006, 0.007)
    with pytest.raises(RuntimeError, match="fit loss did not improve"):
        require_fit_integrity(1.0e-4, 1.0e-4, 0.006, 0.007)
    with pytest.raises(RuntimeError, match="fit teacher MAE"):
        require_fit_integrity(1.0e-4, 5.0e-5, 0.008, 0.007)


def test_topology_guard_broadcasts_base_mask_and_rejects_real_undercrossing():
    from utils.a7c_r1_4vp_r2 import evaluate_topology_guard

    base = np.array([0.3, 0.4])
    candidate = np.array([[0.29, 0.39], [0.28, 0.38]])
    passed = evaluate_topology_guard(base, candidate, threshold=0.2)
    assert passed["passed"] is True
    assert passed["mismatch_count"] == 0
    broken = candidate.copy()
    broken[0, 0] = 0.19
    failed = evaluate_topology_guard(base, broken, threshold=0.2)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1


def test_topology_slack_is_reported_only_for_a5_selected_carriers():
    from utils.a7c_r1_4vp_r2 import evaluate_topology_guard

    result = evaluate_topology_guard(
        np.array([0.3, 0.1]), np.array([[0.29, 0.09], [0.28, 0.08]]), 0.2
    )
    assert result["passed"] is True
    assert result["minimum_slack"] == pytest.approx(0.08)


def test_terminal_classification_preserves_negative_audit_status():
    from utils.a7c_r1_4vp_r2 import classify_terminal_status

    assert classify_terminal_status(0, "CANARY_PROMOTED") == "completed"
    assert classify_terminal_status(2, "CANARY_NEGATIVE") == "rejected"
    assert classify_terminal_status(1, "TRAINING_ERROR") == "failed"


def test_train_fold_uses_fit_only_teacher_and_final_checkpoint(tmp_path):
    from tools.train_a7c_r1_4vp_r2_loss_repair import train_fold

    rng = np.random.default_rng(7)
    samples, carriers, channels = 8, 3, 5
    teacher_mask = np.array([True] * 4 + [False] * 4)
    teacher = np.full((samples, carriers), np.nan, dtype=np.float32)
    teacher[teacher_mask] = 0.965
    contract = {
        "frame_stride": 1,
        "random_seed": 17,
        "view_embedding_dimension": 4,
        "pose_embedding_dimension": 4,
        "gru_hidden_dimension": 4,
        "residual_gate_scale": 0.1,
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "maximum_parameter_count": 50_000,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "training_epochs": 20,
        "gradient_clip_norm": 1.0,
        "gate_huber_delta": 0.01,
        "temporal_huber_delta": 0.005,
        "temporal_loss_weight": 0.25,
        "residual_loss_weight": 0.00001,
        "maximum_fit_teacher_mae": 0.007,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_projection_gate_jump": 0.015,
        "lexicographic_tolerance": 1e-9,
        "solver_primal_tolerance": 1e-9,
        "solver_residual_tolerance": 1e-7,
    }
    summary = train_fold(
        fold=0,
        features=rng.normal(size=(samples, carriers, channels)).astype(np.float32),
        pose=rng.normal(size=(samples, 36)).astype(np.float32),
        adjacency=np.broadcast_to(np.eye(carriers), (samples, carriers, carriers)),
        visibility=np.ones((samples, carriers), dtype=np.float32),
        base_gates=np.full((samples, carriers), 0.97, dtype=np.float32),
        teacher_gates=teacher,
        teacher_mask=teacher_mask,
        prediction_mask=np.ones(samples, dtype=bool),
        camera_index=np.zeros(samples, dtype=np.int64),
        frame_index=np.arange(samples, dtype=np.int64),
        block_ids=np.array([0] * 4 + [1] * 4, dtype=np.int64),
        runtime_mass=np.ones((samples, carriers), dtype=np.float32),
        a5_weight=np.full(carriers, 0.3, dtype=np.float32),
        contract=contract,
        output_dir=tmp_path,
        device="cpu",
    )
    assert summary["checkpoint_epoch"] == 20
    assert summary["final_components"]["loss"] < summary["initial_components"]["loss"]
    assert summary["fit_teacher_mae"] <= 0.007
    assert summary["held_teacher_values_accessed"] is False
    assert summary["residual_loss_weight"] == 0.00001
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "predictions.npz").is_file()


def test_r2_freeze_manifest_requires_exactly_24_learned_artifacts(tmp_path):
    from tools.audit_a7c_r1_4vp_r2_loss_repair import verify_frozen_artifacts

    artifacts = {}
    for fold in range(6):
        root = tmp_path / "training" / f"fold_{fold}"
        root.mkdir(parents=True)
        for name in ("model.pt", "predictions.npz", "projection_certificates.json", "summary.json"):
            path = root / name
            path.write_bytes(f"{fold}:{name}".encode())
            import hashlib
            artifacts[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "models_frozen.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    observed = verify_frozen_artifacts(tmp_path)
    assert observed == artifacts


def test_runner_preserves_audit_exit_two_as_rejected():
    runner = ROOT / "tools/run_a7c_r1_4vp_r2_loss_repair_377.sh"
    text = runner.read_text(encoding="utf-8")
    assert 'if "${PYTHON}" "${ROOT}/tools/audit_a7c_r1_4vp_r2_loss_repair.py"' in text
    assert 'audit_status=$?' in text
    assert 'CANARY_NEGATIVE) mark_terminal rejected' in text
    assert 'CANARY_NEGATIVE) mark_terminal failed' not in text
