import json


def test_v5_3_summary_uses_capacity_and_retrospective_only(tmp_path, monkeypatch):
    import tools.summarize_a7_v5_3_development as module

    contract = {
        "_fingerprint": "c" * 64,
        "freeze_id": "a7_dual_evidence_v5_3_canary_377",
        "retrospective_test_cameras": ["c21", "c22", "c23"],
        "maximum_audit_visibility_response_ratio": 1.0,
        "minimum_audit_target_response_ratio": 0.99,
    }
    candidate = {
        "candidate_id": "dual_evidence_constrained_v5_3",
        "capacity_summary": {
            "camera_ids": list(range(8)),
            "all_folds_passed": True,
            "final": {
                "construction_evaluation": {"passed": True},
                "evaluation": {"passed": True},
            },
        },
    }
    monkeypatch.setattr(module, "load_a7_temporal_contract", lambda *_: contract)
    monkeypatch.setattr(
        module,
        "load_validated_candidate",
        lambda *_: (candidate, "b" * 64),
    )
    temporal_calls = []

    def fake_temporal(*args, **kwargs):
        temporal_calls.append((args, kwargs))
        return ({"outer_ratio": 0.99, "boundary_ratio": 0.99}, [])

    monkeypatch.setattr(module, "_summarize_temporal_group", fake_temporal)
    monkeypatch.setattr(
        module,
        "_summarize_spatial",
        lambda *_: ({"passed": True}, []),
    )
    output = tmp_path / "summary.json"

    assert module.main(
        [
            "--candidate-index", str(tmp_path / "candidate.json"),
            "--a7-contract", str(tmp_path / "contract.json"),
            "--method-freeze", str(tmp_path / "a5.json"),
            "--temporal-root", str(tmp_path / "temporal"),
            "--spatial-root", str(tmp_path / "spatial"),
            "--output", str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "dual_evidence_constrained_v5_3"
    assert payload["construction"]["camera_ids"] == list(range(8))
    assert payload["construction"]["all_folds_passed"] is True
    assert payload["paper_test_eligible"] is False
    assert payload["development_passed"] is True
    assert len(temporal_calls) == 1
    assert temporal_calls[0][1]["cameras"] == ["c21", "c22", "c23"]
    assert temporal_calls[0][1]["constrained_parts"] == module.ACTIVE_PARTS


def test_v5_3_summary_propagates_retrospective_gate_failure(tmp_path, monkeypatch):
    import tools.summarize_a7_v5_3_development as module

    contract = {
        "_fingerprint": "c" * 64,
        "retrospective_test_cameras": ["c21", "c22", "c23"],
        "maximum_audit_visibility_response_ratio": 1.0,
        "minimum_audit_target_response_ratio": 0.99,
    }
    candidate = {
        "candidate_id": "dual_evidence_constrained_v5_3",
        "capacity_summary": {"camera_ids": list(range(8)), "all_folds_passed": True},
    }
    monkeypatch.setattr(module, "load_a7_temporal_contract", lambda *_: contract)
    monkeypatch.setattr(module, "load_validated_candidate", lambda *_: (candidate, "b" * 64))
    monkeypatch.setattr(
        module,
        "_summarize_temporal_group",
        lambda *_, **__: ({}, ["visibility_response"]),
    )
    monkeypatch.setattr(module, "_summarize_spatial", lambda *_: ({}, []))
    output = tmp_path / "summary.json"

    assert module.main(
        [
            "--candidate-index", str(tmp_path / "candidate.json"),
            "--a7-contract", str(tmp_path / "contract.json"),
            "--method-freeze", str(tmp_path / "a5.json"),
            "--temporal-root", str(tmp_path / "temporal"),
            "--spatial-root", str(tmp_path / "spatial"),
            "--output", str(output),
        ]
    ) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["development_passed"] is False
    assert payload["invalid_reasons"] == ["retrospective_visibility_response"]
