import json


def test_v5_4_summary_reports_capacity_and_spatial_without_retrospective(tmp_path, monkeypatch):
    import tools.summarize_a7_v5_4_development as module

    contract = {"_fingerprint": "c" * 64}
    candidate = {
        "candidate_id": "dual_evidence_camera_time_v5_4",
        "capacity_summary": {
            "fold_count": 48,
            "all_folds_passed": True,
            "consensus": {
                "minimum_fold_count": 36,
                "selected_indices": [1, 2],
            },
            "final": {
                "passed": True,
                "lower_outer_gain": 0.016,
                "lower_boundary_gain": 0.0155,
                "block_violation": 0.0,
            },
        },
    }
    monkeypatch.setattr(module, "load_a7_temporal_contract", lambda *_: contract)
    monkeypatch.setattr(module, "load_validated_candidate", lambda *_: (candidate, "b" * 64))
    monkeypatch.setattr(module, "_summarize_spatial", lambda *_: ({"passed": True}, []))
    output = tmp_path / "summary.json"

    assert module.main(
        [
            "--candidate-index", str(tmp_path / "candidate.json"),
            "--a7-contract", str(tmp_path / "contract.json"),
            "--method-freeze", str(tmp_path / "a5.json"),
            "--spatial-root", str(tmp_path / "spatial"),
            "--output", str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["development_passed"] is True
    assert payload["paper_test_eligible"] is False
    assert payload["capacity"]["fold_count"] == 48
    assert payload["capacity"]["minimum_stability_selection_count"] == 36
    assert "retrospective" not in payload


def test_v5_4_summary_propagates_spatial_failure(tmp_path, monkeypatch):
    import tools.summarize_a7_v5_4_development as module

    candidate = {
        "candidate_id": "dual_evidence_camera_time_v5_4",
        "capacity_summary": {
            "fold_count": 48,
            "all_folds_passed": True,
            "consensus": {"minimum_fold_count": 36, "selected_indices": []},
            "final": {"passed": True, "block_violation": 0.0},
        },
    }
    monkeypatch.setattr(module, "load_a7_temporal_contract", lambda *_: {"_fingerprint": "c" * 64})
    monkeypatch.setattr(module, "load_validated_candidate", lambda *_: (candidate, "b" * 64))
    monkeypatch.setattr(module, "_summarize_spatial", lambda *_: ({}, ["soft_iou:lower"]))
    output = tmp_path / "summary.json"

    assert module.main(
        [
            "--candidate-index", str(tmp_path / "candidate.json"),
            "--a7-contract", str(tmp_path / "contract.json"),
            "--method-freeze", str(tmp_path / "a5.json"),
            "--spatial-root", str(tmp_path / "spatial"),
            "--output", str(output),
        ]
    ) == 2
    assert json.loads(output.read_text())["invalid_reasons"] == ["spatial_soft_iou:lower"]
