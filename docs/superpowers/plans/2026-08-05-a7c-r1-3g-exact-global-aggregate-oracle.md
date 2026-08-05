# A7c R1.3-G Exact Global Aggregate Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a deterministic 24-record renderer-oracle replay that saves real gate witnesses and certifies whether the frozen R1.3-P action space passes the unchanged aggregate promotion protocol.

**Architecture:** A frozen JSON contract pins every upstream artifact and threshold. A small pure-function module validates the R1.3-P endpoints, constructs replay requests, checks gate isolation/certificates, and classifies only `CERTIFIED_FEASIBLE`, `UNRESOLVED`, or `ORACLE_ERROR`. One CLI regenerates and saves the six fold witnesses; a separate CLI reloads those saved gates, recomputes renderer metrics, and calls the existing `summarize_records`; a restart-safe shell runner owns timestamps and terminal markers.

**Tech Stack:** Python 3.10, NumPy, SciPy 1.13 HiGHS LP through `solve_fixed_gain_oracle`, PyTorch only for the frozen runtime-mass transform, pytest, Bash.

---

## File Map

- Create `configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json`: immutable experiment contract and source fingerprints.
- Create `utils/a7c_exact_aggregate_oracle.py`: endpoint validation, replay request construction, certificate/isolation validation, and verdict logic.
- Create `tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py`: load frozen sources, solve 24 replay LPs, and write six witness folds plus records/summary.
- Create `tools/audit_a7c_r1_3g_exact_aggregate_oracle.py`: reload saved gates, independently recompute 24 renderer records, and invoke the unchanged formal aggregator.
- Create `tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh`: source checks, restart handling, audit-aware exit handling, timestamps, and exactly one root marker.
- Create `tests/test_a7c_exact_aggregate_oracle.py`: contract, endpoint, replay, isolation, audit, classification, and runner tests.
- Modify `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`: append only the completed R1.3-G command, hashes, verdict, exact metrics, and next-stage decision after the real run.

### Task 1: Freeze The R1.3-G Contract

**Files:**
- Create: `configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json`
- Create: `tests/test_a7c_exact_aggregate_oracle.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json"


def test_r1_3g_contract_freezes_constructive_replay_and_isolation():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["experiment_id"] == "a7c_r1_3g_exact_aggregate_oracle_377_v1"
    assert contract["status"] == "frozen"
    assert contract["source_r1_3p_records_sha256"] == (
        "7a4d3998408a67cb4754d2bcb799e9e4e2ed8518b23fa3254055c4b88f9d3ce8"
    )
    assert contract["source_r1_3p_summary_sha256"] == (
        "82c1d019002a7ee980d0b13c583bf023ae0df5aac8f5d383e513d2cbff12c5c5"
    )
    assert contract["replay_margin"] == 2.0e-5
    assert contract["oracle_bisection_tolerance"] == 1.0e-5
    assert contract["minimum_outer_gain"] == 0.005
    assert contract["maximum_projection_gate_jump"] == 0.015
    assert contract["solver_residual_tolerance"] == 1.0e-7
    assert contract["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert contract["audit_cameras"] == []
    assert contract["forbidden_cameras"] == [
        "c17", "c18", "c19", "c20", "c21", "c22", "c23"
    ]
    assert contract["retrain_predictor"] is False
    assert contract["deployment_eligible"] is False
    assert contract["teacher_eligible"] is False
    assert contract["paper_test_eligible"] is False
```

- [ ] **Step 2: Run the test and verify the missing contract fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_r1_3g_contract_freezes_constructive_replay_and_isolation -v`

Expected: `FAIL` with `FileNotFoundError` for `a7c_r1_3g_exact_aggregate_oracle_377_v1.json`.

- [ ] **Step 3: Create the frozen contract**

Create the JSON by copying the R1.3-P promotion/guard/source fields unchanged, then add these R1.3-G fields exactly:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_3g_exact_aggregate_oracle_377_v1",
  "status": "frozen",
  "subject": "377",
  "fit_cameras": ["c01", "c05", "c09", "c13"],
  "audit_cameras": [],
  "forbidden_cameras": ["c17", "c18", "c19", "c20", "c21", "c22", "c23"],
  "frame_start": 0,
  "frame_end": 570,
  "frame_stride": 5,
  "temporal_block_count": 6,
  "part": "lower",
  "minimum_gate": 0.9,
  "maximum_gate": 1.0,
  "selection_threshold": 0.2,
  "proxy_target_response": 0.995,
  "maximum_projection_gate_jump": 0.015,
  "solver": "highs",
  "solver_primal_tolerance": 1e-9,
  "solver_residual_tolerance": 1e-7,
  "minimum_outer_gain": 0.005,
  "minimum_boundary_gain": 0.005,
  "minimum_positive_block_fraction": 0.9,
  "block_gain_quantile": 0.1,
  "minimum_block_gain_quantile": 0.0,
  "maximum_worst_block_regression": 0.005,
  "minimum_target_response": 0.99,
  "maximum_selection_soft_iou_drop": 0.005,
  "maximum_adjacent_gate_change": 0.02,
  "r1_1_f1_outer_gain": -0.00012761059760764496,
  "r1_1_f1_boundary_gain": 0.023481874880317264,
  "oracle_bisection_tolerance": 1e-5,
  "replay_margin": 2e-5,
  "source_r1_3p_contract": "configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json",
  "source_r1_3p_contract_sha256": "a62d99f65d1358d2b985db3c5dec5221396a7fb1c8cbf287abc8943788f4c61c",
  "source_r1_3p_records": "exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/records.json",
  "source_r1_3p_records_sha256": "7a4d3998408a67cb4754d2bcb799e9e4e2ed8518b23fa3254055c4b88f9d3ce8",
  "source_r1_3p_summary": "exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/summary.json",
  "source_r1_3p_summary_sha256": "82c1d019002a7ee980d0b13c583bf023ae0df5aac8f5d383e513d2cbff12c5c5",
  "source_r1_3g_design": "docs/superpowers/specs/2026-08-05-a7c-r1-3g-exact-global-aggregate-oracle-design.md",
  "source_r1_3g_design_sha256": "839a7624848a56e4d15b3e81e3b146a0ac897f3eadf17c982381083bc2949d92",
  "source_probe": "exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz",
  "source_probe_sha256": "643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411",
  "source_teacher": "exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz",
  "source_teacher_sha256": "698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8",
  "source_evidence": "exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz",
  "source_evidence_sha256": "8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4",
  "source_a5_bank": "exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz",
  "source_a5_bank_sha256": "49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3",
  "preserve_a5_selection_topology": true,
  "retrain_predictor": false,
  "deployment_eligible": false,
  "teacher_eligible": false,
  "paper_test_eligible": false
}
```

- [ ] **Step 4: Run the contract test**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_r1_3g_contract_freezes_constructive_replay_and_isolation -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit the frozen contract**

```bash
git add configs/semantic/a7c_r1_3g_exact_aggregate_oracle_377_v1.json tests/test_a7c_exact_aggregate_oracle.py
git commit -m "test: freeze R1.3-G aggregate oracle contract"
```

### Task 2: Validate Endpoints And Construct Replay Requests

**Files:**
- Create: `utils/a7c_exact_aggregate_oracle.py`
- Modify: `tests/test_a7c_exact_aggregate_oracle.py`

- [ ] **Step 1: Add failing tests for exact endpoint extraction**

```python
import copy
import pytest


def _source_records():
    return [
        {
            "fold": fold,
            "camera_index": camera,
            "boundary_conditioned": {
                "status": "bracketed",
                "feasible_lower": 0.03 + 0.001 * fold + 0.0001 * camera,
                "infeasible_upper": 0.030009 + 0.001 * fold + 0.0001 * camera,
                "interval_width": 9.0e-6,
            },
        }
        for fold in range(6)
        for camera in range(4)
    ]


def test_extract_replay_requests_requires_exact_24_record_grid():
    from utils.a7c_exact_aggregate_oracle import extract_replay_requests

    rows = extract_replay_requests(
        _source_records(), replay_margin=2.0e-5, maximum_interval_width=1.0e-5
    )
    assert len(rows) == 24
    assert rows[0]["minimum_outer_gain"] == 0.005
    assert rows[0]["minimum_boundary_gain"] == pytest.approx(0.02998)

    for mutation, message in (
        (lambda records: records.pop(), "exactly 24"),
        (lambda records: records.append(copy.deepcopy(records[0])), "duplicate"),
        (lambda records: records[0]["boundary_conditioned"].update(status="conditioning_infeasible"), "bracketed"),
        (lambda records: records[0]["boundary_conditioned"].update(interval_width=1.1e-5), "interval width"),
    ):
        broken = _source_records()
        mutation(broken)
        with pytest.raises(ValueError, match=message):
            extract_replay_requests(
                broken, replay_margin=2.0e-5, maximum_interval_width=1.0e-5
            )
```

- [ ] **Step 2: Run the endpoint test and verify import failure**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_extract_replay_requests_requires_exact_24_record_grid -v`

Expected: `FAIL` with `ModuleNotFoundError: utils.a7c_exact_aggregate_oracle`.

- [ ] **Step 3: Implement endpoint validation and request construction**

```python
from __future__ import annotations

import numpy as np


def extract_replay_requests(
    records, *, replay_margin: float, maximum_interval_width: float
) -> list[dict]:
    rows = list(records)
    if len(rows) != 24:
        raise ValueError("replay source requires exactly 24 records")
    expected = {(fold, camera) for fold in range(6) for camera in range(4)}
    observed = [(int(row["fold"]), int(row["camera_index"])) for row in rows]
    if len(set(observed)) != len(observed):
        raise ValueError("duplicate fold-camera replay record")
    if set(observed) != expected:
        raise ValueError("replay source fold-camera grid differs")
    output = []
    for row in sorted(rows, key=lambda value: (value["fold"], value["camera_index"])):
        endpoint = row["boundary_conditioned"]
        if endpoint.get("status") != "bracketed":
            raise ValueError("every replay endpoint must be bracketed")
        lower = float(endpoint["feasible_lower"])
        upper = float(endpoint["infeasible_upper"])
        width = float(endpoint["interval_width"])
        if not np.isfinite([lower, upper, width]).all() or width < 0.0:
            raise ValueError("replay endpoint values must be finite")
        if width > float(maximum_interval_width) or abs((upper - lower) - width) > 1e-12:
            raise ValueError("replay endpoint interval width differs")
        output.append({
            "fold": int(row["fold"]),
            "camera_index": int(row["camera_index"]),
            "source_feasible_lower": lower,
            "source_infeasible_upper": upper,
            "source_interval_width": width,
            "minimum_outer_gain": 0.005,
            "minimum_boundary_gain": lower - float(replay_margin),
        })
    return output
```

- [ ] **Step 4: Run the endpoint tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py -k 'extract_replay_requests' -v`

Expected: all selected tests pass.

- [ ] **Step 5: Commit endpoint construction**

```bash
git add utils/a7c_exact_aggregate_oracle.py tests/test_a7c_exact_aggregate_oracle.py
git commit -m "feat: validate R1.3-G replay endpoints"
```

### Task 3: Build And Certify Fold Witnesses

**Files:**
- Modify: `utils/a7c_exact_aggregate_oracle.py`
- Create: `tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py`
- Modify: `tests/test_a7c_exact_aggregate_oracle.py`

- [ ] **Step 1: Add failing tests for witness isolation and direct certificates**

```python
import numpy as np


def test_insert_replay_segment_isolates_nonheld_samples_and_checks_certificate():
    from utils.a7c_exact_aggregate_oracle import insert_replay_segment

    gates = np.full((8, 2), np.nan)
    replay_mask = np.zeros(8, dtype=bool)
    selected = np.array([True, True, False, False, False, False, False, False])
    solved = {
        "gates": np.array([[0.99, 1.0], [0.98, 0.99]]),
        "metrics": {
            "outer_gain": 0.006,
            "boundary_gain": 0.03,
            "minimum_target_response": 0.995,
            "maximum_soft_iou_drop": 0.001,
            "maximum_adjacent_gate_change": 0.01,
        },
        "certificate": {"maximum_primal_violation": 1e-9, "status": 0},
        "sample_order_fingerprint": "sample-order",
        "carrier_order_fingerprint": "carrier-order",
    }
    row = insert_replay_segment(
        replay_gates=gates,
        replay_mask=replay_mask,
        selected=selected,
        solved=solved,
        request={
            "fold": 0,
            "camera_index": 0,
            "source_feasible_lower": 0.03002,
            "source_infeasible_upper": 0.030029,
            "source_interval_width": 9e-6,
            "minimum_outer_gain": 0.005,
            "minimum_boundary_gain": 0.03,
        },
        frame_index=np.array([0, 5, 0, 5, 0, 5, 0, 5]),
        block_ids=np.zeros(8, dtype=np.int16),
        carrier_ids=np.array([10, 11]),
        residual_tolerance=1e-7,
    )
    assert np.isfinite(gates[selected]).all()
    assert np.isnan(gates[~selected]).all()
    assert replay_mask.tolist() == selected.tolist()
    assert row["minimum_topology_slack"] >= 0.0
    assert row["maximum_primal_violation"] <= 1e-7


def test_fixed_gain_replay_is_deterministic():
    from utils.a7c_feasibility_oracle import solve_fixed_gain_oracle

    base = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    point = base[:, None]
    streams = {
        "objective": {
            "outer": {"base": base, "point": point},
            "boundary": {"base": base, "point": point},
        },
        "guard": {
            "target": {"base": np.ones(5), "point": np.zeros((5, 1))},
            "outer": {"base": base, "point": point},
        },
    }

    kwargs = {
        "runtime_mass": np.zeros((5, 1)),
        "a5_weight": np.array([0.8]),
        "streams": streams,
        "minimum_gate": 0.9,
        "maximum_gate": 1.0,
        "selection_threshold": 0.2,
        "proxy_target_response": 0.995,
        "maximum_gate_jump": 0.015,
        "minimum_target_response": 0.99,
        "maximum_soft_iou_drop": 0.005,
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
    }
    first = solve_fixed_gain_oracle(**kwargs)
    second = solve_fixed_gain_oracle(**kwargs)
    np.testing.assert_array_equal(first["gates"], second["gates"])
    assert first["metrics"] == second["metrics"]
```

- [ ] **Step 2: Run the witness test and verify the missing function fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_insert_replay_segment_isolates_nonheld_samples_and_checks_certificate -v`

Expected: `FAIL` with `ImportError` for `insert_replay_segment`.

- [ ] **Step 3: Implement the fold insertion and certificate function**

Add `insert_replay_segment(...)` to `utils/a7c_exact_aggregate_oracle.py`. It must reject overlap, wrong shape, non-finite gates, solver status other than zero, or `maximum_primal_violation > residual_tolerance`; write only `selected`; and return a flattened JSON certificate containing request fields, direct metrics, solver fields, frame/block/carrier counts, maximum-jump frame/carrier location, source fingerprints, sample-order fingerprint, carrier-order fingerprint, and minimum slack values. Compute topology/bound/proxy slack and the jump location from the actual solved arrays in the evaluator and pass them in `solved["slack"]` / `solved["locations"]`; do not infer proxy slack from renderer metrics.

```python
def insert_replay_segment(
    *, replay_gates, replay_mask, selected, solved, request,
    frame_index, block_ids, carrier_ids, residual_tolerance
) -> dict:
    mask = np.asarray(selected, dtype=bool)
    values = np.asarray(solved["gates"], dtype=np.float64)
    if values.shape != (int(mask.sum()), replay_gates.shape[1]):
        raise ValueError("replay gate shape differs from selected segment")
    if np.any(replay_mask[mask]):
        raise ValueError("replay segments overlap")
    certificate = dict(solved["certificate"])
    if int(certificate["status"]) != 0:
        raise RuntimeError("replay solver is not optimal")
    violation = float(certificate["maximum_primal_violation"])
    if not np.isfinite(values).all() or violation > float(residual_tolerance):
        raise RuntimeError("replay witness certificate failed")
    replay_gates[mask] = values
    replay_mask[mask] = True
    indices = np.flatnonzero(mask)
    return {
        **request,
        **solved["metrics"],
        **certificate,
        **solved.get("locations", {}),
        **solved.get("slack", {
            "minimum_topology_slack": 0.0,
            "minimum_bound_slack": 0.0,
            "minimum_proxy_target_slack": 0.0,
        }),
        "block_index": int(np.unique(np.asarray(block_ids)[mask]).item()),
        "first_frame": int(np.asarray(frame_index)[indices[0]]),
        "last_frame": int(np.asarray(frame_index)[indices[-1]]),
        "sample_count": int(mask.sum()),
        "carrier_count": int(np.asarray(carrier_ids).size),
        "source_fingerprints": dict(solved.get("source_fingerprints", {})),
        "sample_order_fingerprint": str(solved["sample_order_fingerprint"]),
        "carrier_order_fingerprint": str(solved["carrier_order_fingerprint"]),
    }
```

- [ ] **Step 4: Implement the replay CLI by reusing frozen loaders and equations**

In `tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py`:

1. Parse `--contract --source-records --probe --evidence --a5-bank --teacher --output-dir`.
2. Verify contract, design, source records, source summary, probe, evidence, A5 bank, and teacher SHA256 through `verify_source_file`.
3. Require the frozen R1.3-P summary verdict to be `UNRESOLVED`.
4. Load manifests exactly as R1.3-P does and build `runtime_mass`, renderer streams, `sample_block_ids`, and `build_canary_splits`.
5. Call `extract_replay_requests` once, indexed by `(fold, camera_index)`.
6. For each fold and camera 0-3, select `held & fit_mask & (camera_index == camera)`, verify one block and contiguous `frame_stride`, slice streams, then call:

```python
solved = solve_fixed_gain_oracle(
    runtime_mass=runtime_mass[selected],
    a5_weight=a5_weight,
    streams=record_streams,
    minimum_gate=float(contract["minimum_gate"]),
    maximum_gate=float(contract["maximum_gate"]),
    selection_threshold=float(contract["selection_threshold"]),
    proxy_target_response=float(contract["proxy_target_response"]),
    maximum_gate_jump=float(contract["maximum_projection_gate_jump"]),
    minimum_target_response=float(contract["minimum_target_response"]),
    maximum_soft_iou_drop=float(contract["maximum_selection_soft_iou_drop"]),
    minimum_outer_gain=request["minimum_outer_gain"],
    minimum_boundary_gain=request["minimum_boundary_gain"],
    primal_tolerance=float(contract["solver_primal_tolerance"]),
    residual_tolerance=float(contract["solver_residual_tolerance"]),
)
```

7. Write `witness/fold_N/predictions.npz` with `replay_gates`, `replay_mask`, `camera_index`, `frame_index`, `block_ids`, `carrier_ids`, per-sample `requested_outer_gain`, `requested_boundary_gain`, `source_feasible_lower`, `source_infeasible_upper`, source hashes, sample/carrier order fingerprints, and uint8-zero values for all three eligibility flags. Request/endpoint arrays are finite only under `replay_mask` and NaN elsewhere.
8. Write `witness/fold_N/certificates.json`, `witness/summary.json`, root `records.json`, and a provisional root `summary.json` with `execution_status="REPLAY_COMPLETED"`, `verdict="UNRESOLVED"`, `aggregate_audit_opened=false`, and every eligibility flag false.
9. Catch source/solver/numeric errors in `main`, overwrite root `summary.json` with `execution_status="ORACLE_ERROR"`, error type/message, and return 1.

- [ ] **Step 5: Run the focused witness tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py -k 'replay or witness or contract' -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit witness generation**

```bash
git add utils/a7c_exact_aggregate_oracle.py tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py tests/test_a7c_exact_aggregate_oracle.py
git commit -m "feat: generate certified R1.3-G replay witnesses"
```

### Task 4: Independently Audit Saved Gate Arrays

**Files:**
- Create: `tools/audit_a7c_r1_3g_exact_aggregate_oracle.py`
- Modify: `utils/a7c_exact_aggregate_oracle.py`
- Modify: `tests/test_a7c_exact_aggregate_oracle.py`

- [ ] **Step 1: Add failing tests for compensation and verdicts**

```python
def _promotion_contract():
    return {
        "minimum_outer_gain": 0.005,
        "minimum_boundary_gain": 0.005,
        "minimum_positive_block_fraction": 0.9,
        "block_gain_quantile": 0.1,
        "minimum_block_gain_quantile": 0.0,
        "maximum_worst_block_regression": 0.005,
        "minimum_target_response": 0.99,
        "maximum_selection_soft_iou_drop": 0.005,
        "maximum_adjacent_gate_change": 0.02,
        "r1_1_f1_outer_gain": -0.00012761059760764496,
        "r1_1_f1_boundary_gain": 0.023481874880317264,
    }


def test_exact_aggregate_accepts_compensation_below_r1_1_boundary_mean():
    from tools.audit_a7c_r1_2a_quotient_compositor import summarize_records

    records = []
    for index in range(24):
        records.append({
            "outer_gain": 0.006,
            "boundary_gain": 0.02 if index < 3 else 0.03,
            "minimum_target_response": 0.995,
            "maximum_soft_iou_drop": 0.001,
            "maximum_adjacent_gate_change": 0.015,
        })
    summary = summarize_records(records, _promotion_contract())
    assert min(row["boundary_gain"] for row in records) < 0.023481874880317264
    assert summary["boundary_gain"] > 0.023481874880317264
    assert summary["passed"] is True


def test_classify_exact_replay_never_claims_infeasibility():
    from utils.a7c_exact_aggregate_oracle import classify_exact_replay

    assert classify_exact_replay(replay_complete=True, audit_passed=True) == "CERTIFIED_FEASIBLE"
    assert classify_exact_replay(replay_complete=True, audit_passed=False) == "UNRESOLVED"


def test_validate_saved_manifest_rejects_order_or_eligibility_change():
    from utils.a7c_exact_aggregate_oracle import validate_saved_manifest

    expected = {
        "camera_index": np.array([0, 0, 1, 1]),
        "frame_index": np.array([0, 5, 0, 5]),
        "block_ids": np.array([0, 0, 0, 0]),
        "carrier_ids": np.array([10, 11]),
    }
    saved = {**expected, "deployment_eligible": np.array(0),
             "teacher_eligible": np.array(0), "paper_test_eligible": np.array(0)}
    validate_saved_manifest(saved, expected)
    broken = dict(saved)
    broken["frame_index"] = np.array([5, 0, 0, 5])
    with pytest.raises(ValueError, match="frame_index"):
        validate_saved_manifest(broken, expected)
    broken = dict(saved)
    broken["teacher_eligible"] = np.array(1)
    with pytest.raises(ValueError, match="teacher_eligible"):
        validate_saved_manifest(broken, expected)
```

- [ ] **Step 2: Run the tests and verify the verdict function fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py -k 'compensation or classify_exact' -v`

Expected: compensation test passes against the existing aggregator; classification test fails on missing import.

- [ ] **Step 3: Implement exact replay classification**

```python
def classify_exact_replay(*, replay_complete: bool, audit_passed: bool) -> str:
    if not replay_complete:
        raise ValueError("incomplete replay cannot be classified")
    return "CERTIFIED_FEASIBLE" if bool(audit_passed) else "UNRESOLVED"


def validate_saved_manifest(saved: dict, expected: dict) -> None:
    for key in ("camera_index", "frame_index", "block_ids", "carrier_ids"):
        if not np.array_equal(np.asarray(saved[key]), np.asarray(expected[key])):
            raise ValueError(f"saved {key} differs from frozen manifest")
    for key in ("deployment_eligible", "teacher_eligible", "paper_test_eligible"):
        if int(np.asarray(saved[key]).item()) != 0:
            raise ValueError(f"saved {key} must be false")
```

- [ ] **Step 4: Implement the independent audit CLI**

In `tools/audit_a7c_r1_3g_exact_aggregate_oracle.py`, reuse only frozen source loaders, `_build_streams`, `build_canary_splits`, `evaluate_contribution_predictions`, and `summarize_records`. Do not import record metrics from the replay evaluator.

For each fold:

```python
with np.load(fold_root / "predictions.npz", allow_pickle=False) as source:
    gates = np.asarray(source["replay_gates"], dtype=np.float64)
    replay_mask = np.asarray(source["replay_mask"], dtype=bool)
    saved_camera = np.asarray(source["camera_index"])
    saved_frame = np.asarray(source["frame_index"])
    saved_blocks = np.asarray(source["block_ids"])
    saved_carriers = np.asarray(source["carrier_ids"])

expected_mask = np.asarray(held & split["fit_mask"], dtype=bool)
if not np.array_equal(replay_mask, expected_mask):
    raise ValueError("replay mask differs from frozen held split")
if np.any(~np.isfinite(gates[expected_mask])) or np.any(np.isfinite(gates[~expected_mask])):
    raise ValueError("replay finite values cross held split")
```

Also call `validate_saved_manifest` for exact equality of the four saved manifest arrays and eligibility values equal to zero. Verify request/endpoint arrays have the same finite/NaN isolation as gates, verify the saved sample/carrier order fingerprints, and recompute every record from `gates[selected]`. Append `fold` and `camera_index`, require 24 unique rows, then call `summarize_records(records, contract)` unchanged. The source A5-bank SHA256 plus `preserve_a5_selection_topology=true` is the topology/coverage/frozen-part/weight-upper-bound guard; reject either mismatch. Write `audit/held_block_summary.json` containing `stage="r1_3g_exact_aggregate"`, summary, records, source/witness hashes, guard status, `audit_camera_metrics_opened=false`, and false eligibility flags. Return 0 for pass and 2 for a correct but rejected audit; return 1 on integrity/runtime error.

- [ ] **Step 5: Run exact-audit and frozen regression tests**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py tests/test_a7c_feasibility_oracle.py tests/test_a7c_temporal_joint_projection.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the independent audit**

```bash
git add utils/a7c_exact_aggregate_oracle.py tools/audit_a7c_r1_3g_exact_aggregate_oracle.py tests/test_a7c_exact_aggregate_oracle.py
git commit -m "feat: audit R1.3-G saved gate witnesses"
```

### Task 5: Add Restart-Safe Runner And Terminal State

**Files:**
- Create: `tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh`
- Modify: `tests/test_a7c_exact_aggregate_oracle.py`

- [ ] **Step 1: Add the failing runner contract test**

```python
import re


RUNNER = ROOT / "tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh"


def test_r1_3g_runner_is_restart_safe_isolated_and_audit_gated():
    source = RUNNER.read_text(encoding="utf-8")
    assert "/opt/miniconda3/envs/ictrl/bin/python" in source
    assert "evaluate_a7c_r1_3g_exact_aggregate_oracle.py" in source
    assert "audit_a7c_r1_3g_exact_aggregate_oracle.py" in source
    assert source.index("evaluate_a7c_r1_3g") < source.index("audit_a7c_r1_3g")
    assert "audit_status" in source
    assert "CERTIFIED_FEASIBLE" in source
    assert "UNRESOLVED" in source
    assert "ORACLE_ERROR" in source
    assert "check_sha" in source
    for camera in ("c17", "c18", "c19", "c20", "c21", "c22", "c23"):
        assert re.search(rf"\b{camera}\b", source) is None
    for artifact in ("runner.pid", "runner.log", "started_utc.txt", "ended_utc.txt"):
        assert artifact in source
    for marker in (".completed", ".rejected", ".failed"):
        assert marker in source
```

- [ ] **Step 2: Run the runner test and verify the missing script fails**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_r1_3g_runner_is_restart_safe_isolated_and_audit_gated -v`

Expected: `FAIL` with `FileNotFoundError` for the runner.

- [ ] **Step 3: Implement the runner**

Use `tools/run_a7c_r1_3p_temporal_joint_377.sh` as the lifecycle pattern. The R1.3-G runner must:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1}"
PYTHON="/opt/miniconda3/envs/ictrl/bin/python"
WITNESS="${OUT}/witness"
AUDIT="${OUT}/audit"

mark_terminal() {
  local marker="$1"
  rm -f "${OUT}/.completed" "${OUT}/.rejected" "${OUT}/.failed"
  touch "${OUT}/.${marker}"
}
```

It must verify all contract-pinned SHA256 values before computation; skip only when every required artifact, timestamp, and exactly one terminal marker exists; set `.failed` and an ERR trap before work; run the evaluator; run the auditor under `set +e`; accept auditor statuses 0 and 2 only; update root `summary.json` atomically through a small inline call to the same Python interpreter with `execution_status="COMPLETED"`, verdict from `classify_exact_replay`, exact audit summary, false eligibility flags, and `aggregate_audit_opened=true`; write `ended_utc.txt`; and select `.completed` only for `CERTIFIED_FEASIBLE`, `.rejected` only for `UNRESOLVED`, `.failed` only for `ORACLE_ERROR`.

- [ ] **Step 4: Make the runner executable and run its contract test**

Run: `chmod +x tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh`

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py::test_r1_3g_runner_is_restart_safe_isolated_and_audit_gated -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit the runner**

```bash
git add tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh tests/test_a7c_exact_aggregate_oracle.py
git commit -m "feat: orchestrate R1.3-G exact aggregate oracle"
```

### Task 6: Verify The Implementation Before The Real Solve

**Files:**
- Modify only if a failing test identifies an R1.3-G defect.

- [ ] **Step 1: Run static syntax checks**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m py_compile utils/a7c_exact_aggregate_oracle.py tools/evaluate_a7c_r1_3g_exact_aggregate_oracle.py tools/audit_a7c_r1_3g_exact_aggregate_oracle.py`

Expected: exit 0 with no output.

- [ ] **Step 2: Run the focused R1.3-G suite**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_exact_aggregate_oracle.py -v`

Expected: all tests pass.

- [ ] **Step 3: Run frozen R1.3-P regressions**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_feasibility_oracle.py tests/test_a7c_temporal_joint_projection.py -v`

Expected: all tests pass with no modifications to R1.3-P behavior.

- [ ] **Step 4: Check scope and whitespace**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only R1.3-G files from this plan plus pre-existing user changes are present.

### Task 7: Execute The Frozen Oracle And Record The Result

**Files:**
- Generate: `exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/**`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Start the exact oracle in the background**

Run:

```bash
nohup bash tools/run_a7c_r1_3g_exact_aggregate_oracle_377.sh \
  exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1 \
  >/tmp/a7c_r1_3g_launcher.log 2>&1 &
```

Expected: the process starts, `runner.pid`, `runner.log`, and `started_utc.txt` appear, and no training process is launched.

- [ ] **Step 2: Monitor to a terminal marker**

Run:

```bash
while ! compgen -G 'exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/.*ed' >/dev/null; do
  tail -n 20 exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/runner.log
  sleep 30
done
```

Expected: exactly one of `.completed`, `.rejected`, or `.failed` appears and the process recorded in `runner.pid` has exited.

- [ ] **Step 3: Verify artifacts and report the exact verdict**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1')
summary = json.loads((root / 'summary.json').read_text())
audit = json.loads((root / 'audit/held_block_summary.json').read_text())
markers = [name for name in ('.completed', '.rejected', '.failed') if (root / name).exists()]
assert len(markers) == 1, markers
assert summary['paper_test_eligible'] is False
assert audit['paper_test_eligible'] is False
print(json.dumps({'marker': markers[0], 'summary': summary, 'audit': audit['summary']}, indent=2, sort_keys=True))
PY
```

Expected: `CERTIFIED_FEASIBLE` with `.completed` if the constructive witness survives exact replay; otherwise report `UNRESOLVED`/`.rejected` or `ORACLE_ERROR`/`.failed` without modifying thresholds.

- [ ] **Step 4: Append the immutable result to the A7 handoff**

Add a dated R1.3-G section containing the contract/design/source hashes, command, start/end UTC and Beijing times, terminal marker, exact 24-record summary, certificate maximum, artifact paths, and decision:

```text
CERTIFIED_FEASIBLE -> action-space capacity established; separately preregister pose-conditioned A7b using fold-fit oracle supervision only.
UNRESOLVED -> do not train; preregister continuous interval branch-and-bound if further capacity diagnosis is justified.
ORACLE_ERROR -> repair only the identified integrity/runtime defect and rerun the identical frozen contract.
```

- [ ] **Step 5: Validate and commit only the result documentation**

Run: `git diff --check -- docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

Then stage only the newly appended hunk, preserving all unrelated user edits in that file, and commit:

```bash
git commit -m "docs: record R1.3-G exact aggregate result"
```

### Task 8: Enforce The Training Boundary

**Files:**
- No code files in this R1.3-G plan.

- [ ] **Step 1: Stop on non-feasible outcomes**

If the verdict is `UNRESOLVED` or `ORACLE_ERROR`, do not create an A7b contract, labels, checkpoints, or training process. Report the failed guards or exact error from the saved summary.

- [ ] **Step 2: On certified feasibility, begin a separate design cycle**

If the verdict is `CERTIFIED_FEASIBLE`, R1.3-G is complete. Before any training, use brainstorming to write and approve a new pose-conditioned A7b design that enforces:

```text
training labels: current fold fit blocks only
held blocks: audit-only
held labels, normalization, hyperparameter selection, early stopping: forbidden
R1.3-G deployment/teacher/paper eligibility: always false
Task 12 and c17-c23: still closed
```

No A7b training command belongs in the R1.3-G runner or result commit.
