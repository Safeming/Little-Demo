# A7c R1.3-P Temporal Joint Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the preregistered R1.3-P label-free temporal joint hard projector and renderer-evidence feasibility oracle on the frozen CoreView377 R1.2-B held-block predictions.

**Architecture:** A SciPy/HiGHS runtime module performs lexicographic sparse LP projection without importing renderer-evidence code. A separate oracle module adds affine renderer guards and normalized-flicker constraints; thin CLI tools load frozen artifacts, run the 24 held records, audit through the unchanged promotion aggregator, and emit fail-closed certificates.

**Tech Stack:** Python 3.10, NumPy, SciPy sparse matrices, `scipy.optimize.linprog(method="highs")`, pytest, Bash, and existing A7c artifact loaders.

---

## File Map

- Create `configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json`: frozen sources, solver values, margins, and audit gates.
- Create `utils/a7c_temporal_joint_projection.py`: runtime-safe two-stage LP and certificates.
- Create `utils/a7c_feasibility_oracle.py`: renderer constraints, capacity bisection, and verdict.
- Create `tools/project_a7c_r1_3p_temporal_joint.py`: project the 24 held camera-block records.
- Create `tools/audit_a7c_r1_3p_temporal_joint_projection.py`: exact frozen promotion audit.
- Create `tools/evaluate_a7c_r1_3p_feasibility_oracle.py`: run capacity searches and write oracle artifacts.
- Create `tools/run_a7c_r1_3p_temporal_joint_377.sh`: restart-safe formal workflow.
- Create `tests/test_a7c_temporal_joint_projection.py`: runtime solver, CLI, and runner tests.
- Create `tests/test_a7c_feasibility_oracle.py`: oracle math and verdict tests.
- Modify `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`: append the frozen result after the formal run.

### Task 1: Freeze The R1.3-P Contract

**Files:**
- Create: `configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json`
- Create: `tests/test_a7c_temporal_joint_projection.py`

- [ ] **Step 1: Write the failing contract test**

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json"

def test_r1_3p_contract_freezes_runtime_and_oracle_boundaries():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["status"] == "frozen"
    assert c["source_experiment_id"] == "a7c_r1_2b_dense_overlap_set_377_v1"
    assert c["offline_bidirectional"] is True
    assert c["maximum_projection_gate_jump"] == 0.015
    assert c["maximum_adjacent_gate_change"] == 0.02
    assert c["proxy_target_response"] == 0.995
    assert c["minimum_target_response"] == 0.99
    assert c["maximum_selection_soft_iou_drop"] == 0.005
    assert c["solver"] == "highs"
    assert c["solver_residual_tolerance"] == 1e-7
    assert c["oracle_bisection_tolerance"] == 1e-5
    assert c["fit_cameras"] == ["c01", "c05", "c09", "c13"]
    assert c["forbidden_cameras"] == ["c17", "c18", "c19", "c20", "c21", "c22", "c23"]
    assert c["retrain_predictor"] is False
    assert c["paper_test_eligible"] is False
```

- [ ] **Step 2: Run it and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py::test_r1_3p_contract_freezes_runtime_and_oracle_boundaries -q
```

Expected: FAIL because the contract does not exist.

- [ ] **Step 3: Create the frozen contract**

The JSON must copy all R1.2-B promotion fields unchanged and add these exact R1.3-P fields:

```json
{
  "schema_version": 1,
  "experiment_id": "a7c_r1_3p_temporal_joint_projection_377_v1",
  "status": "frozen",
  "subject": "377",
  "source_experiment_id": "a7c_r1_2b_dense_overlap_set_377_v1",
  "fit_cameras": ["c01", "c05", "c09", "c13"],
  "audit_cameras": [],
  "forbidden_cameras": ["c17", "c18", "c19", "c20", "c21", "c22", "c23"],
  "frame_start": 0,
  "frame_end": 570,
  "frame_stride": 5,
  "temporal_block_count": 6,
  "part": "lower",
  "offline_bidirectional": true,
  "minimum_gate": 0.9,
  "maximum_gate": 1.0,
  "selection_threshold": 0.2,
  "proxy_target_response": 0.995,
  "maximum_projection_gate_jump": 0.015,
  "lexicographic_rho_tolerance": 1e-9,
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
  "oracle_boundary_witness_gain": 0.023491874880317264,
  "oracle_bisection_tolerance": 1e-5,
  "source_r1_3p_design": "docs/superpowers/specs/2026-08-04-a7c-r1-3p-temporal-joint-projection-design.md",
  "source_r1_3p_design_sha256": "6204b23695f79c955e45af1b222a61b92209b556aac8f736db5faa21fd5ba9b2",
  "source_r1_2b_contract": "configs/semantic/a7c_r1_2b_dense_overlap_set_377_v1.json",
  "source_r1_2b_contract_sha256": "e2825c1d59e96ff2ea6124bfa1defafb62c73c04a64519c889e909be9ef2f9b5",
  "source_r1_2b_training_summary": "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/training_summary.json",
  "source_r1_2b_training_summary_sha256": "8a604cd5df7407b8b559adcea11304c93de2d9272c0bb8f1f60ca8b4f8efc46d",
  "source_r1_2b_predictions": ["exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_0/predictions.npz", "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_1/predictions.npz", "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_2/predictions.npz", "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_3/predictions.npz", "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_4/predictions.npz", "exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training/fold_5/predictions.npz"],
  "source_r1_2b_prediction_sha256": ["5e53226483194c26ec46e7da08602ee0b72818076d72e3bcafb6190328516dee", "95a56b28b50f538f0e7128954700233904fbb5eaf8762fa242e699284e3f4300", "87e5db5244e1d6bff8342cd2e84b33aaeb4901e0669e4e041c0adc90c36a26e8", "160e54f8e1b0e45d6fef2bcc800817278cd0dc3293d82db701436f4fc8fa7758", "d348a458812105e356855909df06893edb404e484944ffbd21a0eeba69a7a25d", "8c1ef0e5c5fe9b4001e2829c7da536811b6de4cef1761ef878b8adc141ff357c"],
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
  "paper_test_eligible": false
}
```

- [ ] **Step 4: Run the contract test and verify GREEN**

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json tests/test_a7c_temporal_joint_projection.py
git commit -m "实验：冻结A7c R1.3-P求解契约"
```

### Task 2: Implement The Runtime-Safe Projector

**Files:**
- Create: `utils/a7c_temporal_joint_projection.py`
- Modify: `tests/test_a7c_temporal_joint_projection.py`

- [ ] **Step 1: Add failing behavior and leakage tests**

```python
import inspect
import numpy as np

def test_joint_projection_repairs_jump_and_preserves_guards():
    from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection
    raw = np.array([[0.90, 1.00], [1.00, 0.90], [0.90, 1.00]])
    out = solve_temporal_joint_projection(
        raw_gates=raw, runtime_mass=np.ones_like(raw),
        a5_weight=np.array([0.25, 0.8]), minimum_gate=0.9,
        maximum_gate=1.0, selection_threshold=0.2,
        proxy_target_response=0.995, maximum_gate_jump=0.015)
    g = out["gates"]
    assert np.max(np.abs(np.diff(g, axis=0))) <= 0.015 + 1e-8
    assert np.min(np.mean(g, axis=1)) >= 0.995 - 1e-8
    assert out["certificate"]["maximum_primal_violation"] <= 1e-7

def test_joint_projection_zero_mass_keeps_already_feasible_raw():
    from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection
    raw = np.array([[0.93], [0.94], [0.95]])
    out = solve_temporal_joint_projection(
        raw_gates=raw, runtime_mass=np.zeros_like(raw), a5_weight=np.array([0.8]),
        minimum_gate=0.9, maximum_gate=1.0, selection_threshold=0.2,
        proxy_target_response=0.995, maximum_gate_jump=0.015)
    np.testing.assert_allclose(out["gates"], raw, atol=1e-9)

def test_runtime_projector_signature_has_no_renderer_inputs():
    from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection
    names = set(inspect.signature(solve_temporal_joint_projection).parameters)
    assert not names & {"evidence", "target", "outer", "boundary", "teacher_gates"}
```

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py -q
```

Expected: contract test passes; projector import fails.

- [ ] **Step 3: Implement the two LP stages**

Create this public API:

```python
def solve_temporal_joint_projection(
    *, raw_gates, runtime_mass, a5_weight, minimum_gate: float,
    maximum_gate: float, selection_threshold: float,
    proxy_target_response: float, maximum_gate_jump: float,
    rho_tolerance: float = 1e-9, primal_tolerance: float = 1e-9,
    residual_tolerance: float = 1e-7,
) -> dict:
    """Return gates float64[T,C] and a JSON-safe solver certificate."""
```

Stage one variables are `[flatten(g), rho]`; impose gate bounds, both sides of
`abs(g-r)<=rho`, one proxy-target row per frame, and both sides of every adjacent
jump. Stage two variables are `[flatten(g), deviation]`; impose both sides of
`abs(g-r)<=deviation`, freeze `abs(g-r)<=rho_star+1e-9`, and minimize total
deviation. Use sparse CSR matrices and HiGHS with primal/dual tolerance `1e-9`.
Recompute every constraint from returned gates. Raise on non-optimal status,
non-finite values, invalid topology floor, or residual above `1e-7`. Return
solver/SciPy versions, both objectives, displacement statistics, minimum slacks,
maximum jump/location, and maximum recomputed violation.

- [ ] **Step 4: Verify GREEN, then add error and determinism coverage**

Run the Task 2 command. Add tests for malformed shapes, non-finite values,
topology floor above one, and two identical solves. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add utils/a7c_temporal_joint_projection.py tests/test_a7c_temporal_joint_projection.py
git commit -m "方法：实现R1.3-P时间联合硬投影"
```

### Task 3: Implement The Feasibility Oracle

**Files:**
- Create: `utils/a7c_feasibility_oracle.py`
- Create: `tests/test_a7c_feasibility_oracle.py`

- [ ] **Step 1: Write failing affine-guard tests**

```python
import numpy as np

def test_soft_iou_linear_slack_matches_direct_ratio():
    from utils.a7c_feasibility_oracle import soft_iou_linear_slack
    target = np.array([0.98, 0.96]); outer = np.array([0.02, 0.04])
    base_target = np.ones(2); base_outer = np.zeros(2)
    slack = soft_iou_linear_slack(target, outer, base_target, base_outer, 0.005)
    drop = base_target/(base_target+base_outer) - target/(target+outer)
    assert np.array_equal(slack >= -1e-12, drop <= 0.005 + 1e-12)

def test_fixed_gain_oracle_returns_directly_valid_witness():
    from utils.a7c_feasibility_oracle import solve_fixed_gain_oracle
    base = np.array([1., 2., 1., 2., 1.]); point = base[:, None]
    streams = {"objective": {"outer": {"base": base, "point": point},
              "boundary": {"base": base, "point": point}},
              "guard": {"target": {"base": np.ones(5), "point": np.zeros((5,1))},
              "outer": {"base": base, "point": point}}}
    out = solve_fixed_gain_oracle(
        runtime_mass=np.zeros((5,1)), a5_weight=np.array([0.8]), streams=streams,
        minimum_gate=0.9, maximum_gate=1.0, selection_threshold=0.2,
        proxy_target_response=0.995, maximum_gate_jump=0.015,
        minimum_target_response=0.99, maximum_soft_iou_drop=0.005,
        minimum_outer_gain=0.01, minimum_boundary_gain=0.01)
    assert out["metrics"]["outer_gain"] >= 0.01 - 1e-7
    assert out["metrics"]["boundary_gain"] >= 0.01 - 1e-7
    assert out["certificate"]["maximum_primal_violation"] <= 1e-7
```

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_feasibility_oracle.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement fixed-gain feasibility**

Provide `normalized_flicker`, `evaluate_oracle_gates`,
`soft_iou_linear_slack`, and `solve_fixed_gain_oracle`. The LP variables are
gates plus outer/boundary adjacent-difference auxiliaries. Add runtime bounds,
proxy target, gate jump, true target, non-vacuous linear soft-IoU, absolute
signal-difference, and requested normalized-flicker rows. A `None` requested gain
adds no signal constraint. Use zero objective and fail closed after direct
float64 recomputation.

- [ ] **Step 4: Verify GREEN**

Run the Task 3 command. Expected: all tests pass.

- [ ] **Step 5: Test and implement bisection plus three-way verdict**

First add a callback with capacity `0.25` and verify
`lower <= 0.25 <= upper` and `upper-lower <= 1e-5`. Add 24-record synthetic
summaries for all three verdicts. Watch missing-function failures, then implement:

```python
def bisect_feasible_gain(is_feasible, *, lower=-0.01, upper=1.00001,
                         tolerance=1e-5) -> dict:
    low = float(lower)
    high = float(upper)
    if not is_feasible(low):
        raise RuntimeError("oracle lower endpoint is infeasible")
    if is_feasible(high):
        raise RuntimeError("oracle upper endpoint must be infeasible")
    iterations = 0
    while high - low > float(tolerance):
        middle = 0.5 * (low + high)
        if is_feasible(middle):
            low = middle
        else:
            high = middle
        iterations += 1
    return {
        "feasible_lower": low,
        "infeasible_upper": high,
        "interval_width": high - low,
        "iterations": iterations,
    }

def promotion_summary_passes(summary: dict, contract: dict) -> bool:
    improves = (
        summary["outer_gain"] > contract["r1_1_f1_outer_gain"]
        and summary["boundary_gain"] > contract["r1_1_f1_boundary_gain"]
    )
    distribution = all(
        summary[f"{signal}_positive_block_fraction"]
        >= contract["minimum_positive_block_fraction"]
        and summary[f"{signal}_block_gain_quantile"]
        >= contract["minimum_block_gain_quantile"] - 1e-9
        and summary[f"{signal}_worst_block_gain"]
        >= -contract["maximum_worst_block_regression"] - 1e-9
        for signal in ("outer", "boundary")
    )
    return bool(
        summary["outer_gain"] >= contract["minimum_outer_gain"]
        and summary["boundary_gain"] >= contract["minimum_boundary_gain"]
        and improves
        and distribution
    )

def classify_oracle(*, sufficient_audit_passed: bool,
                    optimistic_summary: dict, contract: dict) -> str:
    if sufficient_audit_passed:
        return "CERTIFIED_FEASIBLE"
    if not promotion_summary_passes(optimistic_summary, contract):
        return "CERTIFIED_INFEASIBLE"
    return "UNRESOLVED"
```

Bisection must retain a feasible lower endpoint and infeasible upper endpoint;
only the upper is used for impossibility. Classification returns feasible for an
exact sufficient-witness audit pass, infeasible only if the optimistic summary
fails the unchanged gates, and unresolved otherwise.

- [ ] **Step 6: Run focused tests and commit**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py tests/test_a7c_feasibility_oracle.py -q
git add utils/a7c_feasibility_oracle.py tests/test_a7c_feasibility_oracle.py
git commit -m "诊断：实现R1.3-P可行性oracle"
```

### Task 4: Build Projection And Audit CLIs

**Files:**
- Create: `tools/project_a7c_r1_3p_temporal_joint.py`
- Create: `tools/audit_a7c_r1_3p_temporal_joint_projection.py`
- Modify: `tests/test_a7c_temporal_joint_projection.py`

- [ ] **Step 1: Write failing boundary tests**

Test that projector source contains none of `_build_streams`, `evidence`,
`point_outer`, or `point_boundary`; audit source must import `_build_streams` and
`summarize_records`. A synthetic projection workflow must write NaN outside the
held mask and exactly four finite camera-block segments per fold.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py -q
```

- [ ] **Step 3: Implement runtime-only projection CLI**

Accept contract, probe, A5 bank, teacher, R1.2-B training dir, and output dir.
Verify every source hash. Reuse existing loaders and `runtime_target_mass`. For
each fold, set the output to NaN, select `held & fit_mask`, solve one segment for
each of cameras 0-3, and verify stride and block identity. Write masked raw and
projected gates, projection mask, manifest, fingerprints, and four certificates.
The CLI must not accept or import evidence.

- [ ] **Step 4: Implement separate held audit CLI**

Load evidence only here, build frozen objective/guard streams, verify exact masks
and certificates, call `evaluate_contribution_predictions` for every held record,
and aggregate exactly 24 rows with the unchanged R1.2 `summarize_records`.
Write `.held_block_passed` or `.rejected`; exit `0` or `2`.

- [ ] **Step 5: Verify tests and real-artifact smoke run**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py -q
tmp=$(mktemp -d)
/opt/miniconda3/envs/ictrl/bin/python tools/project_a7c_r1_3p_temporal_joint.py --contract configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json --probe exp/acceptdata/a7c_r1_1_transmittance_ray_context_377_v1/probe/probe.npz --a5-bank exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz --teacher exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz --source-training-dir exp/acceptdata/a7c_r1_2b_dense_overlap_set_377_v1/training --output-dir "$tmp/projection"
rm -rf "$tmp"
```

Expected: six outputs and 24 optimal segment certificates.

- [ ] **Step 6: Commit**

```bash
git add tools/project_a7c_r1_3p_temporal_joint.py tools/audit_a7c_r1_3p_temporal_joint_projection.py tests/test_a7c_temporal_joint_projection.py
git commit -m "实验：接入R1.3-P投影与held审计"
```

### Task 5: Build The Oracle CLI

**Files:**
- Create: `tools/evaluate_a7c_r1_3p_feasibility_oracle.py`
- Modify: `tests/test_a7c_feasibility_oracle.py`

- [ ] **Step 1: Write a failing synthetic workflow test**

Require each record to contain balanced, boundary-conditioned, independent
outer/boundary capacity intervals and sufficient-witness state. Require summary
execution status, registered verdict, optimistic summary, sufficient summary,
fingerprints, and `paper_test_eligible=false`.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_feasibility_oracle.py -q
```

- [ ] **Step 3: Implement the 24-record workflow**

Use the identical split/stream builder as audit. Per record run balanced
`outer,boundary>=gamma`, boundary-conditioned `outer>=0.005`, independent outer,
independent boundary, and sufficient `outer>=0.005,boundary>=0.023491874880317264`.
Build the optimistic audit only from infeasible upper endpoints. Audit actual
sufficient gates only when every witness succeeds. Emit `ORACLE_ERROR` and exit
1 on source/solver error; otherwise emit one registered verdict and exit 0.

- [ ] **Step 4: Verify and commit**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_feasibility_oracle.py tests/test_a7c_temporal_joint_projection.py -q
git add tools/evaluate_a7c_r1_3p_feasibility_oracle.py tests/test_a7c_feasibility_oracle.py
git commit -m "实验：接入R1.3-P容量判定流程"
```

### Task 6: Add The Restart-Safe Runner

**Files:**
- Create: `tools/run_a7c_r1_3p_temporal_joint_377.sh`
- Modify: `tests/test_a7c_temporal_joint_projection.py`

- [ ] **Step 1: Write the failing runner test**

Assert the runner uses the `ictrl` Python, contains no `c17`-`c23` camera token,
checks every frozen SHA, calls projection before audit/oracle, handles expected
audit exit 2 outside `ERR`, writes UTC timestamps, and makes
`.completed/.rejected/.failed` mutually exclusive.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py::test_r1_3p_runner_is_restart_safe_and_camera_isolated -q
```

- [ ] **Step 3: Implement runner and restart checks**

Follow the R1.2-B runner structure. Verify design, parent contract, training
summary, six predictions, probe, teacher, evidence, and A5 hashes. Resume only
missing artifacts. Always run oracle after projection audit, including expected
audit rejection. Set `.completed` for audit pass, `.rejected` for exit 2, and
`.failed` for projection/oracle/unexpected audit failure.

- [ ] **Step 4: Verify and commit**

```bash
bash -n tools/run_a7c_r1_3p_temporal_joint_377.sh
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py tests/test_a7c_feasibility_oracle.py -q
git add tools/run_a7c_r1_3p_temporal_joint_377.sh tests/test_a7c_temporal_joint_projection.py
git commit -m "实验：门控R1.3-P正式运行器"
```

### Task 7: Verify, Run, And Record The Frozen Experiment

**Files:**
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Run focused regression tests**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py tests/test_a7c_feasibility_oracle.py tests/test_a7c_quotient_compositor.py tests/test_a7c_overlap_set_compositor.py -q
```

Expected: all pass with no R1.3-P warnings.

- [ ] **Step 2: Launch formal runner**

```bash
mkdir -p exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1
nohup bash tools/run_a7c_r1_3p_temporal_joint_377.sh exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1 > exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/runner.log 2>&1 &
echo $! > exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/runner.pid
```

After four segment solves, estimate remaining LP wall time, convert to
Asia/Shanghai, and report the Beijing completion estimate. Call it a projection/
oracle solve, not training.

- [ ] **Step 3: Monitor to exactly one terminal marker**

Require six prediction files, six certificate files, projection summary/audit,
oracle records/summary, start/end timestamps, and one terminal marker. Do not
launch a duplicate process.

- [ ] **Step 4: Re-run exact audit and summarize evidence**

Confirm stable audit exit code and summary SHA. Report 24-record promotion
metrics, displacement/residual maxima, capacity interval distributions,
optimistic audit, sufficient-witness audit, and oracle verdict. State explicitly
that no R1.3-P outcome authorizes Task 12, LOSO, or a paper claim.

- [ ] **Step 5: Append the formal result to the A7 handoff**

Append a `Task 11 A7c R1.3-P` section with Beijing start/end, duration, source
and output hashes, exact metrics, oracle verdict, terminal marker, and frozen
next decision. Preserve all existing uncommitted document edits. Stage only this
hunk if it can be isolated; otherwise leave it uncommitted and report that fact.

- [ ] **Step 6: Final verification**

```bash
git diff --check
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_a7c_temporal_joint_projection.py tests/test_a7c_feasibility_oracle.py tests/test_a7c_quotient_compositor.py tests/test_a7c_overlap_set_compositor.py -q
```

Expected: no R1.3-P whitespace errors and all selected tests pass.
