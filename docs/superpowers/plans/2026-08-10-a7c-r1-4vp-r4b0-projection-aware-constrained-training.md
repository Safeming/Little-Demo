# A7c R1.4-VP-R4-B0 Projection-Aware Constrained Training Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Keep the approved R4-B0 design frozen, use TDD, commit only R4-B0-owned files, and stop before held audit on any pre-registered negative gate.

**Goal:** Train the unchanged 9,073-parameter R4-A compositor through the exact HiGHS deployment projection so the optimized forward gate and deployed gate are identical, then decide fold-0 entry without exposing held renderer or teacher rows.

**Architecture:** Add a small R4-B0 policy module for straight-through exact projection, differentiable renderer/gain/action components, global scale freezing, observability, and fold-entry checks. Fork the R4-A trainer only where the loss consumes gates: every complete fit segment is projected exactly in the forward pass, while the identity straight-through gradient updates the unchanged compositor. Reuse the inherited projection solver, dataset construction, six-fold protocol, and held auditor.

**Tech Stack:** Python 3.10, PyTorch, NumPy, SciPy/HiGHS, pytest, Bash, existing A7c compositor/projection/audit utilities.

---

## Frozen Experiment Contract

- Experiment ID: `a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1`
- Output: `exp/acceptdata/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1`
- Design source: `docs/superpowers/specs/2026-08-10-a7c-r1-4vp-r4b0-projection-aware-constrained-training-design.md`
- Parent sources: the exact committed R4-A contract, policy, trainer, auditor, runner, and fold-0 prediction artifact.
- Frozen model/optimizer: 9,073 parameters, no attention, no carrier embedding, AdamW, 400 epochs, learning rate 0.001, weight decay 0.0001, gradient clip 1.0, R4-A seed/order/splits.
- No held c17-c23 access before six fit folds pass. No Task 12 or LOSO.

### Task 1: Freeze the R4-B0 contract and executable ownership

**Files:**
- Create: `configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json`
- Create: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

**Step 1: Write the failing contract test**

Assert the experiment ID, design/source hashes, 9,073 parameter budget, disabled attention/carrier embedding, unchanged optimizer schedule, exact projection backend, STE formula, component deltas, global median scale rule, observability thresholds, fold-0 thresholds, status routing, and held-data prohibition. Also assert that the future runner contains the R4-B0 contract path and status markers.

**Step 2: Run the focused test and confirm RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py -x
```

Expected: fail because the R4-B0 contract does not exist.

**Step 3: Add the minimal frozen contract**

Copy only inherited immutable values from R4-A, then add explicit R4-B0 fields:

```json
{
  "experiment_id": "a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1",
  "projection_training_mode": "exact_highs_straight_through",
  "straight_through_forward": "raw_plus_stop_gradient_exact_minus_raw",
  "scale_scope": "global_median_over_fit_segments_at_initialization",
  "projection_consistency_scale": 0.0002,
  "minimum_observability_gradient_norm": 1e-12,
  "maximum_fit_projected_teacher_mae": 0.0065,
  "minimum_fit_outer_recovery": 0.75,
  "minimum_fit_boundary_recovery": 0.75,
  "minimum_fit_positive_segment_fraction": 0.95,
  "minimum_fit_action_cosine": 0.90,
  "minimum_fit_top_k_overlap": 0.45,
  "maximum_fit_missed_suppression_fraction": 0.55,
  "maximum_fit_raw_to_exact_mae": 0.0002,
  "maximum_fit_projection_changed_fraction": 0.05
}
```

Pin SHA256 for every approved source. Do not reference mutable working-tree inputs without a hash.

**Step 4: Run the focused contract test and confirm GREEN**

Run the Step 2 command. Expected: contract assertions pass; runner assertion remains skipped until Task 5 only if the test isolates missing executable checks.

**Step 5: Commit**

```bash
git add configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
git commit -m "test: freeze R4-B0 projection-aware contract"
```

### Task 2: Implement exact-forward STE and registered loss mathematics

**Files:**
- Create: `utils/a7c_r1_4vp_r4b0.py`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

**Step 1: Add failing pure-function tests**

Cover:

```python
deployed = raw + (exact - raw).detach()
torch.testing.assert_close(deployed, exact)
deployed.sum().backward()
torch.testing.assert_close(raw.grad, torch.ones_like(raw))
```

Also test exact projection/certificate parity with `solve_temporal_joint_projection`, exact normalized flicker/gain parity with the NumPy evaluator, signed trajectory parity with R4-A, target Huber, gate plus temporal Huber, action cosine including zero-action handling, projection consistency normalization, global median scale freezing, and rejection of missing/nonfinite/zero components.

Use a gate perturbation where raw and exact differ and verify every renderer/preservation component changes with deployed gates but not with the detached raw forward value.

**Step 2: Run focused tests and confirm RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py -x
```

Expected: import failure for `utils.a7c_r1_4vp_r4b0`.

**Step 3: Implement minimal policy functions**

Implement:

- `exact_projected_straight_through(raw, runtime_mass, a5_weight, contract)` returning deployed gates and a detached certificate;
- `normalized_flicker(values, epsilon)` and `flicker_gain(base, edited, epsilon)` using `mean(abs(diff(v))) / max(abs(mean(v)), epsilon)`;
- `projection_aware_components(deployed, raw, teacher, base, streams, ...)` with keys `trajectory_outer`, `trajectory_boundary`, `gain_outer`, `gain_boundary`, `target`, `gate`, `action`, `projection`;
- `freeze_global_median_scales(segment_components, minimum)` for the seven non-projection components;
- `projection_aware_loss(components, scales, residual)` where renderer and preservation groups are unweighted means and residual weight is exactly `1e-5`;
- `projection_diagnostics(raw, exact, threshold=1e-12)`;
- `evaluate_fit_projected_entry(summary, contract)` returning pass/failure reasons without accessing held data.

The exact solver is forward-only. Preserve the gradient solely through the STE expression.

**Step 4: Run focused tests and confirm GREEN**

Run the Step 2 command.

**Step 5: Run inherited math regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
```

Expected: all pass.

**Step 6: Commit**

```bash
git add utils/a7c_r1_4vp_r4b0.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
git commit -m "feat: add R4-B0 exact projection-aware loss"
```

### Task 3: Add fail-closed gradient observability preflight

**Files:**
- Modify: `utils/a7c_r1_4vp_r4b0.py`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

**Step 1: Add failing observability tests**

Construct a tiny deterministic model/projector harness. Test a positive result and each independent rejection:

- certificate failure;
- nonfinite component;
- nonfinite gradient;
- aggregate gradient norm `<=1e-12`;
- ephemeral AdamW step does not reduce exact-reprojected aggregate deployed loss;
- held teacher or renderer rows become finite/readable.

Assert the clone is discarded and the source model state remains byte-identical.

**Step 2: Run focused observability tests and confirm RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py -k observability -x
```

**Step 3: Implement the minimal preflight helper**

Add `run_gradient_observability_preflight(...)`. It must clone the model, use the frozen AdamW signature for exactly one ephemeral step, reproject the updated forward gates, compare aggregate deployed loss before/after, report every finite/certificate/held-access check, and never save the clone or optimizer.

Return a JSON-safe payload with verdict `FEATURE_OBSERVABILITY_POSITIVE` or `FEATURE_OBSERVABILITY_NEGATIVE` plus explicit failure reasons.

**Step 4: Run focused and full R4-B0 tests**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
```

**Step 5: Commit**

```bash
git add utils/a7c_r1_4vp_r4b0.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
git commit -m "feat: add R4-B0 gradient observability preflight"
```

### Task 4: Implement the fold trainer with exact deployed gates

**Files:**
- Create: `tools/train_a7c_r1_4vp_r4b0_projection_aware.py`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

**Step 1: Add failing trainer integration tests**

Use the R4-A synthetic fold fixture, but assert:

- exact projection executes for every complete fit segment;
- all forward losses receive deployed gates;
- global initial scales are one median dictionary, not per-segment dictionaries;
- parameter count is exactly 9,073 and optimizer signature is unchanged;
- observability runs before epoch 1 and failure returns status 2 without `model.pt`;
- prediction artifacts save both raw and exact gates and exact certificates;
- summary includes initial/final raw and normalized components, flicker gains, action and projection diagnostics, and false held-access flags;
- fold-0 entry evaluates exact gates and rejects each individual threshold without invoking held audit.

Patch training epochs only inside synthetic unit tests; the formal contract remains 400.

**Step 2: Run trainer tests and confirm RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py -k 'trainer or fold_entry' -x
```

**Step 3: Fork the R4-A trainer minimally**

Reuse R4-A input validation, normalization, segment packing, model construction, source verification, frozen input loading, and output schema. Change only:

1. project each raw segment before computing registered components;
2. freeze the global median initial scales across all 20 fit segments;
3. run observability before formal epoch 1;
4. train all 400 epochs through deployed exact gates;
5. compute fold diagnostics and entry from exact gates;
6. return status 2 and `FIT_PROJECTED_ENTRY_NEGATIVE` before any held action on failure.

Keep held rows as NaN and index them only after verifying a fit/prediction mask.

**Step 4: Run trainer integration tests and inherited regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
```

**Step 5: Commit**

```bash
git add tools/train_a7c_r1_4vp_r4b0_projection_aware.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
git commit -m "feat: train R4-B0 through exact deployed gates"
```

### Task 5: Add inherited held auditor and fail-closed runner

**Files:**
- Create: `tools/audit_a7c_r1_4vp_r4b0_projection_aware.py`
- Create: `tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

**Step 1: Add failing lifecycle tests**

Assert:

- `FEATURE_OBSERVABILITY_NEGATIVE` creates `.observability_rejected`, status 2, and never calls held audit;
- `FIT_PROJECTED_ENTRY_NEGATIVE` creates `.fit_rejected`, status 2, and never starts folds 1-5;
- exactly six positive fit folds are frozen and hashed before held audit;
- the auditor delegates to the unchanged R4-A/R3/R2 gate set and only changes the stage name;
- final markers are mutually exclusive and all eligibility flags remain false.

**Step 2: Run lifecycle tests and confirm RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py -k 'runner or audit or routing' -x
```

**Step 3: Implement the minimal auditor and runner**

The runner must verify contract/source SHA256 values before making the output root, write PID/log/start/end timestamps, launch fold 0 first, route every status fail-closed, launch folds 1-5 only after fold 0 entry passes, freeze manifest hashes, and invoke the inherited held audit exactly once only after all six fits pass.

The auditor wraps the inherited R4-A auditor and changes only the stage label to `r1_4vp_r4b0_projection_aware_held_canary`.

**Step 4: Run focused lifecycle and syntax checks**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
bash -n tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh
```

**Step 5: Commit**

```bash
git add tools/audit_a7c_r1_4vp_r4b0_projection_aware.py \
  tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
git commit -m "feat: audit and run R4-B0 projection-aware canary"
```

### Task 6: Verify, launch once, wait for the preregistered terminal verdict, and document

**Files:**
- Modify after terminal result: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`
- Generated only: `exp/acceptdata/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1/**`

**Step 1: Verify all R4-B0-owned source hashes and tests**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest -q \
  tests/test_a7c_r1_4vp_r2_loss_repair.py \
  tests/test_a7c_r1_4vp_r3_crw.py \
  tests/test_a7c_r1_4vp_r4a_signed_renderer.py \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
bash -n tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh
git diff --check -- \
  configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json \
  utils/a7c_r1_4vp_r4b0.py \
  tools/train_a7c_r1_4vp_r4b0_projection_aware.py \
  tools/audit_a7c_r1_4vp_r4b0_projection_aware.py \
  tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh \
  tests/test_a7c_r1_4vp_r4b0_projection_aware.py
```

Expected: all pass, syntax clean, no whitespace errors.

**Step 2: Perform launch preflight**

Verify required frozen inputs exist and match hashes, no active A7 runner owns the GPU, GPU memory is available, the formal output root does not exist, and no held c17-c23 artifact is opened. If any invariant fails, stop without launching.

**Step 3: Launch exactly once**

```bash
nohup bash tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh \
  exp/acceptdata/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1 \
  > /tmp/a7c_r4b0_launch.log 2>&1 &
```

Record the actual PID, UTC start time, Beijing start time, and an evidence-based Beijing ETA from measured observability/fold epoch throughput. Confirm the process is alive and `runner.log` advances.

**Step 4: Wait for a terminal marker without relaunching**

Poll at short intervals while continuing to report progress. Terminal markers are `.observability_rejected`, `.fit_rejected`, `.rejected`, `.completed`, or `.failed`. Do not treat an unchanged log as a failure while the PID is alive.

**Step 5: Analyze the terminal result**

Report observability, fold-0 entry, projected outer/boundary recovery and positive fractions, exact teacher MAE, action diagnostics, projection displacement, certificate status, and held-access flags. State whether the pre-registered route authorizes R4-B1, held audit, or neither.

**Step 6: Append the reproducible result to the A7 handoff document**

Append experiment ID, source hashes, timestamps, command, verdict, metrics, and next-route decision to `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`. Because that file already contains user-owned edits, do not stage or commit it unless its diff can be isolated without capturing unrelated content.

**Step 7: Final verification before reporting completion**

Re-run the focused R4-B0 tests, inspect terminal summaries directly, verify PID termination, and run `git status --short` to distinguish committed R4-B0 source from pre-existing user changes.

## Execution Choice

The user explicitly selected current-session inline execution in the preceding workflow and has now approved training. Execute this plan in the current workspace with review checkpoints after each task; do not create a separate worktree and do not push remotely.

## Pre-Launch Review Remediation

The independent review after Task 5 found four launch blockers. The user
approved the fit-only snapshot design recorded in the design's
`Launch-Blocking Isolation Amendment`. Complete these checks before Task 6.

### Task 5A: Correct global median validation

**Files:**
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`
- Modify: `utils/a7c_r1_4vp_r4b0.py`

1. Add a failing test with finite per-segment zeros but a positive global
   median. Assert that the scale freezes successfully.
2. Run the focused test and confirm the existing per-element check rejects it.
3. Permit finite values below the minimum before aggregation; reject only
   nonfinite values and medians `<= initial_scale_minimum`.
4. Run the full R4-B0 policy suite and commit.

### Task 5B: Verify the 36-file R4-B0 schema

**Files:**
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`
- Modify: `tools/audit_a7c_r1_4vp_r4b0_projection_aware.py`

1. Add a failing integration test that constructs and hashes all 36 expected
   fold artifacts, then calls the real R4-B0 verifier.
2. Confirm RED against the inherited R3 30-file verifier.
3. Implement `verify_frozen_artifacts` for the six R4-B0 filenames and inject
   it into the inherited audit chain only for the duration of `_run`.
4. Test missing, extra, and changed artifacts fail closed; run auditor
   regressions and commit.

### Task 5C: Stage and enforce fit-only immutable inputs

**Files:**
- Create: `tools/stage_a7c_r1_4vp_r4b0_fit_inputs.py`
- Modify: `tools/train_a7c_r1_4vp_r4b0_projection_aware.py`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`
- Generate: `exp/acceptdata/a7c_r1_4vp_r4b0_fit_only_inputs_377_v1/**`

1. Add failing tests for deterministic camera-only staging, provenance hashes,
   forbidden-camera rejection in the trainer, and NaN expansion from 456 fit
   predictions back to the 912-row audit schema.
2. Implement a staging tool that selects only camera indices `0..3`, writes the
   minimal probe/evidence/manifest and six base/teacher files, and hashes every
   output in `manifest.json`.
3. Change the trainer to require exactly the staged four-camera manifest and
   use no full evidence/base/teacher path. Build objective streams from the
   three staged renderer sequences only.
4. Save prediction bundles in original 912-row order by placing fit predictions
   at staged source indices and leaving all other rows NaN/masked false.
5. Generate the formal bundle once, verify its manifest independently, run
   trainer and inherited regression tests, and commit source changes. Generated
   experiment data remains uncommitted.

### Task 5D: Freeze final sources and launch routing

**Files:**
- Modify: `configs/semantic/a7c_r1_4vp_r4b0_projection_aware_constrained_377_v1.json`
- Modify: `tools/run_a7c_r1_4vp_r4b0_projection_aware_377.sh`
- Modify: `tests/test_a7c_r1_4vp_r4b0_projection_aware.py`

1. Add failing tests that require contract hashes for the policy, trainer,
   auditor, staging tool, fit-only manifest/artifacts, and a runner-generated
   `source_fingerprints.json` containing its own hash plus committed HEAD.
2. Pin final hashes after Tasks 5A-5C and update the design hash.
3. Point training arguments only at fit-only paths while leaving full 912-row
   paths exclusively in the post-freeze auditor invocation.
4. Require R4-B0-owned paths to be clean relative to HEAD, write the source
   manifest before training, and include it in every terminal completeness
   check.
5. Run the complete verification suite, request a second independent review,
   and only then perform Task 6 launch.
