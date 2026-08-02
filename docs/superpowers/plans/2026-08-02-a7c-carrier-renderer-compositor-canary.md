# A7c Carrier Renderer Compositor Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and gate a CoreView377 carrier-resolved renderer-visible target-preserving compositor canary without using camera/frame IDs or c21-c23.

**Architecture:** Export deterministic per-carrier runtime probes from the existing Gaussian render loop, train a shared bounded MLP against construction-only carrier oracle gates, and audit predicted gates with the existing renderer contribution evidence. Compose the final edit delta with alpha/transmittance-weighted lower support so target-heavy pixels approach the unchanged A5 edit.

**Tech Stack:** Python, PyTorch, NumPy, SciPy HiGHS, existing Gaussian rasterizer, JSON/NPZ, pytest, Bash.

---

### Task 1: Freeze Contract And Core Invariants

**Files:**
- Create: `configs/semantic/a7c_carrier_compositor_canary_377_v1.json`
- Create: `utils/a7c_renderer_compositor.py`
- Create: `tests/test_a7c_renderer_compositor.py`

- [ ] **Step 1: Write failing contract and invariant tests**

Test that the contract fixes cameras, frames, six blocks, `[0.9,1.0]` gates,
the `32 -> 16 -> 1` MLP, forbidden fields, and `paper_test_eligible=false`.
Test `target_preserving_gate(point_gate, lower_support)` at support zero and one,
and reject camera/frame/subject/Gaussian IDs from feature schemas.

- [ ] **Step 2: Verify RED**

Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest -q tests/test_a7c_renderer_compositor.py`.
Expected: import failure for `utils.a7c_renderer_compositor`.

- [ ] **Step 3: Implement the frozen contract and pure helpers**

The JSON contains exact fit/audit cameras, frame range, feature names, MLP
dimensions, optimizer budget, source fingerprints, and promotion thresholds.
The utility exports:

```python
def validate_feature_schema(names: list[str]) -> tuple[str, ...]: ...
def contiguous_block_ids(frame_indices, block_count: int) -> np.ndarray: ...
def target_preserving_gate(point_gate, lower_support):
    return 1.0 - (1.0 - point_gate) * (1.0 - lower_support)
def normalized_flicker(values) -> float: ...
```

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test, then commit only the contract, utility, and test with
message `方法：冻结A7c载体合成canary契约`.

### Task 2: Export Carrier Oracle Teacher

**Files:**
- Modify: `tools/evaluate_a7c_oracle_capacity.py`
- Modify: `utils/a7c_oracle_capacity.py`
- Modify: `tests/test_a7c_oracle_capacity.py`
- Output: `exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz`

- [ ] **Step 1: Write a failing teacher-artifact test**

Test that camera-wise point gates can be assembled in source sample order and
that carrier IDs, camera/frame indices, gate bounds, input fingerprints, and
`paper_test_eligible=false` survive an NPZ round trip.

- [ ] **Step 2: Verify RED**

Run the new named test and confirm it fails because teacher assembly/export is
missing.

- [ ] **Step 3: Add explicit teacher export**

Add `--teacher-output` and `--subjects`. Retain the selected point-oracle gate
matrix from each camera. Refuse to overwrite a teacher whose source
fingerprints or carrier IDs differ. Do not export global/ray labels.

- [ ] **Step 4: Generate and verify the 377 teacher**

Run the exact oracle contract for subject 377 with teacher output under
`exp/acceptdata/a7c_carrier_compositor_canary_377_v1/teacher/teacher.npz`.
Verify exact sample count, carrier count, bounds, and fingerprint; commit code
and tests, not generated data.

### Task 3: Collect Renderer Runtime Probes

**Files:**
- Create: `tools/build_a7c_renderer_probe.py`
- Modify: `utils/a7c_renderer_compositor.py`
- Modify: `tests/test_a7c_renderer_compositor.py`
- Output: `exp/acceptdata/a7c_carrier_compositor_canary_377_v1/probe/probe.npz`

- [ ] **Step 1: Write failing probe extraction tests**

Use synthetic tensors to test finite feature extraction, train-only
normalization, invisible-carrier encoding, carrier subsetting, schema order,
and fingerprint validation.

- [ ] **Step 2: Verify RED, then implement the minimal extractor**

For each configured sample, reuse `scene.convert_gaussians` and
`rasterize_gaussians`. Export only selected R0 carrier rows with:

```text
visibility, log1p_radius, camera_x_over_z, camera_y_over_z, log_depth,
view_dir_x, view_dir_y, view_dir_z, opacity, footprint_proxy,
a5_lower_weight, selected_lower
```

Camera/frame indices are manifest fields only. No target mask, contribution
label, or oracle gate enters the feature tensor.

- [ ] **Step 3: Run deterministic dry run and a two-frame smoke probe**

The dry run checks all input fingerprints and prints the expected tensor shape.
The smoke probe uses c01 frames 0 and 5 and must reload with the same output
fingerprint.

- [ ] **Step 4: Launch the full eight-camera probe**

Run on GPU 0 with one CPU thread per numerical backend, logging every sample and
writing `.done` only after reload/fingerprint validation.

### Task 4: Train Shared Bounded Carrier MLP

**Files:**
- Create: `tools/train_a7c_carrier_compositor.py`
- Modify: `utils/a7c_renderer_compositor.py`
- Modify: `tests/test_a7c_renderer_compositor.py`
- Output: `exp/acceptdata/a7c_carrier_compositor_canary_377_v1/training/`

- [ ] **Step 1: Write failing model and split tests**

Test deterministic initialization at gate `>=0.999`, output bounds, no ID
embedding, normalization fitted only on fit cameras, six contiguous held-block
folds, audit-camera exclusion, and byte-identical predictions across batch
orders.

- [ ] **Step 2: Verify RED, then implement the fixed trainer**

Use the frozen `input -> 32 -> 16 -> 1` SiLU model, AdamW, fixed seed and fixed
epoch budget from the contract. Fit only c01/c05/c09/c13 teacher rows. Save
normalization statistics, schema, model state, losses, split manifest, source
fingerprints, and predictions for every construction sample.

- [ ] **Step 3: Run six held-block folds**

Each fold writes an independent checkpoint and prediction artifact. Stop before
held-camera audit if topology, gate bounds, target response, soft-IoU, adjacent
gate change, or held-block temporal gates fail.

### Task 5: Target-Preserving Compositor Audit

**Files:**
- Create: `tools/audit_a7c_carrier_compositor.py`
- Modify: `utils/a7c_renderer_compositor.py`
- Modify: `tests/test_a7c_renderer_compositor.py`
- Output: `exp/acceptdata/a7c_carrier_compositor_canary_377_v1/audit/`

- [ ] **Step 1: Write failing contribution and compositor metric tests**

Test carrier prediction alignment, target-preserving limiting cases, block
aggregation, target response, positive-block fraction, q10, worst regression,
and rejection when any camera regresses beyond the contract.

- [ ] **Step 2: Implement held-block gate and held-camera audit**

Use unweighted renderer contribution evidence for hard guards. Held-camera
predictions are opened only after held-block `.passed` exists. Render A5 lower
support and gate numerator with identical Gaussian state, form the pixel gate,
and verify its values are finite and in `[0.9,1.0]`.

- [ ] **Step 3: Write auditable summaries**

Write per-camera JSON, six-block CSV, aggregate JSON, fingerprints, and exactly
one of `.passed` or `.rejected`. Always keep `paper_test_eligible=false`.

### Task 6: Queue, Launch, And Record

**Files:**
- Create: `tools/run_a7c_carrier_compositor_canary_377.sh`
- Modify: `tests/test_a7c_renderer_compositor.py`
- Modify: `docs/A7时序可靠性校准实施计划与新对话交接_20260728.md`

- [ ] **Step 1: Test queue dry-run, resume, and stop behavior**

The queue must resume teacher/probe/training/audit stages independently, stop on
`.rejected` or `.failed`, and never reference c21-c23.

- [ ] **Step 2: Run focused and regression verification**

Run both A7c test modules, Bash syntax validation, and `git diff --check`.

- [ ] **Step 3: Launch the canary queue**

Launch on GPU 0 with a timestamped log and PID file. Confirm the process is
alive, the active stage advances, and GPU memory is allocated. Estimate the
completion time from measured smoke-probe throughput plus fixed training and
audit budgets, then report the Beijing completion window.

- [ ] **Step 4: Record immutable launch metadata**

Append contract fingerprint, code commit, PID, command, start time, estimated
Beijing completion time, and output root to the A7 handoff document.
