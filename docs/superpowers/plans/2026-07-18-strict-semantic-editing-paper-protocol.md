# Strict Semantic Editing Paper Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a leakage-free CoreView_377 semantic editing paper protocol with fair previews, matched-retention curves, standard semantic metrics, and B0-B5 baselines.

**Architecture:** A shared protocol module owns camera/frame splits, validation, fingerprints, and provenance. Existing calibration, projection, and preview tools consume protocol-selected records, while a focused paper evaluator constructs B0-B5 predictions and produces semantic metrics plus matched-retention reports. A formal shell runner orchestrates strict semantic training, three asset exports, validation selection, and frozen test evaluation.

**Tech Stack:** Python 3, NumPy, OpenCV, PyTorch, pytest/unittest, Hydra, Bash, existing 3DGS renderer and semantic asset tools.

---

## File Structure

- Create `configs/semantic/coreview377_strict_paper_protocol.json`: single source of truth for strict splits and validation grids.
- Create `utils/semantic_eval_protocol.py`: protocol parsing, validation, record selection, fingerprints, frozen-config checks, and provenance.
- Create `tests/test_semantic_eval_protocol.py`: protocol behavior and split-leakage tests.
- Modify `tools/calibrate_evidence_soft_edit_weights.py`: protocol-selected calibration records and provenance.
- Modify `tools/analyze_projected_soft_edit_leakage.py`: protocol-selected evaluation records and protocol metadata.
- Modify `tools/analyze_multiview_semantic_consistency.py`: forward protocol selection to the projection evaluator.
- Modify `tools/make_semantic_edit_render_preview.py`: formal mode that rejects parser-backed compositing and records fairness metadata.
- Modify `tests/test_evidence_calibrated_semantic_bank.py`: calibration split rejection tests.
- Modify `tests/test_projected_soft_edit_leakage.py`: evaluation split selection tests.
- Modify `tests/test_semantic_edit_render_preview.py`: formal preview fairness tests.
- Create `utils/semantic_paper_metrics.py`: standard semantic metrics, boundary metrics, curves, interpolation, baseline weight resolution, and validation selection.
- Create `tests/test_semantic_paper_metrics.py`: metric and matched-retention tests.
- Create `tools/evaluate_semantic_editing_paper_protocol.py`: unified B0-B5 test evaluator and report writer.
- Create `tests/test_semantic_editing_paper_protocol.py`: evaluator report and baseline contract tests.
- Create `tools/select_semantic_editing_validation_config.py`: deterministic validation selection and frozen config writer.
- Create `tests/test_select_semantic_editing_validation_config.py`: retention constraint and tie-break tests.
- Create `tools/run_strict_semantic_editing_paper_protocol.sh`: end-to-end strict runner.
- Create `tests/test_strict_semantic_editing_paper_protocol_script.py`: dry-run and split wiring tests.

### Task 1: Strict Protocol Core

**Files:**
- Create: `configs/semantic/coreview377_strict_paper_protocol.json`
- Create: `utils/semantic_eval_protocol.py`
- Create: `tests/test_semantic_eval_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Cover loading the approved manifest, rejecting camera or record overlap, requiring calibration to be a subset of semantic training, selecting exact camera/frame records, stable fingerprints independent of JSON key order, and rejecting a frozen config with the wrong protocol/checkpoint fingerprint.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_semantic_eval_protocol.py -q
```

Expected: import failure for `utils.semantic_eval_protocol`.

- [ ] **Step 3: Implement the protocol module and manifest**

Use normalized integer camera/frame lists, SHA-256 over canonical JSON, explicit `(cam_id, frame_id)` record keys, and JSON provenance containing selected record names and fingerprints.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: all protocol tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/semantic/coreview377_strict_paper_protocol.json utils/semantic_eval_protocol.py tests/test_semantic_eval_protocol.py
git commit -m "feat: add strict semantic evaluation protocol"
```

### Task 2: Enforce Protocol Splits in Existing Tools

**Files:**
- Modify: `tools/calibrate_evidence_soft_edit_weights.py`
- Modify: `tools/analyze_projected_soft_edit_leakage.py`
- Modify: `tools/analyze_multiview_semantic_consistency.py`
- Modify: `tests/test_evidence_calibrated_semantic_bank.py`
- Modify: `tests/test_projected_soft_edit_leakage.py`

- [ ] **Step 1: Write failing split-enforcement tests**

Test that calibration accepts only `calibration`, evaluation accepts only `validation` or `test`, selected records match the manifest exactly, and test records cannot be passed to calibration even through a mixed asset root.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_projected_soft_edit_leakage.py -q
```

Expected: missing protocol CLI/helper behavior.

- [ ] **Step 3: Add protocol CLI and selection**

Add `--protocol` and `--protocol-split`. Load all records, select through `select_protocol_records`, assert exact split membership, and attach protocol/record fingerprints to summaries. Preserve legacy behavior only when `--protocol` is absent.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command. Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/calibrate_evidence_soft_edit_weights.py tools/analyze_projected_soft_edit_leakage.py tools/analyze_multiview_semantic_consistency.py tests/test_evidence_calibrated_semantic_bank.py tests/test_projected_soft_edit_leakage.py
git commit -m "feat: enforce semantic protocol record splits"
```

### Task 3: Fair Formal Preview

**Files:**
- Modify: `tools/make_semantic_edit_render_preview.py`
- Modify: `tests/test_semantic_edit_render_preview.py`

- [ ] **Step 1: Write failing formal-preview tests**

Test that `--formal-paper-mode` rejects `--screen-mask-composite`, rejects parser-backed footprint gating, accepts raw Hard/Ours rendering, and writes `uses_test_parser_composite=false`.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_semantic_edit_render_preview.py -q
```

- [ ] **Step 3: Implement formal-paper validation**

Add a pure validation helper called by `make_render_preview` before loading masks. In formal mode, Hard and Soft use the same uncomposited rasterizer output and the summary records the fairness flags.

- [ ] **Step 4: Run and verify GREEN**

Run the same pytest command.

- [ ] **Step 5: Commit**

```bash
git add tools/make_semantic_edit_render_preview.py tests/test_semantic_edit_render_preview.py
git commit -m "fix: make semantic paper previews fair"
```

### Task 4: Standard Metrics and Matched Retention

**Files:**
- Create: `utils/semantic_paper_metrics.py`
- Create: `tests/test_semantic_paper_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Use small binary/soft masks with hand-computed expected values for per-part IoU, macro mIoU excluding empty targets, micro IoU, precision, recall, soft IoU, boundary precision/recall/F1/IoU at tolerance two, curve normalization, shared-retention discovery, interpolation, and no-extrapolation errors.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_semantic_paper_metrics.py -q
```

- [ ] **Step 3: Implement minimal metric primitives**

Use NumPy arrays, OpenCV distance transforms or deterministic binary dilation for boundary tolerance, fuzzy `min/max` intersection/union for soft IoU, sorted monotonic curve points, and linear interpolation only within observed retention ranges.

- [ ] **Step 4: Run and verify GREEN**

Run the same pytest command.

- [ ] **Step 5: Commit**

```bash
git add utils/semantic_paper_metrics.py tests/test_semantic_paper_metrics.py
git commit -m "feat: add semantic paper metrics and retention curves"
```

### Task 5: Unified B0-B5 Evaluator

**Files:**
- Create: `tools/evaluate_semantic_editing_paper_protocol.py`
- Create: `tests/test_semantic_editing_paper_protocol.py`

- [ ] **Step 1: Write failing evaluator tests**

Test baseline metadata and weight resolution:

```text
B0 parser oracle -> current-view parser prediction, oracle=true
B1 voting        -> voting bank hard/soft fields
B2 hard          -> editable_label equality
B3 raw prob      -> semantic_probs
B4 conf/margin   -> compute_soft_edit_weights inputs
B5 ours          -> edit_target_weights/edit_support_weights
```

Also test required CSV columns, protocol fingerprints, oracle labeling, and failure on missing baseline fields.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_semantic_editing_paper_protocol.py -q
```

- [ ] **Step 3: Implement evaluator**

Reuse projection/rasterization helpers from `analyze_projected_soft_edit_leakage.py`. Generate per-view prediction maps once per baseline/part/threshold, then aggregate semantic metrics, raw/actionable footprint leakage, curves, matched-retention rows, and summary CSV/JSON. Plot figures with Matplotlib if available; CSV/JSON remain mandatory.

- [ ] **Step 4: Run and verify GREEN**

Run the same pytest command.

- [ ] **Step 5: Commit**

```bash
git add tools/evaluate_semantic_editing_paper_protocol.py tests/test_semantic_editing_paper_protocol.py
git commit -m "feat: add unified semantic editing paper evaluator"
```

### Task 6: Validation Selection

**Files:**
- Create: `tools/select_semantic_editing_validation_config.py`
- Create: `tests/test_select_semantic_editing_validation_config.py`

- [ ] **Step 1: Write failing selector tests**

Test rejection when every candidate has aggregate retention below 0.60. Test deterministic ordering by actionable footprint leakage, boundary F1, then smaller radius. Test frozen config fingerprints and selected thresholds.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_select_semantic_editing_validation_config.py -q
```

- [ ] **Step 3: Implement selector**

Read validation candidate JSON/CSV, enforce the retention floor, sort deterministically, and write `frozen_validation_config.json` containing protocol/checkpoint/bank fingerprints and all selected parameters.

- [ ] **Step 4: Run and verify GREEN**

Run the same pytest command.

- [ ] **Step 5: Commit**

```bash
git add tools/select_semantic_editing_validation_config.py tests/test_select_semantic_editing_validation_config.py
git commit -m "feat: freeze semantic validation configuration"
```

### Task 7: Strict End-to-End Runner

**Files:**
- Create: `tools/run_strict_semantic_editing_paper_protocol.sh`
- Create: `tests/test_strict_semantic_editing_paper_protocol_script.py`
- Modify: `tools/formal/run_377_v338_semantic_train.sh`

- [ ] **Step 1: Write failing dry-run tests**

Test that the runner uses the manifest-derived semantic training views/frames, creates separate calibration/validation/test export commands, passes the correct protocol split to every tool, disables parser compositing, and blocks test evaluation without a frozen config.

- [ ] **Step 2: Run and verify RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_strict_semantic_editing_paper_protocol_script.py -q
```

- [ ] **Step 3: Implement runner**

Provide stages `validate`, `semantic-train`, `export-calibration`, `export-validation`, `export-test`, `build-banks`, `calibrate`, `select-validation`, `evaluate-test`, and `all`. Reuse the existing semantic train/export scripts through environment variables. Persist a state file and timestamped logs.

- [ ] **Step 4: Run and verify GREEN**

Run the same pytest command and a manual dry run:

```bash
DRY_RUN=1 bash tools/run_strict_semantic_editing_paper_protocol.sh all
```

- [ ] **Step 5: Commit**

```bash
git add tools/run_strict_semantic_editing_paper_protocol.sh tools/formal/run_377_v338_semantic_train.sh tests/test_strict_semantic_editing_paper_protocol_script.py
git commit -m "feat: add strict semantic paper protocol runner"
```

### Task 8: Regression Verification

**Files:**
- Test only

- [ ] **Step 1: Run focused suite**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_semantic_eval_protocol.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_projected_soft_edit_leakage.py \
  tests/test_multiview_semantic_consistency.py \
  tests/test_semantic_edit_render_preview.py \
  tests/test_semantic_paper_metrics.py \
  tests/test_semantic_editing_paper_protocol.py \
  tests/test_select_semantic_editing_validation_config.py \
  tests/test_strict_semantic_editing_paper_protocol_script.py -q
```

- [ ] **Step 2: Run existing semantic bank regressions**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_suppress_projected_soft_leakage.py -q
```

- [ ] **Step 3: Inspect changes**

Run `git diff --check` and confirm formal defaults contain no parser composite.

### Task 9: Run Strict CoreView_377 Experiment

**Files:**
- Generated outputs under `exp/acceptdata/strict_semantic_paper_protocol_20260718/`

- [ ] **Step 1: Validate inputs**

```bash
bash tools/run_strict_semantic_editing_paper_protocol.sh validate
```

- [ ] **Step 2: Start strict semantic training and capture ETA**

Start the `semantic-train` stage with the existing v395/base checkpoint, record Beijing start time, measure iteration throughput from the first completed interval, and calculate the expected Beijing completion time. Keep the process attached to a resumable execution session and do not leave an untracked background process.

- [ ] **Step 3: Export isolated assets**

Run calibration, validation, and test exports and verify their record fingerprints and exact camera/frame lists.

- [ ] **Step 4: Build baselines and calibrate**

Build trained and voting banks from training/calibration assets only, run evidence calibration only on calibration records, and create validation sweeps.

- [ ] **Step 5: Freeze validation and evaluate test**

Select the validation configuration, run B0-B5 test evaluation, standard metrics, matched-retention curves, radius sensitivity, multiview metrics, and fair previews.

- [ ] **Step 6: Verify formal outputs**

Check provenance, fingerprints, required CSV/JSON/figures, retention constraint, and absence of parser composites in formal previews.
