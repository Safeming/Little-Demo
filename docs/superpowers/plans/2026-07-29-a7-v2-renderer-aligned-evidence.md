# A7 V2 Renderer-Aligned Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CoreView_377 A7 v2 canary whose temporal evidence uses differentiable rasterizer attribution, freezes parts without valid A5 coverage, and preserves the A5 thresholded carrier topology.

**Architecture:** A single differentiable selection render uses RGB gradient channels to extract target, outer, and boundary alpha/transmittance contributions for each Gaussian. The evidence accumulator stores raw renderer contributions and normalized compatibility fields for the existing reliability/calibration pipeline. Candidate calibration freezes `face,upper,shoes,skin` for the 377 canary and prevents every A5-selected carrier from crossing below the frozen `0.2` threshold.

**Tech Stack:** Python 3, NumPy, PyTorch autograd/CUDA, diff-gaussian-rasterization, JSON/NPZ, pytest, Bash.

---

### Task 1: Renderer contribution attribution and temporal accumulation

**Files:**
- Create: `utils/renderer_aligned_temporal_evidence.py`
- Create: `tests/test_renderer_aligned_temporal_evidence.py`

- [ ] Write a synthetic differentiable-render test for RGB target/outer/boundary gradients.
- [ ] Verify the test fails because the attribution API is absent.
- [ ] Implement `extract_renderer_region_contributions()` and deterministic online accumulation of target/outer/boundary contributions.
- [ ] Export raw contribution means/flicker plus normalized `temporal_*` compatibility arrays.
- [ ] Run the focused tests and verify finite, non-negative, deterministic outputs.

### Task 2: Frozen parts and topology-preserving calibration

**Files:**
- Modify: `utils/temporal_reliability_calibration.py`
- Modify: `tools/calibrate_temporal_reliable_a7_weights.py`
- Modify: `tests/test_temporal_reliability_calibration.py`
- Modify: `tests/test_calibrate_temporal_reliable_a7_weights.py`

- [ ] Add failing tests requiring frozen parts to remain bitwise identical to A5.
- [ ] Add a failing test requiring zero threshold crossings at `0.2`.
- [ ] Implement frozen-part preservation and selected-carrier lower bounds.
- [ ] Make v2 candidate proxy consume renderer target/outer/boundary contribution fields.
- [ ] Reject candidates with any topology crossing.

### Task 3: V2 contract and renderer evidence CLI

**Files:**
- Create: `configs/semantic/frozen_a7_renderer_aligned_v2_canary_377.json`
- Create: `tools/build_renderer_aligned_temporal_evidence.py`
- Create: `tests/test_build_renderer_aligned_temporal_evidence.py`
- Modify: `utils/frozen_semantic_method.py`
- Modify: `tests/test_frozen_a7_temporal_method.py`

- [ ] Freeze renderer attribution mode, contribution epsilon, coverage threshold, frozen parts, and topology threshold.
- [ ] Implement the formal `c01,c05,c09,c13 / 0:570:5` CLI using one render and six RGB-gradient backward calls per frame.
- [ ] Multiply raster contribution coefficients by per-Gaussian recolor sensitivity.
- [ ] Save formal provenance, fingerprints, raw renderer arrays, and compatibility arrays.
- [ ] Run parser/contract/unit tests and a one-frame CUDA canary.

### Task 4: Formal 377 canary runner

**Files:**
- Create: `exp/acceptdata/a7_renderer_aligned_v2_canary_377/run_377_v2_validation.sh`

- [ ] Build formal renderer-aligned evidence and 24 candidates in an independent output root.
- [ ] Select at most two valid candidates without opening c21.
- [ ] Run `c17-c20 / 0:570:5` temporal validation with dense matched-response strengths.
- [ ] Run spatial guards and record BJT start/end timestamps.
- [ ] Stop if fewer than two candidates pass evidence/topology guards.
