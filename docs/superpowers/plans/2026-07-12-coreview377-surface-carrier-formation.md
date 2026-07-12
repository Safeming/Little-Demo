# CoreView377 Surface Carrier Formation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a deterministic, one-command 64k CoreView_377 from-zero run whose initial Gaussian support is surface-area-aware and whose first 12k iterations use gradual patch-level carrier competition with automatic image-collapse gates.

**Architecture:** A pure NumPy sampler creates deterministic surface points, per-point surface areas, patch IDs, anisotropic scales, and rotations. `GaussianModel` consumes this enriched point cloud, while a pure PyTorch training helper regularizes per-patch opacity/support without hard checkpoint surgery. A thin launcher reuses the verified 64k long-horizon pipeline with the new dataset mode, competition option, gate schedule, and diagnostics.

**Tech Stack:** Python 3.9, NumPy, PyTorch, trimesh/SMPL data, Hydra YAML, Bash, pytest.

---

### Task 1: Deterministic Surface Sampler And Cache Identity

**Files:**
- Create: `utils/surface_carrier.py`
- Modify: `dataset/zjumocap.py`
- Create: `tests/test_surface_carrier_sampling.py`

- [ ] Write failing tests proving cache hashes change with seed/quota changes, sampling is deterministic, and head/shoulder quotas remain within configured bounds.
- [ ] Run `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_surface_carrier_sampling.py -q` and verify RED because the module does not exist.
- [ ] Implement `surface_carrier_cache_identity()` and `sample_surface_carriers()` using exclusive region masks, area-weighted face selection, deterministic barycentric sampling, represented-area accounting, local frames, and quaternion rotations.
- [ ] Update `ZJUMoCapDataset.readPointCloud()` so `surface_carrier_v1` uses a hash-qualified PLY plus JSON/NPZ manifest and refuses a mismatched cache.
- [ ] Re-run the focused test and verify all tests pass.

### Task 2: Enriched Point Cloud Initialization

**Files:**
- Modify: `utils/graphics_utils.py`
- Modify: `scene/gaussian_model.py`
- Extend: `tests/test_surface_carrier_sampling.py`

- [ ] Write failing tests for a `SurfacePointCloud` payload and for Gaussian initialization preferring supplied anisotropic scales/rotations over nearest-neighbor scales.
- [ ] Run the focused tests and verify RED on the missing payload behavior.
- [ ] Add `SurfacePointCloud` with backward-compatible point/color/normal fields plus scales, rotations, patch IDs, represented areas, region IDs, and cache identity.
- [ ] Make `create_from_pcd()` consume enriched scales and rotations only when present, retain the legacy path otherwise, and retain surface metadata as non-parameter tensors on the Gaussian model.
- [ ] Re-run focused tests and the existing Gaussian-model tests.

### Task 3: Early Patch Carrier Competition

**Files:**
- Modify: `train.py`
- Create: `tests/test_surface_carrier_competition.py`

- [ ] Write failing CPU tests showing the loss is differentiable, preserves group support budget, penalizes redundant mid-opacity carriers, and is zero when surface metadata is absent.
- [ ] Run the focused test and verify RED on the missing helper.
- [ ] Implement `_compute_surface_carrier_competition_loss()` with scheduled weight, patch scatter-add support budgets, opacity polarization, tangent coverage floor, and diagnostics.
- [ ] Integrate it into `base_loss`, progress logs, and wandb metrics without changing behavior when disabled.
- [ ] Add an image gate helper that records 2k/5k/8k/12k LPIPS, emits `SURFACE_CARRIER_GATE_FAILED` only for a greater-than-0.01 regression from the preceding gate, and otherwise leaves the uninterrupted 64k process running.
- [ ] Re-run focused and existing anchor-tether tests.

### Task 4: Configuration And One-Command Launcher

**Files:**
- Create: `configs/option/stageA_377_multiview_explicit_hq_fromzero_surface_carrier_v1.yaml`
- Create: `exp/zero_train_to_v395/coreview377_surface_carrier_formation_20260712_bjt/launch_surface_carrier_formation.sh`
- Modify: `exp/zero_train_to_v395/coreview377_neutral_longhorizon_20260710_bjt/launch_neutral_longhorizon.sh`
- Create: `tests/test_coreview377_surface_carrier_formation_pipeline.py`

- [ ] Write failing tests for 50k initialization, deterministic seed/hash inputs, 64k horizon, competition ramp ending at 12k, gate milestones, no pretrained checkpoint, and inherited long-horizon schedules.
- [ ] Run the pipeline test and verify RED because the option and launcher do not exist.
- [ ] Allow the base launcher to override initialization settings without changing existing defaults.
- [ ] Add the surface-carrier option and wrapper with `TRAIN_ITERS=64000`, tests at 2k/5k/8k/12k and later long-horizon milestones, and final selection output.
- [ ] Re-run the pipeline test and all related launcher tests.

### Task 5: Verification And Formal Launch

**Files:**
- Runtime output: `exp/zero_train_to_v395/coreview377_surface_carrier_formation_20260712_bjt/`

- [ ] Run `python -m py_compile` on all modified Python files.
- [ ] Run the focused pytest suite for sampler, competition, densification, tether, and launchers.
- [ ] Generate a fresh 50k carrier cache and verify point count, quota distribution, deterministic hash, finite scales, positive represented areas, and tangent/normal anisotropy.
- [ ] Run a 20-iteration GPU smoke and verify checkpoint, metrics, carrier logs, and clean process exit.
- [ ] Start the formal detached 64k run on the first free GPU and record PID, run directory, start time, and initial progress.
- [ ] Estimate Beijing completion time from the measured 20-step startup plus the latest verified 64k wall time, and provide `grep`/`tail` tracking commands that do not require `rg`.
