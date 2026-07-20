# Validation Guard And CoreView 387/392 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one subject-independent validation rule that preserves mIoU and reduces leakage, add an empty-small-part B3 fallback for B5, and run untouched CoreView_387/CoreView_392 experiments sequentially.

**Architecture:** The evaluator remains the single source of B0-B5 metrics. A guarded selector reads fixed-threshold validation reports, enforces the same mIoU/leakage gates for every subject, derives fallback parts only from validation when B5 predicts an empty non-empty target and B3 has support, and writes those choices into the frozen config. The existing generic multi-subject pipeline consumes the frozen config for test; CoreView_387 and CoreView_392 use independent dataset/protocol configs and a persistent single-GPU queue.

**Tech Stack:** Python 3.9, pytest, NumPy, OmegaConf/Hydra, Bash, PyTorch/CUDA, Hulk CIHP parser.

---

### Task 1: Add CoreView_387 And CoreView_392 Protocol Configs

**Files:**
- Create: `configs/dataset/zjumocap_387_multiview_hq.yaml`
- Create: `configs/dataset/zjumocap_392_multiview_hq.yaml`
- Create: `configs/semantic/coreview387_strict_paper_protocol.json`
- Create: `configs/semantic/coreview392_strict_paper_protocol.json`
- Modify: `tests/test_coreview386_multisubject_pipeline.py`

- [ ] Add failing assertions for subjects, frame limits, camera splits, and untouched test frames.
- [ ] Run `python -m pytest -q tests/test_coreview386_multisubject_pipeline.py` and confirm failure.
- [ ] Add configs using cameras 1-16 for semantic train/calibration, 17-20 for validation, and 21-23 for test. Use base frame end 570 for 387 and 550 for 392.
- [ ] Re-run the test and confirm it passes.
- [ ] Commit with a Chinese message.

### Task 2: Add The Validation-Gated Selector

**Files:**
- Create: `tools/select_guarded_semantic_validation_config.py`
- Create: `tests/test_select_guarded_semantic_validation_config.py`

- [ ] Write failing tests requiring candidates to pass `B5 mIoU >= B1 mIoU - 0.02` and lower B5 leakage than B1 at retention 0.5 and 0.6.
- [ ] Add a failing test requiring deterministic selection by lowest leakage, then highest mIoU, then highest boundary F1.
- [ ] Add a failing test deriving fallback parts only when B5 predicts zero pixels for a non-empty validation target and B3 predicts non-zero pixels.
- [ ] Implement report loading from `baseline_summary.csv`, `matched_retention.csv`, and `per_part_metrics.csv`, then write fingerprints, threshold values, gate evidence, and `b5_fallback_parts` to frozen JSON.
- [ ] Run the new test file and existing selector tests.
- [ ] Commit with a Chinese message.

### Task 3: Add B5 Empty-Part Fallback Evaluation

**Files:**
- Modify: `tools/evaluate_semantic_editing_paper_protocol.py`
- Modify: `tests/test_semantic_editing_paper_protocol.py`

- [ ] Write failing tests showing B5 uses `semantic_probs` for a frozen fallback part and retains `edit_target_weights` for all other parts.
- [ ] Add CLI support for `--b5-fallback-part` and frozen-config `selected.b5_fallback_parts`.
- [ ] Pass the frozen fallback set through fixed metrics and leakage curves without using test masks for selection.
- [ ] Run evaluator and strict-protocol regression tests.
- [ ] Commit with a Chinese message.

### Task 4: Validate And Freeze The 386 Guarded Candidate

**Files:**
- Generate: `exp/acceptdata/coreview386_multisubject_strict_20260719/evaluation/validation_guarded_final/`
- Generate: `exp/acceptdata/coreview386_multisubject_strict_20260719/frozen_validation_config_guarded_final.json`

- [ ] Feed the completed 0.05-0.50 validation reports to the guarded selector.
- [ ] Re-evaluate validation using the frozen threshold and fallback list.
- [ ] Require assessment pass, non-zero face/shoes/skin IoU where B3 provides support, and no test evaluation.
- [ ] Record the final validation metrics and fingerprints.

### Task 5: Prepare 387 And 392 Data And Hulk Masks

**Files:**
- Generate: `data/ZJUMoCap/CoreView_387/`
- Generate: `data/ZJUMoCap/CoreView_392/`
- Generate: `data/parsers_from_hulk_multiview/CoreView_387/mask_cihp/`
- Generate: `data/parsers_from_hulk_multiview/CoreView_392/mask_cihp/`

- [ ] Convert raw subjects with translation scale 0.001 and symlinked images/masks.
- [ ] Audit 387 as 23 cameras x 654 frames and 392 as 23 cameras x 556 frames.
- [ ] Run Hulk sequentially on GPU 0 and collect parser masks.
- [ ] Verify every label is in CIHP range 0-19.

### Task 6: Queue Full 387 And 392 Experiments

**Files:**
- Generate: `exp/acceptdata/coreview387_multisubject_strict_20260720/`
- Generate: `exp/acceptdata/coreview392_multisubject_strict_20260720/`
- Generate: `exp/acceptdata/multisubject_387_392_queue_20260720/queue_status.txt`

- [ ] Run a 20-step base smoke test for each subject.
- [ ] Start a persistent single-GPU queue: 387 base 40k, semantic 2k, calibration/validation/freeze/test; then the same stages for 392.
- [ ] Use the guarded validation selector identically for both subjects before each untouched test.
- [ ] Record Beijing timestamps, PIDs, checkpoint paths, and stage failures.
- [ ] Measure the first 100 stable base iterations and calculate the final Beijing completion time.

### Task 7: Final Verification

**Files:**
- Modify only the experiment record if required.

- [ ] Run all strict semantic protocol tests.
- [ ] Confirm no 377 adopted-geometry asset is used by 386/387/392 bank, calibration, or evaluation stages.
- [ ] Confirm each test split contains exactly nine frozen records and `uses_test_parser_for_calibration=false`.
- [ ] Commit any remaining tracked record changes with a Chinese message.
