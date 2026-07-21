# CoreView 386 Final Test And 393/377 Unified Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the validation-frozen CoreView_386 final test, prepare and train CoreView_393, then rerun CoreView_377 with the same 40k base plus 2k semantic strict protocol used by the multi-subject paper experiments.

**Architecture:** CoreView_386 reuses its existing checkpoint, banks, exports, and validation-only guarded config; its old test output remains immutable and the new frozen evaluation is written separately. CoreView_393 is prepared from raw ZJU data and Hulk parser output, then CoreView_393 and CoreView_377 run sequentially on the single RTX 4090 with identical base/semantic/validation/test stages. A persistent queue records Beijing timestamps and skips untouched test evaluation whenever guarded validation fails.

**Tech Stack:** Bash, Python 3.9, PyTorch/CUDA, Hydra/OmegaConf, Hulk CIHP parser, pytest, NumPy.

---

### Task 1: Complete CoreView_386 Guarded Final Test

**Files:**
- Reuse: `exp/acceptdata/coreview386_multisubject_strict_20260719/frozen_validation_config_guarded_final.json`
- Generate: `exp/acceptdata/coreview386_multisubject_strict_20260719/evaluation/test_guarded_final/`
- Generate: `exp/acceptdata/coreview386_multisubject_strict_20260719/audit/test_guarded_final_assessment.json`

- [ ] Verify the frozen config was selected from validation and points to threshold 0.1 with the validation-derived fallback list.
- [ ] Evaluate the existing nine untouched c21-c23 test records without reading test parser masks for calibration.
- [ ] Assess mIoU gap at most 0.02 and B5 leakage no greater than B1 at retention 0.5 and 0.6.
- [ ] Keep the previous failing test directory unchanged for provenance.

### Task 2: Prepare CoreView_393 Data And Hulk Parser

**Files:**
- Generate: `data/ZJUMoCap/CoreView_393/`
- Generate: `/remote-home/ming/Hulk/data/zju393_multiview_cihp/CoreView_393/`
- Generate: `/remote-home/ming/Hulk/experiments/release/generated_zju393_multiview/`
- Generate: `data/parsers_from_hulk_multiview/CoreView_393/mask_cihp/`

- [ ] Convert 23 cameras x 658 frames using camera translation scale 0.001 and symlinked RGB/masks.
- [ ] Prepare Hulk inputs for cameras 1-23 using symlinks.
- [ ] Run Hulk sequentially on GPU 0 and collect each camera's pseudo labels.
- [ ] Verify 15134 parser masks, no dangling links, and CIHP labels in range 0-19.

### Task 3: Add CoreView_393 Strict Configs

**Files:**
- Create: `configs/dataset/zjumocap_393_multiview_hq.yaml`
- Create: `configs/semantic/coreview393_strict_paper_protocol.json`
- Modify: `tests/test_coreview386_multisubject_pipeline.py`

- [ ] Add a failing test for CoreView_393 subject identity, frame range, and strict split.
- [ ] Run the focused test and confirm failure because the configs do not exist.
- [ ] Add the dataset config with base frame end 570 and the shared strict split: cameras 1-16 train/calibration, 17-20 validation, 21-23 test.
- [ ] Re-run the focused test and strict protocol tests.
- [ ] Commit only these tracked files with a Chinese commit message.

### Task 4: Run CoreView_393 Full Strict Experiment

**Files:**
- Generate: `exp/acceptdata/coreview393_multisubject_strict_20260721/`

- [ ] Run the 20-step base smoke test.
- [ ] Train base checkpoints at 10k, 20k, 30k, and 40k.
- [ ] Train the semantic adapter from 40k to 42k.
- [ ] Export calibration and validation assets, build banks, calibrate voting, and run the guarded validation selector.
- [ ] Only after validation passes, export nine untouched test records and evaluate B0-B5.

### Task 5: Run CoreView_377 Unified Full Strict Experiment

**Files:**
- Generate: `exp/acceptdata/coreview377_multisubject_strict_20260721/`

- [ ] Reuse the complete processed data and Hulk parser assets.
- [ ] Train a fresh 40k base using the same multi-subject recipe as CoreView_386/387/393.
- [ ] Train the semantic adapter to 42k and use `coreview377_strict_paper_protocol.json`.
- [ ] Run guarded validation before the untouched test and preserve the historical v395-based result separately.

### Task 6: Persistent Queue And Final Verification

**Files:**
- Generate: `exp/acceptdata/multisubject_393_377_queue_20260721/run_queue.sh`
- Generate: `exp/acceptdata/multisubject_393_377_queue_20260721/queue_status.txt`
- Generate: `exp/acceptdata/multisubject_393_377_queue_20260721/queue.pid`

- [ ] Run CoreView_393 first and CoreView_377 second on GPU 0.
- [ ] Record start/completion/failure timestamps in Beijing time for every stage.
- [ ] Estimate completion from measured Hulk throughput and the completed CoreView_387/392 base-stage durations.
- [ ] Confirm each final test has nine records and `uses_test_parser_for_calibration=false`.
- [ ] Run strict semantic regression tests and report the final four-subject status.
