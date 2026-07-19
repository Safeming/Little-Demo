# CoreView_386 Multi-Subject Strict Semantic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert CoreView_386 into the CoreView_377 project/Hulk layout, train an independent 40k base and 2k semantic adapter, and evaluate frozen Scheme A without subject-specific tuning.

**Architecture:** Reuse the existing raw ZJU converter and parameterized Hulk tools. Add subject-generic semantic train/export launchers, a fixed-config materializer, a 386 dataset/protocol config, and a thin strict orchestrator. Base reconstruction remains parser-free; Hulk supervision enters only semantic training and semantic asset export.

**Tech Stack:** Bash, Python 3.9, Hydra/OmegaConf, PyTorch/CUDA, NumPy, Hulk CIHP parser, pytest, existing 3DGS renderer/evaluator.

---

### Task 1: CoreView_386 dataset and strict protocol configs

**Files:**
- Create: `tests/test_coreview386_multisubject_pipeline.py`
- Create: `configs/dataset/zjumocap_386_multiview_hq.yaml`
- Create: `configs/semantic/coreview386_strict_paper_protocol.json`

- [ ] **Step 1: Write failing config tests**

Add tests that load both files and assert the 386 subject, cameras 1-20 for base training, cameras 21-23 held out, 768x768 images, and the exact semantic/calibration/validation/test split copied from 377.

```python
def test_coreview386_dataset_uses_multiview_paper_split():
    payload = OmegaConf.load("configs/dataset/zjumocap_386_multiview_hq.yaml")
    assert payload.dataset.subject == "CoreView_386"
    assert [str(v) for v in payload.dataset.train_views] == [str(v) for v in range(1, 21)]
    assert [str(v) for v in payload.dataset.val_views] == ["21", "22", "23"]
    assert list(payload.dataset.img_hw) == [768, 768]

def test_coreview386_protocol_matches_cross_subject_split():
    payload = json.loads(Path("configs/semantic/coreview386_strict_paper_protocol.json").read_text())
    assert payload["subject"] == "CoreView_386"
    assert payload["semantic_train"]["camera_ids"] == list(range(1, 17))
    assert payload["semantic_train"]["frame_ids"] == [0, 120, 240, 360, 480]
    assert payload["validation"] == {"camera_ids": [17, 18, 19, 20], "frame_ids": [60, 300]}
    assert payload["test"] == {"camera_ids": [21, 22, 23], "frame_ids": [180, 420, 540]}
```

- [ ] **Step 2: Run RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview386_multisubject_pipeline.py -q`

Expected: FAIL because both config files are absent.

- [ ] **Step 3: Create both configs**

Copy the 377 multiview dataset structure and strict protocol. Change only dataset/protocol names and subject to CoreView_386. Keep all cameras, frames, parts, adjacency pairs, grids, retention targets, and boundary tolerance identical.

- [ ] **Step 4: Run GREEN and commit**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_coreview386_multisubject_pipeline.py -q`

Commit only the three task files with message `配置：增加386多主体严格协议`.

### Task 2: Cross-subject fixed evaluation config materializer

**Files:**
- Create: `tools/materialize_fixed_semantic_evaluation_config.py`
- Create: `tests/test_materialize_fixed_semantic_evaluation_config.py`

- [ ] **Step 1: Write failing tests**

Test this pure API and missing-key rejection:

```python
payload = materialize_fixed_config(
    protocol={"protocol_name": "coreview386_strict_paper_v1", "subject": "CoreView_386"},
    template={"selected": {"soft_threshold": "0.5", "support_threshold": "0.3", "boundary_radius": "6"}},
    checkpoint_fingerprint="ckpt386",
    bank_fingerprint="bank386",
)
assert payload["selected"] == {"soft_threshold": 0.5, "support_threshold": 0.3, "boundary_radius": 6}
assert payload["selection_mode"] == "cross_subject_fixed_from_template"
```

- [ ] **Step 2: Run RED**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_materialize_fixed_semantic_evaluation_config.py -q`

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement tool and CLI**

Implement `materialize_fixed_config()` using `protocol_fingerprint()`. Require `soft_threshold`, `support_threshold`, and `boundary_radius` in the template. CLI arguments are `--protocol`, `--template`, `--checkpoint-fingerprint`, `--bank-fingerprint`, and `--output`.

- [ ] **Step 4: Run GREEN and commit**

Run the focused test and commit the tool/test with message `工具：物化跨主体固定评估配置`.

### Task 3: Subject-generic semantic train and export launchers

**Files:**
- Create: `tools/formal/run_subject_semantic_train.sh`
- Create: `tools/formal/run_subject_semantic_export.sh`
- Modify: `tests/test_coreview386_multisubject_pipeline.py`

- [ ] **Step 1: Add failing launcher tests**

Require both launchers to use mandatory `SUBJECT`, contain no `assets/adopted_geometry/377` and no `explicit_binding_render_preset`, and pass `dataset.subject=$SUBJECT`. Require semantic adapter-only training in the train launcher and semantic editable export in the export launcher.

- [ ] **Step 2: Run RED**

Run the CoreView386 pipeline test. Expected: FAIL because launchers are absent.

- [ ] **Step 3: Implement generic semantic train launcher**

Validate base config/checkpoint, subject data, parser masks, compact mapping, and Python. Load the base Hydra config and checkpoint. Enable semantic logits adapters and stageB semantic loss, freeze geometry/appearance/pose/deformers, use cameras 1-16 and the protocol's five semantic frames, and train `TRAIN_STEPS=2000`. Do not pass a 377 formal preset.

- [ ] **Step 4: Implement generic export launcher**

Load the semantic experiment config/checkpoint with `render.py`, pass subject/views/frames/parser root, enable direct parser semantic editable assets and compact-head export, and do not pass a 377 formal preset.

- [ ] **Step 5: Verify and commit**

Run `bash -n` on both launchers and the focused pytest. Commit with message `功能：增加主体通用语义训练与导出入口`.

### Task 4: Multi-subject strict Scheme A orchestrator

**Files:**
- Create: `tools/run_multisubject_strict_semantic_protocol.sh`
- Modify: `tests/test_coreview386_multisubject_pipeline.py`

- [ ] **Step 1: Add failing orchestrator tests**

Require stages `validate`, `semantic-train`, `export-calibration`, `export-validation`, `export-test`, `build-banks`, `calibrate-voting`, `evaluate-validation`, `evaluate-test`, and `all`. Require use of the fixed-config materializer and voting bank calibration. Forbid `select_semantic_editing_validation_config.py`.

- [ ] **Step 2: Run RED**

Run the focused pipeline test. Expected: FAIL because the orchestrator is absent.

- [ ] **Step 3: Implement the orchestrator**

Require `SUBJECT`, `PROTOCOL`, `OUTPUT_ROOT`, `BASE_EXP`, `BASE_CKPT`, `PARSER_ROOT`, `DATA_ROOT`, and `FROZEN_TEMPLATE`. Build raw and voting banks, calibrate the voting bank with penalty 0.2 and the fixed ten support pairs, materialize the 386 frozen config from the 377 template, evaluate validation with threshold 0.5/radius 6, and evaluate test only from the frozen config.

- [ ] **Step 4: Verify and commit**

Run `bash -n`, a `DRY_RUN=1` full command expansion, and focused pytest. Commit with message `功能：增加多主体方案A严格协议入口`.

### Task 5: Convert CoreView_386 and produce Hulk parser masks

**Generated outputs:**
- `data/ZJUMoCap/CoreView_386/`
- `/remote-home/ming/Hulk/data/zju386_multiview_cihp/CoreView_386/`
- `/remote-home/ming/Hulk/experiments/release/generated_zju386_multiview/`
- `data/parsers_from_hulk_multiview/CoreView_386/mask_cihp/`

- [ ] **Step 1: Convert raw data**

Run one line:

```bash
/opt/miniconda3/envs/ictrl/bin/python tools/prepare_zju_from_raw.py --raw-subject-dir /remote-home/ming/dataSet/CoreView_386 --output-root data/ZJUMoCap --subject-name CoreView_386 --mask-source mask --camera-translation-scale 0.001
```

- [ ] **Step 2: Audit conversion**

Require 23 camera directories, 646 JPG and PNG files per camera, 646 model NPZ files, and `camera_translation_scale=0.001`.

- [ ] **Step 3: Prepare and run Hulk**

Prepare all 23 camera inputs with `prepare_hulk_zju377_multiview_cihp.py`, custom `zju386_multiview_cihp` root, and symlinks. Run `run_hulk_zju377_multiview_parsing.py` with generated config dir `generated_zju386_multiview` and experiment prefix `zju386_mv_hulk`.

- [ ] **Step 4: Collect and audit outputs**

Collect with `collect_hulk_zju377_multiview_parser.py --link`. Require exactly 14,858 parser PNG links and sampled palette labels within `[0,19]`.

### Task 6: Base smoke and formal 40k reconstruction

**Generated outputs:**
- `exp/acceptdata/coreview386_multisubject_strict_20260719/base_smoke/`
- `exp/acceptdata/coreview386_multisubject_strict_20260719/base_train_40k/`

- [ ] **Step 1: Validate data through existing subject wrapper**

Set `SUBJECT=CoreView_386`, `DATASET_CONFIG=zjumocap_386_multiview_hq`, and the new parser root, then run stage `validate`.

- [ ] **Step 2: Run 20-step smoke**

Use the existing base wrapper with `BASE_SMOKE_STEPS=20` and all test/save/checkpoint steps set to 20. Require `ckpt20.pth`, no traceback/non-finite loss, and nonzero Gaussian count.

- [ ] **Step 3: Start fresh formal 40k training**

Use a new base directory, `BASE_FULL_STEPS=40000`, and checkpoints `10000,20000,30000,40000`. After 500-1000 steps, estimate finish time from actual throughput and report Beijing time.

- [ ] **Step 4: Verify completion**

Require `ckpt40000.pth`, `.hydra/config.yaml`, a `Training complete.` log line, and no non-finite loss record.

### Task 7: Semantic 2k, assets, Scheme A banks, validation and one-shot test

**Generated outputs:**
- `exp/acceptdata/coreview386_multisubject_strict_20260719/semantic_train_strict/`
- `exp/acceptdata/coreview386_multisubject_strict_20260719/assets/`
- `exp/acceptdata/coreview386_multisubject_strict_20260719/banks/`
- `exp/acceptdata/coreview386_multisubject_strict_20260719/evaluation/`

- [ ] **Step 1: Train semantic adapter 2k**

Use the generic semantic train launcher with base `ckpt40000.pth`; require `ckpt42000.pth`.

- [ ] **Step 2: Export calibration and validation assets**

Run orchestrator export stages and prune to exactly 80 calibration and 8 validation records.

- [ ] **Step 3: Build banks and calibrate final Scheme A**

Build raw and projected voting banks. Calibrate the voting bank and verify `edit_target_weights` plus `edit_support_weights`.

- [ ] **Step 4: Evaluate fixed validation**

Materialize fixed threshold 0.5, support 0.3, radius 6 from the 377 template. Run fixed validation and assessor at retention 0.5/0.6. Do not select subject-specific parameters.

- [ ] **Step 5: Export and evaluate test once**

After validation provenance audit, export exactly nine test records and run test once with the frozen config. Write `test_assessment.json`.

### Task 8: Verification and Chinese experiment record

**Files:**
- Create: `docs/CoreView386多主体严格语义实验记录_20260719.md`

- [ ] **Step 1: Run regression suite**

Run the new tests plus part-label-bank, evidence calibration, evaluator, and assessor tests.

- [ ] **Step 2: Audit provenance and fingerprints**

Confirm calibration only c01-c16, validation only c17-c20, test only c21-c23, and frozen checkpoint/bank fingerprints match actual files.

- [ ] **Step 3: Write the record**

Record data/Hulk counts, base and semantic finish Beijing times, checkpoints, B1/B5 validation/test mIoU, boundary F1, leakage at 0.5/0.6, per-part deltas, failures, and output paths.

- [ ] **Step 4: Commit**

Commit only the new record with message `文档：记录386多主体严格语义实验`.
