# Strict Semantic Editing Paper Protocol Design

**Date:** 2026-07-18

**Goal:** Replace the current same-view calibration/evaluation loop with a reproducible, leakage-free protocol that produces fair qualitative comparisons, matched-retention curves, standard semantic metrics, and a complete baseline table for CoreView_377.

## 1. Scope

This work covers the five highest-priority paper requirements:

1. Strict semantic training, calibration, validation, and test splits.
2. Fair Hard/Ours qualitative rendering without test-view parser compositing.
3. Leakage-retention curves and matched-retention comparisons.
4. mIoU, per-part IoU, boundary F1, and boundary IoU.
5. A common evaluation entry point for parser, multi-view voting, hard-label, raw-probability, confidence/margin, and evidence-calibrated baselines.

It does not add a new avatar reconstruction model, tune CoreView_377 rendering quality, or implement cross-subject experiments. The resulting protocol and runner must be subject-agnostic so later subjects can reuse them unchanged.

## 2. Strict Split

The initial CoreView_377 paper protocol is:

```text
semantic training:
  cameras: 1-16
  poses:   0,120,240,360,480

evidence calibration:
  cameras: 1-16
  poses:   0,120,240,360,480

validation:
  cameras: 17-20
  poses:   60,300

held-out test:
  cameras: 21-23
  poses:   180,420,540
```

The RGB reconstruction backbone may have been trained with cameras 1-20. The semantic adapter must be retrained using only the semantic-training split above. Parser labels from validation and test records must not contribute to semantic adapter optimization or evidence calibration.

Validation may select thresholds and one global boundary radius. Test records may only evaluate a configuration already frozen from validation.

## 3. Protocol Manifest

Create `configs/semantic/coreview377_strict_paper_protocol.json` with:

```json
{
  "protocol_name": "coreview377_strict_paper_v1",
  "subject": "CoreView_377",
  "semantic_train": {"camera_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "frame_ids": [0, 120, 240, 360, 480]},
  "calibration": {"camera_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "frame_ids": [0, 120, 240, 360, 480]},
  "validation": {"camera_ids": [17, 18, 19, 20], "frame_ids": [60, 300]},
  "test": {"camera_ids": [21, 22, 23], "frame_ids": [180, 420, 540]},
  "parts": ["face", "hair", "upper", "lower", "shoes", "skin"],
  "allowed_adjacency": {
    "face": ["hair"],
    "hair": ["face"],
    "upper": ["skin", "lower"],
    "lower": ["upper", "skin"],
    "shoes": ["lower", "skin"],
    "skin": ["upper", "lower"]
  },
  "validation_grid": {
    "soft_thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5],
    "support_thresholds": [0.1, 0.2, 0.3],
    "boundary_radii": [0, 2, 4, 6]
  }
}
```

The manifest is the single source of truth. Shell scripts may accept a protocol path but must not independently redefine split lists.

## 4. Shared Protocol Module

Add `utils/semantic_eval_protocol.py` with focused interfaces:

```python
load_protocol(path) -> dict
validate_protocol(protocol) -> None
select_protocol_records(records, protocol, split_name) -> list[dict]
assert_record_set_matches_split(records, split) -> None
assert_no_forbidden_overlap(protocol) -> None
protocol_fingerprint(protocol) -> str
record_fingerprint(records) -> str
write_protocol_provenance(output_dir, protocol, selected_records, frozen_config=None) -> None
```

Required validation:

- `semantic_train`, `validation`, and `test` camera sets are disjoint.
- `semantic_train`, `validation`, and `test` `(camera_id, frame_id)` records are disjoint.
- Calibration records are a subset of semantic-training records.
- Every selected record exactly matches the requested split.
- Test evaluation requires a frozen validation configuration.
- Provenance records protocol and record fingerprints, source asset paths, selected parameters, and command arguments.

## 5. Asset Export and Calibration

The formal runner exports three independent asset roots from the strictly trained semantic checkpoint:

```text
assets/calibration/test-view/semantic_editable_assets
assets/validation/test-view/semantic_editable_assets
assets/test/test-view/semantic_editable_assets
```

The directory suffix remains `test-view` because that is the renderer's existing split name; the surrounding directory and provenance manifest define its protocol role.

`tools/calibrate_evidence_soft_edit_weights.py` gains `--protocol` and `--protocol-split`. The calibration command only accepts `--protocol-split calibration`. It must reject validation or test records even if they are present under the supplied asset root.

Calibration produces an evidence-calibrated bank but does not choose final thresholds. Validation sweeps the protocol grid and writes `frozen_validation_config.json`. Selection uses a deterministic objective:

```text
minimize mean actionable footprint leakage
subject to aggregate target activation retention >= 0.60
then maximize mean boundary F1
then prefer the smaller boundary radius
```

If no candidate reaches 0.60 retention, the runner reports failure instead of silently accepting a low-retention result.

## 6. Baselines

All baselines use the same test records, part definitions, valid masks, footprint rasterization, and metrics.

```text
B0 parser oracle
  Uses the direct 2D parser mask for each evaluated test image.
  It is labeled as an online 2D oracle and is not presented as a persistent asset.

B1 projected multi-view voting
  Builds a Gaussian label bank from semantic-training/calibration views only.

B2 hard trained label
  Uses editable_label == target part.

B3 raw semantic probability
  Uses semantic_probs[:, target] as the continuous edit strength.

B4 confidence/margin
  Uses semantic probability multiplied by confidence, semantic margin, and reliability.

B5 evidence-calibrated target/support
  Uses validation-frozen edit_target_weights and edit_support_weights.
```

No baseline may read test masks when constructing Gaussian ownership or selecting thresholds. B0 is the only method that reads the current test image parser mask, and its oracle status must appear in CSV/JSON output.

## 7. Matched-Retention Evaluation

Add a common curve evaluator that consumes per-view target and leakage activation for every baseline.

For continuous baselines, sweep thresholds from all distinct weights or a deterministic quantile grid. For hard-label and parser baselines, sweep a scalar edit-strength multiplier so comparisons can be made at the same total target activation without altering spatial ownership.

Report:

```text
target_activation
target_activation_retention_vs_hard
raw_footprint_leakage_ratio
actionable_footprint_leakage_ratio
selected_count
edit_strength
threshold
```

Matched-retention values are obtained by monotonic interpolation on each baseline curve at shared retention targets:

```text
0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
```

The main comparison uses retention levels supported by both compared methods. Extrapolation outside a method's observed range is forbidden.

## 8. Standard Semantic Metrics

For each baseline and part, rasterize its screen-space selection into a prediction map on the same held-out views used for leakage evaluation.

Hard prediction metrics use the validation-frozen threshold. Continuous metrics additionally report soft IoU based on fuzzy intersection and union.

Required metrics:

```text
per-part IoU
macro mIoU
micro IoU
precision
recall
boundary precision
boundary recall
boundary F1
boundary IoU
```

Boundary metrics use a fixed evaluation tolerance of two pixels. This tolerance is separate from the actionable-leakage radius sweep and cannot be selected on test data.

Empty target parts are excluded from macro averages and recorded explicitly. They are not assigned an automatic score of one.

## 9. Fair Qualitative Rendering

Formal preview mode disables:

```text
screen_mask_composite
footprint_edit_gate based on current-view parser masks
test-mask-based connected-component cleanup
```

Hard and Ours render through the same rasterizer and differ only in Gaussian edit weights. The preview summary records `uses_test_parser_composite=false` and fails formal mode if any parser-backed composite option is enabled.

Parser-composited previews may remain available as a diagnostic mode, but their filenames and manifests must contain `oracle_composite` and they cannot be included in formal Hard/Ours figures.

## 10. Outputs

The formal test runner writes:

```text
protocol_manifest.json
protocol_provenance.json
frozen_validation_config.json
baseline_summary.csv
per_part_metrics.csv
per_view_metrics.csv
leakage_retention_curve.csv
matched_retention.csv
boundary_radius_sensitivity.csv
summary.json
figures/leakage_retention.png
figures/baseline_semantic_metrics.png
figures/fair_edit_preview.png
```

Every report contains the protocol fingerprint, semantic checkpoint, asset root, bank source, baseline name, and whether current-view parser masks were used.

## 11. Formal Runner

Add `tools/run_strict_semantic_editing_paper_protocol.sh` with stages:

```text
validate
semantic-train
export-calibration
export-validation
export-test
build-banks
calibrate
select-validation
evaluate-test
all
```

The runner supports `DRY_RUN=1`, persists stage outputs, and refuses to evaluate test data until `frozen_validation_config.json` exists and passes fingerprint validation.

## 12. Error Handling

The formal pipeline fails on:

- split overlap;
- records outside the selected split;
- calibration using validation or test records;
- test evaluation without a frozen validation configuration;
- frozen configuration created for another protocol or checkpoint;
- formal preview with parser compositing enabled;
- missing baseline inputs;
- matched-retention comparisons with no shared retention range;
- no validation candidate meeting the minimum retention constraint.

Failure messages identify the offending camera/frame records and source path.

## 13. Testing

Tests follow red-green-refactor and cover:

1. Protocol parsing, overlap detection, record filtering, and fingerprints.
2. Calibration rejection of test records.
3. Validation selection with the retention constraint and deterministic tie-breaking.
4. Matched-retention interpolation without extrapolation.
5. Per-part IoU, macro/micro IoU, soft IoU, boundary F1, and boundary IoU.
6. Baseline weight construction for B1-B5 and oracle labeling for B0.
7. Formal preview rejection of parser composite flags.
8. Runner dry-run commands and provenance wiring.
9. Existing part-label-bank, calibration, leakage, multiview, and preview regression tests.

## 14. Acceptance Criteria

Implementation is complete when:

- the strict semantic checkpoint is trained only on the configured semantic-training records;
- calibration, validation, and test asset roots have disjoint protocol roles and verified record lists;
- the test evaluator cannot run with a bank calibrated from test records;
- formal Hard/Ours previews contain no parser-based compositing;
- B0-B5 are reported using one evaluator;
- mIoU, per-part IoU, boundary F1, and boundary IoU are produced;
- leakage-retention and matched-retention CSV/figures are produced;
- all focused and regression tests pass;
- a complete CoreView_377 formal output directory is generated with protocol provenance.
