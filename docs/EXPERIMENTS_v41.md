# v4.1 Experiment Workflow

This document consolidates the recommended workflow for the `body/cloth v4.1` binding branch.

## Main goal

The current mainline method is:
- explicit Gaussian-SMPL binding
- hierarchical `rigid / soft / free` deformation
- body/cloth-aware interpretable region binding
- temporal consistency for slip suppression
- paper-ready interpretability exports

## Recommended checkpoints

For the main paper result, prefer the latest numbered checkpoint from the main training run.

Example main experiment:
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main`

Preferred checkpoint policy:
- use `ckpt15000.pth` for the final mainline render
- optionally inspect `best_ckpt.pth` for metric-best analysis

## One-click interpretability pipeline

Run the full test-view + test-video interpretability pipeline:

```bash
/opt/miniconda3/envs/anim/bin/python tools/run_full_interpretability_pipeline.py \
  --main-exp exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main \
  --copy-assets
```

This will:
- render `test-view` interpretability maps
- summarize `test-view` statistics and keyframes
- build paper montage panels
- render `test-video` temporal maps
- summarize `test-video` statistics and keyframes

## Core outputs

### Main render outputs
- `test-view/renders`
- `test-view/binding_maps`
- `test-view/binding_analysis`
- `test-view/paper_montages`

### Temporal outputs
- `test-video/renders`
- `test-video/binding_maps`
- `test-video/binding_analysis`

## Interpretability maps

Supported interpretability maps:
- `layer`
- `region`
- `body_prob`
- `soft_prob`
- `cloth_prob`
- `semantic`
- `temporal`
- `thin`

## Summary + keyframes

Generate or refresh summary files manually:

```bash
/opt/miniconda3/envs/anim/bin/python tools/summarize_binding_interpretability.py \
  --exp-dir exp/<interp_exp_dir> \
  --split test-view \
  --copy-assets
```

Outputs:
- `binding_analysis/aggregate.json`
- `binding_analysis/keyframes.json`
- `binding_analysis/selected_assets/`

## Paper montage

Create paper-ready montage panels:

```bash
/opt/miniconda3/envs/anim/bin/python tools/make_binding_paper_montage.py \
  --exp-dir exp/<interp_exp_dir> \
  --split test-view \
  --panels gt render layer region body_prob cloth_prob thin semantic
```

## Compare experiments

Compare multiple experiments into one bundle:

```bash
/opt/miniconda3/envs/anim/bin/python tools/compare_experiments.py \
  --exp-dirs \
    exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717 \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main \
  --labels baseline v3 v4.1 \
  --split test-view \
  --output-dir exp/comparisons/v41_main
```

Outputs:
- `comparison.json`
- `comparison.csv`
- `comparison.md`

## Export CSV / LaTeX tables

```bash
/opt/miniconda3/envs/anim/bin/python tools/export_binding_tables.py \
  --exp-dirs \
    exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717 \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main \
  --labels baseline v3 v4.1 \
  --split test-view \
  --output-dir exp/comparisons/v41_main/tables
```

Outputs:
- `metrics_table.csv`
- `binding_table.csv`
- `metrics_table.tex`
- `binding_table.tex`

## Suggested paper package

For the current paper draft, the recommended final package is:
- main quantitative metrics from `best_test_metrics.json`
- `test-view` interpretability summary
- `test-video` temporal summary
- montage figures from representative keyframes
- one comparison bundle across `baseline / v3 / v4.1`
- one table bundle exported as CSV and LaTeX
