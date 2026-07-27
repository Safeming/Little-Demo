# Coverage-Constrained Real Editing and Flicker Failure Analysis Design

## Goal

Complete the remaining paper statistics without retraining by producing coverage-constrained pooled real-editing tables, fixed-versus-adaptive temporal flicker diagnostics, and an explicit formal failure-case report.

## Inputs

Use only frozen existing artifacts:

```text
exp/acceptdata/five_subject_real_editing_matched_strength_20260723/CoreView_{subject}/metrics.csv
exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/all_metrics.csv
exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_matched_retention/
```

Subjects are `377, 386, 387, 393, 394`; parts are `hair, face, upper, lower, shoes, skin`; retentions are `0.25, 0.50`.

## Real-Editing Formal Rule

Evaluate recolor, removal, and texture separately. For each task and retention, a part is formally eligible only when A5 reaches the retention on at least 80% of Voting-supported records in every subject.

For each eligible subject-task-part cell and method:

```text
pooled outer burden = sum(matched outer delta) / sum(matched target delta)
pooled boundary burden = sum(matched boundary delta) / sum(matched target delta)
```

Pool eligible parts within each subject before division, then treat five subjects equally. A5-minus-Voting paired bootstrap confidence intervals use 10,000 subject resamples. Excluded parts remain in coverage and failure tables.

Write results under:

```text
exp/acceptdata/five_subject_real_editing_matched_strength_20260723/
  aggregate/formal_coverage_constrained/
```

Required files are `coverage_table.csv`, `part_table.csv`, `subject_table.csv`, `formal_table.csv`, `paired_statistics.csv`, and `summary.json`.

## Fixed-Versus-Adaptive Flicker Rule

Use the same reachable frame set and the same task-independent formal eligible parts already selected by the matched-retention temporal protocol.

For every subject-part-retention-method sequence, compute:

- `fixed`: original full-strength outer and boundary delta means on the common reachable frames;
- `adaptive`: the existing per-frame matched-retention scaled values;
- normalized consecutive-frame flicker, using only frame pairs with index difference exactly one;
- compensation penalty: `adaptive_flicker - fixed_flicker`.

Voting has a constant retention multiplier, so its normalized fixed and adaptive flicker should agree up to floating-point tolerance. A5 may differ because its adaptive strength varies per frame. Report subject-equal paired A5-minus-Voting statistics for both modes and A5 adaptive-minus-fixed compensation statistics.

Write results under:

```text
exp/acceptdata/five_subject_semantic_temporal_stability_20260724/
  aggregate/formal_flicker_diagnostic/
```

## Failure Cases

Generate a paper-ready Markdown and CSV failure inventory covering:

- face, skin, and shoes coverage failures for all three editing tasks;
- CoreView_377 lower pooled-burden failure;
- temporal outer and boundary flicker regressions;
- available qualitative assets: the three real-editing paper sheets and 15 temporal videos.

Each failure entry records the experiment, subject/task/part, retention, metric, Voting value, A5 value, direction, interpretation, and artifact path. Do not hide unsupported cells or create replacement results through test-driven method tuning.

Write the report to:

```text
docs/正式论文失败案例与时序诊断_20260727.md
```

and the machine-readable table beside the formal experiment outputs.

## Execution

Implement two standalone tested summarizers plus one queue script. The queue runs on CPU, records start/end/status in Beijing time, and never launches training or rerendering.

## Verification

- unit tests lock exact pooled ratios, per-task cross-subject eligibility, common support, and fixed/adaptive flicker behavior;
- all CSV/JSON numeric outputs must be finite;
- source fingerprints and statistical assumptions must be recorded;
- rerunning to an independent temporary directory must reproduce byte-identical outputs;
- commit only new scripts, tests, specifications, plans, and the requested failure report, preserving unrelated worktree changes.
