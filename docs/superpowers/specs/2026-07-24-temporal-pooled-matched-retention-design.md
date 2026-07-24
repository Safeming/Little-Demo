# Temporal Pooled Matched-Retention Formal Tables Design

## Goal

Replace unstable per-frame ratio averages in the five-subject temporal experiment with pooled burden, explicit coverage constraints, and target-matched temporal statistics, using the existing 34,200 full-strength recolor records without rerendering.

## Exact Matching

Recolor output deltas are linear in the global edit strength because Gaussian geometry and opacity remain fixed and only precomputed colors are linearly interpolated. For each subject, part, and frame, Voting at full strength is the target-effect reference.

For target retention `r` in `{0.25, 0.50}`:

- Voting is evaluated at strength `r`.
- A5 is reachable when `A5_target_full / Voting_target_full >= r`.
- Reachable A5 strength is `r / (A5_target_full / Voting_target_full)`.
- Target, outer, and boundary deltas are scaled by the corresponding exact strength.

Frames with no Voting target effect are unsupported references and never enter burden denominators.

## Coverage Constraint

Each subject-part cell reports reference count, supported count, and coverage rate. A part is eligible for the formal table at a retention only when its coverage rate is at least 0.80 for every one of the five subjects. This cross-subject rule is fixed before inspecting matched burden values and prevents selective subject or part inclusion.

All ineligible parts remain visible in `coverage_table.csv` and `part_table.csv`, including face, skin, and shoes failure cases.

## Formal Metrics

For each reachable subject-part sequence and method:

- pooled outer burden: `sum(outer_delta_sum) / sum(target_delta_sum)`;
- pooled boundary burden: `sum(boundary_outer_delta_sum) / sum(target_delta_sum)`;
- pooled screen selection leakage: `sum(outside_selection_mass) / sum(inside_selection_mass)`;
- matched outer and boundary temporal flicker, computed only from consecutive supported frame pairs and normalized by the sequence mean.

The formal five-subject table first pools eligible parts within each subject, then treats subjects equally. A5-minus-Voting paired bootstrap confidence intervals use 10,000 subject resamples.

## Outputs

Write corrected tables under:

`exp/acceptdata/five_subject_semantic_temporal_stability_20260724/aggregate/formal_matched_retention/`

Required files:

- `coverage_table.csv`;
- `part_table.csv`;
- `subject_table.csv`;
- `formal_table.csv`;
- `paired_statistics.csv`;
- `summary.json`.

The summary records the exact matching assumption, coverage threshold, eligible parts for each retention, excluded parts, and source aggregate fingerprint.
