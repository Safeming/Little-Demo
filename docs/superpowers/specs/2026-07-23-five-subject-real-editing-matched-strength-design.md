# Five-Subject Real Editing Matched-Strength Design

## Goal

Add a strict matched-strength evaluation for recolor, opacity removal, and texture replacement across the frozen five-subject A5 protocol. The experiment must separate leakage reduction from weaker target edits and must report method coverage instead of dividing by zero-reference targets.

## Frozen Protocol

- Subjects: CoreView 377, 386, 387, 393, and 394.
- Test records: c21-c23 at frames 180, 420, and 540.
- Parts: hair, face, upper, lower, shoes, and skin.
- Methods: Raw Hard, Voting, and frozen A5.
- Tasks: recolor, opacity removal, and canonical texture replacement.
- Strength grid: `0.2, 0.4, 0.6, 0.8, 1.0`.
- Matched target retention levels: `0.25` and `0.50`.
- Reference target magnitude: Voting at strength `1.0` for the same subject, view, part, and task.
- A record is reference-supported only when its target mask has positive support and Voting@1.0 produces positive target delta.
- Test masks are used only to measure target/outer/boundary deltas and interpolate evaluation curves. They do not select Gaussian weights, thresholds, banks, or model parameters.

## Approaches Considered

1. Scale the existing full-strength recolor/texture metrics analytically and sweep removal only. This is faster but depends on a linearity argument and stored uint8 measurements.
2. Render all five strengths for all three tasks. This is selected because every curve point is directly measured by the same renderer and is easiest to defend in review.
3. Choose one per-record strength directly from test masks and rerender. This is rejected because it hides curve coverage and resembles test-time tuning.

## Renderer Extension

`tools/render_semantic_real_editing_paper_suite.py` will accept `--edit-strengths` in addition to the existing single `--edit-strength`. The default remains one strength, preserving the completed fixed-strength experiment. A `--metrics-only` option suppresses frame export for the sweep while retaining the exact same render and metric path.

The output contains one row for every view, part, task, method, and requested strength. Multiple-strength frame names include the strength token when frame export is enabled.

## Curve And Matching Rules

For each reference-supported record and method:

1. prepend the exact origin `(retention=0, burden=0)`;
2. compute retention as method target delta divided by Voting@1.0 target delta;
3. compute outer and boundary burden by dividing the corresponding delta sum by Voting@1.0 target delta;
4. sort curve points by retention and collapse duplicate retention values by keeping the lowest burden only when their target delta is numerically equal;
5. linearly interpolate burden at retention `0.25` and `0.50` only when the method reaches that retention.

For each A5-vs-Voting or A5-vs-Raw comparison, metrics use the common record set where both methods reach the target retention. Coverage is reported for each method and for the common set. No missing record is converted to zero.

## Statistics

Outputs include:

- per-record strength curves;
- per-record matched-retention values;
- per-subject equal-weight means;
- five-subject mean/std;
- paired A5-baseline deltas and deterministic subject-level bootstrap 95% intervals;
- reference support, method coverage, and common comparison coverage by task and retention;
- per-part coverage and failure tables.

Primary metrics are matched outer burden and matched boundary burden. Target retention is fixed by construction and coverage is a co-primary constraint.

## Execution

`tools/run_five_subject_real_editing_matched_strength.sh` runs five subjects sequentially on one GPU, reuses completed subject sweeps, records Beijing timestamps, and invokes the matched-strength summarizer. The expected render count is `5 subjects x 9 views x 6 parts x 3 tasks x 3 methods x 5 strengths = 12150`.

## Verification

- Unit tests verify strength parsing, backward-compatible single-strength behavior, metrics-only output, interpolation, zero-reference exclusion, common coverage, subject-equal aggregation, and queue routing.
- A CoreView 377 one-record smoke must produce `270` metric rows.
- The final aggregate must contain all five subjects, no infinite/NaN matched metrics, and explicit coverage columns.
