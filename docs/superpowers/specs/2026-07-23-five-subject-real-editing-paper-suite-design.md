# Five-Subject Real Editing Paper Suite Design

## Goal

Build a frozen, reproducible five-subject paper experiment for real semantic edits. The suite compares Raw Hard, multi-view Voting, and frozen A5 under the same renderer for recolor, opacity removal, and canonical texture replacement.

## Protocol

- Subjects: CoreView 377, 386, 387, 393, and 394.
- Test records: c21-c23 at frames 180, 420, and 540.
- Parts: hair, face, upper, lower, shoes, and skin.
- Methods:
  - `raw_hard`: hard labels from the raw trained semantic bank.
  - `voting`: hard labels from the projected multi-view voting bank.
  - `a5`: `soft_edit_weights` from the frozen A5 footprint-evidence bank, thresholded by the subject's five-subject LOSO configuration.
- Edit tasks:
  - `recolor`: fixed part-specific target color with a common edit alpha.
  - `removal`: multiply opacity by `1 - edit_weight`.
  - `texture`: replace selected Gaussian colors with a deterministic two-color stripe pattern derived from canonical Gaussian coordinates.
- Test parser masks are evaluation-only. They must not change Gaussian weights, rendered colors, opacity, or texture coordinates.
- All methods use the same deformed Gaussians, camera, rasterizer, background, and edit parameters.

## Approaches Considered

1. Run the existing preview tool separately for every bank and task. This minimizes new code but repeatedly loads checkpoints, cannot render opacity removal, and risks inconsistent method metadata.
2. Add a formal multi-method suite that loads a subject once and renders all methods and tasks through one rasterizer interface. This is the selected approach because it is faster, auditable, and enforces fair rendering.
3. Modify checkpoints or save edited Gaussian models. This creates unnecessary persistent assets and makes accidental method-specific state more likely, so it is rejected.

## Architecture

`utils/semantic_real_editing.py` contains pure, CPU-testable functions for resolving method weights, generating deterministic texture colors, applying recolor/removal/texture edits, and computing screen-space delta metrics.

`gaussian_renderer.rasterize_gaussians` gains an optional opacity override. The default remains `pc.get_opacity`, preserving all existing callers. Color overrides continue to use `colors_precomp`.

`tools/render_semantic_real_editing_paper_suite.py` loads the checkpoint and three banks once per subject, renders every frozen test record, and writes frame images plus one row per subject/view/part/task/method to CSV and JSON.

`tools/summarize_semantic_real_editing_paper_suite.py` aggregates subject-level means, paired A5-minus-baseline deltas, and bootstrap confidence intervals. It also creates compact paper contact sheets from a fixed, declared subset without selecting by test performance.

`tools/run_five_subject_real_editing_paper_suite.sh` validates all inputs, runs subjects sequentially on one GPU, records timestamps/status, reuses completed subject summaries on restart, and invokes the final summarizer.

## Metrics

For each rendered edit, absolute RGB delta from the unedited render is accumulated over evaluation masks:

- target delta mean and sum;
- outer delta mean and sum;
- outer-to-target delta ratio;
- boundary outer delta mean in the frozen protocol boundary band;
- target edit retention relative to the corresponding Voting edit for the same subject/view/part/task.

Removal additionally records rendered opacity change where available. The primary comparison is outer delta at matched target edit magnitude; raw ratios remain auxiliary because different edit tasks have different natural magnitudes.

## Failure Handling And Provenance

- Point counts must match across checkpoint and all three banks.
- A5 must expose two-dimensional `soft_edit_weights`.
- LOSO held-out subject and method-freeze fingerprint must match the subject and frozen A5 configuration.
- Missing records, masks, banks, or checkpoint fail the subject rather than silently skipping it.
- Every summary records bank paths/fingerprints, threshold, task parameters, record names, and `uses_test_parser_for_edit_selection=false`.

## Verification

Unit tests cover opacity overrides, method weight routing, deterministic texture generation, edit transforms, metrics, CLI dry-run routing, and restart behavior. A one-record smoke run must produce all nine method-task combinations before the five-subject queue starts.

## Scope Boundary

This suite does not add video/temporal metrics or external methods. Those are separate paper experiments after the real-edit suite completes.
