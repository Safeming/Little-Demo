# Five-Subject Semantic Temporal Stability Design

## Goal

Evaluate whether the frozen canonical semantic Gaussian selection remains stable over continuous avatar motion, without retraining or changing the frozen A5 method.

## Protocol

- Subjects: CoreView 377, 386, 387, 393, and 394.
- Camera: held-out camera 21 for every subject.
- Frames: 0 through 569 at step 1.
- Methods: projected multi-view Voting and frozen A5.
- Parts: hair, face, upper, lower, shoes, and skin.
- Frozen inputs: the existing 42k checkpoints, five-subject LOSO configuration, Voting bank, A5 footprint-evidence bank, and `frozen_a5_main_method_v1.json`.
- Ground truth is used only for evaluation. Hulk masks from camera 21 never affect method selection, thresholds, calibration, or the A5 bank.

The continuous sequence contains poses excluded by the training stride for most frames, while camera 21 is outside the training-camera set. This tests held-out view and predominantly held-out pose behavior.

## Metrics

For each method, part, and frame, render the selected canonical Gaussian weights as a screen-space soft mask and report:

- screen-space soft IoU;
- screen-space hard IoU, precision, and recall at a predeclared 0.2 screen threshold;
- selected mass inside and outside the parser target;
- selection leakage ratio;
- target, outer, and boundary edit delta for a fixed recolor operation.

For every subject/method/part sequence, report the mean, sample standard deviation, and coefficient of variation. Temporal flicker is the normalized mean absolute difference between adjacent frames for target edit magnitude and leakage. Subject-equal aggregate statistics and paired subject-level bootstrap confidence intervals compare A5 with Voting.

## Videos

Generate one synchronized MP4 per subject for `upper`, `hair`, and `shoes`. Each frame contains the original render, Voting and A5 selection overlays, and Voting and A5 recolor results. Videos use camera 21, frames 0-569, 25 FPS, and stream directly to H.264 without storing the complete PNG sequence.

## Outputs

The output root is `exp/acceptdata/five_subject_semantic_temporal_stability_20260724/` with:

- per-subject `metrics.csv`, `summary.json`, and MP4 files;
- aggregate per-subject and five-subject tables;
- paired bootstrap statistics;
- queue status, process ID, timestamps, and logs.

## Failure Handling

The queue validates all frozen inputs before GPU work. A subject is reusable only when its summary has exactly 6,840 rows (`570 frames x 6 parts x 2 methods`), the expected protocol provenance, and all three videos. Missing compact parser masks, mismatched Gaussian counts, invalid LOSO provenance, non-finite metrics, or failed video encoding stop the queue.
