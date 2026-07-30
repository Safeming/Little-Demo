# A7 v5.4 Camera-Time Stability Selection Design

## Goal

Improve lower-part temporal stability without using CoreView377 c21-c23 for
candidate selection, while reducing carrier-selection variance and preventing
full-sequence averages from hiding local temporal regressions.

## Diagnosis

The v5.3 optimizer scores each construction camera over its complete 114-frame
sequence. Its final candidate passes eight-camera construction, but only 37 of
48 fixed camera-time outer blocks improve. The final 25-carrier lower solution
also contains seven carriers selected in at most half of the eight camera LOCO
folds, including one carrier selected in no fold. This is selection variance,
not missing renderer evidence.

c21-c23 have been inspected since v5.1. They are permanently development-only
and must not influence v5.4 ranking, thresholds, candidate acceptance, or
runner success.

## Construction Units

Reuse the frozen v5.3 evidence over c01/c05/c09/c13/c17/c18/c19/c20 and frames
0:570:5. Divide each camera sequence into six contiguous blocks of 19 samples,
producing 48 fixed camera-time units. Block boundaries are derived from sample
order and pinned by the contract.

## Nested Stability Selection

Run 48 construction folds. Fold `(camera, block)` excludes the complete camera
and the selected temporal block from every remaining camera. Support counting
must preserve temporal segment boundaries and must not create a false adjacent
pair across the removed block.

Each fold constructs a lower candidate with the existing v4 seed and 5%/10%
actions. A carrier enters the final consensus only when it is selected in at
least 36 of 48 folds. Its final level is the most frequent fold-selected level,
with the smaller reduction used for deterministic ties. No full-data-only
carrier may be appended after consensus.

Hair remains exact A5. Face, upper, shoes, and skin remain frozen. Selection
topology, coverage policy, maximum weight, action grid, and maximum changed
fraction remain unchanged.

## Lexicographic Objective

Candidate moves are ranked in this order:

1. camera-level target, visibility, and evidence soft-IoU constraint violation;
2. camera-time block gate violation;
3. worst-decile camera-time normalized outer/boundary flicker;
4. worst full-camera normalized outer/boundary flicker;
5. mean normalized flicker and raw adjacent change.

This avoids adding another freely tuned scalar objective. A candidate with a
better average cannot outrank one with fewer constraint or block violations.

## Frozen Development Gates

```text
construction aggregate outer gain >= 0.75%
construction aggregate boundary gain >= 0.75%
lower-only mean outer gain >= 1.5%
lower-only mean boundary gain >= 1.5%
positive outer block fraction >= 44/48
positive boundary block fraction >= 44/48
10th-percentile outer block gain >= 0
10th-percentile boundary block gain >= 0
worst block regression <= 0.5%
training target response >= 0.995
training visibility response <= 0.998
evidence soft-IoU drop <= 0.005
stability selection frequency >= 36/48
hair exact A5
paper_test_eligible = false
```

If no consensus candidate passes, v5.4 is rejected. The implementation must
not expand the action grid, add hair moves, relax the block gates, or inspect
c21-c23 to rescue it.

## Development Runner

The runner performs only evidence-based 48-fold capacity construction and the
existing spatial guard audit. It does not render c21-c23. Success freezes the
candidate for a later one-shot multi-subject LOSO experiment; it is not a paper
promotion result.

## Integrity

- CoreView377 c01-c23 is development-only after v5.4.
- The runner writes `paper_test_eligible=false` in its manifest and summary.
- Candidate identity includes the 48-fold selection record and block policy.
- Multi-subject and untouched-subject outputs are not opened during v5.4.
- Failure ends the route without automatic v5.4b or A7b expansion.
