# A7 v5.2 Dual-Margin Development Design

## Goal

Produce a deterministic lower-carrier replacement candidate that protects both
visibility-normalized target stability and mean target response without changing
the sparse action space, hair policy, topology, coverage, or frozen parts.

## Root Cause

The v5.1 formal diagnostic failed only at `c22/lower`: raw adjacent target
change improved, but mean target response fell slightly more, so normalized
visibility flicker reached `1.000077`. The four evidence cameras did not expose
that normalization gap. The existing optimizer treated target and visibility as
hard gates and optimized only outer/boundary flicker after those gates passed.

## Frozen v5.2 Method

- Reuse the v5 dual renderer evidence and v4 lower seed.
- Keep hair exactly A5 unless the existing temporal-shortfall rule opens at
  most three compensation moves.
- Keep 5%/10% coordinate reductions, 20% lower changed fraction, fixed A5
  topology, frozen parts exact, and every weight no greater than A5.
- Require construction visibility response ratio `<= 0.9990`.
- Require construction target response ratio `>= 0.9975`.
- Keep held-out/diagnostic visibility response ratio `<= 1.0` and target
  response ratio `>= 0.99`.
- Keep evidence soft-IoU drop `<= 0.005` and formal part soft-IoU drop
  `<= 0.01`.

The carrier IDs are not hard-coded. A read-only counterfactual run produced the
expected equal-capacity replacement: restore lower carriers `424` and `2102`,
add `1786` and `2050`, retain 23 lower carriers, and keep hair unchanged.

## Evaluation Integrity

CoreView377 `c21-c23` have already been opened by v5.1. v5.2 therefore treats
them as retrospective diagnostics, not a fresh paper test. The development run
uses `c17-c20` as validation, then reports `c21-c23` retrospective behavior and
the existing protocol test spatial metrics. A successful development run only
authorizes freezing the algorithm for the five-subject LOSO stage; it does not
by itself satisfy paper promotion.

## Development Gates

- Capacity LOCO: 4/4 folds pass construction and held-out gates.
- Validation `c17-c20`: aggregate outer and boundary improve by at least 0.5%,
  every camera moves in the improving direction, visibility ratio is at most
  `0.9995`, and target response is at least `0.995`.
- Retrospective `c21-c23`: existing formal temporal gates pass, including
  visibility at most `1.0` and target response at least `0.99`.
- Spatial: burden, coverage, part soft-IoU, macro mIoU, micro IoU, topology,
  provenance, and bank identity all pass the existing v5.1 thresholds.

## Paper Promotion

After the v5.2 algorithm and all hyperparameters are frozen, five-subject LOSO
uses unopened held-out subject tests. Paper promotion requires every held-out
subject to pass the existing temporal/spatial gates and a subject-level paired
bootstrap confidence interval showing improvement rather than relying on the
already-opened CoreView377 diagnostics.
