# A7 v5.1 Visibility Margin Design

## Goal

Repair the single v5 LOCO failure without weakening the formal promotion gate or
changing the sparse action space.

## Frozen Method

- Reuse the v5 dual renderer evidence with SHA-256
  `3e04bd3c49dfa32c6dd74ca3313669536df85083e5c3dccd9b198acc5a10a141`.
- Keep hair exactly at A5 unless the existing temporal shortfall rule opens at
  most three compensation moves.
- Keep lower initialized from the v4 sparse bank.
- Keep coordinate actions at 5% and 10%, topology fixed, frozen parts exact,
  and all weights no greater than A5.
- Require training-camera visibility response ratio `<= 0.9995`.
- Keep LOCO held-out and formal audit visibility response ratio `<= 1.0`.
- Keep target response `>= 0.99` and evidence soft-IoU drop `<= 0.005`.

## Data Flow

The candidate calibrator loads the frozen v5 evidence, verifies its SHA-256 and
source contract fingerprint, and runs the same fold-local support and v4-seed
construction. The optimizer receives the stricter construction threshold,
while held-out evaluation receives the unchanged audit threshold.

If all four folds and the full-evidence candidate pass, the runner writes a
freeze manifest before opening c21-c23. Temporal audit uses all frames at stride
5; spatial audit uses the protocol test split.

## Success Criteria

- Four of four LOCO folds pass the unchanged held-out gates.
- Full-evidence outer and boundary flicker each improve by at least 0.5%.
- Hair/lower target, soft-IoU, visibility, topology, and frozen-part constraints pass.
- c21-c23 are not opened before the candidate and manifest are frozen.

