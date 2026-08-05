# A7c R1.3-G Exact Global Aggregate Feasibility Oracle Design

## Status And Scope

This document preregisters the CoreView377 R1.3-G diagnostic before any replay
gate is regenerated or any new formal aggregate result is opened. R1.3-G answers
one question left unresolved by R1.3-P: does the frozen lower-carrier action
space contain one set of 24 held-record gate sequences that passes the exact
aggregate promotion protocol?

R1.3-G is a deterministic feasibility solve, not predictor training. It does
not change R1.2-B, the runtime-safe R1.3-P projector, the carrier set, or any
promotion threshold. Its renderer-oracle gates are diagnostic artifacts and are
not eligible for deployment or teacher supervision.

## Frozen R1.3-P Result

R1.3-P repaired the gate-space temporal failure but did not pass renderer-space
promotion:

```text
maximum adjacent gate change    0.015000       required <= 0.02       PASS
outer mean gain                +0.378596%      required >= 0.5%       FAIL
outer positive records         15/24           required >= 90%        FAIL
outer q10                      -0.305641%      required >= 0          FAIL
boundary mean gain             +0.264706%      required >= 0.5%       FAIL
boundary positive records      21/24           required >= 90%        FAIL
minimum target response         0.999532       required >= 0.99       PASS
maximum soft-IoU drop          -0.000000363    required <= 0.005      PASS
```

Its first feasibility oracle returned `UNRESOLVED`. Independent optimistic
capacity passed the aggregate gates, but the sufficient witness required every
record to exceed the R1.1-F1 boundary mean (`2.348187%`). Three records could not
meet that unnecessarily strong per-record condition.

The existing R1.3-P boundary-conditioned feasible endpoint metrics have already
been observed. A read-only aggregation of those saved metrics passes the formal
summary:

```text
outer mean gain                 1.283506%
boundary mean gain              3.448923%
outer/boundary positive         24/24
outer q10                       0.500000%
boundary q10                    2.343608%
minimum target response         0.994824
maximum soft-IoU drop           0.000994
maximum adjacent gate change    0.015000
```

This is not yet a formal witness because the gate arrays were not saved in the
R1.3-P records artifact. R1.3-G must regenerate the gates, certify every LP, and
audit the actual arrays.

## Alternatives Considered

### Constructive endpoint replay, selected

Each record already has a certified feasible boundary capacity lower endpoint
under `outer gain >= 0.005` and all hard guards. Replaying a slightly interior
point produces actual gates. Passing the unchanged aggregate auditor is a direct
existence certificate; feasibility does not require finding a global optimum.

### Discrete Pareto-grid MILP, rejected

A MILP could choose one precomputed outer/boundary pair per record, but it would
only be exact over a newly introduced gain grid. The grid is unnecessary when a
constructive witness is already latent in the frozen endpoints.

### Continuous interval branch-and-bound, deferred

Branch-and-bound over two-dimensional per-record gain boxes could eventually
certify either feasibility or infeasibility to tolerance. It has substantially
more implementation and verification risk. It is permitted only as a separately
preregistered fallback if endpoint replay is unresolved.

## Frozen Sources

R1.3-G freezes these upstream artifacts before implementation:

```text
R1.3-P contract
  configs/semantic/a7c_r1_3p_temporal_joint_projection_377_v1.json
  SHA256 a62d99f65d1358d2b985db3c5dec5221396a7fb1c8cbf287abc8943788f4c61c

R1.3-P oracle records
  exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/records.json
  SHA256 7a4d3998408a67cb4754d2bcb799e9e4e2ed8518b23fa3254055c4b88f9d3ce8

R1.3-P oracle summary
  exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/oracle/summary.json
  SHA256 82c1d019002a7ee980d0b13c583bf023ae0df5aac8f5d383e513d2cbff12c5c5

runtime probe
  SHA256 643c541af20f732a9de2c4ac6c20ea804ac27be8ad6dad13b1ead5efb6f8b411
renderer evidence
  SHA256 8b655f48fad664ba308f51d3291971382d7f9037fc7d69e38fca37907efd77f4
A5 bank
  SHA256 49ba86b05c4f87eaa8b98ef47822c7083a31fdf050a35bd8cf3a88843f8a45d3
teacher manifest
  SHA256 698f61e195a78849c72be14b8cf9073f281b94124d804013988e7bf605304aa8
```

The source records must contain exactly 24 unique `fold x camera_index` rows,
six folds, cameras 0-3, and no duplicate. Every `boundary_conditioned` search
must have status `bracketed`, a non-null feasible lower endpoint, and interval
width at most `1e-5`.

## Replay Request

For record `r`, define the frozen boundary request:

```text
boundary_request[r]
  = R1.3-P boundary_conditioned.feasible_lower[r] - 0.00002
```

The replay margin `0.00002` is fixed before gate regeneration. It is twice the
R1.3-P bisection tolerance and moves the request into the certified feasible
interior without changing the carrier set or formal audit threshold.

Every record is solved independently with:

```text
outer normalized-flicker gain >= 0.005
boundary normalized-flicker gain >= boundary_request[r]
runtime proxy target response >= 0.995 per frame
true renderer target response >= 0.99 per frame
true selection soft-IoU drop <= 0.005 per frame
adjacent gate change <= 0.015
topology_floor[i] <= gate[t,i] <= 1.0
```

The implementation reuses the frozen `solve_fixed_gain_oracle` equations and
HiGHS tolerances. It does not change the LP objective, introduce gain weights,
or use a second solver.

## Gate Artifact And Isolation

Each fold output has the full frozen sample/carrier shape. Only its four fit-
camera held-block segments contain finite replay gates. Every train block and
every non-fit camera position must remain NaN. The prediction artifact includes:

```text
replay_gates
replay_mask
camera_index, frame_index, block_ids, carrier_ids
requested outer/boundary gains
source endpoint and source fingerprints
deployment_eligible = false
teacher_eligible = false
paper_test_eligible = false
```

Oracle gates must not be copied into R1.3-P runtime predictions, a part-label
bank, or an A7b training set. Camera and frame entries only identify the frozen
record and sample order.

## Per-Record Certificate

Every replay record emits:

```text
fold, camera, block, first/last frame, sample and carrier counts
source feasible lower/upper endpoints and interval width
requested outer and boundary gains
solver status/message/iterations and SciPy/HiGHS identity
direct outer and boundary gains
minimum target response and maximum soft-IoU drop
maximum adjacent gate change and location
minimum topology/bound/proxy-target slack
maximum recomputed primal violation
source, manifest, sample-order, and carrier-order fingerprints
```

A non-optimal solver result, non-finite gate, mismatched fingerprint, or maximum
primal violation above `1e-7` fails closed before aggregate audit.

## Exact Aggregate Audit

The audit loads the 24 saved gate sequences and frozen renderer streams, directly
recomputes every record metric, and calls the unchanged R1.2/R1.3
`summarize_records`. No metric may be copied from the source endpoint JSON into
the formal audit.

All existing promotion conditions remain mandatory:

```text
mean outer gain >= 0.5%
mean boundary gain >= 0.5%
outer and boundary positive fraction >= 0.90
outer and boundary q10 >= 0
worst outer and boundary regression <= 0.5%
minimum target response >= 0.99
maximum soft-IoU drop <= 0.005
maximum adjacent gate change <= 0.02
mean outer and boundary gains improve over frozen R1.1-F1
A5 topology, coverage, frozen parts, and weight upper bounds unchanged
```

The internal `0.015` jump remains a construction margin. Aggregate compensation
is allowed: an individual boundary gain may be below the R1.1-F1 mean as long as
the exact 24-record summary passes all formal distribution and aggregate gates.

## Verdict And Terminal State

R1.3-G has these execution outcomes:

- `CERTIFIED_FEASIBLE`: all 24 LPs are optimal with residual at most `1e-7`,
  all artifacts pass isolation checks, and the exact aggregate audit has
  `passed=true`. The root terminal marker is `.completed`.
- `UNRESOLVED`: replay executes correctly but any record or aggregate promotion
  guard fails. The root terminal marker is `.rejected`.
- `ORACLE_ERROR`: a source, manifest, solver, serialization, or numerical check
  fails. The root terminal marker is `.failed`.

Constructive replay cannot return `CERTIFIED_INFEASIBLE`. An unresolved result
does not prove action-space insufficiency.

## Artifacts

The frozen output root is:

```text
exp/acceptdata/a7c_r1_3g_exact_aggregate_oracle_377_v1/
```

Required outputs are:

```text
witness/fold_0..5/predictions.npz
witness/fold_0..5/certificates.json
witness/summary.json
audit/held_block_summary.json
records.json
summary.json
runner.log, runner.pid, started_utc.txt, ended_utc.txt
exactly one root terminal marker: .completed, .rejected, or .failed
```

Every JSON summary sets `paper_test_eligible=false`.

## Verification Requirements

Focused tests must cover:

- exact extraction of 24 unique bracketed endpoints;
- rejection of missing, duplicate, non-bracketed, or wider-than-`1e-5` rows;
- exact application of the `0.00002` replay margin;
- a synthetic aggregate where three records are below the R1.1 boundary mean
  but compensation produces a valid formal pass;
- deterministic fixed-gain replay and direct metric recomputation;
- target, soft-IoU, jump, topology, bounds, and proxy-target certificate checks;
- finite held gates and NaN isolation everywhere else;
- `deployment_eligible=false` and `teacher_eligible=false` propagation;
- source, manifest, sample-order, and carrier-order mismatch rejection;
- mutually exclusive verdicts and terminal markers;
- restart behavior and no c17-c23 path;
- unchanged R1.3-P and fixed-gain oracle regression tests.

## Stop Rules And A7b Boundary

- R1.3-G does not train a predictor, avatar, compositor, or part-label bank.
- It does not change endpoint values, replay margin, promotion thresholds,
  solver tolerances, carrier set, or renderer streams after preregistration.
- It does not open c17-c20 or c21-c23.
- A rejected replay cannot be repaired in place; continuous branch-and-bound
  requires a new design.
- No R1.3-G result authorizes Task 12, LOSO, or a paper claim.
- Only `CERTIFIED_FEASIBLE` establishes that R1.3-P is blocked by runtime
  prediction/observability rather than aggregate action-space capacity.
- After `CERTIFIED_FEASIBLE`, pose-conditioned A7b requires its own design and
  contract. Per-fold oracle supervision may be generated only from that fold's
  fit blocks. Held blocks remain audit-only and cannot supply training labels,
  normalization, checkpoint selection, or hyperparameters.
