# A7c R1.3-P Temporal Joint Hard Projection And Feasibility Oracle Design

## Status And Scope

This document preregisters the CoreView377 R1.3-P development experiment before
implementation or any new held-block result is opened. R1.3-P tests two separate
questions raised by the rejected R1.2-B compositor:

1. Can a label-free, full-sequence projection remove projection-induced gate
   jumps while preserving the useful R1.2-B raw prediction?
2. Does the frozen lower-carrier action space contain any gate sequence that can
   jointly satisfy the temporal, renderer, topology, and promotion constraints?

R1.3-P does not train a new predictor and does not change the R1.2-B network. It
adds one runtime-safe temporal projector and one renderer-evidence feasibility
oracle. The oracle is diagnostic only and cannot become a deployment input. A
pass cannot authorize Task 12 or a paper claim by itself.

## Frozen Failure And Root-Cause Hypothesis

The frozen R1.2-B held-block result is:

```text
outer mean gain              +0.519637%   required >= +0.500000%
outer positive blocks        17/24        required >= 90%
outer q10                    -0.278741%   required >= 0
outer worst block            -0.659769%   required >= -0.500000%
boundary mean gain           +0.286637%   required >= +0.500000%
boundary positive blocks     23/24        required >= 90%
boundary q10                 +0.067905%   required >= 0
boundary worst block         -0.172289%   required >= -0.500000%
minimum target response       0.999484    required >= 0.99
maximum soft-IoU drop        -0.000000363 required <= 0.005
maximum adjacent gate jump    0.069361    required <= 0.02
```

The current per-frame target-budget projection is beneficial in aggregate but
is not temporally coupled. Its independently computed scale changes sharply
between adjacent frames: adjacent scale-difference quantiles are approximately
`0.0204/0.2877/0.4073/0.6105` at p50/p90/p95/p99, with maximum `0.8123`.
Forty-seven of 200 audited jump violations had an R1.2-B raw-gate jump already
below `0.02`; the independent projection created the violation. Conversely,
removing projection entirely worsens outer consistency and target preservation,
so projection must be repaired rather than deleted.

The hypothesis is therefore split into a projection gap and an action-space
capacity question. These must be measured separately before adding a recurrent
or pose-conditioned predictor.

## Alternatives Considered

### Sparse linear programming with HiGHS, selected

`scipy.optimize.linprog(method="highs")` is already available in the frozen
environment. The constraints below are linear, sparse, and admit the identity
gate as a deployment-projection feasibility witness. HiGHS provides an explicit
solver state and primal residual suitable for a formal certificate without a
new dependency.

### SLSQP or trust-constr quadratic projection, rejected

A quadratic objective would directly minimize L2 displacement from the raw
gates, but nonlinear convergence status is weaker as a feasibility certificate
and is less predictable on the full sequence. It is not selected for R1.3-P.

### Custom alternating or differentiable projection, deferred

Dykstra-style or differentiable projection could later support end-to-end
training, but it introduces convergence and stopping-rule choices before the
action space itself has been certified. It is outside this canary.

## Sequence Boundary And Offline Contract

The user approved full contiguous-sequence, offline, bidirectional projection.
Each solve is isolated to one registered `camera x temporal block` segment:

- no constraint crosses a camera boundary;
- no constraint crosses a temporal-block boundary or missing-frame gap;
- samples must be ordered by the frozen manifest and differ by the registered
  frame stride;
- each of the six fold models supplies raw gates only for its own held block;
- the final-fit model may be projected over all six fit-camera blocks only after
  the held-block configuration is frozen.

This is not a causal streaming method. Any later causal claim requires a new
preregistration and cannot reuse the R1.3-P result as evidence.

## Runtime-Safe Projection Inputs

The deployment projector may read only:

```text
R1.2-B raw gates
runtime alpha/transmittance target mass
frozen A5 soft-edit weight for each registered carrier
selection threshold and gate bounds from the frozen contract
manifest adjacency used only to form isolated temporal segments
```

It must not read objective or guard renderer contributions, teacher gates,
target/outer/boundary labels, masks, subject ID, camera ID, frame ID, Gaussian
ID, audit result, or any held-derived statistic. Camera and frame entries may be
used only for grouping and ordering, never as optimization coefficients.

## Temporal Joint Hard Projection

For frame `t`, carrier `i`, raw gate `r[t,i]`, runtime mass `m[t,i]`, and frozen
A5 weight `w[i]`, define:

```text
topology_floor[i] = max(0.9, selection_threshold / max(w[i], 1e-8))
```

The projected gate `g[t,i]` must satisfy all constraints simultaneously:

```text
topology_floor[i] <= g[t,i] <= 1.0
sum_i m[t,i] * g[t,i] >= 0.995 * sum_i m[t,i]       for every frame
abs(g[t,i] - g[t-1,i]) <= 0.015                     for every adjacency
```

The `0.015` internal jump limit reserves `0.005` for the formal `0.02` audit
limit. If `topology_floor[i] > 1 + 1e-12`, the input contract is invalid. Values
within numerical tolerance are clipped to one. Zero-mass frames retain the
bounds and temporal constraints and make the proxy target inequality vacuous.

The projection objective is lexicographic and frozen:

1. Introduce `rho` and minimize `rho` subject to
   `abs(g[t,i] - r[t,i]) <= rho`.
2. Constrain `rho <= rho_star + 1e-9`, introduce absolute-deviation auxiliaries,
   and minimize `sum(abs(g[t,i] - r[t,i]))`.

Both stages use sparse `float64` matrices and HiGHS. There is no renderer term,
temporal loss weight, post-solve smoothing, carrier-specific penalty, or result-
dependent retry. Since the all-one gate satisfies valid bounds, proxy target,
and temporal constraints, runtime projection infeasibility indicates an input or
implementation error rather than a model-capacity failure.

## Projection Certificate And Fail-Closed Behavior

Every segment emits a machine-readable certificate containing:

```text
solver name/version, SciPy version, status, message, and iteration counts
stage-one and stage-two objective values
rho_star and observed maximum/mean/total raw-gate displacement
minimum lower-bound, upper-bound, topology, and proxy-target slack
maximum adjacent gate change and its frame/carrier location
maximum recomputed primal violation
input, contract, source, sample-order, and carrier-order fingerprints
```

The implementation recomputes every constraint from the returned gates instead
of trusting solver metadata alone. A non-optimal stage, a non-finite value, a
fingerprint mismatch, or maximum primal violation greater than `1e-7` fails
closed. Failed output is not audited for promotion and cannot be relabeled by
manually editing a marker.

## Renderer-Evidence Feasibility Oracle

The oracle uses the frozen objective and guard contribution streams from the
same 24 fit-camera held-block records. Unlike the runtime projector, it may read
true renderer evidence because its output is a capacity diagnosis, not a gate
source for deployment. Oracle gates must never be copied into runtime prediction
artifacts or used as teacher targets in R1.3-P.

For every candidate sequence, the oracle retains the runtime projection bounds,
proxy target constraint, and `0.015` temporal constraint, and adds:

```text
true target response >= 0.99 for every frame
true selection soft-IoU drop <= 0.005 for every frame
requested outer normalized-flicker gain for the held block
requested boundary normalized-flicker gain for the held block
```

Renderer target, outer, and boundary values are affine functions of the gates.
For base soft IoU `b` and limit `k = b - 0.005`, the guard is represented by the
linear inequality `(1-k) * target_candidate - k * outer_candidate >= 0` after
verifying positive denominators. If `k <= 0`, the soft-IoU lower bound is
vacuous; it is recorded but no inequality is added. Absolute adjacent signal
differences use nonnegative auxiliary variables. A requested normalized-flicker
gain `gamma` is imposed linearly as:

```text
sum_t difference_aux[t]
  <= (1-gamma) * base_normalized_flicker
     * (frame_count-1) / frame_count
     * sum_t candidate_signal[t]
```

All coefficients and all post-solve metrics are computed in `float64`. The
existing frozen auditor remains the authority for reported promotion metrics.

## Oracle Searches And Three-Way Verdict

The oracle performs deterministic bisection to `1e-5` gain precision with no
post-result threshold change:

1. **Balanced capacity:** maximum per-record common gain `gamma` required of
   both outer and boundary under all hard guards.
2. **Boundary-conditioned capacity:** maximum per-record boundary gain while
   outer gain is fixed at least `0.005`.
3. **Independent optimistic bounds:** separately maximize outer and boundary for
   each record under all hard guards. These optima need not share gates and are
   therefore upper bounds, not deployable candidates.
4. **Sufficient joint witness:** solve every record with outer gain at least
   `0.005` and boundary gain at least `0.023491874880317264`, the frozen R1.1-F1
   boundary floor plus a preregistered `1e-5` strict-comparison margin, then run
   the exact 24-record auditor on the returned gates.

Every bisection retains a certified feasible lower endpoint and an infeasible
upper endpoint. Capacity reports include both. Only the infeasible upper endpoint
is used as the optimistic value in an impossibility test; using the last feasible
gate would underestimate capacity and is forbidden. No gate is claimed for an
upper endpoint.

The final oracle verdict has exactly three values:

- `CERTIFIED_FEASIBLE`: all sufficient-witness LPs are optimal, their residuals
  are at most `1e-7`, and the exact frozen 24-record audit passes.
- `CERTIFIED_INFEASIBLE`: the promotion summary formed from independent
  per-record infeasible upper endpoints still fails at least one outer/boundary
  mean, positive-fraction, q10, worst-block, or R1.1 improvement requirement.
  Coordinate-wise monotonicity makes this a valid impossibility certificate.
- `UNRESOLVED`: the sufficient witness fails but the optimistic upper bound can
  still pass. This means the action space has not been disproved, but the current
  LP certificate is insufficient; it does not authorize model expansion.

Solver failure is reported separately as execution status `ORACLE_ERROR` and
cannot be converted to `CERTIFIED_INFEASIBLE`. Independent feasible-endpoint
gates are never combined, and no infeasible upper endpoint is presented as a
gate solution.

## Split, Audit, And Promotion

R1.3-P uses only the six leave-one-contiguous-block-out predictions for
`c01/c05/c09/c13`, yielding exactly 24 held records. It does not open c17-c20 or
c21-c23. All R1.2-B promotion conditions remain mandatory:

```text
mean outer gain >= 0.5%
mean boundary gain >= 0.5%
outer and boundary positive-block fraction >= 0.90
outer and boundary q10 block gain >= 0
worst outer and boundary block regression <= 0.5%
minimum target response >= 0.99
maximum soft-IoU drop <= 0.005
maximum adjacent projected gate change <= 0.02
gate range [0.9, 1.0]
A5 topology, coverage, frozen parts, and weight upper bounds unchanged
mean outer and boundary gains both improve over frozen R1.1-F1
```

The internal `0.015` jump constraint and `0.995` proxy target response are
additional construction margins; they do not replace the formal audit values.
There is no validation-set selection, threshold sweep, carrier sweep, or seed
sweep.

## Artifacts

The frozen output root is:

```text
exp/acceptdata/a7c_r1_3p_temporal_joint_projection_377_v1/
```

Required artifacts are:

```text
projection/fold_0..5/predictions.npz
projection/fold_0..5/segment_certificates.json
projection/summary.json
projection/audit/held_block_summary.json
oracle/records.json
oracle/summary.json
runner.log, started_utc.txt, ended_utc.txt, runner.pid
exactly one terminal marker: .completed, .rejected, or .failed
```

Prediction artifacts store both raw and projected gates plus manifest and source
fingerprints. Oracle artifacts store requested thresholds, capacity intervals,
candidate metrics, upper-bound metrics, solver certificates, and the three-way
verdict. Every JSON summary sets `paper_test_eligible=false`.

## Verification Requirements

Focused tests must cover:

- topology-floor, bound, per-frame proxy target, and bidirectional jump rows;
- a synthetic case where independent per-frame projection creates a jump and
  the joint projection removes it;
- identity feasibility and lexicographic minimum-displacement behavior;
- camera/block/gap isolation and frozen sample/carrier order;
- no renderer evidence access in the deployment projection path;
- exact soft-IoU linearization against direct ratio evaluation;
- normalized-flicker LP constraints against direct evaluation;
- feasible, infeasible, unresolved, and solver-error oracle verdicts;
- optimistic upper-bound monotonicity and non-combination of its gates;
- deterministic CPU output, certificate residual recomputation, source hash
  rejection, restart behavior, and terminal-marker exclusivity;
- exact 24-record aggregation through the unchanged frozen auditor.

## Stop Rules And Next Decision

- Any source, contract, manifest, sample-order, carrier-order, or split mismatch
  stops before solving.
- Any runtime projection coefficient derived from renderer labels invalidates the
  experiment.
- A runtime LP that is not optimal or exceeds `1e-7` residual ends `.failed`.
- Results cannot be repaired in place by changing `0.015`, `0.995`, bisection
  precision, objective order, promotion gates, carrier set, or evidence stream.
- R1.3-P does not retrain R1.2-B, enlarge the 5%/10% action grid, restore dense
  scaling, add hair carriers, or open c17-c23.
- If projection passes promotion, the next step is a separately preregistered
  untouched-camera audit; it is not opened automatically.
- If projection fails and the oracle is `CERTIFIED_FEASIBLE`, the remaining gap
  is prediction/observability, supporting a separately designed pose-conditioned
  A7b rather than a larger static bank.
- If the oracle is `CERTIFIED_INFEASIBLE`, the frozen carrier/action space or
  promotion constraints are the blocker; network expansion cannot fix it.
- If the oracle is `UNRESOLVED`, an exact global feasibility formulation must be
  preregistered before predictor training.
- No outcome in this experiment authorizes Task 12, LOSO, or a paper claim.
