# CoreView377 Optimizer-Preserving 96k Continuation Design

## Goal

Determine whether the remaining gap between the verified from-zero neutral64k result (`FG_LPIPS=0.1304382086`) and v395 (`FG_LPIPS~=0.1273284`) is primarily unfinished base-model annealing. Continue the exact neutral trajectory from schedule iteration 64k to 96k without introducing a new loss, module, topology operation, or historical subject state.

This run is a diagnostic continuation from a from-zero checkpoint. A successful tail will later be embedded into a single input-to-final from-zero command.

## Evidence

- Neutral64k is the current from-zero best and reaches its best metric at the final 64k checkpoint.
- Its FG LPIPS improves from `0.1331702` at 60k to `0.1304382` at 64k, so the trajectory has not demonstrated a plateau.
- Historical v395 was produced near 140k schedule iterations.
- Anchor tether, surface-carrier formation, residual reallocation, and structured/high-frequency shadow branches all failed to beat neutral64k.
- The checkpoint already stores Gaussian Adam state, converter Adam state, and converter scheduler state, but Gaussian restore currently discards its saved optimizer state.

## Architecture

### Exact continuation state

Load neutral64k `ckpt64000.pth` and restore:

- all Gaussian parameters and binding state;
- Gaussian Adam moments and parameter-group learning rates;
- converter parameters and Adam moments;
- converter exponential scheduler state;
- the checkpoint iteration as the schedule offset.

Gaussian optimizer restoration is strict for this experiment. Missing, incompatible, or failed state restoration terminates the run instead of silently starting a fresh optimizer.

### Unchanged model path

Retain the exact neutral64k model and loss configuration:

- 80k `focus_head_hands` initialization topology already present in the checkpoint;
- no densification, pruning, opacity reset, compression, reallocation, anchor tether, shadow branch, or new appearance module;
- the same base shallow texture MLP, deformation, pose correction, masks, and image losses;
- XYZ remains at its configured post-24k floor;
- the restored converter scheduler continues its original 64k exponential gamma, reducing converter rates from `0.1x` at 64k toward about `0.0316x` at 96k.

### Evaluation and selection

Run 32k local steps, corresponding to schedule iterations 64k-96k. Evaluate every 2k schedule iterations from 66k through 96k, save milestones every 8k, and select the global best candidate by `best_eval` FG LPIPS. A worse final checkpoint cannot replace the 64k baseline because the launcher records both the continuation best and the immutable baseline metric.

## Success Criteria

- checkpoint load logs confirm Gaussian optimizer, converter optimizer, and converter scheduler restoration;
- the first validation reproduces the 64k base metric within normal deterministic evaluation tolerance;
- no topology or point-count change occurs;
- continuation best beats `0.1304382086`;
- strong success is `FG_LPIPS<=0.1285` by 96k;
- v395-level success is `FG_LPIPS<=0.1273284` with FG PSNR near or above `22.1021`.

## Validation

- a unit test proves Gaussian Adam moments survive capture, restore, and a subsequent `training_setup` call;
- a strict-mode test proves missing optimizer state raises instead of falling back;
- pipeline tests prove the exact 64k from-zero checkpoint, 32k local horizon, schedule offset, optimizer/scheduler restore flags, and disabled topology operations;
- a real-checkpoint GPU smoke verifies restoration logs, schedule iteration 64001+, checkpoint creation, and one validation cycle before the formal run starts.
