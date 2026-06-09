# v399 Diagnostic Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build v399 as a diagnostic-first extension of v398: sweep only over-support caps, then upgrade bad-frame selection from all-or-nothing vetoes into severe veto plus penalty ranking.

**Architecture:** Keep `train.py` and `utils/boundary_support_bank.py` support-bank semantics unchanged for the first diagnostic sweep. Add a v399 wrapper that controls over cap values through existing config keys, then extend the shared v396 controller selector to emit cap saturation, bad-frame image aggregation, failure reason counts, and hard-veto-plus-penalty ranking.

**Tech Stack:** Bash wrappers, inline Python selector in `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`, pytest script tests, existing raw contour gate TSV files, existing support bank diagnostics.

---

## File Structure

- Create: `tools/run_377_explicit_binding_v399_diagnostic_sweep.sh`
  - Thin wrapper around `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`.
  - Enables v398 support-bank defaults.
  - Keeps under caps at v398 values.
  - Accepts `V399_OVER_EFFECTIVE_RATIO` and `V399_OVER_NEW_ONLY_RATIO`.
  - Uses selector names and output names under `v399`.

- Modify: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`
  - Add selector-only support for cap diagnostic columns in JSON.
  - Add bad-frame hard veto plus penalty mode.
  - Add bad-frame image aggregation output.
  - Add failure reason counts for stable-window debugging.
  - Preserve v396/v397/v398 compatibility by defaulting new behavior off unless env flags enable it.

- Modify: `tests/test_v396_raw_gate_paths.py`
  - Add static tests for v399 wrapper defaults.
  - Add static tests for hard veto plus penalty fields.
  - Add static tests for cap saturation diagnostics.
  - Add static tests for failure reason counts and image aggregation output paths.

- Create: `tools/analyze_377_v399_diagnostic_sweep.py`
  - Optional but recommended standalone analyzer for comparing 2-3 sweep runs.
  - Reads `v399_selected_checkpoint.json`, `support_bank_summary.tsv`, and raw contour summaries.
  - Emits a compact TSV ranking sweep runs by hard-floor recovery, bad-frame severity, cap saturation, and selected artifact status.

- Create: `tests/test_v399_sweep_analyzer.py`
  - Unit tests for the optional analyzer using small synthetic JSON/TSV fixtures.

## Task 1: Add v399 Wrapper Tests First

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Create later: `tools/run_377_explicit_binding_v399_diagnostic_sweep.sh`

- [ ] **Step 1: Add failing wrapper test**

Append this test:

```python
def test_v399_wrapper_sweeps_only_over_caps_and_keeps_under_caps():
    wrapper = ROOT / "tools" / "run_377_explicit_binding_v399_diagnostic_sweep.sh"
    text = wrapper.read_text(encoding="utf-8")

    assert "V399_OVER_EFFECTIVE_RATIO" in text
    assert "V399_OVER_NEW_ONLY_RATIO" in text
    assert "boundary_support_bank_under_max_effective_ratio=0.30" in text
    assert "boundary_support_bank_under_max_new_only_ratio=0.24" in text
    assert "boundary_support_bank_over_max_effective_ratio=$V399_OVER_EFFECTIVE_RATIO" in text
    assert "boundary_support_bank_over_max_new_only_ratio=$V399_OVER_NEW_ONLY_RATIO" in text
    assert "BAD_FRAME_SELECTOR_MODE" in text
    assert "v399_diagnostic_sweep" in text
    assert "run_377_explicit_binding_v396_generalized_boundary_controller.sh" in text
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_v399_wrapper_sweeps_only_over_caps_and_keeps_under_caps -q
```

Expected:

```text
FileNotFoundError: ... run_377_explicit_binding_v399_diagnostic_sweep.sh
```

- [ ] **Step 3: Create the minimal wrapper**

Create `tools/run_377_explicit_binding_v399_diagnostic_sweep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_ID="${RUN_ID:-v399_diagnostic_sweep_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/formal/377_v399_diagnostic_sweep_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v399_diagnostic_sweep_${RUN_ID}}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$EXP_DIR/hydra_runtime}"

SELECTOR_NAME="${SELECTOR_NAME:-v399}"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$LOG_DIR/v399_raw_contour_checkpoint_summary.tsv}"
SELECTED_JSON="${SELECTED_JSON:-$LOG_DIR/v399_selected_checkpoint.json}"
SELECTOR_SCHEMA_NAME="${SELECTOR_SCHEMA_NAME:-v399_diagnostic_sweep}"

SUPPORT_BANK_TRAIN_ENABLE="${SUPPORT_BANK_TRAIN_ENABLE:-true}"
SUPPORT_BANK_DIAGNOSTIC_ENABLE="${SUPPORT_BANK_DIAGNOSTIC_ENABLE:-true}"
SUPPORT_BANK_SELECTOR_ENABLE="${SUPPORT_BANK_SELECTOR_ENABLE:-true}"
SUPPORT_BANK_SUMMARY="${SUPPORT_BANK_SUMMARY:-$LOG_DIR/support_bank_summary.tsv}"

BAD_FRAME_SELECTOR_ENABLE="${BAD_FRAME_SELECTOR_ENABLE:-true}"
BAD_FRAME_SELECTOR_MODE="${BAD_FRAME_SELECTOR_MODE:-penalty}"
BAD_FRAME_OUTER_HARD_VETO="${BAD_FRAME_OUTER_HARD_VETO:-8.0}"
BAD_FRAME_HARD_HARD_VETO="${BAD_FRAME_HARD_HARD_VETO:-0.0005}"
BAD_FRAME_HARD_PENALTY="${BAD_FRAME_HARD_PENALTY:-0.00005}"
BAD_FRAME_IMAGE_SUMMARY="${BAD_FRAME_IMAGE_SUMMARY:-$LOG_DIR/bad_frame_image_summary.tsv}"

STABLE_WINDOW_TARGET="${STABLE_WINDOW_TARGET:-3}"
V392_ADOPTED_LOST_MAX="${V392_ADOPTED_LOST_MAX:-0}"

V399_OVER_EFFECTIVE_RATIO="${V399_OVER_EFFECTIVE_RATIO:-0.50}"
V399_OVER_NEW_ONLY_RATIO="${V399_OVER_NEW_ONLY_RATIO:-0.46}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_under_max_effective_ratio=0.30"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_under_max_new_only_ratio=0.24"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_over_max_effective_ratio=$V399_OVER_EFFECTIVE_RATIO"
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS ++opt.boundary_support_bank_over_max_new_only_ratio=$V399_OVER_NEW_ONLY_RATIO"

export RUN_ID EXP_DIR LOG_DIR HYDRA_RUN_DIR
export SELECTOR_NAME SELECTOR_SUMMARY SELECTED_JSON SELECTOR_SCHEMA_NAME
export SUPPORT_BANK_TRAIN_ENABLE SUPPORT_BANK_DIAGNOSTIC_ENABLE SUPPORT_BANK_SELECTOR_ENABLE SUPPORT_BANK_SUMMARY
export BAD_FRAME_SELECTOR_ENABLE BAD_FRAME_SELECTOR_MODE BAD_FRAME_OUTER_HARD_VETO BAD_FRAME_HARD_HARD_VETO
export BAD_FRAME_HARD_PENALTY BAD_FRAME_IMAGE_SUMMARY
export STABLE_WINDOW_TARGET V392_ADOPTED_LOST_MAX EXTRA_TRAIN_ARGS

exec "$ROOT/tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh" "$@"
```

- [ ] **Step 4: Re-run wrapper test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_v399_wrapper_sweeps_only_over_caps_and_keeps_under_caps -q
```

Expected:

```text
1 passed
```

## Task 2: Add Selector Tests for Bad-Frame Penalty Mode

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify later: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add failing static selector test**

Append:

```python
def test_selector_supports_bad_frame_hard_veto_plus_penalty_mode():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_SELECTOR_MODE" in text
    assert "bad_frame_hard_veto_pass" in text
    assert "bad_frame_penalty" in text
    assert "bad_frame_penalty_reasons" in text
    assert "BAD_FRAME_OUTER_HARD_VETO" in text
    assert "BAD_FRAME_HARD_HARD_VETO" in text
    assert "hard_delta_positive_penalty" in text
    assert "fg_positive_count_penalty" in text
    assert "boundary_positive_count_penalty" in text
    assert "edge_positive_count_penalty" in text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_supports_bad_frame_hard_veto_plus_penalty_mode -q
```

Expected failure contains:

```text
AssertionError: assert 'BAD_FRAME_SELECTOR_MODE' in ...
```

- [ ] **Step 3: Extend selector env parsing**

In `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`, add env defaults near existing bad-frame settings:

```bash
BAD_FRAME_SELECTOR_MODE="${BAD_FRAME_SELECTOR_MODE:-veto}"
BAD_FRAME_OUTER_HARD_VETO="${BAD_FRAME_OUTER_HARD_VETO:-$BAD_FRAME_OUTER_VETO}"
BAD_FRAME_HARD_HARD_VETO="${BAD_FRAME_HARD_HARD_VETO:-$BAD_FRAME_HARD_VETO}"
BAD_FRAME_IMAGE_SUMMARY="${BAD_FRAME_IMAGE_SUMMARY:-$LOG_DIR/bad_frame_image_summary.tsv}"
```

Add these variables to `run_info.txt`, `export`, and the selector Python argv list.

- [ ] **Step 4: Update `load_bad_frame_stats()` logic**

In the inline selector Python, keep candidate-row filtering and compute:

```python
stats["bad_frame_hard_veto_pass"] = True
stats["bad_frame_penalty"] = 0.0
stats["bad_frame_penalty_reasons"] = []
```

After reading rows:

```python
if stats["bad_frame_max_outer_delta"] > bad_frame_outer_hard_veto:
    stats["bad_frame_hard_veto_pass"] = False
    stats["bad_frame_reject_reasons"].append("outer_hard_veto")
if stats["bad_frame_max_hard_delta"] > bad_frame_hard_hard_veto:
    stats["bad_frame_hard_veto_pass"] = False
    stats["bad_frame_reject_reasons"].append("hard_hard_veto")
if stats["bad_frame_max_hard_delta"] > 0.0:
    stats["bad_frame_penalty"] += stats["bad_frame_max_hard_delta"] * 10000.0
    stats["bad_frame_penalty_reasons"].append("hard_delta_positive_penalty")
if stats["bad_frame_hard_penalty_count"] > 0:
    stats["bad_frame_penalty"] += stats["bad_frame_hard_penalty_count"] * 0.25
if stats["bad_frame_fg_positive_count"] > 0:
    stats["bad_frame_penalty"] += stats["bad_frame_fg_positive_count"] * 0.05
    stats["bad_frame_penalty_reasons"].append("fg_positive_count_penalty")
if stats["bad_frame_boundary_positive_count"] > 0:
    stats["bad_frame_penalty"] += stats["bad_frame_boundary_positive_count"] * 0.05
    stats["bad_frame_penalty_reasons"].append("boundary_positive_count_penalty")
if stats["bad_frame_edge_positive_count"] > 0:
    stats["bad_frame_penalty"] += stats["bad_frame_edge_positive_count"] * 0.05
    stats["bad_frame_penalty_reasons"].append("edge_positive_count_penalty")
```

For compatibility:

```python
if bad_frame_selector_mode == "penalty":
    row["bad_frame_gate_pass"] = not bad_frame_selector_enable or row["bad_frame_hard_veto_pass"]
else:
    row["bad_frame_gate_pass"] = not bad_frame_selector_enable or len(bad_frame_stats["bad_frame_reject_reasons"]) == 0
```

- [ ] **Step 5: Include penalty in `rank_key()`**

Insert `row.get("bad_frame_penalty", 0.0)` after `safety_miss`:

```python
return (
    0 if row.get("stable_window_member", False) else 1,
    safety_miss,
    row.get("bad_frame_penalty", 0.0),
    inner_miss,
    row["inner_delta"],
    row["hard_delta"],
    row["outer_delta"],
    row["opacity_outer_delta"],
    row["edge_delta"],
)
```

- [ ] **Step 6: Re-run selector static test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_supports_bad_frame_hard_veto_plus_penalty_mode -q
```

Expected:

```text
1 passed
```

## Task 3: Add Cap Saturation and Failure Reason Diagnostics

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify later: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add failing diagnostics test**

Append:

```python
def test_selector_reports_cap_saturation_and_failure_reason_counts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "support_cap_saturated" in text
    assert "support_under_cap_saturation" in text
    assert "support_over_cap_saturation" in text
    assert "failure_reason_counts" in text
    assert "v392_floor_miss" in text
    assert "bad_frame_veto" in text
    assert "cap_saturation" in text
    assert "fg_boundary_edge_regression" in text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_reports_cap_saturation_and_failure_reason_counts -q
```

Expected failure contains missing diagnostic strings.

- [ ] **Step 3: Extend support summary parsing**

When parsing `support_bank_summary.tsv`, add:

```python
item[f"{direction}_effective_count"] = int(float(row["effective_count"]))
item[f"{direction}_new_only"] = int(float(row["new_only"]))
item[f"{direction}_adopted_count"] = int(float(row["adopted_count"]))
```

Use environment cap ratios to compute limits:

```python
point_count = max(
    row.get("under_effective_count", 0),
    row.get("over_effective_count", 0),
)
row["support_under_cap_saturation"] = row.get("under_new_only", 0) / max(1, int(point_count * under_new_only_ratio))
row["support_over_cap_saturation"] = row.get("over_new_only", 0) / max(1, int(point_count * over_new_only_ratio))
row["support_cap_saturated"] = (
    row["support_under_cap_saturation"] >= 0.999
    or row["support_over_cap_saturation"] >= 0.999
)
```

If exact cap ratio env values are not provided, emit `null` saturation values and keep `support_cap_saturated=False`.

- [ ] **Step 4: Add per-row failure reason function**

Add inline Python helper:

```python
def failure_reasons(row):
    reasons = []
    if not (
        row["status"] == "strict_pass"
        and row["fg_delta"] <= 0.0
        and row["boundary_delta"] <= 0.0
        and row["edge_delta"] <= 0.0
    ):
        reasons.append("fg_boundary_edge_regression")
    if not row.get("support_floor_pass", True):
        reasons.append("support_floor_miss")
    if not (
        row["outer_delta"] <= outer_floor
        and row["hard_delta"] <= hard_floor
        and row["opacity_outer_delta"] <= opacity_outer_floor
    ):
        reasons.append("v392_floor_miss")
    if not row.get("bad_frame_gate_pass", True):
        reasons.append("bad_frame_veto")
    if row.get("support_cap_saturated", False):
        reasons.append("cap_saturation")
    return reasons
```

For each row:

```python
row["failure_reasons"] = failure_reasons(row)
```

Aggregate:

```python
failure_reason_counts = {}
for row in rows:
    for reason in row.get("failure_reasons", []):
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
```

- [ ] **Step 5: Add diagnostics to JSON payload**

Add to `payload`:

```python
"failure_reason_counts": failure_reason_counts,
"cap_diagnostics": {
    "selected_support_cap_saturated": selected_support.get("support_cap_saturated"),
    "selected_support_under_cap_saturation": selected_support.get("support_under_cap_saturation"),
    "selected_support_over_cap_saturation": selected_support.get("support_over_cap_saturation"),
},
```

Also include these sections in `selected_checkpoint_metrics.json`.

- [ ] **Step 6: Re-run diagnostics test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_reports_cap_saturation_and_failure_reason_counts -q
```

Expected:

```text
1 passed
```

## Task 4: Add Bad-Frame Image Aggregation

**Files:**
- Modify: `tests/test_v396_raw_gate_paths.py`
- Modify later: `tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh`

- [ ] **Step 1: Add failing image aggregation test**

Append:

```python
def test_selector_writes_bad_frame_image_summary():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "BAD_FRAME_IMAGE_SUMMARY" in text
    assert "bad_frame_image_summary.tsv" in text
    assert "bad_frame_image_summary" in text
    assert "image_failure_aggregate" in text
    assert "candidate_bad_frame_rows" in text
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_writes_bad_frame_image_summary -q
```

Expected failure contains missing image aggregation strings.

- [ ] **Step 3: Implement candidate bad-frame row collection**

In `load_bad_frame_stats()`, collect candidate rows:

```python
stats["candidate_bad_frame_rows"] = []
...
if str(row.get("variant", "")).startswith("candidate_"):
    stats["candidate_bad_frame_rows"].append(dict(row))
```

- [ ] **Step 4: Aggregate after all checkpoint rows are loaded**

Add:

```python
image_failure_aggregate = {}
for row in rows:
    for bad_row in row.get("candidate_bad_frame_rows", []):
        image = bad_row.get("image", "")
        item = image_failure_aggregate.setdefault(image, {
            "image": image,
            "count": 0,
            "worsen_score_sum": 0.0,
            "worsen_score_max": 0.0,
            "outer_delta_max": 0.0,
            "hard_delta_max": 0.0,
            "fg_positive_count": 0,
            "boundary_positive_count": 0,
            "edge_positive_count": 0,
        })
        item["count"] += 1
        worsen = _float_value(bad_row, "worsen_score")
        item["worsen_score_sum"] += worsen
        item["worsen_score_max"] = max(item["worsen_score_max"], worsen)
        item["outer_delta_max"] = max(item["outer_delta_max"], _float_value(bad_row, "outer_delta"))
        item["hard_delta_max"] = max(item["hard_delta_max"], _float_value(bad_row, "hard_delta"))
        item["fg_positive_count"] += int(_float_value(bad_row, "fg_delta") > 0.0)
        item["boundary_positive_count"] += int(_float_value(bad_row, "boundary_delta") > 0.0)
        item["edge_positive_count"] += int(_float_value(bad_row, "edge_delta") > 0.0)
```

Write `BAD_FRAME_IMAGE_SUMMARY`:

```python
with bad_frame_image_summary.open("w", encoding="utf-8", newline="") as handle:
    fields = [
        "image", "count", "worsen_score_sum", "worsen_score_max",
        "outer_delta_max", "hard_delta_max",
        "fg_positive_count", "boundary_positive_count", "edge_positive_count",
    ]
    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
    writer.writeheader()
    for item in sorted(image_failure_aggregate.values(), key=lambda x: (-x["worsen_score_sum"], x["image"])):
        writer.writerow(item)
```

- [ ] **Step 5: Add top images to selected JSON**

Add:

```python
"bad_frame_image_summary": str(bad_frame_image_summary),
"top_bad_frame_images": sorted(
    image_failure_aggregate.values(),
    key=lambda x: (-x["worsen_score_sum"], x["image"]),
)[:10],
```

- [ ] **Step 6: Re-run image aggregation test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v396_raw_gate_paths.py::test_selector_writes_bad_frame_image_summary -q
```

Expected:

```text
1 passed
```

## Task 5: Optional Sweep Analyzer

**Files:**
- Create: `tools/analyze_377_v399_diagnostic_sweep.py`
- Create: `tests/test_v399_sweep_analyzer.py`

- [ ] **Step 1: Add failing analyzer test**

Create `tests/test_v399_sweep_analyzer.py`:

```python
import json
from pathlib import Path

from tools.analyze_377_v399_diagnostic_sweep import summarize_run


def test_summarize_run_reports_floor_bad_frame_and_cap_status(tmp_path):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    selected = {
        "selected": {
            "iteration": "141160",
            "hard_delta": -0.000231,
            "fg_delta": -0.1,
            "boundary_delta": -0.1,
            "edge_delta": -0.1,
        },
        "v392_floor": {"hard_delta": -0.00023074},
        "bad_frame_diagnostics": {
            "selected_bad_frame_max_outer_delta": 7.0,
            "selected_bad_frame_max_hard_delta": 0.0003,
        },
        "cap_diagnostics": {
            "selected_support_cap_saturated": True,
            "selected_support_over_cap_saturation": 1.0,
        },
    }
    (log_dir / "v399_selected_checkpoint.json").write_text(json.dumps(selected), encoding="utf-8")

    row = summarize_run(log_dir)

    assert row["iteration"] == "141160"
    assert row["hard_floor_pass"] == "1"
    assert row["bad_frame_not_worse_than_v398"] == "1"
    assert row["over_cap_saturation"] == "1.0000"
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v399_sweep_analyzer.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'tools.analyze_377_v399_diagnostic_sweep'
```

- [ ] **Step 3: Implement `summarize_run()` and CLI**

Create a small Python module that:

- accepts log dirs as argv;
- loads `v399_selected_checkpoint.json`;
- reads selected metrics and cap diagnostics;
- prints TSV columns:
  - `log_dir`
  - `selected`
  - `iteration`
  - `hard_delta`
  - `hard_floor_pass`
  - `fg_boundary_edge_safe`
  - `bad_frame_max_outer`
  - `bad_frame_max_hard`
  - `bad_frame_not_worse_than_v398`
  - `support_cap_saturated`
  - `over_cap_saturation`

- [ ] **Step 4: Run analyzer test**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_v399_sweep_analyzer.py -q
```

Expected:

```text
1 passed
```

## Task 6: Full Test and Syntax Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused pytest**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_boundary_support_bank.py tests/test_v396_raw_gate_paths.py tests/test_v399_sweep_analyzer.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n tools/run_377_explicit_binding_v396_generalized_boundary_controller.sh
bash -n tools/run_377_explicit_binding_v398_stable_generalization.sh
bash -n tools/run_377_explicit_binding_v399_diagnostic_sweep.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run Python compile checks**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m py_compile tools/analyze_377_v399_diagnostic_sweep.py train.py scene/gaussian_model.py utils/boundary_support_bank.py
```

Expected: no output and exit code `0`.

## Task 7: Diagnostic Sweep Runs

**Files:**
- No code changes.
- Outputs under `exp/formal/` and `exp/formal/logs/`.

- [ ] **Step 1: Run mid cap**

Run:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_cap_mid_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.48 \
V399_OVER_NEW_ONLY_RATIO=0.44 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_cap_mid_launch.log 2>&1 &
```

- [ ] **Step 2: Run high cap**

Run:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_cap_high_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.50 \
V399_OVER_NEW_ONLY_RATIO=0.46 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_cap_high_launch.log 2>&1 &
```

- [ ] **Step 3: Run max cap only if mid/high do not recover hard floor**

Run:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_cap_max_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.52 \
V399_OVER_NEW_ONLY_RATIO=0.48 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_cap_max_launch.log 2>&1 &
```

- [ ] **Step 4: Compare sweep outputs**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python tools/analyze_377_v399_diagnostic_sweep.py \
  exp/formal/logs/377_v399_diagnostic_sweep_v399_cap_mid_* \
  exp/formal/logs/377_v399_diagnostic_sweep_v399_cap_high_* \
  exp/formal/logs/377_v399_diagnostic_sweep_v399_cap_max_*
```

Expected:

- Choose the smallest cap where `hard_floor_pass=1`;
- `fg_boundary_edge_safe=1`;
- `bad_frame_not_worse_than_v398=1`;
- `adopted_lost=0` in `support_bank_summary.tsv`;
- support counts do not exceed configured caps.

## Task 8: Full v399 Run

**Files:**
- No code changes after selecting cap.

- [ ] **Step 1: Launch full run if `v399_cap_mid` is the smallest passing sweep**

Run this command only if the sweep analyzer shows `v399_cap_mid` is the smallest run with `hard_floor_pass=1`, `fg_boundary_edge_safe=1`, and `bad_frame_not_worse_than_v398=1`:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_full_cap_mid_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.48 \
V399_OVER_NEW_ONLY_RATIO=0.44 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_full_cap_mid_launch.log 2>&1 &
```

- [ ] **Step 2: Launch full run if `v399_cap_high` is the smallest passing sweep**

Run this command only if `v399_cap_mid` misses and `v399_cap_high` passes:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_full_cap_high_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.50 \
V399_OVER_NEW_ONLY_RATIO=0.46 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_full_cap_high_launch.log 2>&1 &
```

- [ ] **Step 3: Launch full run if only `v399_cap_max` passes**

Run this command only if both lower caps miss and `v399_cap_max` passes:

```bash
cd /remote-home/ming/3dgs-avatar-release-main
GPU=0 \
RUN_ID=v399_full_cap_max_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt') \
V399_OVER_EFFECTIVE_RATIO=0.52 \
V399_OVER_NEW_ONLY_RATIO=0.48 \
TRAIN_STEPS=1500 \
TEST_INTERVAL=250 \
SAVE_ITERATIONS='[250,500,750,1000,1250,1500]' \
CHECKPOINT_ITERATIONS='[250,500,750,1000,1250,1500]' \
nohup bash tools/run_377_explicit_binding_v399_diagnostic_sweep.sh \
  > exp/formal/logs/v399_full_cap_max_launch.log 2>&1 &
```

- [ ] **Step 4: Verify full acceptance**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python - <<'PY'
import csv, json
from pathlib import Path

log_dir = sorted(Path("exp/formal/logs").glob("377_v399_diagnostic_sweep_v399_full_*"))[-1]
payload = json.loads((log_dir / "v399_selected_checkpoint.json").read_text())
support_rows = list(csv.DictReader((log_dir / "support_bank_summary.tsv").open(), delimiter="\t"))

assert payload["selected"] is not None
assert payload["num_stable_window_pass"] >= 3
assert payload["selected"]["fg_delta"] <= 0.0
assert payload["selected"]["boundary_delta"] <= 0.0
assert payload["selected"]["edge_delta"] <= 0.0
assert payload["selected"]["hard_delta"] <= payload["v392_floor"]["hard_delta"]
assert all(int(float(row["adopted_lost"])) == 0 for row in support_rows)
assert (log_dir / "selected_checkpoint_path.txt").exists()
assert (log_dir / "selected_checkpoint_metrics.json").exists()
print(log_dir)
print(json.dumps(payload["selected"], indent=2))
PY
```

Expected:

- prints the v399 full log dir;
- selected checkpoint exists;
- all assertions pass.

## Experiment Matrix

| Run | `V399_OVER_EFFECTIVE_RATIO` | `V399_OVER_NEW_ONLY_RATIO` | When to run | Pass condition |
| --- | ---: | ---: | --- | --- |
| `v399_cap_mid` | `0.48` | `0.44` | Always | hard floor restored and bad-frame severity not worse than v398 |
| `v399_cap_high` | `0.50` | `0.46` | Always | same, used if mid misses floor |
| `v399_cap_max` | `0.52` | `0.48` | Only if needed | near-v397 support without exceeding controlled cap |

## Risk and Rollback

- Risk: over cap relaxation restores hard floor but worsens local bad frames.
  - Rollback: choose the lower cap and keep bad-frame penalty ranking; do not select a severe veto failure.

- Risk: no cap variant restores hard floor.
  - Rollback: stop before selector changes are used for full selection; return to root-cause investigation of support score distribution and residual response.

- Risk: penalty mode hides real local regressions.
  - Rollback: keep `BAD_FRAME_SELECTOR_MODE=veto` available. v398 behavior remains selectable.

- Risk: v396/v397/v398 wrappers break.
  - Rollback: all new env defaults must preserve current behavior unless v399 wrapper sets them. Run `bash -n` and existing tests before any GPU run.

- Risk: selected artifacts are generated for a weak run.
  - Rollback: selector should only write `selected_checkpoint_path.txt` and `selected_ckpt.pth` when `selected` is non-null. Rejected runs keep `selected_checkpoint_metrics.json` as a report only.

## Self-Review

- Spec coverage: support cap sweep, bad-frame veto/penalty, diagnostics, stable-window failure reasons, test points, validation commands, risk and rollback are covered.
- Placeholder scan: the only angle-bracket values are in the full-run launch command because they must be filled with the chosen sweep result after diagnostics.
- Compatibility: v399 defaults are isolated in a new wrapper; shared controller changes default to v398/v396-compatible behavior.
