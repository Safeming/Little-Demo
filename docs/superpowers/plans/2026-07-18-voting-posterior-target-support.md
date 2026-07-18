# Voting Posterior Target/Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先在 CoreView_377 严格 calibration/validation 协议上执行 voting-first target/support 方案 A；若未达到 B1 对比标准，再实现并执行 trained/voting posterior 融合方案 B。

**Architecture:** 方案 A 直接以现有 multi-view voting bank 的连续 `semantic_probs`、confidence、margin、reliable mask 和 soft weights 为 calibration 输入，复用现有 footprint evidence 与 support-aware calibration。新增一个 validation 固定阈值/radius override 和一个纯 CSV/JSON 候选判定器，确保成功标准自动执行。只有方案 A 判定失败时，方案 B 才新增 posterior 融合 helper/CLI，并对三个 alpha 候选走相同 calibration 和 validation 流程。

**Tech Stack:** Python 3.9、NumPy、pytest、现有 3DGS/CUDA 渲染工具、Bash 严格协议 runner。

---

### Task 1: Validation 固定参数复评入口

**Files:**
- Modify: `tools/evaluate_semantic_editing_paper_protocol.py`
- Modify: `tests/test_semantic_editing_paper_protocol.py`

- [ ] **Step 1: 写失败测试，要求 validation 支持显式阈值和 radius**

在 `tests/test_semantic_editing_paper_protocol.py` 增加：

```python
def test_parse_args_accepts_validation_metric_overrides():
    from tools.evaluate_semantic_editing_paper_protocol import parse_args

    args = parse_args([
        "--protocol", "protocol.json",
        "--protocol-split", "validation",
        "--trained-bank", "trained.npz",
        "--voting-bank", "voting.npz",
        "--checkpoint", "ckpt.pth",
        "--asset-root", "assets",
        "--output-dir", "out",
        "--soft-threshold", "0.05",
        "--boundary-radius", "6",
    ])

    assert args.soft_threshold == 0.05
    assert args.boundary_radius == 6
```

- [ ] **Step 2: 运行 RED**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_semantic_editing_paper_protocol.py::test_parse_args_accepts_validation_metric_overrides -q
```

Expected: FAIL，因为参数尚不存在。

- [ ] **Step 3: 实现最小 override**

在 `parse_args()` 增加：

```python
parser.add_argument("--soft-threshold", type=float, default=None)
parser.add_argument("--boundary-radius", type=int, default=None)
```

在 `evaluate_scene()` 中，将 validation 默认值和 test frozen 值统一为：

```python
fixed_threshold = float(
    args.soft_threshold
    if args.soft_threshold is not None
    else selected_config.get("soft_threshold", 0.20)
)
boundary_radius = int(
    args.boundary_radius
    if args.boundary_radius is not None
    else selected_config.get("boundary_radius", 2)
)
```

test split 仍必须通过 frozen fingerprint 校验；override 只用于 validation 复评，不允许改变 test frozen 参数。若 test 传 override，抛出 `ValueError`。

- [ ] **Step 4: 运行 GREEN 和 evaluator 回归**

Run:

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_semantic_editing_paper_protocol.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tools/evaluate_semantic_editing_paper_protocol.py \
  tests/test_semantic_editing_paper_protocol.py
git commit -m "功能：支持验证集固定参数复评"
```

### Task 2: 方案 A 候选自动判定器

**Files:**
- Create: `tools/assess_voting_posterior_candidate.py`
- Create: `tests/test_assess_voting_posterior_candidate.py`

- [ ] **Step 1: 写失败测试，覆盖通过和失败条件**

测试纯函数：

```python
def test_assess_candidate_passes_when_b5_beats_b1_at_required_retentions():
    from tools.assess_voting_posterior_candidate import assess_candidate

    result = assess_candidate(
        baseline_rows=[
            {"baseline": "B1", "macro_miou": 0.62},
            {"baseline": "B5", "macro_miou": 0.61},
        ],
        matched_rows=[
            {"baseline": "B1", "retention": 0.5, "actionable_leakage": 0.03},
            {"baseline": "B1", "retention": 0.6, "actionable_leakage": 0.03},
            {"baseline": "B5", "retention": 0.5, "actionable_leakage": 0.02},
            {"baseline": "B5", "retention": 0.6, "actionable_leakage": 0.025},
        ],
        required_retentions=(0.5, 0.6),
        max_miou_gap=0.02,
    )

    assert result["passed"] is True
```

再增加：缺少 retention 0.6、B5 leakage 高于 B1、mIoU gap 大于 0.02 时失败。

- [ ] **Step 2: 运行 RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_assess_voting_posterior_candidate.py -q
```

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现判定器和 CLI**

实现：

```python
def assess_candidate(
    baseline_rows,
    matched_rows,
    *,
    required_retentions=(0.5, 0.6),
    max_miou_gap=0.02,
) -> dict:
    ...
```

CLI 参数：

```text
--baseline-summary
--curve
--required-retention 0.5 0.6
--max-miou-gap 0.02
--output
```

输出 JSON 必须包含：

```text
passed
required_retentions
b1_miou
b5_miou
miou_gap
per_retention
failure_reasons
```

CLI 在候选失败时仍写 JSON，但返回 exit code 1；通过返回 0。

- [ ] **Step 4: 运行 GREEN**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_assess_voting_posterior_candidate.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tools/assess_voting_posterior_candidate.py \
  tests/test_assess_voting_posterior_candidate.py
git commit -m "功能：自动判定投票后验候选"
```

### Task 3: 执行方案 A

**Files:**
- Generated: `exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/voting_evidence_target_support/`
- Generated: `exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_a/`
- Generated: `exp/acceptdata/strict_semantic_paper_protocol_20260718/audit/voting_posterior_a/`

- [ ] **Step 1: 运行现有语义测试基线**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_semantic_editing_paper_protocol.py \
  tests/test_assess_voting_posterior_candidate.py -q
```

Expected: PASS。

- [ ] **Step 2: 从 voting bank 执行 support-aware calibration**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  tools/calibrate_evidence_soft_edit_weights.py \
  --part-label-bank exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/multiview_voting/part_label_bank.npz \
  --checkpoint exp/acceptdata/strict_semantic_paper_protocol_20260718/semantic_train_strict_v3/ckpt141910.pth \
  --asset-root exp/acceptdata/strict_semantic_paper_protocol_20260718/assets/calibration/test-view/semantic_editable_assets \
  --output exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/voting_evidence_target_support/part_label_bank.npz \
  --summary-json exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/voting_evidence_target_support/summary.json \
  --mode support-aware \
  --parts face hair upper lower shoes skin \
  --outer-penalty-power 0.2 \
  --support-penalty-power 0.2 \
  --support-pair face:hair \
  --support-pair hair:face \
  --support-pair upper:skin \
  --support-pair upper:lower \
  --support-pair lower:upper \
  --support-pair lower:skin \
  --support-pair shoes:lower \
  --support-pair shoes:skin \
  --support-pair skin:upper \
  --support-pair skin:lower \
  --protocol configs/semantic/coreview377_strict_paper_protocol.json \
  --protocol-split calibration
```

Expected: `processed_views=80`，输出含 `edit_target_weights` 和 `edit_support_weights`。

- [ ] **Step 3: 运行 validation sweep**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  tools/evaluate_semantic_editing_paper_protocol.py \
  --protocol configs/semantic/coreview377_strict_paper_protocol.json \
  --protocol-split validation \
  --validation-sweep \
  --trained-bank exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/voting_evidence_target_support/part_label_bank.npz \
  --voting-bank exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/multiview_voting/part_label_bank.npz \
  --checkpoint exp/acceptdata/strict_semantic_paper_protocol_20260718/semantic_train_strict_v3/ckpt141910.pth \
  --asset-root exp/acceptdata/strict_semantic_paper_protocol_20260718/assets/validation/test-view/semantic_editable_assets \
  --output-dir exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_a
```

- [ ] **Step 4: 冻结方案 A validation 参数**

计算 checkpoint/bank fingerprint，使用现有 selector 写：

```text
evaluation/validation_voting_posterior_a/frozen_validation_config.json
```

不得覆盖根目录原 `frozen_validation_config.json`。

- [ ] **Step 5: 用选定 threshold/radius 复评 validation mIoU**

从方案 A frozen config 读取 `soft_threshold` 和 `boundary_radius`，运行 evaluator 到：

```text
evaluation/validation_voting_posterior_a/fixed_metrics/
```

命令必须传 `--soft-threshold` 和 `--boundary-radius`，不传 `--validation-sweep`。

- [ ] **Step 6: 自动判定方案 A**

```bash
/opt/miniconda3/envs/ictrl/bin/python \
  tools/assess_voting_posterior_candidate.py \
  --baseline-summary exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_a/fixed_metrics/baseline_summary.csv \
  --curve exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_a/leakage_retention_curve.csv \
  --required-retention 0.5 0.6 \
  --max-miou-gap 0.02 \
  --output exp/acceptdata/strict_semantic_paper_protocol_20260718/audit/voting_posterior_a/assessment.json
```

判定器读取 curve 时，应先为 B1/B5 在 required retention 内插 matched values；不得外推。

- [ ] **Step 7: 分支判断**

```text
assessment.passed=true：停止，记录方案 A 成功；
assessment.passed=false：保留全部 A 资产，进入 Task 4。
```

### Task 4: 方案 B posterior 融合实现（仅 A 失败）

**Files:**
- Modify: `utils/part_label_bank.py`
- Create: `tools/fuse_semantic_part_label_banks.py`
- Create: `tests/test_fuse_semantic_part_label_banks.py`

- [ ] **Step 1: 写失败测试**

覆盖：

```python
def test_fuse_semantic_posteriors_blends_and_normalizes():
    ...

def test_fuse_semantic_posteriors_rejects_shape_mismatch():
    ...

def test_fuse_semantic_posteriors_uses_uniform_fallback_for_zero_rows():
    ...
```

CLI roundtrip 测试要求输出包含：

```text
semantic_probs
confidence
semantic_margin
reliable_mask
editable_label
soft_edit_weights
source_type=fused_trained_voting_semantic_probs
trained_bank_fingerprint
voting_bank_fingerprint
fusion_alpha
```

- [ ] **Step 2: 运行 RED**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_fuse_semantic_part_label_banks.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现纯 NumPy helper**

```python
def fuse_semantic_posteriors(trained_probs, voting_probs, *, voting_alpha: float) -> np.ndarray:
    ...
```

要求 `0 <= alpha <= 1`，逐行归一化；零和行回退到均匀六类分布。

- [ ] **Step 4: 实现融合 CLI**

CLI 读取两个 bank，验证 point count/part names/checkpoint 一致，使用现有：

```text
finalize_trained_semantic_probs
compute_semantic_margin
compute_semantic_reliable_mask
compute_soft_edit_weights
save_part_label_bank
```

- [ ] **Step 5: 运行 GREEN 和回归**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_fuse_semantic_part_label_banks.py \
  tests/test_part_label_bank.py -q
```

- [ ] **Step 6: 提交**

```bash
git add utils/part_label_bank.py \
  tools/fuse_semantic_part_label_banks.py \
  tests/test_fuse_semantic_part_label_banks.py
git commit -m "功能：融合训练语义与投票后验"
```

### Task 5: 执行方案 B alpha 网格（仅 A 失败）

**Files:**
- Generated: `exp/acceptdata/strict_semantic_paper_protocol_20260718/banks/fused_voting_target_support_alpha*/`
- Generated: `exp/acceptdata/strict_semantic_paper_protocol_20260718/evaluation/validation_voting_posterior_b/`

- [ ] **Step 1: 生成 alpha 0.50/0.75/0.90 融合 bank**

每个 alpha 先生成 fused base bank，再运行与方案 A 相同的 support-aware calibration，输出独立目录。

- [ ] **Step 2: 对每个 alpha 运行 validation sweep、参数冻结和 fixed-metrics 复评**

不得覆盖方案 A 或原严格结果。

- [ ] **Step 3: 对每个 alpha 运行候选判定器**

输出：

```text
audit/voting_posterior_b/alpha050_assessment.json
audit/voting_posterior_b/alpha075_assessment.json
audit/voting_posterior_b/alpha090_assessment.json
```

- [ ] **Step 4: 选择方案 B 最优候选**

只在通过 retention 0.5/0.6 和 mIoU gap 条件的候选中，按：

```text
最低 mean matched actionable leakage；
最高 mIoU；
最大 alpha；
```

写 `audit/voting_posterior_b/selection.json`。

- [ ] **Step 5: 停止**

方案 B 成功或三个 alpha 全失败后停止。不得运行 test。

### Task 6: 最终验证与中文记录

**Files:**
- Modify: `docs/严格语义论文协议执行记录与下一步投稿缺口_20260718.md`

- [ ] **Step 1: 运行相关测试**

```bash
/opt/miniconda3/envs/ictrl/bin/python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_semantic_editing_paper_protocol.py \
  tests/test_assess_voting_posterior_candidate.py \
  tests/test_fuse_semantic_part_label_banks.py -q
```

若方案 B 未执行，可省略其测试文件。

- [ ] **Step 2: 审计无 test 访问**

确认本轮所有新 provenance 的 split 只有：

```text
calibration
validation
```

并确认 `evaluation/test/` 的文件时间和 fingerprint 未变化。

- [ ] **Step 3: 更新中文记录**

记录：

```text
方案 A/B 是否执行；
结束时间（北京时间）；
validation retention 范围；
B1/B5 matched leakage；
B1/B5 mIoU；
成功或失败原因；
下一步停止条件。
```

- [ ] **Step 4: 提交记录**

```bash
git add docs/严格语义论文协议执行记录与下一步投稿缺口_20260718.md
git commit -m "文档：记录投票后验候选验证结果"
```
