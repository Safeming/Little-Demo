# Voting Posterior Target/Support 设计

日期：2026-07-18

分支：`feat/strict-semantic-paper-protocol`

## 1. 目标

在不读取 test、不重新训练 Avatar backbone 的前提下，将 calibration 视角生成的连续 multi-view voting posterior 接入现有 confidence/margin、footprint evidence 和 target/support calibration，验证新 B5 是否能在 CoreView_377 validation 上接近或超过 B1 voting baseline。

执行顺序固定为：

```text
先执行方案 A；
方案 A 未达到成功标准时，再执行方案 B；
两种方案都只使用 calibration 和 validation；
在方法重新冻结前禁止读取或重跑 test。
```

## 2. 已有输入

最终 checkpoint：

```text
exp/acceptdata/strict_semantic_paper_protocol_20260718/
  semantic_train_strict_v3/ckpt141910.pth
```

连续 voting bank：

```text
exp/acceptdata/strict_semantic_paper_protocol_20260718/
  banks/multiview_voting/part_label_bank.npz
```

该 bank 已包含：

```text
per_part_votes
vote_count
visible_vote_count
semantic_probs
confidence
semantic_margin
reliable_mask
soft_edit_weights
editable_label
```

因此方案 A 不需要重新导出资产，也不需要新增网络训练。

## 3. 方案 A：Voting-First Evidence Calibration

### 3.1 数据流

```text
calibration parser masks
  -> projected multi-view votes
  -> voting semantic_probs
  -> voting confidence / semantic margin / reliable mask
  -> voting soft_edit_weights
  -> multi-view footprint evidence
  -> evidence target/support calibration
  -> voting_evidence_target_support bank
  -> validation B1 vs new B5
```

### 3.2 资产隔离

新候选写入独立目录：

```text
exp/acceptdata/strict_semantic_paper_protocol_20260718/
  banks/voting_evidence_target_support/
  evaluation/validation_voting_posterior_a/
```

不得覆盖：

```text
banks/evidence_target_support/
evaluation/test/
frozen_validation_config.json
```

### 3.3 参数

复用当前严格方法的固定参数：

```text
calibration split        = calibration
parts                    = face,hair,upper,lower,shoes,skin
penalty power            = 0.2
target retention floor   = 0.6
support threshold        = 0.35
minimum support views    = 5
support pairs            = 当前固定十组 adjacency pairs
```

validation 仍使用协议中的阈值和 radius 网格，不引入新的融合系数。

### 3.4 成功标准

方案 A 通过需要同时满足：

```text
1. validation aggregate target retention >= 0.60；
2. 在共同支持的 retention 0.5 和 0.6 上生成 matched rows；
3. new B5 actionable leakage <= B1 actionable leakage；
4. new B5 mIoU 与 B1 的差距不超过 0.02；
5. protocol/checkpoint/bank fingerprints 完整；
6. uses_test_parser_for_calibration=false。
```

如果 retention 0.6 不在双方共同观测范围内，方案 A 判定失败，不允许外推。

## 4. 方案 B：Trained/Voting Posterior 融合

仅在方案 A 失败后执行。

### 4.1 融合定义

对每个 Gaussian 和 part：

```text
fused_probs = alpha * voting_probs + (1 - alpha) * trained_probs
```

然后重新归一化，并从 fused probabilities 计算：

```text
confidence
semantic_margin
reliable_mask
soft_edit_weights
```

再执行与方案 A 相同的 footprint evidence 和 target/support calibration。

### 4.2 validation 网格

只在 validation 上选择：

```text
alpha = 0.50, 0.75, 0.90
```

选择规则：

```text
先满足 retention >= 0.60；
再最小化 matched actionable leakage；
再最大化 mIoU；
最后优先更大的 voting 权重 alpha。
```

所有 alpha 候选使用相同 calibration records、support pairs、penalty power 和 evaluator。

### 4.3 输出隔离

```text
banks/fused_voting_target_support_alpha050/
banks/fused_voting_target_support_alpha075/
banks/fused_voting_target_support_alpha090/
evaluation/validation_voting_posterior_b/
```

方案 B 未冻结前同样禁止 test。

## 5. 代码边界

方案 A 优先复用现有工具，不修改 bank schema：

```text
tools/calibrate_evidence_soft_edit_weights.py
tools/evaluate_semantic_editing_paper_protocol.py
utils/part_label_bank.py
```

方案 B 若需要新增融合逻辑，添加一个纯 NumPy helper 和独立 CLI，避免把融合规则塞进 evaluator：

```text
utils/part_label_bank.py
  fuse_semantic_posteriors()

tools/fuse_semantic_part_label_banks.py
```

## 6. 测试

方案 A 必须验证：

```text
voting bank 字段完整；
calibration 输出继承 voting posterior；
输出包含 edit_target_weights / edit_support_weights；
calibration provenance 只有 80 条 calibration records；
validation 输出不创建或修改 test 目录。
```

方案 B 必须使用 TDD 验证：

```text
alpha 边界；
shape mismatch；
归一化；
零概率 fallback；
输出 bank schema；
融合来源 fingerprint。
```

## 7. 停止条件

```text
方案 A 达标：停止，不执行方案 B；
方案 A 未达标：归档 A，再执行方案 B；
方案 B 仍不能接近 B1：停止方法微调，不读取 test，按 workshop/协议论文定位处理。
```

本轮不修改 Avatar backbone，不启动新训练，不调整 CoreView_377 渲染质量。
