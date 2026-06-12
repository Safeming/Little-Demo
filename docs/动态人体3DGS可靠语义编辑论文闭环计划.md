# 动态人体3DGS可靠语义编辑论文闭环计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前 3DGS 人体语义编辑项目推进到可投稿论文版本，重点完成可靠度软编辑、实验评估闭环、多视角/动态一致性验证，而不是继续追逐单版渲染边缘微调。

**Architecture:** 当前项目已经把 label bank 主来源从多视角 2D mask voting 转到训练出的 `binding_compact_semantic_probs_asset`，并导出了 `semantic_probs/confidence/semantic_margin/reliable_mask/editable_label`。下一阶段在此基础上增加 soft edit weight 导出、编辑泄露评估、多视角/动态一致性指标和论文消融脚本，形成“方法-数据-指标-图表”的闭环。渲染质量精修只作为附属实验，不作为论文主贡献。

**Tech Stack:** Python, NumPy, PyTorch, existing `train.py`, `render.py`, `tools/semantic_viewer/build_part_label_bank.py`, `utils/part_label_bank.py`, Editor/viewer bundle, `part_label_bank.npz`, JSON/CSV experiment reports.

**Last Updated:** 2026-06-10

---

## 1. 当前项目所处阶段

当前项目已经具备论文原型的核心基础：

- 已经有动态人体 3DGS / explicit binding / deformer / semantic asset adapter 主线。
- `part_label_bank.npz` 已经不再主要依赖 2D mask voting，而是从训练语义资产 `binding_compact_semantic_probs_asset` 导出。
- 已经有 `face_guard`、`lower_guard`、`reliability`、`neighbor_fill` 等后处理。
- 已经实现并验证 `soft_edit_weights` 显式导出。
- 已经恢复可复现的 2D mask voting baseline 导出路径。
- 当前推荐给 Editor / 论文主线使用的 soft-edit bundle 是：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_reliability_neighborfill_softedit_20260610/manifest.json
```

当前已复现的 2D voting baseline 是：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_baseline_20260610/part_label_bank.npz
```

注意：上面这版是投影 Y 轴修正前的旧 baseline，仅保留作回溯对比。当前推荐用于论文消融的新 2D voting baseline 是：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_baseline_yfixed_20260610/part_label_bank.npz
```

当前主要问题不是“系统能不能跑”，而是论文还缺少完整证明：

- soft edit selection 已经作为一等输出，但 Editor 侧是否读取还需要同步确认。
- hard label edit 和 reliability soft edit 已有 per-Gaussian 初版对比脚本，但真正有论文说服力的编辑泄露率需要改成 projection-space / boundary-space 评估。
- trained semantic asset 和 2D voting baseline 已有第一版定量对比；修正投影 Y 轴后，2D voting baseline 质量显著改善，但仍不能直接把 voting per-Gaussian label 当作 soft edit leakage 真值。
- 多视角/动态一致性还缺指标。
- projection-space / boundary-space 编辑泄露率已经跑通第一版，下一步需要把它扩展成多视角一致性和论文可视化证据。
- 交互 recipe 反馈训练还没有闭环，但它可以放到 future work 或扩展版本。

## 2. 论文主张建议

推荐论文主张收窄为：

```text
面向动态人体 3D Gaussian Splatting 的训练式部件语义绑定与可靠语义编辑
```

不要把论文主张写成“提升 3DGS 渲染质量”。当前渲染质量已经可用于语义编辑展示，但不是最强贡献。

建议贡献点：

1. **训练式 per-Gaussian semantic asset**
   - 从 2D parser supervision 学到每个 Gaussian 的 compact semantic probability。
   - 避免最终 label bank 只依赖多视角投票。

2. **Reliability-aware semantic label bank**
   - 使用 `confidence`、`semantic_margin`、`reliable_mask`、`editable_label` 区分可编辑点和不确定点。
   - 对 face/hair/skin、upper/lower/skin 等边界污染做保守处理。

3. **Soft semantic edit selection**
   - 编辑不是只看硬标签，而是使用每个 Gaussian 属于目标部件的软权重。
   - 减少非目标区域被影响。

4. **Dynamic/multi-view consistency evaluation**
   - 证明语义和编辑权重在不同视角、不同动作帧下保持稳定。

## 3. 下一阶段不要优先做什么

不建议继续把主要时间放在：

- 单独追求 render PSNR/LPIPS 的小幅提升。
- 只针对当前人物做 face/hair/skin 的手工规则微调。
- 立刻做复杂的交互 recipe 反馈训练闭环。
- 继续堆更多 guard，导致 label bank 后处理越来越臃肿。

原因：

- 最近的 `render_quality_refine_boundary_color_v1_20260610` 已证明渲染指标只能小幅提升，边缘 hard score 没有稳定变好。
- 论文更需要证明“语义编辑更可靠”，不是证明“这版图片稍微亮一点”。
- 交互 recipe 反馈训练很有价值，但会扩大工程范围，适合放 future work 或第二篇/扩展版。

## 4. 必须完成的论文闭环

### 4.1 Soft Edit Weight 导出

目标：让 Editor 不再只用硬标签选择点，而是能读取每个部件的软编辑权重。

推荐公式：

```text
edit_weight[target] =
    semantic_probs[target]
    * confidence
    * reliable_factor
    * margin_factor
    * optional_boundary_falloff
```

字段建议：

```text
soft_edit_weights       float32 [N, C]
soft_edit_confidence    float32 [N]
soft_edit_margin        float32 [N]
soft_edit_reliable      uint8   [N]
soft_edit_part_names    json/list in manifest
```

最小版本可以先不存 `[N, C]` 全矩阵，只存当前 `semantic_probs` 并在 Editor 侧按公式计算。但为了可复现和论文实验，建议导出一份显式 `soft_edit_weights`。

涉及文件：

- `utils/part_label_bank.py`
- `tools/semantic_viewer/build_part_label_bank.py`
- `tests/test_part_label_bank.py`
- Editor 读取 bundle 的代码，如果 Editor 在另一个仓库，需要同步字段说明。

验收标准：

- `part_label_bank.npz` 中包含 soft edit 字段。
- `manifest.json` 中声明 `soft_edit_weight_field`。
- 旧 Editor 即使不读 soft edit 字段，也能继续使用 `editable_label`。
- 新 Editor 可按 soft weight 调整编辑强度。

当前状态（2026-06-10）：

- 已完成代码实现：
  - `utils/part_label_bank.py`
    - `compute_soft_edit_weights()`
    - `save_part_label_bank()` / `validate_part_label_bank_arrays()` 支持 `soft_edit_weights`
  - `tools/semantic_viewer/build_part_label_bank.py`
    - `--export-soft-edit-weights`
    - `--soft-edit-reliable-floor`
    - `--soft-edit-margin-power`
    - `--soft-edit-confidence-power`
  - `tests/test_part_label_bank.py`
- 已生成 soft-edit bundle：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_reliability_neighborfill_softedit_20260610
```

- 验证字段：

```text
soft_edit_weights: float32 [46801, 6]
manifest.soft_edit_weight_field: soft_edit_weights
manifest.soft_edit_part_names: hair, face, upper, lower, shoes, skin
```

### 4.2 Hard Label vs Soft Edit 对比实验

目标：证明 soft edit 能减少非目标区域误编辑。

实验设置：

- 输入同一个 viewer bundle。
- 对每个 target part 分别模拟编辑：
  - `hair`
  - `face`
  - `skin`
  - `upper`
  - `lower`
  - `shoes`
- hard baseline：
  - `editable_label == target`
- soft method：
  - `soft_edit_weights[:, target] >= threshold`
  - 或直接按 weight 连续作用。

指标：

```text
target_coverage
non_target_activation
leakage_ratio = non_target_activation / target_activation
boundary_leakage_ratio
unknown_activated_count
mean_selected_confidence
mean_selected_margin
```

输出：

```text
exp/acceptdata/soft_edit_ablation_*/summary.json
exp/acceptdata/soft_edit_ablation_*/per_part.csv
exp/acceptdata/soft_edit_ablation_*/figures/*.png
```

验收标准：

- soft edit 的 `leakage_ratio` 低于 hard label edit。
- target coverage 不应下降过多。
- face/hair/skin 的边界泄露有明显下降。

当前状态（2026-06-10）：

- 已完成 per-Gaussian 初版评估脚本：
  - `tools/analyze_soft_edit_leakage.py`
  - `tests/test_soft_edit_leakage.py`
- 已支持两种 reference：
  - 自身 `part_label`
  - 外部 `--reference-label-bank`
- 已生成结果：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/soft_edit_ablation_v395_softedit_threshold025
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/soft_edit_ablation_v395_reference_trained_threshold025
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/soft_edit_ablation_with_2d_voting_reference_threshold025
```

重要发现：

- 用自身 `part_label` 当 reference 时，hard leakage 天然接近 0，指标区分力不足。
- 用 trained semantic 早期 bundle 当 reference 时，仍然过于同源，hard/soft leakage 也难以拉开。
- 用 2D voting per-Gaussian label 当 reference 时，出现极端 ratio：

```text
mean_hard_leakage_ratio = 109.45
mean_soft_leakage_ratio = 82.20
```

这不应直接写成论文结论。根因是 2D voting baseline 和 trained semantic asset 的 per-Gaussian label 口径严重错位，部分部件 target activation 接近 0，导致 ratio 爆炸。下一步应把编辑泄露率改成 **projection-space / boundary-space leakage**，即把 hard/soft selection 投影回 2D mask 和边界带上评估。

### 4.3 Trained Semantic Asset vs 2D Voting Baseline

目标：证明当前主线不是“换了个文件名”，而是比原始多视角 2D voting 更稳定。

对比对象：

1. `2D mask voting label bank`
2. `trained semantic asset label bank`
3. `trained semantic asset + reliability`
4. `trained semantic asset + reliability + neighbor fill`
5. `trained semantic asset + reliability + soft edit`

指标：

```text
part_count_distribution
unknown_count
low_margin_count
face_hair_confusion
skin_upper_confusion
skin_lower_confusion
multi_view_projection_agreement
temporal_label_stability
edit_leakage_ratio
```

涉及代码：

- `tools/semantic_viewer/build_part_label_bank.py`
  - 保留旧 voting 路径作为 baseline，不作为默认主线。
- 新增：
  - `tools/analyze_semantic_label_bank_ablation.py`
  - `tests/test_semantic_label_bank_ablation.py`

验收标准：

- 训练语义资产在边界一致性、低置信点识别、编辑泄露方面优于 2D voting。
- voting baseline 可以复现，不需要作为最终推荐版本。

当前状态（2026-06-10）：

- 已恢复 `build_part_label_bank.py` 中的 2D voting baseline 导出路径：
  - `--label-bank-source trained-semantic`
  - `--label-bank-source projected-2d-voting`
- 已新增/完成：
  - `collect_trained_semantic_bank()`
  - `collect_projected_2d_voting_bank()`
  - `tools/analyze_semantic_label_bank_ablation.py`
  - `tests/test_semantic_label_bank_ablation.py`
- 已生成 voting baseline：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_baseline_20260610/part_label_bank.npz
```

旧 baseline 风险：

- 该版本使用了错误的投影 Y 轴约定；训练标签投影到 2D mask 时出现上下倒置。
- 典型症状：
  - hair / face 选中点主要命中 shoes / skin mask。
  - shoes 选中点主要命中 hair / face mask。
  - projection leakage 中 hair / face / shoes 的 target activation 为 0，导致 ratio 不可信。

已修正代码：

```text
tools/semantic_viewer/build_part_label_bank.py::_project_points()
tests/test_part_label_bank.py::test_project_points_uses_asset_mask_y_axis_convention
```

修正后的 voting baseline：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_baseline_yfixed_20260610/part_label_bank.npz
```

- voting baseline 验证：

```text
旧 baseline:
  unknown_count = 1332
  known_count = 45469
  mean_confidence ≈ 0.6569

修正后 baseline:
  source_type = multiview_2d_mask_votes
  point_count = 46801
  max_vote_count = 30
  unknown_count = 908
  known_count = 45893
  mean_confidence = 0.8044
```

- 已生成修正后的消融表：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_label_bank_ablation_with_2d_voting_yfixed/ablation.csv
```

关键数值：

```text
2D voting y-fixed:
  unknown_count = 908
  known_count = 45893
  mean_confidence = 0.8044

trained semantic + reliability + neighborfill + softedit:
  unknown_count = 57
  known_count = 46744
  mean_confidence = 0.8559
  low_margin_count = 1961
  low_margin_rate = 0.04195
  label_agreement_with_2d_voting_yfixed = 0.7240
```

风险说明：

- `label_agreement_with_reference` 使用修正后 2D voting 作为 reference 时约 `0.724`，已经不再是旧版 `0.20` 的错位状态。
- 这能作为 trained semantic asset 与 2D mask voting 的可解释消融，但仍不能单独证明 trained asset 更“准”。
- 后续要用 projection-space agreement / boundary leakage / temporal stability 来支撑论文主张。

### 4.4 多视角一致性评估

目标：证明同一个 3D Gaussian 的语义和编辑权重不会因为观察视角变化而明显漂移。

评估思路：

- 选择同一个 checkpoint / bundle。
- 对多个 camera view 渲染 semantic/part projection。
- 对每个 part 统计：
  - projected target area
  - projected soft weight mean
  - target/non-target overlap
  - view-to-view variance
  - cross-view IoU 或 agreement

输出：

```text
exp/acceptdata/multiview_semantic_consistency_*/summary.json
exp/acceptdata/multiview_semantic_consistency_*/per_view.csv
exp/acceptdata/multiview_semantic_consistency_*/montage/*.png
```

验收标准：

- trained semantic asset 的 view-to-view variance 低于 2D voting baseline。
- soft edit 在多视角下的 target 区域更稳定。

### 4.5 动态一致性评估

目标：证明在动作序列中，语义选择和编辑权重不会明显闪烁。

评估维度：

- frame-to-frame part count change
- frame-to-frame soft weight mean change
- target selected Gaussian overlap
- low-confidence boundary fluctuation
- edit leakage over time

推荐指标：

```text
temporal_part_count_std
temporal_weight_mean_std
temporal_selection_jaccard
temporal_leakage_std
worst_frame_leakage
```

涉及工具：

- 新增 `tools/analyze_dynamic_semantic_consistency.py`
- 读取同一个 bundle 和一组 frame/camera。
- 如果需要渲染辅助图，调用现有 `render.py` 或已有 semantic export。

验收标准：

- soft edit selection 的 temporal leakage 比 hard label selection 更低。
- 关键部件 `face/hair/skin/upper/lower/shoes` 在动态帧中没有明显跳变。

### 4.6 论文图表与消融矩阵

论文必须至少准备以下图表：

1. 方法流程图
   - 2D parser supervision
   - semantic asset adapter training
   - per-Gaussian semantic probability
   - reliability-aware label bank
   - soft edit selection

2. 语义 label bank 可视化
   - voting baseline
   - trained semantic asset
   - reliability/neighbor fill
   - soft edit heatmap

3. 编辑结果图
   - hard edit vs soft edit
   - face/hair/skin 边界
   - upper/lower/skin 边界

4. 动态一致性图
   - 多帧编辑结果
   - temporal leakage 曲线

5. 消融表

推荐表格：

```text
Method                         Semantic Acc   Leakage ↓   Temporal Std ↓   Unknown ↓
2D voting                      ...
Trained semantic asset          ...
+ face/lower guard              ...
+ reliability                   ...
+ neighbor fill                 ...
+ soft edit                     ...
```

## 5. 具体执行计划

### Task 1: 实现 soft edit weight 导出

**Files:**

- Modify: `utils/part_label_bank.py`
- Modify: `tools/semantic_viewer/build_part_label_bank.py`
- Modify: `tests/test_part_label_bank.py`

- [x] Step 1: 在 `tests/test_part_label_bank.py` 增加 soft edit 权重测试。
- [x] Step 2: 运行测试，确认当前缺少字段而失败。
- [x] Step 3: 在 `utils/part_label_bank.py` 增加 `compute_soft_edit_weights()`。
- [x] Step 4: 在 `save_part_label_bank()` 和 `validate_part_label_bank_arrays()` 中支持 `soft_edit_weights`。
- [x] Step 5: 在 `tools/semantic_viewer/build_part_label_bank.py` 增加导出参数：

```text
--export-soft-edit-weights
--soft-edit-reliable-floor
--soft-edit-margin-power
--soft-edit-confidence-power
```

- [x] Step 6: 重新生成当前推荐 viewer bundle 的 soft edit 版本。
- [x] Step 7: 验证 `manifest.json` 和 `part_label_bank.npz` 字段。

### Task 2: 实现 hard vs soft edit 评估

**Files:**

- Create: `tools/analyze_soft_edit_leakage.py`
- Create: `tests/test_soft_edit_leakage.py`

- [x] Step 1: 用合成 label bank 写 leakage 单测。
- [x] Step 2: 实现读取 `part_label_bank.npz`。
- [x] Step 3: 实现 hard selection 和 soft selection。
- [x] Step 4: 输出 per-part leakage 指标。
- [x] Step 5: 在当前推荐 bundle 上跑评估。
- [x] Step 6: 输出 `summary.json` 和 `per_part.csv`。

### Task 2b: 实现 projection-space / boundary-space edit leakage 评估

**Files:**

- Create: `tools/analyze_projected_soft_edit_leakage.py`
- Create: `tests/test_projected_soft_edit_leakage.py`
- Reuse: `tools/semantic_viewer/build_part_label_bank.py`

背景：

- Per-Gaussian leakage 只能作为诊断，不适合作为论文主指标。
- 2D voting baseline 和 trained semantic asset 的 per-Gaussian 标签口径差异过大，直接计算 ratio 会产生不稳定极值。
- 论文需要的是“编辑投影后是否越过目标 2D mask / boundary band”，所以应在投影视角下评估。

建议指标：

```text
projected_target_activation
projected_non_target_activation
projected_leakage_ratio
boundary_band_activation
outer_boundary_leakage_ratio
inner_target_coverage
unknown_or_invalid_projection_count
```

最小实现思路：

- 输入：
  - softedit `part_label_bank.npz`
  - 同一 bundle 的 `semantic_editable_assets/view_records.json`
  - checkpoint / config / asset-root
  - target part list
- 对每个 selected Gaussian：
  - hard: `editable_label == target`
  - soft: `soft_edit_weights[:, target] >= threshold`
- 对每个 view：
  - 用现有 `_project_points()` 投影 Gaussian。
  - 读取 `_load_record_masks()` 的 target part mask、foreground、valid。
  - 在 2D mask 空间统计 target / non-target / boundary band activation。
- 输出：

```text
exp/acceptdata/semantic_editing_paper_loop_20260610/projected_soft_edit_leakage_*/summary.json
exp/acceptdata/semantic_editing_paper_loop_20260610/projected_soft_edit_leakage_*/per_part.csv
exp/acceptdata/semantic_editing_paper_loop_20260610/projected_soft_edit_leakage_*/per_view.csv
```

验收标准：

- 能在当前 v395 softedit bundle 上跑通。
- soft selection 的 projected leakage ratio 低于 hard selection，或至少 boundary leakage 更低。
- 如果指标仍不区分，应输出 threshold sweep，而不是只给单一 threshold。

当前状态（2026-06-10）：

- 已完成第一版实现：
  - `tools/analyze_projected_soft_edit_leakage.py`
  - `tests/test_projected_soft_edit_leakage.py`
- 已修正投影 Y 轴约定，并用真实场景交叉表验证：
  - 修正前：hair/face 命中 shoes/skin，shoes 命中 hair/face。
  - 修正后：hair/lower/shoes/skin 明显对角，face 与 hair 仍有头部边界混淆但不再上下倒置。
- 已生成修正后的 threshold sweep：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/projected_soft_edit_leakage_v395_yfixed_threshold_sweep.csv
```

关键数值：

```text
threshold  hard_leakage  soft_leakage  delta   hard_boundary  soft_boundary  boundary_delta
0.10       0.4657        0.4658        +0.0002 0.2242         0.2207         -0.0034
0.20       0.4657        0.4658        +0.0001 0.2242         0.2205         -0.0036
0.25       0.4657        0.4759        +0.0102 0.2242         0.2228         -0.0013
0.35       0.4657        0.4740        +0.0083 0.2242         0.2219         -0.0023
0.50       0.4657        0.4686        +0.0029 0.2242         0.2193         -0.0049
```

解读：

- projection-space 指标现在可信，不再有 target activation 为 0 的假异常。
- 当前 soft selection 的总体 leakage ratio 没有显著优于 hard selection，不能直接写成“soft 明显降低全部外泄”。
- boundary leakage 稳定略低，尤其阈值 `0.10/0.20/0.50` 有更清晰优势。
- per-part 上 face/hair/skin 有改善，lower/shoes 有反向趋势；下一步应把论文主张收敛到“可靠边界控制/软权重可解释性”，并补多视角一致性与可视化，而不是只看单一 mean leakage。

完成项：

- [x] Step 1: 写 pure NumPy 单测覆盖 boundary band、target/outer/boundary activation。
- [x] Step 2: 实现 `tools/analyze_projected_soft_edit_leakage.py`。
- [x] Step 3: 在真实 scene 上复现并定位投影错位。
- [x] Step 4: 修正 `_project_points()` 的 Y 轴约定并补回归测试。
- [x] Step 5: 重新生成 y-fixed 2D voting baseline。
- [x] Step 6: 在 v395 softedit bundle 上跑 `0.10 / 0.20 / 0.25 / 0.35 / 0.50` threshold sweep。

### Task 3: 实现 trained asset vs voting baseline 对比

**Files:**

- Create: `tools/analyze_semantic_label_bank_ablation.py`
- Create: `tests/test_semantic_label_bank_ablation.py`

- [x] Step 1: 收集 baseline label bank 路径。
- [x] Step 2: 统一读取多个 label bank。
- [x] Step 3: 统计 part distribution、unknown、margin、confusion pairs。
- [x] Step 4: 输出 ablation summary。
- [x] Step 5: 生成论文表格草稿 CSV。

补充完成项：

- [x] 恢复 2D voting baseline 导出：

```text
--label-bank-source projected-2d-voting
```

- [x] 生成 baseline：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_baseline_yfixed_20260610/part_label_bank.npz
```

### Task 4: 实现多视角一致性评估

**Files:**

- Create: `tools/analyze_multiview_semantic_consistency.py`
- Create: `tests/test_multiview_semantic_consistency.py`

- [x] Step 1: 定义 view-level projection 输入格式。
- [x] Step 2: 复用 y-fixed `_project_points()` / `_load_record_masks()` / projection leakage row。
- [x] Step 3: 计算 view-to-view mean/std/CV。
- [x] Step 4: 对 hard/soft 两种选择分别输出结果。
- [x] Step 5: 保存 per-view CSV、per-part CSV 和 summary JSON。

当前状态（2026-06-10）：

- 已完成第一版实现：
  - `tools/analyze_multiview_semantic_consistency.py`
  - `tests/test_multiview_semantic_consistency.py`
- 已在 v395 softedit bundle 上跑 30-view 评估：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/multiview_semantic_consistency_v395_yfixed_threshold020
```

关键 summary：

```text
soft_threshold = 0.20
view_count = 30
part_count = 6
mean_leakage_std_delta_soft_minus_hard = -0.1767
mean_boundary_leakage_std_delta_soft_minus_hard = -0.00335
mean_target_activation_cv_delta_soft_minus_hard = +0.00294
```

解读：

- soft selection 的跨视角 leakage ratio 波动总体低于 hard，主要由 face/hair/skin 的改善贡献。
- boundary leakage std 也平均略低，但幅度较小。
- lower/shoes/upper 仍有反向趋势，说明 soft 权重还不能直接宣称所有部件都更稳定。
- 论文表述建议写成“soft/reliability improves boundary-aware stability for ambiguous regions, while lower/shoes require part-specific tuning”，不要写成全局绝对优势。

### Task 5: 实现动态一致性评估

**Files:**

- Create: `tools/analyze_dynamic_semantic_consistency.py`
- Create: `tests/test_dynamic_semantic_consistency.py`

- [ ] Step 1: 定义 frame-level 输入格式。
- [ ] Step 2: 统计 part count、soft weight、selected set 的时间变化。
- [ ] Step 3: 计算 temporal selection Jaccard。
- [ ] Step 4: 计算 worst-frame leakage。
- [ ] Step 5: 输出论文可用 CSV。

### Task 6: 生成论文实验总表

**Files:**

- Create: `tools/make_semantic_editing_paper_tables.py`

- [ ] Step 1: 读取 Task 2-5 的 summary JSON。
- [ ] Step 2: 合并成 `paper_tables/*.csv`。
- [ ] Step 3: 输出一份 `paper_tables/README.md` 说明每列含义。
- [ ] Step 4: 记录所有实验使用的 bundle/checkpoint/command。

## 6. 推荐实验路径

推荐所有论文闭环实验放在：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_YYYYMMDD
```

推荐当前起点 bundle：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_reliability_neighborfill_20260609
```

推荐不要把下面这版作为主线：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/render_quality_refine_boundary_color_v1_20260610_export
```

原因：它 RGB 指标略好，但边缘 hard score 和 halo 没有更稳，不适合作为语义编辑论文主版本。

## 7. 投稿前最低完成标准

投稿前至少要满足：

- [x] soft edit weight 已导出并可被 Editor/评估脚本读取。
- [x] hard label edit vs soft edit 有 per-Gaussian 初版定量结果。
- [x] trained semantic asset vs 2D voting 有第一版定量结果。
- [x] hard label edit vs soft edit 有 projection-space / boundary-space 定量结果。
- [x] face/hair/skin 和 upper/lower/skin 有第一版边界泄露分析。
- [x] 多视角一致性有 summary。
- [x] 识别质量有第一版可视化 QA sheet。
- [ ] 动态一致性有 summary。
- [ ] 至少 3 组可视化图：
  - label bank 可视化
  - edit result 可视化
  - temporal consistency 可视化
- [ ] 所有实验命令、bundle 路径、checkpoint 路径可复现。

## 8. 可以放到 Future Work 的内容

以下内容有价值，但不是第一篇必须完成：

- 交互式 edit recipe 反馈训练。
- 人工修正 label 后重新训练 semantic adapter。
- 更复杂的 boundary-aware semantic loss。
- 面向更多人物的大规模泛化实验。
- RGB 渲染质量 SOTA 对比。
- 复杂物理/几何编辑。

## 9. 当前最建议立刻执行的下一步

截至 2026-06-10，Task 1 / Task 2 / Task 2b / Task 3 / Task 4 的工程闭环已经完成第一版。当前额外补了一轮识别可视化 QA。当前最建议马上做：

```text
Task 6 的前半部分：把已有结果汇总成 paper tables
Task 5: 实现动态一致性评估
```

原因：

- 现在 per-Gaussian leakage 已经证明“指标链路能跑”，但不能作为最终论文主指标。
- 2D voting baseline 的投影 Y 轴错位已经修正，旧 baseline 结果只作为 debug 记录，不应进入论文表格。
- projection-space leakage 已经可用，但当前结果显示 soft selection 的总体 leakage ratio 优势不明显，边界泄露略优。
- 多视角一致性第一版已经补上：soft 的 leakage std 平均更低，但仍有部件反向趋势。
- 已有结果分散在多个 summary/per_part CSV 中，需要尽快生成统一 paper tables，防止后续混用旧 baseline。

推荐执行顺序：

1. 实现 `tools/make_semantic_editing_paper_tables.py` 的最小版，先汇总 soft-edit、label-bank ablation、projection leakage sweep、multiview consistency。
2. 生成 label bank / soft weight / leakage heatmap 三类可视化。
3. 实现 `tools/analyze_dynamic_semantic_consistency.py`，补动态帧 summary。
4. 根据 paper tables 判断是否需要为 lower/shoes 做 part-specific soft threshold 或权重公式消融。

## 10. 识别准确性 QA 与优化判断

当前已经新增识别质量可视化 QA 工具：

```text
tools/make_semantic_recognition_qa_sheet.py
tests/test_semantic_recognition_qa_sheet.py
```

已生成总览 QA sheet：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020/semantic_recognition_qa_sheet.png
```

已生成分部件 QA sheet：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_hair/semantic_recognition_qa_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_face/semantic_recognition_qa_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_upper/semantic_recognition_qa_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_lower/semantic_recognition_qa_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_shoes/semantic_recognition_qa_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/semantic_recognition_qa_v395_threshold020_skin/semantic_recognition_qa_sheet.png
```

4-view QA 外泄统计：

```text
part   hard_outer  soft_outer  soft_minus_hard
hair   26453.0     25375.2     -1077.8
face   2687.0      2014.2      -672.8
upper  65154.0     54642.7     -10511.3
lower  84102.0     54791.7     -29310.3
shoes  35160.0     33997.5     -1162.5
skin   27528.0     17017.4     -10510.6
```

视觉结论：

- 当前识别不是完全失败；目标区域中心基本能对上。
- `upper/lower` 投影选区外扩明显，尤其 lower 的 Gaussian 点云散到短裤/腿 mask 外；soft 主要降低强度，但没有完全收紧空间范围。
- `skin` 相对合理，soft 明显减少腿部/衣服污染，但边界仍有红色外泄点。
- `face/hair` 需要继续看边界局部，但当前优先级低于 lower/upper 的空间外扩。

下一步优化建议：

1. 不要立即重训 parser 或 semantic adapter。
2. 先做 **part-specific soft threshold / margin_power 消融**：
   - hair / face / skin 可以提高 threshold。
   - upper 可以小幅提高 threshold，但收益有限。
   - lower / shoes 不应靠提高 threshold 修，因为 sweep 显示阈值升高没有改善，甚至变差。
3. QA 可视化里 hard/soft 当前使用 Gaussian 中心点 rasterization，下一版应加入 footprint/radius-aware 视图，避免把“点云散点显示”误判成最终 splat footprint。

已完成 part-specific threshold sweep：

```text
tools/analyze_part_specific_soft_thresholds.py
tests/test_part_specific_soft_thresholds.py
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/part_specific_soft_threshold_sweep_v395_yfixed
```

推荐阈值（约束 target retention >= 0.5）：

```text
hair  -> 0.70  leakage=0.2731  retention=0.924
face  -> 0.70  leakage=0.4075  retention=0.864
upper -> 0.50  leakage=0.7625  retention=0.954
lower -> 0.10  leakage=0.4624  retention=1.000
shoes -> 0.10  leakage=0.4953  retention=1.000
skin  -> 0.70  leakage=0.2655  retention=0.855
```

排查结论：

- hair / face / skin：提高 soft threshold 是有效修复方向，能降低 leakage 且 coverage 仍保留较多。
- upper：阈值提高到 0.50 只有很小收益，说明问题不是单纯阈值太低。
- lower / shoes：阈值提高无效。lower 在 `0.10` 最好，0.25 之后 target retention 大幅下降且 leakage 变差；shoes 几乎全程变差。这说明 lower/shoes 的外扩不是 soft threshold 问题，更可能来自：
  - 2D part mask 与 Gaussian 中心投影 footprint 不匹配；
  - lower/shoes label 本身覆盖了会投到 mask 外的 3D 点；
  - 当前点中心 rasterization 夸大了稀疏/边界 splat 的外泄。

下一步修复优先级：

1. 对 hair / face / skin 做 part-specific threshold 参数化。
2. 对 lower / shoes 不调高阈值，先做 footprint-aware QA / leakage metric。
3. 如果 footprint-aware 后 lower/shoes 仍外扩，再考虑 lower/shoes 的可靠 mask 或 geometry-aware guard。

为避免过拟合当前人物，已新增 subject-adaptive calibration 第一步：

```text
tools/calibrate_subject_semantic_editing.py
tests/test_calibrate_subject_semantic_editing.py
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260610/subject_adaptive_calibration_v395
```

该工具不把 v395 阈值写死为全局规则，而是读取每个 subject 的 threshold sweep，自动分类：

```text
threshold_fixable      阈值提升后 leakage 明显下降且 target retention 足够
weak_threshold_fixable 阈值提升有弱收益，需要视觉复查
not_threshold_fixable  阈值提升无效，转入 footprint-aware / geometry 排查
```

v395 当前校准输出：

```text
face  threshold_fixable       threshold=0.70  next=use_threshold
skin  threshold_fixable       threshold=0.70  next=use_threshold
hair  weak_threshold_fixable  threshold=0.70  next=use_threshold_and_visual_review
upper weak_threshold_fixable  threshold=0.50  next=use_threshold_and_visual_review
lower not_threshold_fixable   threshold=0.10  next=footprint_aware_required
shoes not_threshold_fixable   threshold=0.10  next=footprint_aware_required
```

泛化策略：

- 新人物不复用 v395 的阈值。
- 新人物必须先跑 `analyze_part_specific_soft_thresholds.py`，再跑 `calibrate_subject_semantic_editing.py`。
- 论文/系统默认使用“自动校准流程”，而不是固定阈值表。
- 只有 `threshold_fixable / weak_threshold_fixable` 部件进入 threshold config；`not_threshold_fixable` 部件必须进入 footprint-aware 排查。

## 11. 2026-06-11 footprint-aware 识别修复进展

本轮目标不是继续手调 v395 的 lower/shoes 阈值，而是修复中心点投影对真实 Gaussian footprint 的误判。已新增：

```text
tools/analyze_footprint_aware_soft_edit_leakage.py
tests/test_footprint_aware_soft_edit_leakage.py
```

并扩展：

```text
tools/semantic_viewer/build_part_label_bank.py
tools/analyze_projected_soft_edit_leakage.py
tools/make_semantic_recognition_qa_sheet.py
utils/part_label_bank.py
```

核心变化：

- `rasterize_projected_activation()` 支持 per-Gaussian `radii` disk footprint。
- `build_part_label_bank.py --label-bank-source projected-2d-voting` 支持：
  - `--vote-footprint-mode footprint`
  - `--vote-use-render-radii`
  - `--vote-min-footprint-hit-ratio`
- voting bank 现在导出 vote-normalized `semantic_probs`，因此可以继续走 reliability / neighbor-fill / `soft_edit_weights` 流程。
- QA / leakage 工具支持 hard-only label bank 的 one-hot fallback，便于诊断中间产物。

第一步诊断输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/footprint_aware_soft_edit_leakage_v395_lower_shoes
```

诊断结论：

```text
center lower/shoes mean hard leakage = 0.4638
center lower/shoes mean soft leakage = 0.5055
footprint lower/shoes mean hard leakage = 1.0991
footprint lower/shoes mean soft leakage = 0.9493
```

解释：

- lower 的部分“外扩”确实来自中心点度量和 footprint 覆盖不一致。
- shoes 在真实 footprint 下仍有外泄，说明鞋区域还存在真实边界混杂，不应只靠 threshold 解决。

随后构建了 hard-only footprint voting label bank：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_20260611/part_label_bank.npz
```

summary：

```text
unknown_count=914
mean_confidence=0.8105
conflicted_point_count=177
lower_count=15773
shoes_count=4920
```

hard-only projection leakage：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_leakage_v395_footprint_voting_lower_shoes
```

结果：

```text
lower hard leakage 0.1372
shoes hard leakage 0.4038
mean hard leakage 0.2705
```

对比旧 y-fixed center voting 近似基线：

```text
lower hard leakage 0.4085 -> 0.1372
shoes hard leakage 0.5191 -> 0.4038
```

这说明 footprint-aware voting 是有效的通用修复，不是针对当前人物的阈值过拟合。

最后构建了可进入 soft-edit 流程的完整候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_20260611/part_label_bank.npz
```

构建参数要点：

```text
--label-bank-source projected-2d-voting
--vote-footprint-mode footprint
--vote-use-render-radii
--vote-min-footprint-hit-ratio 0.50
--reliability-enable
--neighbor-fill-enable
--export-soft-edit-weights
```

soft-edit projection leakage：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_footprint_voting_softedit_lower_shoes
```

结果：

```text
lower hard leakage 0.0988, soft leakage 0.0630
shoes hard leakage 0.3368, soft leakage 0.2066
mean hard leakage 0.2178, mean soft leakage 0.1348
mean soft-hard delta -0.0830
```

QA 图：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/semantic_recognition_qa_v395_footprint_voting_softedit_lower_shoes/semantic_recognition_qa_sheet.png
```

视觉结论：

- lower soft 选区保留短裤/下装主体，外侧红色散点明显减少。
- shoes soft 选区仍有少量边缘外泄，但比 hard 和旧 soft 明显更集中。
- 当前更像“有效收紧”而不是单纯删空，因为主体区域仍可见。

当前注意事项：

- 完整候选的 `editable_unknown_count=9665`，明显高于原 softedit 主 bundle，需要后续用 target retention / edit render 进一步确认不会过度保守。
- `reliability_min_opacity=0.005` 对 voting bank 可能偏严；下一步建议做 reliability/soft-floor 消融，而不是直接把该候选替换为主结果。
- footprint voting 当前逐点 disk 统计，离线可用，但后续若频繁使用应做向量化或缓存优化。

下一步建议：

1. 对 footprint-voting softedit bundle 跑全 part projection leakage 和 multiview consistency。
2. 做 `reliability_min_opacity` 与 `soft_edit_reliable_floor` 小消融，确认 lower/shoes 收紧不是以过低 target retention 换来的。
3. 生成 edit render 级预览，不只看投影 QA。
4. 若全 part 指标稳定，再把 footprint-aware voting 作为新 subject 的 `not_threshold_fixable` 部件默认排查/修复路径。

## 12. 2026-06-11 footprint voting 候选验证结果

执行计划：

```text
/root/.config/superpowers/worktrees/3dgs-avatar-release-main/semantic-edit-soft-weights/docs/superpowers/plans/2026-06-11-footprint-voting-validation.md
```

### 12.1 全部件 projection leakage

候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_20260611/part_label_bank.npz
```

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_footprint_voting_softedit_all_parts
```

summary：

```text
mean_hard_leakage_ratio = 0.2608
mean_soft_leakage_ratio = 0.1507
mean_soft-hard_delta    = -0.1101
mean_hard_boundary      = 0.1944
mean_soft_boundary      = 0.1638
```

per-part soft leakage：

```text
hair  0.0276
face  0.2258
upper 0.2580
lower 0.0630
shoes 0.2066
skin  0.1234
```

结论：全部件 soft leakage 都低于 hard，没有发现明显被 footprint voting 带坏的部件。

### 12.2 多视角一致性

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/multiview_semantic_consistency_v395_footprint_voting_softedit_threshold020
```

对比旧 y-fixed softedit：

```text
旧 baseline mean_leakage_std_delta_soft_minus_hard = -0.1767
新 footprint mean_leakage_std_delta_soft_minus_hard = +0.5707

旧 baseline mean_boundary_std_delta = -0.00335
新 footprint mean_boundary_std_delta = +0.01646
```

解释：

- 新候选的 mean leakage 明显更低，但 leakage ratio 的跨视角 std 变差。
- 主要风险来自 face：soft target activation 较小，ratio 对少量外泄更敏感。
- 这说明原始 footprint softedit 不宜直接作为最终主结果，需要 soft floor / reliability 消融。

### 12.3 消融：soft_edit_reliable_floor=0.25

候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_floor025_20260611/part_label_bank.npz
```

summary：

```text
editable_unknown_count = 9665
mean_weight            = 0.0463
```

projection leakage：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_footprint_voting_softedit_floor025_all_parts

mean_hard_leakage_ratio = 0.2608
mean_soft_leakage_ratio = 0.1462
mean_soft-hard_delta    = -0.1147
```

per-part soft leakage：

```text
hair  0.0258
face  0.2270
upper 0.2495
lower 0.0503
shoes 0.2019
skin  0.1226
```

multiview：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/multiview_semantic_consistency_v395_footprint_voting_softedit_floor025_threshold020

mean_leakage_std_delta_soft_minus_hard = +0.5419
mean_boundary_std_delta                = +0.0164
```

结论：floor=0.25 比原候选略降低 mean leakage，并增加 soft target activation；但多视角 std 仍未恢复到旧 baseline。视觉 QA 更干净，是目前更平衡的 footprint-voting 候选。

QA：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/semantic_recognition_qa_v395_footprint_voting_softedit_floor025_all_parts/semantic_recognition_qa_sheet.png
```

### 12.4 消融：reliability_min_opacity=0.0

候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_noopacity_20260611/part_label_bank.npz
```

summary：

```text
editable_unknown_count = 6315
mean_weight            = 0.0915
```

projection leakage：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_footprint_voting_softedit_noopacity_all_parts

mean_hard_leakage_ratio = 0.2729
mean_soft_leakage_ratio = 0.1649
mean_soft-hard_delta    = -0.1081
```

multiview：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/multiview_semantic_consistency_v395_footprint_voting_softedit_noopacity_threshold020

mean_leakage_std_delta_soft_minus_hard = +0.3833
mean_boundary_std_delta                = +0.0205
```

结论：no-opacity 明显提高 coverage，并降低 leakage std delta，但 mean leakage 和视觉外泄都比 floor025 更高。它更适合作为 coverage 上限参考，不宜直接提升为主候选。

QA：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/semantic_recognition_qa_v395_footprint_voting_softedit_noopacity_all_parts/semantic_recognition_qa_sheet.png
```

### 12.5 当前接受决策

当前不建议把原始 footprint-voting softedit 直接替换主 bundle。

更合理的当前决策：

- 保留原主 bundle 作为稳定 baseline。
- 将 `footprint_radii_softedit_floor025` 作为 lower/shoes 修复方向的第一候选。
- 将 `noopacity` 作为 coverage 上限参考。
- 下一步必须进入 edit-render 级预览，而不能只依赖 projection QA。

原因：

- footprint voting 确实显著改善 lower/shoes 和全 part mean leakage。
- floor025 在 mean leakage 与视觉干净程度上最好。
- 但多视角 leakage std 仍不如旧 baseline，尤其 face ratio 波动提示 soft target activation 仍偏保守。

下一步：

1. 生成 floor025 的实际 edit render 预览。
2. 对 face 单独看 target activation/coverage，不急着改 lower/shoes。
3. 若 edit render 视觉可接受，再把 floor025 作为论文候选表格的一列，而不是直接覆盖主结果。

## 13. 2026-06-11 实际 edit render 预览

为避免只依赖 projection QA，已新增 renderer-level 语义编辑预览工具：

```text
tools/make_semantic_edit_render_preview.py
tests/test_semantic_edit_render_preview.py
```

工具逻辑：

- 读取 `part_label_bank.npz` 的 `editable_label` / `soft_edit_weights`。
- 对指定 part 生成 hard / soft 两组选择权重。
- 在 `scene.convert_gaussians()` 后直接混入目标颜色，再调用 `gaussian_renderer.rasterize_gaussians()` 渲染。
- 输出 RGB / Hard edit / Soft edit 的真实 3DGS renderer 预览，而不是 2D mask recolor。

### 13.1 floor025 候选 lower/shoes 预览

候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_floor025_20260611/part_label_bank.npz
```

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_footprint_floor025_lower_shoes/semantic_edit_render_preview_sheet.png
```

统计：

```text
lower hard_selected=13825
lower soft_selected=10348
lower soft_weight_sum=4922.8
lower soft_weight_mean_selected=0.4757

shoes hard_selected=4161
shoes soft_selected=2630
shoes soft_weight_sum=1269.4
shoes soft_weight_mean_selected=0.4826
```

视觉结论：

- lower soft edit 能清楚改变短裤/下装主体，没有删空。
- shoes soft edit 能清楚改变鞋主体，范围比 hard 更收敛。
- 真实 render 中没有看到明显大面积污染上衣、皮肤或背景。

### 13.2 原主 bundle lower/shoes 对照

对照：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_reliability_neighborfill_softedit_20260610/part_label_bank.npz
```

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_main_softedit_lower_shoes/semantic_edit_render_preview_sheet.png
```

统计：

```text
lower hard_selected=17859
lower soft_selected=15560
lower soft_weight_sum=8045.6
lower soft_weight_mean_selected=0.5171

shoes hard_selected=5167
shoes soft_selected=5022
shoes soft_weight_sum=4506.3
shoes soft_weight_mean_selected=0.8973
```

对照结论：

- 原主 bundle 的 lower/shoes 编辑也可见，但 shoes soft 几乎覆盖 hard 选择，且权重很高。
- floor025 的 shoes soft 更保守、更集中，和 projection leakage 降低一致。
- floor025 在真实 render 中没有表现为“删空”，因此可以作为 lower/shoes 修复候选继续进入论文表格/更多视角预览。

当前建议：

- `footprint_radii_softedit_floor025` 可以作为 lower/shoes 识别修复的候选结果保留。
- 仍不建议直接覆盖原主 bundle，因为多视角 leakage std 指标尚未优于旧 baseline。
- 下一步优先做：
  1. floor025 的更多视角/更多动作帧 edit preview；
  2. face 的 target activation / ratio 波动排查；
  3. 将原主 bundle、floor025、noopacity 进入 paper table 汇总。

## 14. 2026-06-11 Hybrid lower/shoes soft-channel 修复

针对“全量 footprint floor025 会改善 lower/shoes，但 face 等部位多视角 std 变差”的问题，本轮没有继续全局替换主 bundle，而是新增 hybrid 构建工具：

```text
tools/build_hybrid_part_label_bank.py
tests/test_hybrid_part_label_bank.py
```

主仓库同步：

- `utils/part_label_bank.py` 已支持保存/校验 `soft_edit_weights`。
- 单元测试覆盖：只替换指定 part 的 soft channel；`part_label` / `editable_label` / `semantic_probs` 默认保持 base。

### 14.1 构建规则

base 使用稳定主 bundle：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_reliability_neighborfill_softedit_20260610/part_label_bank.npz
```

override 使用 floor025 footprint 候选：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_2d_voting_footprint_radii_softedit_floor025_20260611/part_label_bank.npz
```

hybrid 输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_softchannels_20260611/part_label_bank.npz
```

具体策略：

- 只替换 `soft_edit_weights[:, lower]` 和 `soft_edit_weights[:, shoes]`。
- `hair/face/upper/skin` 的 soft 权重保留原主 bundle。
- `part_label`、`editable_label`、`semantic_probs`、reliability 字段保留原主 bundle。
- 这不是针对单个人体形状写死规则，而是一个通用“base 稳定包 + 指定部位 soft channel override”的组合工具；换人物时只需要重新生成对应人物的 base/override bank。

权重变化：

```text
lower base_weight_sum    = 8764.6
lower override_weight_sum= 5446.4
lower mean_abs_delta     = 0.1499

shoes base_weight_sum    = 4521.7
shoes override_weight_sum= 1439.5
shoes mean_abs_delta     = 0.0739
```

### 14.2 Projection leakage 结果

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_hybrid_main_floor025_lower_shoes_all_parts
```

summary：

```text
mean_hard_leakage_ratio = 0.4657
mean_soft_leakage_ratio = 0.3441
mean_soft-hard_delta    = -0.1215

mean_hard_boundary_leakage_ratio = 0.2242
mean_soft_boundary_leakage_ratio = 0.1926
mean_boundary_delta              = -0.0316
```

per-part soft leakage：

```text
hair  0.2855
face  0.4487
upper 0.7626
lower 0.0488
shoes 0.2003
skin  0.3190
```

关键变化：

- lower soft leakage 从 hard 的 `0.3475` 降到 `0.0488`。
- shoes soft leakage 从 hard 的 `0.4692` 降到 `0.2003`。
- upper 仍是主要短板，hybrid 本轮没有试图修 upper。

### 14.3 多视角一致性结果

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/multiview_semantic_consistency_v395_hybrid_main_floor025_lower_shoes_threshold020
```

summary：

```text
mean_leakage_std_delta_soft_minus_hard       = -0.2053
mean_boundary_leakage_std_delta_soft_minus_hard = -0.0060
mean_target_activation_cv_delta_soft_minus_hard = -0.0072
```

对比前一轮：

- 全量 floor025 的 `mean_leakage_std_delta_soft_minus_hard = +0.5419`，不适合直接替换主包。
- hybrid 变为 `-0.2053`，说明保留 face/hair/upper/skin 主包 soft 通道后，多视角稳定性恢复。
- lower/shoes 自身也改善：lower std delta `-0.0227`，shoes std delta `-0.0707`。

### 14.4 实际 edit render 预览

输出：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_main_floor025_lower_shoes/semantic_edit_render_preview_sheet.png
```

统计：

```text
lower hard_selected=17859
lower soft_selected=10348
lower soft_weight_sum=4922.8
lower soft_weight_mean_selected=0.4757

shoes hard_selected=5167
shoes soft_selected=2630
shoes soft_weight_sum=1269.4
shoes soft_weight_mean_selected=0.4826
```

视觉检查：

- lower soft edit 没有删空，下装主体清楚；但在背面/肩背附近仍有少量边界污染，说明 lower 还不是最终论文级完美结果。
- shoes soft edit 明显比 hard 更收敛，鞋主体清楚，腿部污染比原主 bundle 小。
- 当前 hybrid 更适合作为“lower/shoes 修复候选”，而不是宣称全身语义识别已经全部解决。

### 14.5 当前判断与下一步

当前效果：

- 完成度：lower/shoes 识别修复约 `70%`，已经从“可疑候选”推进到“可进入更多视角验证的候选”。
- 稳定性：hybrid 比全量 floor025 更可靠，因为没有让 face/hair/upper/skin 跟着 footprint 候选一起波动。
- 泛化性：修复方式是按 bundle 通道组合，不依赖当前人物的固定空间阈值或手写身体位置，因此比前面的单主体空间 guard 更不容易过拟合。

仍需修复：

1. lower 在背面视角仍有上衣/肩背边界污染，需要再做 lower/upper 分界的 render-level QA 或更细粒度 threshold。
2. upper soft leakage 仍高，不能把 hybrid 当作全 part 最终结果。
3. 下一步应做 8-12 个视角/动作帧的 hybrid edit render sheet，并把原主 bundle、floor025、hybrid 三列放进统一 paper table。

当前建议：

- 将 `viewer_bundle_v395_hybrid_main_floor025_lower_shoes_softchannels_20260611` 作为下一轮 lower/shoes 主候选。
- 暂不替换所有部位的主 bundle。
- 后续优先优化 upper/lower 边界，而不是继续改 face 或全局 soft 权重。

## 15. 2026-06-11 Worst-view render QA 与阈值诊断

本轮目标不是继续修改 bank，而是按 systematic debugging 先确认：

- upper 高 leakage 是否是真实 render 污染。
- lower 背面边界污染是否能通过更高 soft threshold 缓解。
- shoes 在最差姿态下是否仍保留主体。

### 15.1 Worst-view QA

根据 per-view leakage，选取 upper/lower/shoes 最差视角生成 renderer-level QA：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_worst_views_upper_lower_shoes_threshold020/semantic_edit_render_preview_sheet.png
```

视角集合：

```text
upper: c23_f000540, c21_f000060, c22_f000480, c21_f000240
lower: c21_f000300, c22_f000300, c23_f000300, c23_f000360
shoes: c22_f000240, c21_f000240, c23_f000240
```

统计：

```text
upper hard_selected=12476
upper soft_selected=11899
upper soft_weight_sum=8684.6
upper soft_weight_mean_selected=0.7299

lower hard_selected=17859
lower soft_selected=10348
lower soft_weight_sum=4922.8
lower soft_weight_mean_selected=0.4757

shoes hard_selected=5167
shoes soft_selected=2630
shoes soft_weight_sum=1269.4
shoes soft_weight_mean_selected=0.4826
```

视觉观察：

- upper 在 worst-view sheet 中蓝色编辑肉眼看起来并没有大面积污染，projection leakage 高值很可能部分来自 2D mask/遮挡/target activation 偏低造成的 ratio 放大。
- lower 在 worst-view 中主体完整，但 soft threshold=0.20/0.25 时仍可见少量上衣/肩背边界染色。
- shoes 在最差姿态中主体仍清楚，污染主要体现在 ratio 指标对鞋子小目标更敏感。

### 15.2 Lower/shoes threshold sweep

对 lower/shoes 最差视角生成 threshold sweep：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_lower_shoes_worst_threshold025/semantic_edit_render_preview_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_lower_shoes_worst_threshold030/semantic_edit_render_preview_sheet.png
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_lower_shoes_worst_threshold035/semantic_edit_render_preview_sheet.png
```

统计对比：

```text
threshold=0.25
lower soft_selected=10182, soft_weight_sum=4887.7, mean=0.4800
shoes soft_selected=2573,  soft_weight_sum=1257.1, mean=0.4886

threshold=0.30
lower soft_selected=3660, soft_weight_sum=3255.1, mean=0.8894
shoes soft_selected=1166, soft_weight_sum=903.6,  mean=0.7750

threshold=0.35
lower soft_selected=3593, soft_weight_sum=3233.6, mean=0.9000
shoes soft_selected=1135, soft_weight_sum=893.5,  mean=0.7873
```

结论：

- `0.25` 与 `0.20` 视觉和选择量接近，边界污染没有明显减少。
- `0.30` 会把 lower/shoes soft 选择明显收缩，但真实 render 中主体仍保留，没有删空。
- `0.35` 与 `0.30` 差异很小，收益不明显。

当前建议：

- lower/shoes 的实际编辑预览与论文图优先使用 `soft_threshold=0.30`。
- bank 本身暂不重建；先把 `0.30` 作为 part-specific render/edit threshold 验证更多视角。
- upper 暂时不急着修 bank，因为 worst-view render 未显示明显大面积污染；下一步需要补 upper 的 mask alignment/target activation 诊断，而不是直接提高阈值。

下一步：

1. 用 `soft_threshold=0.30` 生成 8-12 视角 lower/shoes 正式预览。
2. 单独为 upper 做 projection mask overlay 或 target activation 诊断，确认高 leakage ratio 是否主要来自 mask/遮挡。
3. 若 `0.30` 在更多视角稳定，再将 hybrid + part-specific threshold 写入 paper table。

## 16. 2026-06-11 下一轮识别准确度提升计划：多视角泄露反向压制

用户追问：仅用 `soft_threshold=0.30` 生成更多预览，本质上是提高编辑干净度/展示稳定性，不是提高底层识别准确度。下一轮若目标是继续提高识别准确度，应从“调阈值”切换到“利用多视角错误投影证据修 soft weights”。

本节按 systematic debugging 的思路记录下一轮要做什么，供新对话直接接续执行。

### 16.1 当前根因判断

已有证据：

- hybrid lower/shoes 比原主包更干净，但 lower 背面/肩背边界仍有轻微染色。
- shoes 在 `f000240` 附近 ratio 偏高，主要因为 shoes 是小目标，target activation 小，少量错误投影会放大 leakage ratio。
- `soft_threshold=0.30` 能明显收缩 lower/shoes 选择点，且 render 主体仍保留；但这只是过滤低权重点，不改变 bank。
- upper projection leakage 高，但 worst-view render 没看到明显大面积污染，暂时不应先动 upper bank。

核心假设：

```text
lower/shoes 的剩余污染来自少量 soft weight 较高、但多视角稳定投到非目标 mask 的 3D Gaussians。
如果能在 per-Gaussian 层面识别这些稳定泄露点，并只压低对应 part 的 soft weight，
就能提高 lower/shoes 的识别/选择准确度，而不是只靠更高 threshold 临时过滤。
```

### 16.2 下一轮目标

构建一个新的 suppressed hybrid bank：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_leak_suppressed_20260611/part_label_bank.npz
```

它以当前 hybrid 为输入：

```text
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_softchannels_20260611/part_label_bank.npz
```

只修改：

```text
soft_edit_weights[:, lower]
soft_edit_weights[:, shoes]
```

不修改：

```text
part_label
editable_label
semantic_probs
hair/face/upper/skin soft channels
```

这样可以保持 hybrid 的稳定性，同时进一步压低 lower/shoes 的稳定泄露点。

### 16.3 拟新增工具

建议新增：

```text
tools/suppress_projected_soft_leakage.py
tests/test_suppress_projected_soft_leakage.py
```

工具输入：

```text
--part-label-bank 当前 hybrid bank
--checkpoint ckpt139910.pth
--asset-root semantic_editable_assets
--output 输出 suppressed bank
--parts lower shoes
--soft-threshold 0.20 或 0.25
--min-observed-views 5
--min-outer-views 3
--max-target-hit-ratio 0.35
--min-outer-hit-ratio 0.55
--suppress-factor 0.25
--boundary-cap 0.30
--summary-json
--per-point-csv
```

工具逻辑：

1. 读取 bank 的 `soft_edit_weights`。
2. 对每个 view 投影所有 Gaussians，读取对应 part 的 2D mask / foreground / valid mask。
3. 对每个 part、每个 Gaussian 统计：

```text
observed_view_count
target_hit_count
outer_hit_count
boundary_hit_count
target_weight_sum
outer_weight_sum
boundary_weight_sum
```

4. 计算 per-point 诊断量：

```text
target_hit_ratio = target_hit_count / observed_view_count
outer_hit_ratio  = outer_hit_count  / observed_view_count
stable_leak_score = outer_hit_ratio - target_hit_ratio
```

5. 只压制“多视角稳定外泄”的点：

```text
severe_leak =
  soft_weight >= soft_threshold
  observed_view_count >= min_observed_views
  outer_hit_count >= min_outer_views
  outer_hit_ratio >= min_outer_hit_ratio
  target_hit_ratio <= max_target_hit_ratio
```

6. 更新权重：

```text
if severe_leak:
    new_weight = old_weight * suppress_factor
elif boundary_dominated:
    new_weight = min(old_weight, boundary_cap)
else:
    keep old_weight
```

注意：

- 只降权，不升权，避免引入新污染。
- 对边界点用 cap 而不是清零，避免 lower/shoes 主体被削空。
- 对只在 1-2 个视角异常的点不处理，避免过拟合 mask 偶发错误。

### 16.4 TDD 验收点

先写单元测试再实现：

1. `compute_point_leakage_stats()` 能正确累计 target/outer/boundary hit。
2. `classify_suppression_mask()` 只标记多视角稳定 outer、高 target 低的点。
3. `apply_soft_weight_suppression()` 只修改指定 part channel。
4. 保存的新 bank 保留 base 的 `part_label` / `editable_label` / 非目标 soft channels。
5. point count / part names 不一致时必须报错。

### 16.5 评估流程

生成 suppressed bank 后跑三类评估：

projection leakage：

```text
tools/analyze_projected_soft_edit_leakage.py
输出到：
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/projected_soft_edit_leakage_v395_hybrid_leak_suppressed_lower_shoes
```

multiview consistency：

```text
tools/analyze_multiview_semantic_consistency.py
输出到：
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/multiview_semantic_consistency_v395_hybrid_leak_suppressed_lower_shoes_threshold020
```

render preview：

```text
tools/make_semantic_edit_render_preview.py
输出到：
/remote-home/ming/3dgs-avatar-release-main/exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_leak_suppressed_lower_shoes
```

重点对比：

```text
当前 hybrid lower soft leakage = 0.0488
当前 hybrid shoes soft leakage = 0.2003
当前 hybrid lower/shoes threshold=0.30 视觉较干净，但 bank 未变
```

期望：

- lower leakage 不反弹，最好继续下降或保持接近。
- shoes worst-view leakage 下降，尤其 `f000240` 相关视角。
- render preview 不删空 lower/shoes 主体。
- 多视角 std 不劣化。

### 16.6 风险与止损规则

风险：

- 2D mask 本身有错，直接根据 outer 投影降权可能误伤真实目标。
- shoes 是小目标，过强 suppress 会让鞋编辑变弱。
- lower 边界点可能是真实衣服边界，不应全部清零。

止损规则：

- 如果 lower/shoes render 主体明显变薄或断裂，回退 suppress 参数。
- 如果 projection leakage 降低但 render 更差，以 render 为准。
- 如果 suppression 需要大量硬编码空间规则，停止，改回数据驱动的投影证据规则。
- 若连续三版 suppression 都出现新污染或主体削空，应重新审视 footprint voting / mask alignment，而不是继续叠规则。

### 16.7 2026-06-11 复查结论：旧 suppressed 不是可信有效修复

本轮按第 16 节流程恢复并校准了 evaluation 工具，重新检查 projection / multiview / render preview。关键发现是：最早生成的

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_leak_suppressed_20260611/part_label_bank.npz
```

不应作为论文候选。它是在投影 Y 轴约定修正前生成的 suppressed bank，虽然用修正后的 evaluator 评估时 projection leakage 看起来下降，但实际是旧错位投影证据造成的大规模误压。

旧 suppressed 的异常信号：

```text
lower changed_count = 4322
shoes changed_count = 2579
shoes target_hit_ratio mean = 0.0
shoes outer_hit_ratio mean = 1.0
```

render preview 也验证了过抑制：

```text
hybrid baseline shoes soft_selected_count = 2630
old suppressed shoes soft_selected_count = 608
hybrid baseline shoes soft_weight_sum = 1269.35
old suppressed shoes soft_weight_sum = 149.46
```

因此对用户问题“这次修复没效果吗”的回答是：

```text
不是完全没效果，而是旧结果的下降主要来自错误投影证据导致的过抑制；
按 y-fixed 可信口径重新生成后，suppression 只有微弱改善，不足以作为有效修复主线。
```

已重新用 y-fixed 投影证据生成可信 suppressed bank：

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_leak_suppressed_yfixed_20260611/part_label_bank.npz
```

重生成摘要：

```text
lower changed_count = 10
shoes changed_count = 33
total_changed_count = 43
total_removed_weight_sum = 8.6891
```

公平 projection 对比使用同一 `soft_threshold=0.20`、同一 lower/shoes 部件集合：

```text
baseline:
  lower soft leakage = 0.0503225681
  shoes soft leakage = 0.2019229909
  mean soft leakage  = 0.1261227876

y-fixed suppressed:
  lower soft leakage = 0.0501450325
  shoes soft leakage = 0.1970070933
  mean soft leakage  = 0.1235760599
```

multiview consistency：

```text
y-fixed suppressed mean_leakage_std_delta_soft_minus_hard = -0.0470266930
y-fixed suppressed mean_boundary_leakage_std_delta_soft_minus_hard = -0.0054441465
```

render preview：

```text
exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_leak_suppressed_yfixed_lower_shoes/semantic_edit_render_preview_sheet.png

y-fixed lower soft_selected_count = 10338
y-fixed shoes soft_selected_count = 2597
y-fixed lower soft_weight_sum = 4920.31
y-fixed shoes soft_weight_sum = 1260.26
```

视觉结论：y-fixed suppressed 与 hybrid baseline 基本一致，主体没有削空，但改善幅度很小。

下一步建议：

1. 不再沿用旧 `leak_suppressed_20260611` 作为候选或论文结果。
2. `leak_suppressed_yfixed_20260611` 可作为 sanity-check 产物，但不值得提升为主候选。
3. 第 16 节这种中心点 projection 反向压制已经到达收益上限；继续调 `suppress_factor / outer_ratio / target_ratio` 大概率只会在“没效果”和“误杀鞋主体”之间摆动。
4. 若继续提高 lower/shoes，应回到 footprint/mask alignment：
   - 对 shoes 做 mask dilation / footprint hit 诊断，而不是中心点 outer hit。
   - 对 lower 保留 hybrid floor025 主线，优先扩大 worst-view render QA。
   - 若要做新工具，应把 suppression evidence 从 center-hit 改成 footprint-overlap hit，并加入 per-part target retention guard。

### 16.8 2026-06-11 evidence-calibrated soft weight 修复

针对上面的结论，本轮没有继续做 suppression 参数微调，而是修复了代码链路和权重来源逻辑：

1. 把 soft-edit / footprint-voting 的可复现入口补回主仓：

```text
utils/part_label_bank.py
tools/semantic_viewer/build_part_label_bank.py
```

新增/恢复能力：

```text
finalize_votes() 导出 vote-normalized semantic_probs
compute_soft_edit_weights()
--label-bank-source projected-2d-voting
--vote-footprint-mode footprint
--vote-use-render-radii
--export-soft-edit-weights
```

2. 新增 evidence-calibrated 权重校准：

```text
tools/calibrate_evidence_soft_edit_weights.py
tests/test_evidence_calibrated_semantic_bank.py
```

核心逻辑：

```text
先计算 per-Gaussian / per-part footprint_target_ratio
再计算 footprint_outer_ratio / view_support_count / conflict_ratio
最后用统一公式校准 soft_edit_weights
```

这与旧 suppression 的区别：

```text
旧 suppression: 发现 center projection 外泄点后乘 suppress_factor
新 calibration: 用 footprint overlap 作为证据连续调权，并用 target_retention_floor 保护真实目标主体
```

当前第一版使用 8-view evidence 生成候选：

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_evidence_calibrated_8view_20260611/part_label_bank.npz
```

生成摘要：

```text
processed_views = 8
lower calibrated_count = 41598
lower changed_count = 13607
lower old_weight_sum = 5446.43
lower new_weight_sum = 4949.67

shoes calibrated_count = 41598
shoes changed_count = 4380
shoes old_weight_sum = 1439.50
shoes new_weight_sum = 1213.55
```

Projection 对比，统一 `soft_threshold=0.20`：

```text
baseline lower soft leakage   = 0.0503225681
evidence lower soft leakage   = 0.0352662618

baseline shoes soft leakage   = 0.2019229909
evidence shoes soft leakage   = 0.1723651962

baseline lower soft target activation = 123309.73
evidence lower soft target activation = 114552.60

baseline shoes soft target activation = 21067.34
evidence shoes soft target activation = 17358.75
```

Multiview：

```text
mean_leakage_std_delta_soft_minus_hard = -0.0476542917
mean_boundary_leakage_std_delta_soft_minus_hard = -0.0025351172
mean_target_activation_cv_delta_soft_minus_hard = -0.0171893665
```

Render preview：

```text
exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_evidence_calibrated_8view_lower_shoes/semantic_edit_render_preview_sheet.png

baseline lower soft_selected = 10348
evidence lower soft_selected = 9755
baseline lower soft_weight_sum = 4922.80
evidence lower soft_weight_sum = 4493.21

baseline shoes soft_selected = 2630
evidence shoes soft_selected = 2317
baseline shoes soft_weight_sum = 1269.35
evidence shoes soft_weight_sum = 1044.23
```

视觉结论：

- lower / shoes 主体仍然可编辑，没有出现旧 suppressed 的削空。
- lower 和 shoes projection leakage 都有实质下降。
- 这版是“有证据的收紧”，比旧 center suppression 更接近要修的识别逻辑。

当前不足：

- 8-view evidence 证明方向有效，但视角采样不足，不能作为最终论文候选。
- hybrid bank 仍只保存校准后的 `soft_edit_weights`；证据矩阵已改为 sidecar 保存，后续如要进入 Editor 或论文 artifact，可再提升为正式 schema 字段。
- 仍需补 worst-view render QA，确认极端姿态/遮挡视角下没有目标主体漏选。

已执行下一步（2026-06-11）：

1. 优化 `tools/calibrate_evidence_soft_edit_weights.py`：
   - 只对目标 part 的 `soft_edit_weights >= 0.05` 或 `editable_label == part` 候选点算 footprint evidence。
   - 输出 evidence sidecar：

```text
footprint_evidence.npz
```

新增/验证接口：

```text
build_part_candidate_mask()
build_footprint_evidence_record(..., candidate_mask=...)
save_footprint_evidence_npz()
--candidate-soft-min-weight
--evidence-output
```

TDD / 回归：

```text
conda run -n ictrl python -m pytest tests/test_evidence_calibrated_semantic_bank.py -q
6 passed

conda run -n ictrl python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_hybrid_part_label_bank.py \
  tests/test_projected_soft_edit_leakage.py \
  tests/test_multiview_semantic_consistency.py \
  tests/test_semantic_edit_render_preview.py -q
43 passed
```

2. 生成 30-view optimized evidence-calibrated 候选：

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_evidence_calibrated_30view_20260611/part_label_bank.npz
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_evidence_calibrated_30view_20260611/summary.json
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_evidence_calibrated_30view_20260611/footprint_evidence.npz
```

生成摘要：

```text
processed_views = 30
candidate_counts.lower = 19898
candidate_counts.shoes = 5682

lower calibrated_count = 19558
lower changed_count = 11338
lower old_weight_sum = 5446.43
lower new_weight_sum = 5017.34
lower removed_weight_sum = 429.09

shoes calibrated_count = 5130
shoes changed_count = 4286
shoes old_weight_sum = 1439.50
shoes new_weight_sum = 1191.92
shoes removed_weight_sum = 247.58
```

3. Projection 对比，统一 `soft_threshold=0.20`：

```text
baseline lower soft leakage = 0.0503225681
8-view   lower soft leakage = 0.0352662618
30-view  lower soft leakage = 0.0360276987

baseline shoes soft leakage = 0.2019229909
8-view   shoes soft leakage = 0.1723651962
30-view  shoes soft leakage = 0.1624963636

baseline lower soft target activation = 123309.73
8-view   lower soft target activation = 114552.60
30-view  lower soft target activation = 115722.67

baseline shoes soft target activation = 21067.34
8-view   shoes soft target activation = 17358.75
30-view  shoes soft target activation = 17244.38
```

4. Multiview 对比：

```text
baseline lower soft leakage mean = 0.0512149397
8-view   lower soft leakage mean = 0.0357502861
30-view  lower soft leakage mean = 0.0365768413

baseline shoes soft leakage mean = 0.2234885955
8-view   shoes soft leakage mean = 0.1956272609
30-view  shoes soft leakage mean = 0.1881356293

8-view mean_leakage_std_delta_soft_minus_hard = -0.0476542917
30-view mean_leakage_std_delta_soft_minus_hard = -0.0396129382

8-view mean_target_activation_cv_delta_soft_minus_hard = -0.0171893665
30-view mean_target_activation_cv_delta_soft_minus_hard = -0.0160610989
```

5. Render preview 对比：

```text
baseline lower soft_selected = 10348
8-view   lower soft_selected = 9755
30-view  lower soft_selected = 9829

baseline lower soft_weight_sum = 4922.80
8-view   lower soft_weight_sum = 4493.21
30-view  lower soft_weight_sum = 4540.50

baseline shoes soft_selected = 2630
8-view   shoes soft_selected = 2317
30-view  shoes soft_selected = 2187

baseline shoes soft_weight_sum = 1269.35
8-view   shoes soft_weight_sum = 1044.23
30-view  shoes soft_weight_sum = 999.82
```

Preview sheet：

```text
exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_evidence_calibrated_30view_lower_shoes/semantic_edit_render_preview_sheet.png
```

当前结论：

- 30-view evidence calibration 是当前 lower/shoes 识别修复主候选。
- 相比 baseline，30-view lower / shoes projection leakage 都明显下降；shoes 在 30-view 下继续优于 8-view。
- 30-view lower 比 8-view 略保守，leakage 小幅回升但 target activation 和 selected count 也略回升，更像是修正 8-view 采样偏差，而不是过拟合式削权重。
- Render preview 视觉检查显示 lower / shoes 主体仍然可编辑，没有出现旧 center suppression 的削空问题。
- 后续不再沿用旧 `leak_suppressed_20260611`；论文主线建议使用 30-view evidence-calibrated soft edit，并补 worst-view QA 和跨 subject/动作泛化验证。

### 16.9 2026-06-12 shoes worst-view center/footprint 口径修复

针对 shoes 在 `c21/c22/c23_f000240` 的 worst-view center leakage 偏高，本轮没有继续做视角特判或 suppression，而是先定位 center metric 与 3DGS footprint 影响范围的口径错位：

```text
projection center leakage: 只看 Gaussian 中心点是否落在 shoes mask 内。
真实 soft edit 影响: Gaussian splat 按半径覆盖 footprint 区域。
30-view calibration: 已经使用 footprint overlap 作为主要证据。
```

诊断图：

```text
exp/acceptdata/semantic_editing_paper_loop_20260611/shoes_worst_view_diagnostics_20260612/c22_f000240_shoes_center_overlay.png
exp/acceptdata/semantic_editing_paper_loop_20260611/shoes_worst_view_diagnostics_20260612/c21_f000240_shoes_center_overlay.png
exp/acceptdata/semantic_editing_paper_loop_20260611/shoes_worst_view_diagnostics_20260612/c23_f000240_shoes_center_overlay.png
exp/acceptdata/semantic_editing_paper_loop_20260611/shoes_worst_view_diagnostics_20260612/c22_f000060_shoes_center_overlay.png
```

图中红点主要集中在脚踝、鞋面上缘和 shoes mask 边界，不是上身/大腿等大块错标。对 shoes mask 做仅用于诊断的膨胀后，worst-view center leakage 显著下降：

```text
c22_f000240 dilate 0 leak = 0.8811
c22_f000240 dilate 2 leak = 0.6512
c22_f000240 dilate 4 leak = 0.4608

c21_f000240 dilate 0 leak = 0.8471
c21_f000240 dilate 2 leak = 0.5987
c21_f000240 dilate 4 leak = 0.4310
```

这说明 worst-view 主要是小部件 mask 边界过紧 + center sampling 指标放大，而不是 shoes 语义整体崩坏。

本轮新增代码能力：

```text
tools/calibrate_evidence_soft_edit_weights.py
  build_center_consistency_evidence_record()
  accumulate_center_consistency_evidence()
  --min-center-views
  --center-penalty-power
  --center-target-retention-floor
  footprint_center_evidence.npz

tools/analyze_projected_soft_edit_leakage.py
  compute_footprint_leakage_for_selection()
  per_view.csv 额外输出 hard_footprint / soft_footprint rows
```

TDD / 回归：

```text
conda run -n ictrl python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_hybrid_part_label_bank.py \
  tests/test_projected_soft_edit_leakage.py \
  tests/test_multiview_semantic_consistency.py \
  tests/test_semantic_edit_render_preview.py -q
47 passed
```

生成并验证了两个 center-consistency 候选：

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_center_consistent_30view_20260612/part_label_bank.npz
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_shoes_center_consistent_30view_conservative_20260612/part_label_bank.npz
```

结果显示，center penalty 能降低总体 leakage，但没有真正解决 worst-view center ratio，因为 target 分母也被压低：

```text
old 30-view shoes soft leakage = 0.1624963636
strong center-consistent shoes soft leakage = 0.1490926623
conservative shoes-only center-consistent shoes soft leakage = 0.1539215193

old 30-view c22_f000240 shoes center leak = 0.8811
strong center-consistent c22_f000240 shoes center leak = 0.8997
conservative c22_f000240 shoes center leak = 0.9028
```

同一 worst-view 下，footprint-aware 指标明显低于 center 指标：

```text
strong center-consistent c22_f000240 shoes center leak = 0.8997
strong center-consistent c22_f000240 shoes footprint leak = 0.7121

conservative c22_f000240 shoes center leak = 0.9028
conservative c22_f000240 shoes footprint leak = 0.7111
```

当前结论：

- center-consistency evidence 是有价值的诊断字段，但当前 center-penalty 权重版不应提升为论文主候选。
- 主候选仍保持 `evidence_calibrated_30view_20260611`，因为它在平均 leakage、target retention、preview 之间更平衡。
- 论文评估应新增 footprint-aware projection leakage，并把 center leakage 作为辅助/worst-view 指标；否则 shoes 这种小部件会被 center-only metric 系统性放大。
- 后续若继续优化权重，不能简单惩罚 center outer ratio；应改为 view-robust footprint/center combined risk，例如 high-quantile footprint outer ratio、mask-boundary aware tolerance，或者在训练/资产侧改善 shoes mask 边界，而不是进一步全局压 shoes 权重。

### 16.10 2026-06-12 boundary-aware footprint 修复

按照 16.9 的根因判断，本轮没有继续惩罚 center outer ratio，而是把 parser hard mask 边界改成 soft-boundary target，用同一口径进入 footprint evidence 和 footprint-aware projection evaluation。

新增代码能力：

```text
tools/calibrate_evidence_soft_edit_weights.py
  build_soft_boundary_target_mask()
  build_footprint_evidence_record(..., soft_boundary_radius, soft_boundary_min_value)
  --soft-boundary-radius
  --soft-boundary-min-value

tools/analyze_projected_soft_edit_leakage.py
  compute_footprint_leakage_for_selection(..., use_soft_target=True)
  --soft-boundary-radius
  --soft-boundary-min-value
  per_part/summary 汇总 hard_footprint / soft_footprint

tools/analyze_multiview_semantic_consistency.py
  ensure_projected_arg_defaults()
  --soft-boundary-radius
  --soft-boundary-min-value
```

TDD / 回归：

```text
conda run -n ictrl python -m pytest \
  tests/test_part_label_bank.py \
  tests/test_evidence_calibrated_semantic_bank.py \
  tests/test_hybrid_part_label_bank.py \
  tests/test_projected_soft_edit_leakage.py \
  tests/test_multiview_semantic_consistency.py \
  tests/test_semantic_edit_render_preview.py -q
52 passed
```

生成 boundary-aware 30-view candidate：

```text
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_boundary_aware_30view_20260612/part_label_bank.npz
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_boundary_aware_30view_20260612/summary.json
exp/acceptdata/viewer_bundle_v395_hybrid_main_floor025_lower_shoes_boundary_aware_30view_20260612/footprint_boundary_evidence.npz
```

参数：

```text
soft_boundary_radius = 2
soft_boundary_min_value = 0.25
center_penalty_power = 0.0
```

生成摘要：

```text
old 30-view lower removed_weight_sum = 429.09
boundary-aware lower removed_weight_sum = 365.23

old 30-view shoes removed_weight_sum = 247.58
boundary-aware shoes removed_weight_sum = 221.28
```

这说明 soft-boundary evidence 对边界目标点更宽容，不是进一步压权重。

Projection 对比，统一 `soft_threshold=0.20`：

```text
baseline lower center leakage = 0.0503225681
old 30-view lower center leakage = 0.0360276987
boundary-aware lower center leakage = 0.0374641451

baseline shoes center leakage = 0.2019229909
old 30-view shoes center leakage = 0.1624963636
boundary-aware shoes center leakage = 0.1665751998

boundary-aware lower soft_footprint leakage = 0.0394426768
boundary-aware shoes soft_footprint leakage = 0.1368107833
```

shoes worst-view:

```text
old 30-view c22_f000240 shoes center leak = 0.8811
boundary-aware c22_f000240 shoes center leak = 0.8699

boundary-aware c22_f000240 shoes soft_footprint leak = 0.6038
boundary-aware c21_f000240 shoes soft_footprint leak = 0.5097
boundary-aware c23_f000240 shoes soft_footprint leak = 0.3774
```

Multiview：

```text
old 30-view lower soft leakage mean = 0.0365768413
boundary-aware lower soft leakage mean = 0.0380389292

old 30-view shoes soft leakage mean = 0.1881356293
boundary-aware shoes soft leakage mean = 0.1916295674

old 30-view shoes target_activation_mean = 574.81
boundary-aware shoes target_activation_mean = 590.49
```

Render preview：

```text
exp/acceptdata/semantic_editing_paper_loop_20260611/edit_render_preview_v395_hybrid_boundary_aware_30view_lower_shoes/semantic_edit_render_preview_sheet.png

old 30-view lower soft_selected = 9829
boundary-aware lower soft_selected = 9887

old 30-view shoes soft_selected = 2187
boundary-aware shoes soft_selected = 2229
```

当前结论：

- boundary-aware 不是 suppression，而是把监督/评估从 hard center mask 推向 soft-boundary footprint mask，更符合 3DGS splat 的真实影响范围。
- 它在 center leakage 均值上略弱于 old 30-view，但 target activation、preview selected 和 shoes worst-view soft_footprint 都更合理。
- 如果论文主指标采用 footprint-aware leakage，boundary-aware 30-view 是更根源、更可解释的候选。
- 如果必须沿用 center-only leakage 排名，old 30-view 数字更低；但 center-only 对 shoes 这类小部件存在系统性偏差，不建议作为唯一主指标。
- 后续最值得做的是把 boundary-aware footprint evidence 蒸馏进 semantic asset 训练，而不是继续做后处理权重压制。

### 16.11 新对话建议开场

下一次对话可以直接说：

```text
请按照 docs/动态人体3DGS可靠语义编辑论文闭环计划.md 第 16 节，
实现 tools/suppress_projected_soft_leakage.py，
先 TDD，再生成 leak_suppressed hybrid bank，
并跑 projection / multiview / render preview 验证。
```
