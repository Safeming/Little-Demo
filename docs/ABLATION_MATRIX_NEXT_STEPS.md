# 消融实验下一步计划（出门前可直接开跑）

本文档是当前 `v4.1` 主线的消融实验执行单，目标是在你离开 2–3 小时时间内，把最关键的 ablation 继续推进到可分析状态。

当前已知结果：
- 主线 `v4.1 main`：
  - `PSNR = 30.7344`
  - `SSIM = 0.97788`
  - `LPIPS = 0.02468`
- `w/o body-cloth`：
  - `PSNR = 29.3717`
  - `SSIM = 0.97326`
  - `LPIPS = 0.02961`

这已经足以说明：
- `body/cloth` 分区是有贡献的
- 去掉 body/cloth 后，主渲染指标明显下降

因此，接下来最值得补的两个消融是：
1. `w/o temporal`
2. `w/o semantic`

---

## 1. 当前消融矩阵状态

### 已完成
- `baseline`
- `v3`
- `v4.1 main`
- `v4.1 w/o body-cloth`

### 接下来要补
- `v4.1 w/o temporal`
- `v4.1 w/o semantic`

### 可选补充
- `v4.1 w/o temporal` 的 interpretability 导出，重点看 `test-video temporal`
- `v4.1 w/o semantic` 的 `test-view semantic` 图

---

## 2. 实验 A4：v4.1 w/o temporal

### 目的
验证 temporal consistency 是否真的抑制连续帧滑移。

### 方法
关闭 temporal loss 和 temporal cache 的有效作用：
- `opt.lambda_binding_temporal=0.0`
- `model.deformer.rigid.temporal_momentum=0.0`
- `model.deformer.rigid.temporal_cache_size=1`

### 预期
如果 temporal consistency 有效，那么去掉后应出现：
- `temporal slip` 指标变差
- 连续帧局部区域更抖
- 极端动作或边界位置更容易滑移

### 你回来后重点看什么
- 主渲染指标：`PSNR / SSIM / LPIPS`
- `test-video` 的 temporal map
- 和 `v4.1 main` 的 temporal summary 对比

---

## 3. 实验 A5：v4.1 w/o semantic

### 目的
验证 semantic anchor 相关项是否真正提升了绑定稳定性与 anchor 可读性。

### 方法
关闭 semantic loss 和 semantic 权重：
- `opt.lambda_binding_semantic=0.0`
- `model.deformer.rigid.semantic_skinning_weight=0.0`
- `model.deformer.rigid.semantic_normal_weight=0.0`
- `model.deformer.rigid.semantic_prior_weight=0.0`

### 预期
如果 semantic 相关项有效，那么去掉后应出现：
- `semantic stability` 下降
- 绑定更碎或更不稳定
- 局部区域解释图更混乱

### 你回来后重点看什么
- 主渲染指标：`PSNR / SSIM / LPIPS`
- `semantic` 可视化图
- `binding_analysis/aggregate.json` 中的 semantic 统计

---

## 4. 为什么现在不再继续追 PSNR 调参

已经确认失败的方向：
- `30k + shallow_mlp`：全面退化
- `15k + mlp texture`：全面退化

说明当前主线的高分不是靠更长训练或更大模型得来的，而是来自一个已经比较好的结构平衡点。

因此现在最有论文价值的不是继续试：
- 更长训练
- 更大纹理头

而是补齐：
- temporal 消融
- semantic 消融
- 对应解释图与统计

---

## 5. 出门前最推荐的自动执行顺序

### 第一项
训练 `w/o temporal`

### 第二项
渲染 `w/o temporal`

### 第三项
对 `w/o temporal` 跑完整 interpretability pipeline
- 尤其要导出 `test-video temporal`

### 第四项
训练 `w/o semantic`

### 第五项
渲染 `w/o semantic`

### 第六项
输出一个汇总表，显示：
- `v4.1 main`
- `w/o body-cloth`
- `w/o temporal`
- `w/o semantic`

---

## 6. 你回来后怎么判断结果

### 如果 `w/o temporal` 明显变差
说明：
- temporal consistency 是有效模块
- 可以在论文里明确主张“抑制非物理滑移”

### 如果 `w/o semantic` 明显变差
说明：
- semantic anchor 相关项是有效模块
- 可以主张“提升 anchor 稳定性与解释性”

### 如果两者都影响不大
说明：
- 当前主线的主要贡献更集中在：
  - explicit binding
  - body/cloth 分区
- temporal / semantic 可能更偏展示层而不是强性能项

---

## 7. 最终最小可发版消融矩阵

建议最终论文里至少保留：
- `baseline`
- `v3`
- `v4.1 main`
- `v4.1 w/o body-cloth`
- `v4.1 w/o temporal`

如果 `w/o semantic` 结果也明显，则再加入：
- `v4.1 w/o semantic`

这套已经足够构成一张像样的消融表。
