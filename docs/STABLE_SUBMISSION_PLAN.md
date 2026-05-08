# 更稳投稿版执行方案

本文档面向当前项目的 **更稳投稿版** 路线，目标不是继续扩展方法，而是把现有主线整理成一套更有说服力、风险更低、投稿更稳的实验与论文资产包。

适用当前主线：
- `explicit binding`
- `rigid / soft / free` 分层绑定
- `body/cloth` 分区
- `temporal consistency`
- `semantic anchor`
- interpretability 导出与统计

当前状态概览：
- 方法主线已稳定
- 主实验、baseline、v3、多个消融已经具备
- `Table 1 / Table 2 / Figure 1~4` 已初步生成
- 当前最缺的是：
  1. 多主体验证
  2. 论文图表精修
  3. 论文叙事与结果组织
  4. 补充材料与复现资产

---

## 大类 A：多主体验证

这是当前“更稳投稿版”里最重要的部分。

### A1. 选择额外主体

建议从现有配置中再选 **2 个主体**，优先考虑：
- `386`
- `387`
- `392`
- `393`
- `394`

推荐策略：
- 先选 2 个主体
- 优先选动作和外观与 `377` 略有差异的主体
- 如果时间不足，至少补 1 个主体

### A2. 多主体最小实验组合

每个新增主体，至少跑：
- `baseline`
- `v4.1 main`

如果时间允许，再补：
- `v3`

不建议一开始就对新主体跑所有消融，因为成本太高。

### A3. 每个主体需要产出的数据

至少产出：
- `test-view/results.npz`
- `test-view/renders`

如果时间允许，主线 `v4.1` 再补：
- `interpretability maps`
- `binding_analysis`

### A4. 要比较的指标

跨主体主表重点比较：
- `PSNR`
- `SSIM`
- `LPIPS`

如果有 interpretability，再额外比较：
- `semantic_stability`
- `temporal_slip`
- `layer / region` 分布

### A5. 预期目标

通过多主体验证回答：
- 你的方法是否只在 `377` 有效？
- 主线 `v4.1` 是否在多个主体上表现稳定？
- baseline 与 `v4.1` 的相对关系是否一致？

### A6. 完成标准

至少形成：
- 一张跨主体指标表
- 每个新增主体 1 张代表性对比图

---

## 大类 B：主文图精修

你已经有 `Figure 1~4`，接下来要做的是从“自动生成图”提升到“主文可用图”。

### B1. 精修 Figure 1：方法主文图

当前目标：
- 展示 `GT / Render / Layer / Region / Body Prob / Semantic`

具体要做：
- 检查当前帧是否最具代表性
- 看 `Body Prob` 是否比 `Soft Prob` 更适合作为主图
- 看 `Semantic` 是否有足够对比度
- 检查标题是否适合论文风格

预期结果：
- 一张能概括方法可解释性的主图

### B2. 精修 Figure 2：body/cloth 消融图

当前目标：
- 对比 `v4.1` 和 `w/o body-cloth`

具体要做：
- 选一个衣物/身体过渡更明显的帧
- 让 `Region` 和 `Body Prob` 更有区分度
- 必要时做局部 crop

预期结果：
- 一张能直观看出 body/cloth 分区贡献的图

### B3. 精修 Figure 3：temporal 消融图

当前目标：
- 对比 `v4.1` 和 `w/o temporal`

具体要做：
- 检查当前 temporal 图是否足够显著
- 必要时使用更强的 temporal 显示尺度
- 选择一帧能体现局部滑移差异的连续帧图

预期结果：
- 一张能明显表达时序一致性作用的图

### B4. 精修 Figure 4：failure case 图

当前目标：
- 展示 thin accessory / 边界 / 局部变形问题

具体要做：
- 重新确认 crop 区域是否最有说服力
- 对比 `GT / baseline / v4.1 / w/o body-cloth`
- 突出裤绳、鞋带、边缘和手臂轮廓这些区域

预期结果：
- 一张诚实但对你方法有利的 failure case 图

### B5. 主文图完成标准

最终至少要有：
- `Figure 1` 方法图
- `Figure 2` body/cloth 消融图
- `Figure 3` temporal 消融图
- `Figure 4` failure case 图

并且每张图都要达到：
- 标题统一
- 帧选择合理
- 无明显版式问题
- 适合直接放论文主文

---

## 大类 C：表格与统计收口

你已经有初始表格，接下来要把它们变成论文最终版。

### C1. 收口 Table 1：主渲染指标表

当前内容建议保留：
- `baseline`
- `v3`
- `v4.1`

如果补了新主体，则扩展成：
- 每个主体一行，或
- 每个方法跨主体平均值一行

具体要做：
- 统一评估口径（当前建议用 final checkpoint `results.npz`）
- 检查是否需要补跨主体平均值
- 决定是否保留 `TIME`

### C2. 收口 Table 2：消融/解释性统计表

当前建议保留：
- `v4.1`
- `w/o body-cloth`
- `w/o temporal`
- `w/o semantic`

具体要做：
- 检查 `temporal_slip` 是否优先使用 `test-video`
- 检查 `semantic_stability` 统计是否合理
- 决定哪些列放主文，哪些列放 supplementary

### C3. 视情况补一张跨主体表

如果完成多主体验证，建议新增：
- `Table 3: cross-subject generalization`

列可包括：
- Subject
- Method
- PSNR
- SSIM
- LPIPS

### C4. 表格完成标准

至少形成：
- 一张主渲染指标表
- 一张消融/解释性统计表
- 如果有多主体，再补一张跨主体表

---

## 大类 D：论文叙事与结果解释

这部分决定你最后能不能“稳投稿”。

### D1. 明确论文主张

你不应该把论文主张写成：
- “我们在所有渲染指标上全面超过 baseline”

更合理的主张是：
- 提出一种物理可解释的显式高斯-骨骼绑定框架
- 在保持可接受渲染质量的前提下，提高结构先验、绑定解释性与时序稳定性
- 消融证明 body/cloth、temporal、semantic 三个组件均有效

### D2. 结果解释模板

主渲染表解释：
- baseline 在纯渲染指标上仍有优势
- `v4.1` 在方法系列中表现最好
- 说明结构约束与解释性会带来一定拟合-结构权衡

消融表解释：
- `w/o body-cloth` 明显退化，证明分区绑定有效
- `w/o temporal` 明显退化，证明时序一致性有效
- `w/o semantic` 明显退化，证明 semantic anchor 有效

### D3. 失败案例解释

不是回避 failure case，而是说明：
- 哪些局部区域仍然困难
- 相比 baseline 改善在哪里
- 为什么这些失败与 thin accessory / silhouette deformation 相关

### D4. 论文叙事完成标准

应能清楚回答：
- 方法做了什么
- 每个模块为什么存在
- 哪些实验支持这些模块
- 方法的边界和限制是什么

---

## 大类 E：补充材料与复现资产

如果想稳投稿，这一部分也很重要。

### E1. Supplementary 图

建议准备：
- 更多主体的对比图
- 更多 temporal 连续帧图
- 更多 semantic / thin 可视化图
- 更多 crop 图

### E2. 命令与目录整理

具体要做：
- 保存最终训练命令
- 保存最终渲染命令
- 保存最终 comparison / table / figure 生成命令
- 确认所有路径清晰可复现

### E3. README / docs 收口

建议在最终阶段保留：
- 主线工作流文档
- 实验计划文档
- 稳投稿版路线文档（本文档）

### E4. 补充材料完成标准

至少形成：
- 一套 supplementary 图
- 一份最终命令清单
- 一份复现流程说明

---

## 推荐执行顺序

### 第一阶段：先补硬数据
1. 多主体验证
2. 检查并整理跨主体结果

### 第二阶段：精修主文图
3. 精修 `Figure 1~4`
4. 决定哪些图进主文、哪些放 supplementary

### 第三阶段：整理表格
5. 固化 `Table 1 / Table 2`
6. 如有需要补 `Table 3`

### 第四阶段：写作与收口
7. 完成结果分析段落
8. 完成 failure case 描述
9. 补充 supplementary 与复现说明

---

## 最终目标

完成本计划后，你应该至少拥有：
- 稳定的主线方法结果
- 一套最小但完整的消融矩阵
- 至少 1~2 个新主体的泛化验证
- 主文可用图 `Figure 1~4`
- 主文可用表 `Table 1~2`（以及可选 `Table 3`）
- 一套可以直接进入写作阶段的论文资产包

这时项目就不再只是“实验跑通”，而是进入真正的投稿准备阶段。
