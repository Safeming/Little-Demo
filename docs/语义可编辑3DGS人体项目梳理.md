# 语义可编辑 3DGS 人体项目梳理

## 目标定位

当前项目要做的不是单纯的 3DGS 人体重建，也不是只提升渲染 PSNR，而是让 3DGS 人体 Avatar 真正具备稳定的部件语义和可编辑能力。

核心目标可以概括为：

> 让可动画 3DGS 人体中的每个 Gaussian 都知道自己属于哪个身体或服饰部件，并且这个语义在视角变化、姿态变化、动画和编辑操作中保持稳定。

更适合作为研究方向的表述是：

> Persistent Semantic Binding for Animatable 3D Gaussian Human Avatars

或者：

> Semantic-aware Editable Gaussian Avatars

## 当前已有基础

项目目前已经具备一条可工作的原型链路：

- 已有可动画 3DGS 人体 Avatar。
- 已有训练侧 compact semantic probability，即 `binding_compact_semantic_probs_asset`。
- 已经将 label bank 的主来源从多视角 2D mask voting 改为 trained per-Gaussian semantic probability。
- 已经生成 per-Gaussian `part_label_bank.npz`。
- 已经能输出 Editor 可读的 `manifest.json`、render images、semantic assets、motion assets。
- 已经开始修复真实使用中的语义问题：
  - face Gaussian 尺度过大导致语义颜色外溢。
  - hair outlier 飘点。
  - upper/lower 局部错标。
  - 避免重新绕回 raw 2D parser mask。

这说明项目已经不只是一个渲染 demo，而是在接近“可编辑 3DGS 人体资产”的核心问题。

## 已确认的问题和修复方向

### 1. 2D mask voting 不稳定

原来的 3DGS label bank 依赖多视角 2D mask voting。这个路线容易受到以下问题影响：

- 2D parser 本身错误。
- 多视角投影不一致。
- 遮挡区域投票不可靠。
- Gaussian splat 尺度大时，一个点会影响较大图像区域。
- 动作变化后 2D mask 和 3D Gaussian 的对应关系不稳定。

因此后续方向应避免把 2D mask voting 作为主语义来源。

当前更合理的路线是：

> 直接把 `binding_compact_semantic_probs_asset` 作为 per-Gaussian 识别信息主来源，把 label bank 从“投票器”改成“训练语义导出器”。

### 2. face 大尺度 Gaussian 问题

排查发现，face 点本身数量不大，也没有大面积跑到身体上，但部分 face Gaussian 的 scale 偏大，渲染/预览时会把 face 颜色糊到真实脸外面。

当前修复策略：

- `face_prob >= 0.70`
- `face_prob - second_prob >= 0.15`
- `face max scale <= 0.12`
- 不满足时回退到第二高概率类别或 unknown。

已生成过 face guard 版本：

```text
exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_faceguard_20260609/manifest.json
```

### 3. upper/lower 局部错标

在后续截图中，face 已经不是主要问题。进一步统计发现：

- `lower` 有 117 个点落在上半身高度。
- 这些点的第二高概率全部是 `upper`。

因此做了更窄的 lower guard：

- 当前标签是 `lower`。
- canonical `y > 0.30`。
- 空间上位于躯干附近。
- 第二高类别是 `upper`。
- 则回退为 `upper`。

当前推荐的新 bundle：

```text
exp/acceptdata/viewer_bundle_v395_dense_canary_semantic_face_lowerguard_20260609/manifest.json
```

验证结果：

- `lower_y_gt_0.30`: `117 -> 0`
- `lower`: `18869 -> 18752`
- `upper`: `12826 -> 12943`
- `face`: `358`
- `unknown`: `57`

## 当前还缺什么

### 1. 缺正式的方法定义

当前实现更像工程链路：

```text
训练语义概率 -> 导出 label bank -> guard 修正 -> Editor 查看
```

论文或系统工作中，需要抽象成方法：

```text
训练时持久语义绑定 -> 语义可靠性估计 -> 可动画语义资产导出 -> 可编辑应用
```

也就是把当前的工程步骤整理为一个稳定、可复现、可解释的方法框架。

### 2. 缺训练侧语义约束的清晰定义

需要进一步明确：

- `binding_compact_semantic_probs_asset` 是如何监督的。
- 语义 loss 来自 2D parser、SMPL/body prior、多视角一致性，还是融合监督。
- Gaussian densify、split、clone 后语义如何继承。
- 同一个 Gaussian 在不同 pose 下语义是否保持一致。
- 语义概率是否参与训练优化，而不只是后处理导出。

这是从工程系统变成研究方法的关键。

### 3. 缺稳定性评估指标

目前主要靠 Editor 视觉检查，还不够。需要建立量化指标：

- per-Gaussian semantic consistency。
- cross-view part consistency。
- cross-pose part consistency。
- part boundary accuracy。
- edit leakage rate。
- 语义选择准确率。

尤其需要证明：

> trained per-Gaussian semantic bank 比 2D mask voting 更稳定。

### 4. 缺 baseline 和消融实验

建议对比：

- raw 2D parser mask。
- multi-view 2D mask voting。
- compact 2D mask voting。
- trained semantic probability。
- trained semantic probability + face/lower/hair guard。
- trained semantic probability + body/pose prior。

消融实验可以说明每一部分的必要性。

### 5. 缺明确的编辑任务展示

Editor 不能只作为查看工具，还应该作为任务验证工具。建议展示：

- 只修改上衣颜色。
- 只修改裤子颜色。
- 只修改鞋子颜色。
- 只修改头发颜色。
- 只隐藏或选择某个 part。
- 动画过程中编辑结果不漂移。

评价重点不是单帧好看，而是：

> 局部编辑是否稳定、是否泄漏、是否跨姿态一致。

## 这个方向能解决什么问题

传统 3DGS Avatar 的主要能力是高质量渲染和快速动画，但它缺少稳定的结构语义。换句话说，它“看起来像人”，但“不知道自己哪里是头发、脸、皮肤、衣服、裤子、鞋子”。

这个方向要解决的问题包括：

- 3DGS Avatar 无法准确选择身体部件。
- 2D parser mask 在多视角和动画下不稳定。
- 单纯后处理投票无法保证 per-Gaussian 语义一致。
- 动画后语义容易漂移。
- 局部编辑容易影响其它部位。
- 数字人资产只能看，不能稳定地改。

因此，这个方向本质上是把 3DGS Avatar 从：

```text
photorealistic rendering
```

推进到：

```text
semantic controllability
```

也就是从“能渲染的人体”变成“可理解、可选择、可编辑的人体资产”。

## 有什么用

这个方向的应用价值比较明确：

- 虚拟试衣：稳定选择上衣、裤子、鞋子并进行颜色或材质编辑。
- 数字人资产制作：为游戏、影视、虚拟主播提供可编辑 3DGS 人体资产。
- AR/VR Avatar：实时渲染的同时支持局部交互。
- VFX 后期制作：只修改衣服、头发、皮肤，不影响其它区域。
- 语义化人体重建：让 3DGS 不只是外观重建，也包含部件理解。
- 文本驱动编辑：未来可以支持“把上衣改成蓝色”“只调整头发”等指令。

## 相关研究脉络

当前已有相近方向，但还没有完全覆盖“可动画人体 + per-Gaussian 持久语义 + Editor 可编辑资产”这个交叉点。

### 1. 3DGS Avatar

已有工作包括：

- 3DGS-Avatar
- SplattingAvatar
- Human Gaussian Splatting

这些工作主要解决可动画人体、实时渲染、SMPL/mesh 绑定和高质量重建问题。

但它们通常不重点解决可编辑部件语义。

### 2. 3DGS 编辑

GaussianEditor 等工作提出了 semantic tracing，用于更可控的 3D Gaussian 编辑。

这说明“语义 + Gaussian 编辑”本身已经是被认可的研究问题。

### 3. 3DGS 语义和开放词汇

LangSplat、OpenSplat3D 等工作关注 3DGS 的语义场、语言特征、开放词汇分割。

这些工作多数面向静态 scene 或 general scene/object，而不是专门面向可动画人体 Avatar。

### 4. 语义引导人体 Gaussian

也已有 semantically-guided Gaussian avatar 相关尝试，说明“语义引导人体 Gaussian”方向有研究关注。

当前项目的机会在于：

> 面向可动画人体 Avatar 的 per-Gaussian 持久部件语义和可编辑资产输出。

## 建议的下一阶段路线

### 阶段 1：收敛语义类别和资产协议

固定当前 compact parts：

- hair
- face
- skin
- upper
- lower
- shoes

并稳定输出：

- `manifest.json`
- `part_label_bank.npz`
- `semantic_probs`
- `part_label`
- `confidence`
- render images
- motion assets

### 阶段 2：训练时语义绑定

目标是让语义真正成为训练中的一部分，而不是导出后修补。

需要补充：

- semantic loss
- body prior loss
- cross-view consistency
- cross-pose consistency
- densify/split 语义继承机制

### 阶段 3：语义可靠性估计

将当前 heuristic guard 抽象成 reliability estimation：

- probability confidence
- top-1/top-2 margin
- Gaussian scale
- opacity
- body coordinate prior
- pose consistency
- multi-view consistency

输出每个 Gaussian 的：

```text
semantic label + semantic confidence + reliability score
```

### 阶段 4：编辑任务验证

设计固定任务：

- 上衣换色。
- 裤子换色。
- 鞋子换色。
- 头发换色。
- 只隐藏某个 part。
- 动画中保持编辑结果稳定。

### 阶段 5：定量实验

至少需要这些指标：

- semantic accuracy
- cross-view consistency
- cross-pose consistency
- edit leakage rate
- boundary error
- user-visible failure count

并和 2D voting baseline 对比。

## 推荐论文贡献表述

可以整理成三个贡献：

1. 提出面向可动画 3DGS 人体的持久 per-Gaussian 语义绑定方法。
2. 提出结合 semantic confidence、Gaussian scale、body prior 的语义可靠性校准机制。
3. 构建 Editor-ready semantic Gaussian avatar asset，并展示稳定部件选择和局部编辑能力。

推荐核心句：

> We propose a persistent semantic binding framework for animatable 3D Gaussian human avatars, enabling stable part-level selection and editing across views and poses.

## 总结判断

这个方向有学术价值，也值得继续做。

但后续重点不应该是无限修单张截图，而是把当前工程经验整理成一个方法体系：

- 训练时学语义。
- Gaussian 持久携带语义。
- 语义随动画稳定。
- 语义可被 Editor 使用。
- 局部编辑可量化验证。

如果做扎实，它可以形成一篇偏 CV、CG 或多媒体方向的工作。核心价值不是再做一个 3DGS Avatar，而是让 3DGS Avatar 从高质量渲染走向稳定语义控制和可编辑数字人资产。
