# 结构化、可解释且可编辑的人体 Avatar 表示相关工作整理

本文档面向当前项目的新方向整理相关论文。目标方向是：

> 基于显式绑定的语义可分解 avatar 表示，在统一表示中联合建模运动分解、区域语义与局部外观控制，并进一步支持区域级可编辑。

本文不追求把所有 avatar 论文都列全，而是优先整理和当前方向最相关的工作，并标出它们和本项目的关系。

## 1. 最接近当前目标的论文

### 1.1 NECA: Neural Customizable Human Avatar
- 会议：CVPR 2024
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_NECA_Neural_Customizable_Human_Avatar_CVPR_2024_paper.html
- 核心点：
  - 强调 `customizable avatar`
  - 在统一框架中建模几何、反照率、阴影、光照
  - 支持姿态、光照、纹理等编辑
- 和本项目的关系：
  - 这是“从 animatable avatar 走向 editable avatar”的代表工作之一。
  - 它的重点是外观与光照的可控性，不是 body/cloth/hair 的显式语义分解。
- 对本项目的启发：
  - 你可以参考它如何把“编辑能力”写成表示层能力，而不是单独的后处理功能。

### 1.2 Structured 3D Features for Reconstructing Controllable Avatars
- 会议：CVPR 2023
- 链接：https://openaccess.thecvf.com/content/CVPR2023/html/Corona_Structured_3D_Features_for_Reconstructing_Controllable_Avatars_CVPR_2023_paper.html
- 核心点：
  - 用结构化 3D 特征表示人体
  - 支持 animatable、relightable、controllable avatar
  - 特征不只覆盖身体，还可以覆盖头发、配饰和宽松服装
- 和本项目的关系：
  - 这篇和“结构化表示”非常接近。
  - 但它的“可控”更偏重姿态、光照、衣着变化，不是显式的 `body / cloth / hair + rigid / soft / free` 双重分解。
- 对本项目的启发：
  - 你可以把自己的创新点从“解释图”升级成“结构化、可控的表示”，这篇是一个很强的参照。

### 1.3 UV Volumes for Real-Time Rendering of Editable Free-View Human Performance
- 会议：CVPR 2023
- 链接：https://openaccess.thecvf.com/content/CVPR2023/html/Chen_UV_Volumes_for_Real-Time_Rendering_of_Editable_Free-View_Human_Performance_CVPR_2023_paper.html
- 核心点：
  - 强调 editable free-view human performance
  - 用 UV 空间和 3D 体表示解耦几何与外观
  - 支持实时渲染与 retexturing
- 和本项目的关系：
  - 和“区域级外观编辑”比较接近。
  - 但它不强调解释性，也不强调层级语义分解。
- 对本项目的启发：
  - 如果后续要把 `body/hair/cloth` 的 appearance code 做成可编辑接口，这篇可以提供工程思路。

### 1.4 Neural-ABC: Neural Parametric Models for Articulated Body With Clothes
- 期刊：IEEE TVCG 2025
- 链接：https://pubmed.ncbi.nlm.nih.gov/38345957/
- 核心点：
  - 显式强调 `body with clothes`
  - 对 identity、clothing、shape、pose 做 disentangled latent 建模
  - 支持属性编辑
- 和本项目的关系：
  - 这是和“人体-衣物分解 + 参数化编辑”最接近的一篇之一。
  - 它做的是神经参数模型，不是你当前的显式绑定解释图路线。
- 对本项目的启发：
  - 如果你要把当前工作往“结构化、可编辑表示”方向升级，这篇是必须参考的强相关工作。

### 1.5 AttriHuman-3D: Editable 3D Human Avatar Generation with Attribute Decomposition and Indexing
- 会议：CVPR 2024
- 链接：https://cvpr.thecvf.com/virtual/2024/poster/30413
- 核心点：
  - 直接做 attribute decomposition
  - 将 body、hair、clothes 等属性拆开并支持单独编辑
- 和本项目的关系：
  - 它和你未来想做的 `body / cloth / hair` 可编辑非常接近。
  - 但它更偏生成式 3D avatar，不是从单主体视频重建得到的显式绑定 avatar。
- 对本项目的启发：
  - 你后面若强调“区域级 appearance editing”，这篇是 related work 中绕不开的一篇。

### 1.6 SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing
- 会议：CVPR 2025
- 链接：https://research.nvidia.com/labs/dair/publication/li2025simavatar/
- 核心点：
  - 采用 layered hair and clothing 的表示
  - 更强调可分层、可仿真、可操作的 avatar
- 和本项目的关系：
  - 这篇和你想做的 `body / cloth / hair` 分层表示很像。
  - 它偏模拟和层次化 avatar，不是当前这种单目显式绑定解释链路。
- 对本项目的启发：
  - 可以用于支撑“hair / cloth / body 分层表示是有价值的”这个论点。

## 2. 你的主干基础文献：animatable human avatar

这些论文更偏运动驱动、canonical 表示、novel pose 和 monocular/sparse-view avatar，是你当前显式绑定主线的上游背景。

### 2.1 Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies
- 会议：ICCV 2021
- 链接：https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Animatable_Neural_Radiance_Fields_for_Modeling_Dynamic_Human_Bodies_ICCV_2021_paper.html
- 关键词：
  - canonical space
  - skeletal + non-rigid deformation
  - animatable avatar
- 作用：
  - 这是动态人体 avatar 的经典基础文献之一。

### 2.2 HumanNeRF: Free-Viewpoint Rendering of Moving People From Monocular Video
- 会议：CVPR 2022
- 链接：https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html
- 关键词：
  - monocular video
  - canonical appearance volume
  - skeletal rigid + non-rigid motion decomposition
- 作用：
  - 和你现在“显式绑定 + 非刚性 + novel pose”这条线直接相关。

### 2.3 MonoHuman: Animatable Human Neural Field From Monocular Video
- 会议：CVPR 2023
- 链接：https://openaccess.thecvf.com/content/CVPR2023/html/Yu_MonoHuman_Animatable_Human_Neural_Field_From_Monocular_Video_CVPR_2023_paper.html
- 关键词：
  - monocular animatable avatar
  - bidirectional deformation
  - skeletal motion weight + non-rigid motions
- 作用：
  - 对“运动分解”很关键。
  - 但它没有把语义解释和局部编辑作为核心目标。

### 2.4 DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks
- 会议：ECCV 2022
- 链接：https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4883_ECCV_2022_paper.php
- 关键词：
  - disentangled articulated representation
  - body-part correlation
  - unseen pose robustness
- 作用：
  - 这篇能支撑“分解式运动表示”的合理性。
  - 如果你后面要把 rigid / soft / free 写成结构先验，这篇很有参考价值。

## 3. 与局部外观控制相关的论文

### 3.1 Relightable and Animatable Neural Avatar from Sparse-View Video
- 会议：CVPR 2024
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Relightable_and_Animatable_Neural_Avatar_from_Sparse-View_Video_CVPR_2024_paper.html
- 核心点：
  - 在动态人体 avatar 中同时支持 animation 和 relighting
  - 强调材质和光照恢复
- 和本项目的关系：
  - 对“局部外观可控”是上游参考。
  - 但不直接做语义区域级编辑。

### 3.2 TexVocab: Texture Vocabulary-conditioned Human Avatars
- 会议：CVPR 2024
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Liu_TexVocab_Texture_Vocabulary-conditioned_Human_Avatars_CVPR_2024_paper.html
- 核心点：
  - 用 texture vocabulary 建模动态人体外观
  - 将 pose 与 texture 建立结构化关联
- 和本项目的关系：
  - 如果你后面要保存 `skin / hair / cloth` 的 appearance code，这篇有强参考价值。
  - 它更偏 texture representation，不是显式语义分区编辑。

## 4. 综述文献

### 4.1 3D Human Avatar Reconstruction with Neural Fields: A Recent Survey
- 期刊：Image and Vision Computing, 2025
- 链接：https://www.sciencedirect.com/science/article/pii/S0262885624004463
- 用途：
  - 用于全面梳理 avatar reconstruction 的大图景
  - 可作为 related work 开头的综述支撑

## 5. 当前项目和现有文献的关系

如果把当前项目的核心路线概括一下，可以写成：

1. 显式绑定的动态人体 avatar 主干
2. 在渲染阶段导出 `layer / region / body_prob / cloth_prob / thin / semantic` 等解释图
3. 尝试把 `body / cloth / hair` 与 `rigid / soft / free` 变成更稳定的语义分解
4. 进一步朝区域级 appearance editing 升级

和现有文献相比，当前项目已经覆盖了：

- 动态 avatar 的显式/半显式运动建模
- 一定程度上的 region-aware body-cloth 分解
- 面向论文图和 failure analysis 的可解释导出

但还没有完全完成的是：

- 将解释因子显式参数化为可保存、可替换的 appearance / semantic / motion bank
- 在统一训练框架中稳定实现 `body / cloth / hair` 和 `rigid / soft / free` 双分解
- 用这些分解后的因子进行跨主体、跨动作、跨外观的可编辑控制

## 6. 如果你要写成论文，最可能的定位

当前最合适的定位不是：

- “我们做了一个更好的 human parser”
- “我们做了一个后处理换肤色系统”

而更应该写成：

> 我们提出一种基于显式绑定的语义可分解 avatar 表示，在统一表示中联合建模运动层级、区域语义与局部外观控制，使解释图不仅用于分析，还可进一步转化为区域级编辑控制。

这个定位和现有文献的关系是：

- 相比 HumanNeRF / MonoHuman / Animatable NeRF：
  - 你更强调解释性和语义分解
- 相比 NECA / UV Volumes / TexVocab：
  - 你更强调结构语义、body-cloth-hair 分解和显式绑定
- 相比 Neural-ABC / SimAvatar：
  - 你更强调从已有单主体 avatar 重建链路中，抽取并利用解释因子，而不是从头做一个独立的参数模型或仿真模型

## 7. 当前方向可能的创新空白

目前公开工作中，已经有人分别做了：

- animatable avatar
- controllable avatar
- editable avatar
- clothing-aware avatar
- layered hair / clothing avatar

但相对少见的是把下面这几件事真正统一起来：

1. 显式绑定的动态 avatar 表示
2. `rigid / soft / free` 的运动层级解释
3. `body / cloth / hair` 的区域语义分解
4. 基于这些解释因子的局部 appearance editing
5. 将语义、外观、运动都保存为可重组资产

如果本项目后续能把这条链路真正做通，那么相较现有文献，比较可能形成的创新点是：

> 从“可解释 avatar”进一步升级到“可编辑 avatar 表示”，让解释因子本身变成可保存、可调用、可替换的控制变量。

## 8. 推荐优先精读顺序

如果只读最关键的 6 篇，建议顺序如下：

1. HumanNeRF
2. MonoHuman
3. Structured 3D Features
4. UV Volumes
5. NECA
6. Neural-ABC

理由：

- 前两篇补你的 avatar 主干背景
- 中间两篇补“可控/可编辑 avatar”的表示思路
- 后两篇补“可定制/可解耦/可编辑”的强相关方法方向

## 9. 一个实用判断

如果后续实现只是：

- parser 分割
- 几个规则修正
- 简单颜色替换

那么和现有工作相比，创新性不会太强。

如果后续真正做到：

- 统一的显式绑定表示
- 运动、区域语义、appearance 的分解式建模
- 解释因子的资产化保存
- 局部语义编辑和跨动作/跨主体重组

那么这条线会明显更接近“结构化、可解释且可编辑的人体 avatar 表示”这一目标。
