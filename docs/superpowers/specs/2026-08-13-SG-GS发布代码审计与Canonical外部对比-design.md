# SG-GS 发布代码审计与 Canonical 外部对比设计

## 目标

核验 `/remote-home/ming/SGGS` 的来源和可复现性，建立一个可运行、可审计且不夸大官方实现完整度的 SG-GS 外部基线，并在与 A5、SAGA-Canonical、Gaussian Grouping-Canonical 相同的严格语义编辑协议下完成 CoreView 377、386、394 的定量比较。

## 身份与发布状态

- 本地仓库固定为 `Maxwell-Zhao/SGGS`，提交 `27b9ed9c9e4c5663deb169247c2339ccafe1c254`。
- SG-GS 项目主页的 Code 链接直接指向该仓库；arXiv `2408.09665` 的作者、方法名称和项目主页一致。因此该仓库按作者公开的官方仓库处理，不标记为伪造仓库。
- 发布包缺少 README 声称存在的 `environment.yml`、Git submodule 元数据和许可证文件，README 克隆命令仍指向上游 3DGS-Avatar。论文所述 semantic projection、semantic loss 和 neighborhood semantic consistency 在公开 `train.py` 中被注释。因此发布代码不得等同于论文完整实现。
- 论文和实验固定命名为 `SG-GS-Released-Code-Canonical (controlled-input adaptation)`。若后续另行复现论文缺失模块，必须命名为 `SG-GS paper-method reimplementation`，不得与本基线混合。

## 实现边界

### 保留的发布代码要素

- SMPL semantic prior 及每个 canonical Gaussian 的 semantic attribute。
- 发布代码中实际执行的语义初始化、语义渲染和按部位 densification 逻辑。
- 其 3DGS-Avatar 动态人体表示、刚性蒙皮和非刚性变形结构，仅用于最小原生启动审计。
- 如发布代码中存在且实际可执行的拓扑或几何邻域项，保留其原始公式和默认权重，并记录源码位置。

### 允许增加的兼容层

- 从 `/remote-home/ming/3dgs-avatar-release-main/body_models` 只读加载 SMPL 模型和派生参数。
- 从 `/remote-home/ming/dataSet` 只读加载 ZJU-MoCap 数据。
- 补齐依赖路径、CUDA 扩展加载、离线 WandB 和输出目录配置。
- 将发布代码的 SMPL 部件标签确定性映射到严格协议的六类 `hair/face/upper/lower/shoes/skin`。SMPL 无法区分衣物、头发和裸露皮肤的类别必须记录为该基线的先验限制，不使用测试解析掩码补标签。
- 为公平的编辑评测导出 canonical 点级 soft/hard semantic bank、置信度、类别边际、源提交哈希和文件指纹。

### 禁止事项

- 不直接修改或提交 `/remote-home/ming/SGGS` 的 Git 历史；适配代码放在当前项目的独立工具和工具函数中。
- 不取消注释存在明显不可微操作的发布代码片段并声称这是官方论文实现。
- 不使用严格测试视角、测试解析标签或测试编辑结果选择阈值、类别映射、损失权重或 checkpoint。
- 不用 A5 的 evidence-calibrated 权重替换 SG-GS 输出；A5 bank 仅可用于评测器的统一参考和协议校验。
- 不把受控输入适配结果表述为官方仓库在其原生完整训练设置下复现的论文数值。

## 两阶段执行

### 阶段一：官方发布代码最小启动审计

1. 建立依赖清单并确定与 Python/CUDA ABI 匹配的环境。
2. 通过只读路径或受控软链接提供上游 3DGS-Avatar submodule、SMPL 和数据根目录。
3. 生成发布代码要求的 `smpl_semantic.ply`，检查其顶点数量和标签范围。
4. 在 CoreView 377 上解析 Hydra 配置、构造训练集、初始化 Gaussian，并完成至少一次前向、反向和优化器更新。
5. 输出启动审计 JSON：官方提交、依赖版本、数据/SMPL 指纹、实际启用的损失项、成功阶段和错误堆栈。

阶段一只证明发布代码可启动，不直接作为论文主表结果。若发布代码因算法级缺陷无法完成一次更新，记录为 released-code reproduction failure，并转入阶段二的受控适配，不隐瞒失败。

### 阶段二：Canonical 受控输入外部基线

沿用 SAGA/Gaussian Grouping 已验证的公平比较结构：冻结相同的动态人体 canonical geometry、opacity、appearance 和 deformation，只训练或构造 SG-GS 的 semantic attributes。这样排除重建质量、Gaussian 数量和训练视角差异，将比较限定为语义选区与空间泄漏控制。

正式三被试固定为 CoreView 377、386、394。每名被试：

1. 使用与其他外部基线相同的训练视图和 compact-6 监督资产。
2. 只用验证集扫描读出阈值；采用 leave-one-subject-out 选择，并冻结协议、checkpoint、bank 和阈值指纹。
3. 测试集固定为 c21-c23、帧 180/420/540，共 9 个视图。
4. 以 B1 为保留率参考，主表读取 60% Gaussian 目标激活率匹配点。
5. 若 60% 不可达，报告最高可达保留率，不外推、不补造 60% 数值。

## 方法读出与标签映射

SG-GS 发布代码使用 SMPL 拓扑部件，严格协议使用六个可编辑外观部件。适配采用两层输出：

- `released_native_semantics`：原生 SMPL semantic label，保留用于审计和拓扑诊断。
- `compact6_semantics`：由训练视图监督学习的六类读出头；输入仅为 SG-GS semantic/topological features，不读取测试标签。

这种设计保留 SG-GS 的 SMPL 语义拓扑先验，同时避免用一张无法表达头发、衣服和皮肤的硬编码 SMPL 标签表冒充六类外观解析。六类读出训练方式、损失和超参数必须在导出 manifest 中完整记录。

## 评测与对比

主表至少报告：

- 60% matched retention 下的 actionable leakage 和 raw leakage，越低越好。
- macro mIoU 和 mean boundary F1，越高越好。
- 60% 保留率可达被试数。
- 训练时间、峰值显存、点级语义参数量和推理额外开销。

对比对象固定为 A5、SAGA-Canonical、Gaussian Grouping-Canonical 和 SG-GS-Released-Code-Canonical。汇总同时提供逐被试值、共同可比子集均值和相对 A5 的差值；不得用不同被试数的均值直接计算相对提升。

## 产物与可复现性

实验根目录使用带北京时间时间戳的 `exp/external/sggs_released_code_canonical_three_subject_*_bjt`。每名被试至少保存：

- 输入 manifest 和 SHA-256 指纹；
- 训练日志、周期 checkpoint、最终 semantic bank 和完成标记；
- 冻结 LOSO 配置和候选选择轨迹；
- validation/test 的完整严格协议报告；
- 与 A5/SAGA/Gaussian Grouping 的汇总 CSV/JSON。

任何失败必须保留失败状态、日志和最近 checkpoint，队列脚本支持从最近有效 checkpoint 恢复。正式训练前必须通过 dry-run、单次优化更新、短程 canary、bank schema 校验和严格评测器 smoke test。

## 验收标准

- 身份审计能由项目主页、arXiv 元数据、本地 remote 和提交哈希交叉复核。
- CoreView 377 发布代码最小启动审计产生明确的成功或算法级失败证据。
- 三名被试均产生有效 semantic bank；若训练失败则不得用其他方法结果替代。
- 所有测试报告各包含 9 个固定测试视图、有限数值和正确的协议分割标记。
- 阈值选择不读取 held-out 被试测试结果。
- 最终论文表述明确使用 `controlled-input adaptation`，并披露官方发布代码缺失论文模块。
