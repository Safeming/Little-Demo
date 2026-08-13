# Gaussian Grouping-Canonical 外部对比训练设计

## 目标

在与 A5、SAGA-Canonical 相同的冻结动态人体 3DGS、相同训练监督和相同测试协议上，训练 Gaussian Grouping 的 canonical Identity Encoding，形成可进入论文主表的受控输入外部基线。

论文中固定命名为 `Gaussian Grouping-Canonical (controlled-input adaptation)`，不得表述为 Gaussian Grouping 原仓库对动态人体的原生实现。

## 比较范围

- 主体和执行顺序：`CoreView_377`、`CoreView_386`、`CoreView_394`。
- 基础表示：各主体严格协议目录中的 `base_train_40k/ckpt40000.pth`。
- 训练监督：相机 1--16、帧 0/120/240/360/480，共 80 个视图。
- 监督类别：Hulk-CIHP compact-6 的 hair、face、upper、lower、shoes、skin。
- 测试范围：沿用冻结严格协议的相机 21--23 和测试帧，不参与训练、阈值拟合或模型选择。
- 正式训练：每个主体 30,000 次更新，固定随机种子，单张 RTX 4090 顺序执行。

## 受控输入适配

Gaussian Grouping 原仓库以静态 COLMAP 场景为输入，并联合优化重建和分组。当前论文比较的是动态人体 canonical Gaussian 上的局部语义编辑，因此不重新训练静态场景几何，而复用 SAGA-Canonical 已导出的冻结训练视图。

每个冻结视图包含：

- 相同 canonical Gaussian 在该姿态下的 xyz、opacity、scale、rotation；
- 相机矩阵、视场角和分辨率；
- compact-6 像素监督标签；
- 与 40k checkpoint 一致的 canonical 点序。

训练期间冻结 Gaussian 几何、外观、opacity、scale、rotation、形变场、pose/SMPL 模块和点数。唯一可训练内容为：

- 每个 canonical Gaussian 的 16 维 Identity Encoding；
- 将渲染后的 16 维 Identity Encoding 映射为六部位 logits 的共享线性分类头。

该适配保留 Gaussian Grouping 的核心二维 Identity Encoding 监督和三维空间一致性约束，同时排除重建质量和点数差异。

## 训练目标

### 二维身份分类

使用 Gaussian Grouping 定制 rasterizer，将每点 16 维 Identity Encoding 渲染到图像。共享线性分类头输出六类 logits，对有效 compact-6 像素计算交叉熵；ignore/background 像素不进入损失。

为避免人体上不同部位面积差异使 hair/shoes 被大区域类别淹没，每次从当前视图按可见类别均衡采样有效像素。损失仍是原方法的像素分类目标，只改变采样以适配固定六部位协议。

### 三维空间一致性

在 canonical xyz 上构建固定 KNN 图。按原 Gaussian Grouping 形式，对随机采样 canonical 点的类别概率施加邻域一致性损失。默认参数与原仓库保持一致：`k=5`、`lambda=2`、每次最多采样 1000 个点、每 2 次更新计算一次。

总损失为二维交叉熵加三维一致性正则。不得加入 A5 的 evidence、footprint、boundary 或 reliability 权重。

## 输出与评测

每个主体输出：

- 16 维 Identity Encoding 和六类分类头 checkpoint；
- 训练指标 `metrics.jsonl` 和最终 `summary.json`；
- 六类点概率、标签、margin 和置信度；
- 兼容现有严格评测器的 `part_label_bank.npz`；
- 输入 checkpoint、视图 manifest、参数和代码指纹。

点概率由分类头直接作用于 canonical Identity Encoding 得到，softmax 后作为 `semantic_probs`。不使用测试 parser mask 修正点标签或编辑权重。

正式训练完成后，使用与 A5/SAGA 相同的 B1 参考、保留率 sweep、hair/shoes 编辑和严格测试视角进行定量评测。主表采用相对于 B1 的 60% Gaussian 点激活率匹配操作点；不得将其描述为每部位 60% RGB 响应。

## 队列与 ETA

先在 CoreView_377 上运行 100 次更新 canary：

1. 验证 Gaussian Grouping rasterizer 的 16 通道前向和反向传播；
2. 验证二维交叉熵、三维一致性损失、梯度和参数更新均为有限非零值；
3. 记录稳态秒/迭代、峰值显存和导出耗时；
4. 按三个主体各 30,000 次更新加 15% 缓冲估算北京时间完成时刻。

canary 通过后按 `377 → 386 → 394` 顺序启动正式队列。队列持续写入 PID、当前主体、阶段、迭代、更新时间和预计完成北京时间；失败时保留日志与最后 checkpoint，支持断点继续。

## 实现边界

- 训练适配器、单元测试、队列脚本和结果放在 `/remote-home/ming/3dgs-avatar-release-main`，不修改 `/remote-home/ming/gaussian-grouping` 的源码。
- 运行时从 `gaussian_grouping` 环境导入其定制 `diff_gaussian_rasterization` 和复用其 3D 一致性定义。
- 复用已冻结的 SAGA 训练视图，不重新读取测试集，不调用 COLMAP，不生成 DEVA/SAM 伪标签。
- 不覆盖 A5、SAGA 或已有严格协议结果。
- `/remote-home/ming/3dgs-avatar-release-main/body_models` 仅在重新导出冻结视图时作为 SMPL 资产来源；复用已导出视图时不参与训练。

## 异常处理

- checkpoint 指纹、点数、点序、视图数量或六类顺序不一致时立即停止。
- 任一损失、梯度、参数或概率出现 NaN/Inf 时停止并记录失败状态。
- 视图有效标签少于两个类别时跳过该视图；连续无法取得有效视图时停止。
- rasterizer 未实际接收 16 通道 Identity Encoding、梯度为零或参数未更新时 canary 判定失败。
- 输出 bank 点数必须与 40k canonical Gaussian 点数完全一致。

## 测试与验收

1. 纯函数测试覆盖 ignore pixel、类别均衡采样、分类损失和 3D KNN 一致性。
2. 导出测试覆盖 Identity Encoding 到标准 bank 的字段、概率归一化、margin 和点数。
3. 队列测试覆盖主体顺序、环境、输入资产、状态文件、失败状态和断点续训命令。
4. CoreView_377 canary 完成 100 次更新，无 NaN/Inf，Identity Encoding 与分类头均有非零梯度和参数变化。
5. 正式队列启动后 PID 存活、GPU 显存占用合理、首批训练指标有限，并给出北京时间 ETA。
6. 三主体完成后均存在 checkpoint、bank、summary、完整日志和完成标记，之后才可运行主表评测。

