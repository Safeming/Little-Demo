# Gaussian Grouping-Canonical 外部对比训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 377、386、394 的冻结 40k canonical Gaussian 与相同 80 个训练视图上训练 Gaussian Grouping 16 维 Identity Encoding，导出标准语义 bank，并启动可恢复的正式外部对比队列。

**Architecture:** 纯函数模块复刻 Gaussian Grouping 的二维分类数据选择和 canonical KNN KL 正则；GPU 入口直接调用 `gaussian_grouping` 环境中的 16 通道对象 rasterizer，只优化 Identity Encoding 与六类分类头。Shell 队列复用 SAGA 已冻结视图，先执行 canary，再按固定主体顺序训练并持续记录状态与 ETA。

**Tech Stack:** Python 3.8/3.10、PyTorch 2.4、CUDA 12.1、Gaussian Grouping 定制 diff-gaussian-rasterization、NumPy、pytest、Bash。

---

## 文件结构

- Create: `utils/gaussian_grouping_canonical.py`：均衡像素采样、原版形式 3D KL 正则、点概率与 bank 字段导出。
- Create: `tools/train_gaussian_grouping_canonical.py`：冻结视图加载、16 通道渲染、训练、checkpoint、指标和 bank 输出。
- Create: `tools/run_gaussian_grouping_canonical_three_subject.sh`：canary、ETA、377→386→394 持久正式队列和状态记录。
- Create: `tests/test_gaussian_grouping_canonical.py`：纯函数与 CLI/队列契约测试。

### Task 1: Identity Encoding 纯函数

**Files:**
- Create: `tests/test_gaussian_grouping_canonical.py`
- Create: `utils/gaussian_grouping_canonical.py`

- [ ] **Step 1: 写失败测试**

测试 `balanced_pixel_indices` 对有效六类按类最多取固定数量、忽略 `-1` 且固定 seed 可复现；测试 `grouping_3d_consistency_loss` 在邻域概率一致时低于不一致时；测试 `identity_predictions` 输出 `[N,6]` 归一化概率、argmax 标签和非负 margin。

- [ ] **Step 2: 运行失败测试**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_grouping_canonical.py -q`

Expected: FAIL，原因是模块尚不存在。

- [ ] **Step 3: 实现最小纯函数**

`balanced_pixel_indices(labels, samples_per_class, generator)` 对每个出现的非负类别独立抽样后拼接并随机打乱。`grouping_3d_consistency_loss(xyz, probabilities, k=5, lambda_val=2, sample_size=1000, generator=None)` 按原仓库 `loss_cls_3d` 的 KL 公式计算；KNN 包含查询点自身，与原实现一致。`identity_predictions(encodings, classifier)` 对每点调用共享线性头并 softmax。

- [ ] **Step 4: 运行测试确认通过**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_grouping_canonical.py -q`

Expected: PASS。

### Task 2: GPU 训练入口

**Files:**
- Modify: `tests/test_gaussian_grouping_canonical.py`
- Create: `tools/train_gaussian_grouping_canonical.py`

- [ ] **Step 1: 写失败契约测试**

测试参数默认值固定为 16 维、30k、二维 LR `0.0025`、分类头 LR `5e-4`、`reg3d_interval=2`、`k=5`、`lambda=2`、sample size 1000；测试 manifest 校验拒绝非 80 视图、类别顺序错误和点数错误；测试 checkpoint 完成判定支持断点续训。

- [ ] **Step 2: 运行测试确认失败**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_grouping_canonical.py -q`

Expected: FAIL，原因是训练入口尚不存在。

- [ ] **Step 3: 实现训练入口**

直接从 `diff_gaussian_rasterization` 构造与冻结视图匹配的 settings，以零 RGB `colors_precomp` 和 `[N,1,16]` Identity Encoding 调用 rasterizer。二维 loss 对当前视图每类最多抽 512 像素计算 CE，并除以 `log(6)`，与原方法归一化一致。每偶数次更新在 canonical xyz 上计算原形式 3D KL。记录 CE、3D loss、总 loss、梯度范数、参数变化、秒/迭代和峰值显存。

- [ ] **Step 4: 实现 checkpoint 与 bank**

每 5000 次及最后一次保存 Identity Encoding、分类头、两个 optimizer、随机状态和 iteration。训练结束后对 canonical 点直接分类，输出 `identity_encodings.pt`、`classifier.pt`、`part_label_bank.npz`、`summary.json` 和 `COMPLETE`。bank 的 `soft_edit_weights` 等于六类概率，来源标记为 `gaussian_grouping_canonical_controlled_input_identity`。

- [ ] **Step 5: 静态验证**

Run: `/opt/miniconda3/envs/gaussian_grouping/bin/python -m py_compile tools/train_gaussian_grouping_canonical.py utils/gaussian_grouping_canonical.py`

Expected: exit 0。

### Task 3: 三主体队列与 ETA

**Files:**
- Modify: `tests/test_gaussian_grouping_canonical.py`
- Create: `tools/run_gaussian_grouping_canonical_three_subject.sh`

- [ ] **Step 1: 写失败队列测试**

读取 shell 文本并断言环境固定为 `gaussian_grouping`、主体默认顺序为 `377 386 394`、正式默认 30k、canary 默认 100、输入来自冻结 SAGA views、状态包含 PID/subject/stage/iteration/ETA，且正式训练支持已有 checkpoint 恢复。

- [ ] **Step 2: 运行测试确认失败**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_grouping_canonical.py -q`

Expected: FAIL，原因是队列脚本尚不存在。

- [ ] **Step 3: 实现队列**

Shell 先运行 CoreView_377 100 次 canary。用 canary 的后 80% elapsed/iteration 估算三主体 90k 次更新，加 15% 缓冲并写北京时间 ETA。随后顺序执行正式训练，`queue_state.json` 持续记录状态；每主体日志独立保存，失败 trap 写明当前阶段。

- [ ] **Step 4: dry-run 验证**

Run: `DRY_RUN=1 bash tools/run_gaussian_grouping_canonical_three_subject.sh`

Expected: exit 0，输出 canary 和 377、386、394 命令，不启动 GPU。

### Task 4: Canary 与正式队列启动

**Files:**
- Generate: `exp/external/gaussian_grouping_canonical_three_subject_<timestamp>/`

- [ ] **Step 1: 运行专项测试**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_gaussian_grouping_canonical.py tests/test_part_label_bank.py -q`

Expected: PASS。

- [ ] **Step 2: 运行 100 次 canary**

Run: `CANARY_ONLY=1 bash tools/run_gaussian_grouping_canonical_three_subject.sh`

Expected: 100 次更新完成；loss、梯度、参数变化均有限且非零；生成 canary `COMPLETE` 与 summary。

- [ ] **Step 3: 检查 canary 与 ETA**

读取 metrics 后 80% 的吞吐和峰值显存，按 90k 更新与 15% 缓冲计算预计北京时间。若 canary 不符合验收条件，不启动正式队列。

- [ ] **Step 4: 后台启动正式队列**

Run: `nohup bash tools/run_gaussian_grouping_canonical_three_subject.sh > <output>/queue.log 2>&1 &`

Expected: PID 存活，`queue_state.json` 为 running，当前主体 377，GPU 有训练显存占用，首批正式指标有限。

- [ ] **Step 5: 提交源码与协议**

只提交新增源码、测试、设计和计划，不提交大型实验输出。记录输出目录、PID、启动北京时间和预计完成北京时间。

