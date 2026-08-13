# SG-GS 发布代码 Canonical 外部对比 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 审计 SG-GS 官方发布代码的真实可运行状态，在不修改其 Git 仓库的前提下构建 SG-GS-Released-Code-Canonical 受控输入适配，并完成 CoreView 377、386、394 的训练、LOSO 冻结测试和 A5/SAGA/Gaussian Grouping 对比。

**Architecture:** 审计入口以子进程运行官方 `train.py` 并保存依赖、源码和首个阻塞点证据。Canonical 适配复用 SAGA 已冻结的 80 个训练视图，从相同 40k avatar checkpoint 的 SMPL canonical vertices/skinning weights 构造 SG-GS 的 32 维 topology-geometric prior，再仅训练共享 compact-6 读出 MLP；冻结阈值由验证集 leave-one-subject-out 选择，测试集只在配置冻结后读取。

**Tech Stack:** Python 3.9、PyTorch 2.4/CUDA 12.1、SAGA 32-channel diff-gaussian-rasterization、OmegaConf、NumPy、pytest、Bash。

---

## 文件结构

- Create: `utils/sggs_released_code_canonical.py`：仓库审计纯函数、SMPL 原生标签、topology-geometric prior、邻域一致性损失、LOSO 选择。
- Create: `tools/audit_sggs_released_code.py`：官方提交/项目来源/源码启用项/依赖/原生启动审计报告。
- Create: `tools/export_sggs_canonical_prior.py`：从冻结 avatar 与 SAGA 输入导出每点 32 维 SG-GS prior。
- Create: `tools/train_sggs_released_code_canonical.py`：32 维 prior 的 compact-6 读出训练、checkpoint 和 bank 导出。
- Create: `tools/evaluate_sggs_released_code_canonical.py`：验证阈值扫描、LOSO 冻结、严格测试及跨方法汇总。
- Create: `tools/run_sggs_released_code_canonical_three_subject.sh`：审计、canary、ETA、三被试训练和自动评测队列。
- Create: `tests/test_sggs_released_code_canonical.py`：纯函数、CLI、队列和评测协议契约测试。

## Task 1: 官方发布代码审计

**Files:**
- Create: `tests/test_sggs_released_code_canonical.py`
- Create: `utils/sggs_released_code_canonical.py`
- Create: `tools/audit_sggs_released_code.py`

- [ ] **Step 1: 写失败测试**

测试源码审计能识别：Git remote/HEAD、缺失 `environment.yml`/`.gitmodules`/LICENSE、`train.py` 中被注释的 semantic loss、实际启用的 SMPL label initialization，以及缺失的 `diff_gaussian_rasterization_obj`。测试 JSON schema 必须包含 `official_identity`、`release_completeness`、`semantic_code_state`、`dependency_probe`、`native_launch` 和 SHA-256。

- [ ] **Step 2: 运行测试确认失败**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py -q`

Expected: FAIL，原因是审计模块和入口尚不存在。

- [ ] **Step 3: 实现审计纯函数**

实现 `scan_release_tree(repo)`、`scan_semantic_code(train_path)`、`probe_modules(python, modules)` 和 `build_identity_record(repo)`。源码状态通过 Python `tokenize`/逐行结构检查记录具体行号，不以 README 声明替代代码证据。固定官方身份记录：项目主页 Code URL、arXiv `2408.09665`、remote 和 HEAD；网络证据同时保存访问时间和 URL，不把网络失败误判成身份失败。

- [ ] **Step 4: 实现原生启动审计**

`audit_sggs_released_code.py` 在临时工作目录中调用官方 `train.py`，设置 `WANDB_MODE=disabled`，不编辑 SGGS 文件。启动命令设置 377 数据根、SMPL 根和一次迭代意图；按 `imports -> config -> dataset -> scene -> forward -> backward -> optimizer` 记录最远阶段。若发布包在导入 `diff_gaussian_rasterization_obj`/`sparseconvnet` 前失败，保留完整 stderr、return code 和首个根因，不安装未知实现冒充官方依赖。

- [ ] **Step 5: 验证审计输出**

Run: `/opt/miniconda3/envs/ictrl/bin/python tools/audit_sggs_released_code.py --repo /remote-home/ming/SGGS --dataset /remote-home/ming/dataSet --body-models body_models --output /tmp/sggs_release_audit.json`

Expected: exit 0；报告本身成功生成，`native_launch.status` 可为 `blocked`，但必须包含可复现命令和首个阻塞点。

## Task 2: SG-GS topology-geometric prior

**Files:**
- Modify: `tests/test_sggs_released_code_canonical.py`
- Modify: `utils/sggs_released_code_canonical.py`
- Create: `tools/export_sggs_canonical_prior.py`

- [ ] **Step 1: 写失败数学测试**

测试 `native_smpl_labels(skinning_weights)` 按 SG-GS `JOINT_TO_PART` 把最大蒙皮关节映射为 5 个原生部件；测试 `interpolate_smpl_prior` 对 Gaussian 的 K=4 SMPL 邻点作反距离插值并返回归一化 24 维 skinning、5 维 native semantic probability 和有限距离；测试 `build_topology_geometric_features` 精确输出 `[N,32] = 24 skinning + 5 native semantics + 3 normalized canonical xyz`。

- [ ] **Step 2: 运行测试确认失败**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py -q`

Expected: FAIL，原因是 prior 函数尚不存在。

- [ ] **Step 3: 实现 prior 函数**

固定 SG-GS 原生映射 `JOINT_TO_PART=(4,1,1,0,1,1,0,1,1,0,1,1,3,2,2,3,2,2,2,2,2,2,2,2)`。KNN 权重为 `1/(distance+1e-6)` 后按行归一化；xyz 用 SMPL canonical AABB 中心和最大半径缩放到 `[-1,1]`，不使用任何图像或测试标签。额外实现 `topology_consistency_loss(probabilities, knn_indices, knn_weights)`，只约束由同一 SMPL topology 邻域连接的点。

- [ ] **Step 4: 实现导出入口**

入口读取现有 SAGA 输入的 `manifest.json`/`canonical_xyz.pt`，从 manifest 对应 40k config/checkpoint 创建只读 Scene，并读取 `scene.metadata['smpl_verts']` 与 `scene.metadata['skinning_weights']`。输出：`topology_features.pt`、`native_labels.pt`、`native_semantic_probs.pt`、`topology_knn.pt` 和 `manifest.json`。manifest 记录 SGGS HEAD、所有输入指纹、公式、K=4、32 维布局和 frozen/trainable 清单。

- [ ] **Step 5: 377 smoke 导出**

Run: `/opt/miniconda3/envs/ictrl/bin/python tools/export_sggs_canonical_prior.py --input exp/external/saga_canonical_five_subject_20260812_120625_bjt/CoreView_377/frozen_views --output /tmp/sggs_377_prior`

Expected: 点数与 `canonical_xyz.pt` 一致；feature `[N,32]`、KNN `[N,4]`；所有数值有限；原生标签只在 `[0,4]`。

## Task 3: Compact-6 读出训练

**Files:**
- Modify: `tests/test_sggs_released_code_canonical.py`
- Modify: `utils/sggs_released_code_canonical.py`
- Create: `tools/train_sggs_released_code_canonical.py`

- [ ] **Step 1: 写失败训练契约测试**

固定默认值：30k iterations、hidden dim 64、Adam LR `1e-3`、每类 512 像素、topology lambda `0.1`、K=4、每两步计算一致性、seed 0。测试输入拒绝非 80 视图、点数不一致、非 `[N,32]` prior、SGGS HEAD 不一致；测试 resume 优先最近且未完成的 checkpoint。

- [ ] **Step 2: 写失败输出测试**

用小 tensor 测试两层 MLP `32 -> 64 -> 6` 输出点 logits，`compact6_predictions` 返回归一化概率、argmax、confidence 和 top-2 margin。bank 必须包含 `semantic_probs`、`soft_edit_weights`、`confidence`、`semantic_margin`、`part_label`，来源为 `sggs_released_code_canonical_controlled_input`。

- [ ] **Step 3: 运行测试确认失败**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py -q`

Expected: FAIL，原因是训练入口尚不存在。

- [ ] **Step 4: 实现 GPU 训练**

每步随机加载一张冻结视图，对固定 `[N,32]` prior 运行 MLP 得到 `[N,6]` logits，补零为 `[N,32]` 后使用 SAGA 32-channel rasterizer渲染。对 compact-6 有效像素做按类均衡 CE；每偶数步对点 softmax 计算 topology consistency。只优化 MLP，canonical geometry、prior、appearance 和 deformation 全部冻结。

- [ ] **Step 5: 实现 checkpoint 和 bank**

每 5000 步及结尾保存 MLP、optimizer、随机状态和 iteration。记录 CE/topology/total loss、梯度范数、参数变化、吞吐和峰值显存。结尾输出 `readout.pt`、`part_label_bank.npz`、`summary.json`、`metrics.jsonl` 和 `COMPLETE`；summary 明确写入 controlled-input adaptation 和 released-code limitations。

- [ ] **Step 6: 静态与小规模验证**

Run: `/opt/miniconda3/envs/gaussian_splatting/bin/python -m py_compile tools/train_sggs_released_code_canonical.py utils/sggs_released_code_canonical.py`

Expected: exit 0。随后运行 1 iteration smoke，要求 loss/gradient/parameter delta 有限且非零。

## Task 4: 队列、canary 与正式训练

**Files:**
- Modify: `tests/test_sggs_released_code_canonical.py`
- Create: `tools/run_sggs_released_code_canonical_three_subject.sh`

- [ ] **Step 1: 写失败队列契约测试**

断言队列固定 SGGS repo/HEAD、body model/data roots、输入来源、默认主体顺序 `377 386 394`、100-step canary、30k 正式训练、断点续训、BJT 时间、PID/state/ETA 和失败 trap。`DRY_RUN=1` 不得创建训练 checkpoint 或改变已有 queue state。

- [ ] **Step 2: 运行失败测试**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py -q`

Expected: FAIL，原因是队列尚不存在。

- [ ] **Step 3: 实现队列**

队列先生成官方审计，再为三名被试导出 prior；用 377 跑 100-step canary，按后 80% 稳态吞吐估算 90k 更新并加 15% 缓冲。正式训练顺序固定 377、386、394，每名独立日志/状态/checkpoint，根目录为 `exp/external/sggs_released_code_canonical_three_subject_<BJT timestamp>_bjt`。

- [ ] **Step 4: dry-run 与 canary**

Run: `DRY_RUN=1 bash tools/run_sggs_released_code_canonical_three_subject.sh`

Expected: exit 0，只打印审计/导出/canary/三训练命令。

Run: `CANARY_ONLY=1 bash tools/run_sggs_released_code_canonical_three_subject.sh`

Expected: 100 步完成、无 NaN/Inf、MLP 梯度和参数变化非零、bank schema 有效。

- [ ] **Step 5: 启动正式队列**

Run: `nohup bash tools/run_sggs_released_code_canonical_three_subject.sh > <root>/queue.log 2>&1 &`

Expected: PID 存活，`queue_state.json` 为 running，377 首批指标有限；向用户报告基于 canary 的预计北京时间结束时间。

## Task 5: LOSO 冻结、测试和外部对比

**Files:**
- Modify: `tests/test_sggs_released_code_canonical.py`
- Modify: `utils/sggs_released_code_canonical.py`
- Create: `tools/evaluate_sggs_released_code_canonical.py`

- [ ] **Step 1: 写失败选择测试**

测试三被试 leave-one-subject-out 选择：优先要求两个 donor 都能达到 retention 0.60，再最小化 donor mean actionable leakage，随后最大化 macro mIoU/boundary F1、最小化阈值；若 0.60 对所有阈值均不可行，回退到最大共同可达 retention。测试不允许读取 held-out test CSV。

- [ ] **Step 2: 写失败汇总测试**

测试汇总只在共同可比被试上计算 A5 相对提升，输出逐被试和均值的 actionable/raw leakage、mIoU、boundary F1、可达性、训练时间和峰值显存。缺 60% 行必须写 `feasible=false`，不得插值。

- [ ] **Step 3: 实现验证与冻结**

对每名被试验证集扫描阈值 `0.05,0.10,0.15,0.20,0.25,0.35,0.50`，调用统一严格评测器的 B1/B4 路径。写 `frozen_sggs_loso_config.json`，包含 protocol/checkpoint/reference bank/SGGS bank 指纹、donors、候选轨迹和 selection objective。

- [ ] **Step 4: 实现严格测试与汇总**

冻结后只运行一次 test split，固定 c21-c23 与 180/420/540。聚合 A5、SAGA、Gaussian Grouping 和 SG-GS；生成 `sggs_a5_saga_gaussian_grouping_test_comparison.csv/json`。同时检查 27 个视图、有限值和协议指纹。

- [ ] **Step 5: 完成验证**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py tests/test_part_label_bank.py -q`

Expected: PASS。

Run: `/opt/miniconda3/envs/ictrl/bin/python tools/evaluate_sggs_released_code_canonical.py --experiment-root <root>`

Expected: 三名验证/冻结/测试报告完成；汇总表明确标记不可达的 60% 点；不改写 A5/SAGA/Gaussian Grouping 原结果。

## Task 6: 最终审计与提交

**Files:**
- Generate only: `exp/external/sggs_released_code_canonical_three_subject_<timestamp>_bjt/`

- [ ] **Step 1: 完整性检查**

检查官方审计 JSON、三份 prior、三份 COMPLETE、checkpoint/bank 非空、所有验证候选、三份 frozen config、27 个测试视图和数值有限性。记录 `/remote-home/ming/SGGS` HEAD 未改变且工作树未被适配流程修改。

- [ ] **Step 2: 运行全量专项测试**

Run: `/opt/miniconda3/envs/ictrl/bin/python -m pytest tests/test_sggs_released_code_canonical.py tests/test_saga_canonical.py tests/test_gaussian_grouping_canonical.py tests/test_part_label_bank.py -q`

Expected: PASS。

- [ ] **Step 3: 提交实现**

只提交新增源码、测试、设计和计划，不提交大型实验输出，不带入用户已有工作区修改。提交信息：`feat: evaluate SG-GS released code on canonical avatars`。

- [ ] **Step 4: 论文口径核查**

最终汇报必须分别陈述：官方仓库身份是真的、released code 是否原生可运行、canonical 结果属于 controlled-input adaptation、论文缺失模块未被冒充为官方实现。相对提升只从共同被试、相同 60% 保留率的严格测试行计算。
