# 下一步实验计划（v4.1 主线）

本文档给出接下来建议执行的实验顺序、每一步的目的、操作方法、预期结果，以及需要重点对比的参数与指标。

适用主线：
- `body/cloth v4.1`
- 主题：物理可解释的高斯-骨骼绑定
- 当前主干方法：显式绑定 + 分层绑定 + body/cloth 分区 + 时序一致性 + 可解释导出

当前判断：
- 代码工程完成度已经接近可交付状态
- 方法主线已经稳定
- 接下来重点不再是继续发明新模块，而是：
  1. 固化主结果
  2. 补全 baseline / v3 / v4.1 对比
  3. 形成论文图表与定量证据链

---

## 0. 总体策略

接下来实验建议按下面顺序推进：

1. 固化 `v4.1 main` 主结果包
2. 补跑一个真正可复现的 `baseline main`
3. 生成 `baseline / v3 / v4.1` 正式对比表
4. 挑选论文关键帧与 failure case
5. 做 temporal 显示增强版导出
6. 视情况补一组最小消融

建议原则：
- 不要再开新方法分支
- 不要再追更复杂的物理模拟
- 先把现有主线的证据做完整
- 每一步都产出可复用目录，而不是临时结果

---

## 1. 实验目标拆分

### 1.1 当前最重要的三个目标

- 目标 A：把 `v4.1` 打成最终主结果
- 目标 B：让 `baseline / v3 / v4.1` 三组可正式对比
- 目标 C：整理论文展示资产（图、表、关键帧、crop）

### 1.2 每个目标对应的产物

目标 A 需要产物：
- `test-view` 渲染图
- `test-view` interpretability maps
- `test-view` summary + keyframes
- `test-view` paper montage
- `test-video` temporal maps
- `test-video` temporal summary

目标 B 需要产物：
- baseline 主实验目录
- v3 主实验目录
- v4.1 主实验目录
- `comparison.json / csv / md`
- `metrics_table.tex`
- `binding_table.tex`

目标 C 需要产物：
- 最有代表性的 layer / region / semantic / temporal / thin 图
- failure case crop 图
- 主文图与补充材料图候选

---

## 2. 实验 1：固化 v4.1 主结果

### 2.1 目的

把当前主线 `v4.1 main` 的结果固定下来，形成后续所有论文图表的基准版本。

### 2.2 当前主目录

主目录：
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main`

已经完成：
- `15k` 训练
- `test-view` render
- `test-view` interpretability
- `test-video` temporal interpretability

### 2.3 需要检查的目录

重点查看：
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_full/test-view/renders`
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_full/test-view/binding_maps`
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_full/test-view/binding_analysis`
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_full/test-view/paper_montages`
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_video/test-video/binding_maps/temporal`
- `exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main_interp_video/test-video/binding_analysis`

### 2.4 预期结果

应达到：
- `layer` 不是全 `free`
- `region` 不是全 `cloth`
- `body_prob / soft_prob / cloth_prob` 有平滑过渡
- `semantic` 有一定可读性，不是一片纯色
- `temporal` 在 `test-video` 非零，且能看到局部 slip 差异

### 2.5 如果不达预期怎么办

若 `temporal` 图还是太蓝：
- 只调显示参数，不改方法
- 优先试：
  - `+binding_map_temporal_scale=0.001`
  - 若仍不明显，再试 `+binding_map_temporal_scale=0.0005`

若 `semantic` 仍不够清楚：
- 调显示参数：
  - `+binding_map_semantic_quantile=0.8`
  - `+binding_map_semantic_gamma=0.6`

---

## 3. 实验 2：重建真正可复现的 baseline main

### 3.1 目的

当前的 baseline 目录：
- `exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default`

它更像一个渲染目录，而不是完整训练主目录，因此无法稳定接入当前 comparison 工具链。

所以需要重跑一个真正的 baseline 主实验目录。

### 3.2 训练命令

```bash
python train.py \
dataset=zjumocap_377_mono \
rigid=mlp_field \
non_rigid=hashgrid \
pose_correction=direct \
texture=shallow_mlp \
wandb_disable=true \
opt.iterations=15000 \
save_iterations=[5000,10000,15000] \
checkpoint_iterations=[5000,10000,15000] \
test_interval=5000 \
tag=baseline_15k_main
```

### 3.3 训练后渲染命令

```bash
python render.py \
mode=test \
dataset=zjumocap_377_mono \
rigid=mlp_field \
non_rigid=hashgrid \
pose_correction=direct \
texture=shallow_mlp \
wandb_disable=true \
load_ckpt=exp/<baseline_main_dir>/ckpt15000.pth
```

### 3.4 预期结果

需要至少得到：
- `ckpt15000.pth`
- `test-view/renders`
- 若训练中自动保存成功，最好还要有：
  - `best_test_metrics.json`

### 3.5 baseline 需要对比什么

baseline 的核心作用：
- 提供渲染质量对比基准
- 对比 `PSNR / SSIM / LPIPS / L1`

baseline 不一定需要和 `v4.1` 一样完整的 interpretability 图，因为它本身不是显式绑定方法。

建议论文里这样定位：
- `baseline`：渲染效果与泛化基线
- `v3 / v4.1`：解释性与结构先验方法

---

## 4. 实验 3：正式生成 baseline / v3 / v4.1 对比包

### 4.1 目的

把三组实验结果整理成正式 comparison 与论文表格。

### 4.2 对比对象

建议最终固定为：
- baseline：`mlp_field`
- v3：`explicit binding + semantic/anchor 升级版`
- v4.1：`body/cloth main`

### 4.3 comparison 命令

```bash
/opt/miniconda3/envs/anim/bin/python tools/compare_experiments.py \
  --exp-dirs \
    exp/<baseline_main_dir> \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717 \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main \
  --labels baseline v3 v4.1 \
  --split test-view \
  --output-dir exp/comparisons/v41_main
```

### 4.4 表格导出命令

```bash
/opt/miniconda3/envs/anim/bin/python tools/export_binding_tables.py \
  --exp-dirs \
    exp/<baseline_main_dir> \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-expbind_v3_15k-0311-0717 \
    exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main \
  --labels baseline v3 v4.1 \
  --split test-view \
  --output-dir exp/comparisons/v41_main/tables
```

### 4.5 需要重点对比的参数与指标

渲染指标：
- `PSNR`
- `SSIM`
- `LPIPS`
- `L1`

解释性统计：
- `layer_rigid / layer_soft / layer_free`
- `region_body / region_soft / region_cloth`
- `semantic_stability`
- `thin_score`
- `temporal_slip`

### 4.6 预期结果

希望看到：
- `v4.1` 在视觉稳定性和可解释性上优于 baseline
- `v4.1` 与 `v3` 相比，在 `region` 与 `temporal` 展示上更完整
- 即使 PSNR 不一定全方位压制，也要在人眼质量、结构稳定性、解释性上更强

---

## 5. 实验 4：论文关键帧筛选与拼图固化

### 5.1 目的

把论文主文和汇报 PPT 最需要的图固定下来，减少后续反复挑图时间。

### 5.2 先看哪些文件

- `test-view/binding_analysis/keyframes.json`
- `test-video/binding_analysis/keyframes.json`
- `test-view/paper_montages`

### 5.3 重点挑哪几类图

主文建议优先展示：
- `render`
- `layer`
- `region`
- `body_prob`
- `cloth_prob`
- `semantic`
- `temporal`

补充材料建议展示：
- `thin`
- `soft_prob`
- 更多连续帧 temporal
- 更多 crop 对比

### 5.4 预期结果

需要最后形成：
- 2~3 张主文级 montage 图
- 1 张 temporal 连续帧展示图
- 1 张 failure case 局部 crop 图

---

## 6. 实验 5：failure case 分析

### 6.1 目的

把你人眼已经观察到的问题，变成正式分析，而不是口头印象。

你已经观察到的问题包括：
- 裤子处白色小结没有渲染好
- 手臂局部轮廓有轻微变形

### 6.2 怎么做

建议从这些目录挑图：
- baseline 的 `test-view/renders`
- v3 的 `test-view/renders`
- v4.1 的 `test-view/renders`
- GT 原图：`data/ZJUMoCap/CoreView_377/1`

然后做两类 crop：
- 裤绳 / 白结 / 鞋带等 thin accessory 区域
- 手臂、肩膀、肘部这些轮廓容易变形区域

### 6.3 需要对比的内容

重点观察：
- 边界是否毛糙
- 细结构是否断裂
- 局部颜色是否发白 / 漏渲染
- 肢体轮廓是否抖动
- 极端姿态处是否撕裂或滑移

### 6.4 预期结果

论文里不是要求“没有失败案例”，而是要求：
- 失败模式清楚
- 相比 baseline 已经改善
- 失败原因可解释

---

## 7. 实验 6：temporal 显示增强版导出

### 7.1 目的

当前 `test-video` 的 temporal 已经起作用，但图像还不够亮眼。这个实验只为论文展示服务，不改变主方法。

### 7.2 操作方式

在原有 `test-video` 导出基础上，调显示尺度：

```bash
python render.py \
mode=test \
dataset=zjumocap_377_mono \
rigid=explicit_binding \
non_rigid=hashgrid \
pose_correction=direct \
texture=shallow_mlp \
dataset.test_mode=video \
wandb_disable=true \
+export_interpretability=true \
+binding_map_names=[temporal,layer,region,thin,semantic] \
+binding_map_temporal_scale=0.001 \
+exp_dir=exp/<temporal_vis_enhanced_dir> \
load_ckpt=exp/zju_377_mono-direct-explicit_binding-ingp-shallow_mlp-bodycloth_v41_15k-0311-1123-main/ckpt15000.pth \
opt.iterations=15000
```

如果还不够明显，再试：
- `+binding_map_temporal_scale=0.0005`

### 7.3 预期结果

希望达到：
- temporal 不再是几乎纯蓝
- 肩部、肘部、裤边、腿部边缘等局部运动区域更亮
- 能用于论文展示“时序滑移抑制”的视觉证据

---

## 8. 实验 7：最小消融（可选，但很建议做）

### 8.1 目的

如果要冲论文，这一组很关键。它能回答：你的改进到底是哪一部分起作用。

### 8.2 建议最小消融组合

至少做这四个：
- baseline：`mlp_field`
- `explicit binding` 基础版
- `v3`
- `v4.1`

如果还有精力，再做：
- `v4.1` 去掉 body/cloth 分区
- `v4.1` 去掉 temporal consistency

### 8.3 要对比什么

定量：
- `PSNR / SSIM / LPIPS / L1`

解释性：
- `layer` 分布是否合理
- `region` 分布是否合理
- `semantic stability`
- `temporal slip`

定性：
- 局部边界是否更稳定
- thin accessory 是否更自然
- 未见姿态是否更少撕裂

### 8.4 预期结果

论文叙事理想情况：
- `explicit binding` 带来结构可解释性
- `v3` 提升 anchor 稳定性
- `v4.1` 进一步提升 body/cloth 分区表达与时序稳定性

---

## 9. 实验优先级与实际执行顺序

### 第一优先级（必须做）

1. 固化 `v4.1 main` 主结果
2. 重新跑一个完整 `baseline main`
3. 正式生成 `comparison + tables`

### 第二优先级（强烈建议）

4. 挑关键帧和 failure case
5. 跑 temporal 显示增强版

### 第三优先级（论文加分项）

6. 做最小消融

---

## 10. 成功标准

如果下面这些都完成了，就说明你的项目已经从“研究原型”进入“论文提交准备期”：

- `v4.1 main` 有完整主结果包
- `baseline / v3 / v4.1` 有正式 comparison 与 tables
- `test-view` 有可解释拼图
- `test-video` 有 temporal 连续帧证据
- 有 2~3 张可直接放论文主文的图
- 有 1 张 failure case 分析图
- 有一套可以复现的命令和目录结构

---

## 11. 最推荐的下一步执行顺序

建议就按下面顺序执行：

### Step 1
重跑 baseline 完整训练，得到真正的 `baseline main`

### Step 2
重新生成三组正式 comparison 和 tables

### Step 3
挑 `v4.1` 的主文图、temporal 图、failure case 图

### Step 4
如果 temporal 图不够亮，就跑一版显示增强导出

### Step 5
如果准备投稿，再补最小消融

---

## 12. 备注

当前最重要的不是继续发明新模块，而是把已经做出来的主线：
- 显式绑定
- 分层绑定
- body/cloth 分区
- 时序一致性
- 可解释导出

整理成一套清晰、可复现、可展示、可量化的证据链。

这一步完成后，项目的重点就会从“实现功能”切换到“论文包装与实验论证”。
