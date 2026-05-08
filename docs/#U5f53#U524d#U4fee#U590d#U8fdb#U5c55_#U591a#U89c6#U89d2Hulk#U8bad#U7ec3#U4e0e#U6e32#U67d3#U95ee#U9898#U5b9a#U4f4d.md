# 当前修复进展：多视角 Hulk 训练与渲染问题定位

更新时间：2026-03-26

## 1. 当前主线目标

当前项目仍然沿着这条主线推进：

- 显式绑定
- body / cloth 解耦
- interpretability
- semantic editable assets

更具体地说，当前阶段的核心目标是：

**构建一种结构化、可解释、并且最终可编辑的人体 Avatar 表示。**

在这个目标下，这一轮工作的重点不是“最终编辑”，而是先把下面两件事做扎实：

1. 多视角 RGB 训练能够稳定重建出清楚的人体。
2. Hulk 语义标签能够和训练数据一一对应，真正参与到 body / cloth / compact semantic 的监督里。

---

## 2. 这一轮已经完成了什么

### 2.1 多视角 Hulk 数据链路已经打通

之前的核心问题之一，是 Hulk 标签和当前训练数据没有做到严格的一一对应，导致语义监督虽然“接进来了”，但不够可靠。

这一轮已经完成：

- 为 `CoreView_377` 的多视角相机 `1~20` 准备 Hulk 输入。
- 跑通 Hulk 多视角 parsing。
- 将 Hulk 输出整理到当前训练代码可直接读取的目录结构中。

当前多视角 Hulk 解析结果目录：

- `/remote-home/ming/3dgs-avatar-release-main/data/parsers_from_hulk_multiview/CoreView_377/mask_cihp/Camera_B1`
- `/remote-home/ming/3dgs-avatar-release-main/data/parsers_from_hulk_multiview/CoreView_377/mask_cihp/Camera_B2`
- `...`
- `/remote-home/ming/3dgs-avatar-release-main/data/parsers_from_hulk_multiview/CoreView_377/mask_cihp/Camera_B20`

这一步的意义是：

- 训练阶段读取到的 RGB 相机视角和 Hulk 语义标签视角现在是匹配的。
- 后续 body / cloth / compact semantic 的监督不再停留在单视角弱对齐，而是进入了“多视角对应”的路线。

### 2.2 Hulk 工程侧阻塞点已修掉

为了让 Hulk 在当前环境下顺利跑起来，已经修过这些问题：

- 增加本地兼容文件：
  - `/remote-home/ming/Hulk/json_tricks.py`
- 修复 Hulk 的惰性导入问题：
  - `/remote-home/ming/Hulk/core/solvers/solver_mae_devnew.py`
- 修复单进程分布式初始化问题：
  - `/remote-home/ming/Hulk/core/distributed_utils.py`
- 修复 expname 传递问题：
  - `/remote-home/ming/Hulk/test_mae.py`
- 修复多视角配置重写逻辑，确保 `data_path` 被正确替换：
  - `/remote-home/ming/3dgs-avatar-release-main/tools/run_hulk_zju377_multiview_parsing.py`

### 2.3 多视角 Route A 训练已真正跑通

新增或使用的关键脚本包括：

- `/remote-home/ming/3dgs-avatar-release-main/tools/start_377_multiview_direct_hulk_full_pipeline.sh`
- `/remote-home/ming/3dgs-avatar-release-main/tools/run_377_multiview_direct_hulk_finetune.py`

本轮主训练实验目录：

- `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2`

训练日志：

- `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/logs/377_multiview_direct_hulk_v2.train.log`

最优 checkpoint：

- `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2/best_ckpt.pth`

最优指标：

- `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2/best_test_metrics.json`

其中记录的最佳结果为：

- best iteration: `3500`
- selection metric: `psnr_fg`
- PSNR: `32.7703`
- LPIPS: `0.0198`
- FG_PSNR: `21.1016`
- FG_LPIPS: `0.1398`

这说明：

- 训练过程已经不是“直接崩掉”。
- 多视角 + Hulk 监督已经能产出可用的人体结果。
- 但前景人体质量还不够锐，后期还有明显退化。

---

## 3. 这一轮解决掉了哪些具体问题

### 3.1 解决了多视角训练一开始就 OOM 的问题

出现过一次非常关键的错误：

- 多视角训练在数据集初始化阶段就 `CUDA Out Of Memory`

根因已经定位清楚：

- 多视角数据集配置里仍然是 `preload: true`
- `ZJUMoCapDataset.__init__` 会在初始化阶段把全部相机数据预读成 `Camera`
- `Camera` 在构造时又会立刻把图像和 mask 搬到 CUDA
- 多视角下样本数暴涨，导致训练还没真正开始，显存就先炸了

关键位置：

- 数据集预载入：
  - `/remote-home/ming/3dgs-avatar-release-main/dataset/zjumocap.py`
- Camera 构造时搬运到 CUDA：
  - `/remote-home/ming/3dgs-avatar-release-main/scene/cameras.py`

已经做的修复：

- 修改配置：
  - `/remote-home/ming/3dgs-avatar-release-main/configs/dataset/zjumocap_377_multiview_cam1sem.yaml`
  - 设为 `preload: false`
- 启动脚本中强制覆盖：
  - `/remote-home/ming/3dgs-avatar-release-main/tools/run_377_multiview_direct_hulk_finetune.py`
  - 显式传入 `dataset.preload=false`

修复后结果：

- 多视角训练已经能完整跑完，不再在初始化阶段炸显存。

### 3.2 解决了“渲染图看起来和真实图完全不一样”的误判

之前看渲染图时，很容易产生一个直觉：

- “为什么真实图是摄影棚背景，render 是纯黑背景，是不是渲染分支坏了？”

这个问题已经定位清楚，不是 render 分支单独坏掉，而是**训练与评估管线本身就把前景外背景抠成了黑色**。

关键代码：

- `/remote-home/ming/3dgs-avatar-release-main/dataset/zjumocap.py`

其中逻辑是：

- `image[~mask] = 255. if self.white_bg else 0.`

而当前配置：

- `/remote-home/ming/3dgs-avatar-release-main/configs/dataset/zjumocap_377_multiview_cam1sem.yaml`
- `white_background: false`

这意味着：

- 训练用到的 `gt_image` 本来就是“黑背景的人体前景图”
- render 输出和这套 `gt_image` 对比时是合理的
- 如果拿 render 去和原始 JPG 直接比，就一定会看到背景域不一致

已经做过的验证结果：

- render 与 `source_rgb` 的 MAE 只有约 `0.0023 ~ 0.0031`
- render 与原始 raw JPG 的 MAE 约 `0.053`
- `source_rgb` 与 raw JPG 的 MAE 也约 `0.051 ~ 0.054`

结论：

- 当前 render 和“管线内部真值”其实是接近的。
- 它和原始摄影棚 JPG 差得大，主要原因是背景预处理约定不同。

### 3.3 解决了“旧版彻底破碎/错位”的严重渲染问题

和更早那几版相比，这一版已经不是那种大面积碎裂、人体错位、骨架崩掉的状态了。

当前可观察到的情况是：

- 人体整体轮廓基本成立。
- 姿态和相机视角是对上的。
- binding maps 和 compact semantic maps 已经有成块分区结果。

相关目录：

- render 结果：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/renders`
- binding maps：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/binding_maps`
- semantic editable assets：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/semantic_editable_assets`

这一步的意义是：

- 当前的问题已经从“系统级崩坏”转成了“质量级不足”。
- 现在更该做的是提高清晰度和边界质量，而不是继续围绕黑背景误判打转。

---

## 4. 当前仍然存在的主要问题

虽然这版已经能正常训练、正常渲染、正常导出 semantic 相关结果，但当前最主要的问题仍然很明显：

### 4.1 前景人体偏糊

现象包括：

- 四肢边缘偏软
- 手部偏糊
- 鞋子区域偏亮且发虚
- 衣物细节和阴影层次不够
- 整体对比度比原始图更平

这不是背景问题，而是人体本体质量还不够好。

### 4.2 最优结果出现在很早期，后面越训越差

当前 best checkpoint 出现在 `3500` iter，而不是训练末尾。

日志里可以看到：

- `3500` 附近是峰值
- 后续 `FG_PSNR` 逐步掉到 `20` 左右

这说明：

- 后续训练没有继续提升前景质量
- 甚至出现了退化 / 过拟合 / 语义约束后期副作用

### 4.3 语义区分虽然比旧版规整，但人体清晰度还不足以支撑“高质量可编辑”

目前 compact semantic / region binding 已经比之前更成块、更规整，但如果人体前景本身不够清楚，那么：

- 后续 interpretability 输出质量受限
- semantic editable assets 的实用性会打折
- 真正到编辑阶段时，很容易受前景模糊和边界不准影响

---

## 5. 当前问题的根因定位

这一轮已经对“为什么前景会偏糊”做了更深层定位，结论如下。

### 5.1 第一主因：高斯密度被冻结，几何细节长不出来

这一版训练配置非常保守，明确禁止了 resume 后的 redensify。

关键配置：

- `/remote-home/ming/3dgs-avatar-release-main/configs/option/routeA_377_multiview_hulk_direct_v1.yaml`

里面包括：

- `disable_densify_on_resume: true`
- `densify_from_iter: 1000000`
- `densify_until_iter: 0`

而训练日志显示：

- 初始点数只有 `50000`
- 是从旧 checkpoint 继续微调的

这意味着：

- 当前训练不会给手、鞋、细边界、衣物局部新增足够的高斯点
- 只能在已有点集上做保守微调
- 结果就是大轮廓在，但细节不锐

### 5.2 第二主因：纹理头过浅，表达能力不够

当前使用的是：

- `/remote-home/ming/3dgs-avatar-release-main/configs/texture/shallow_mlp.yaml`

其配置非常轻：

- `feature_dim = 32`
- `non_rigid_dim = 16`
- `latent_dim = 16`
- `2` 层 MLP
- `64` hidden neurons

而项目里更强的纹理配置：

- `/remote-home/ming/3dgs-avatar-release-main/configs/texture/mlp.yaml`

其能力明显更强：

- `feature_dim = 128`
- `non_rigid_dim = 64`
- `latent_dim = 64`
- `4` 层 MLP
- `256` hidden neurons

所以当前纹理头更容易产生：

- 颜色块平滑
- 局部阴影不够
- 纹理细节不够
- 视觉上显得“糊”

### 5.3 第三主因：前景边界监督偏软

当前图像重建损失以：

- `L1`
- 小权重 perceptual
- `L1 mask loss`

为主。

这套组合能保证稳定，但不擅长把边界训得非常锋利。

效果上就会表现为：

- 轮廓基本对
- 但边缘像被平均化了一层

### 5.4 Hulk 语义监督不是当前“发糊”的第一主因

这一点已经基本确认。

原因：

- `best_ckpt` 出现在 `3500`
- 而 compact semantic 的权重要到后面才慢慢起来
- 说明前景一开始就不够锐，不是因为语义监督过强把图像拉坏了

更准确的判断是：

- Hulk 监督在当前版本中没有成为主要破坏项
- 但它也还没有强到真正把语义可解释性价值充分推出来
- 当前仍然是“重建质量不够强，限制了语义表现上限”

---

## 6. 当前可直接使用的实验与结果

### 6.1 当前建议优先查看的训练结果

- 训练目录：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2`
- 最优 checkpoint：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2/best_ckpt.pth`
- 指标文件：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2/best_test_metrics.json`

### 6.2 当前建议优先查看的渲染结果

- 完整渲染输出：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full`
- renders：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/renders`
- binding maps：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/binding_maps`
- semantic assets：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/fixedv2/377_multiview_direct_hulk_v2_render_best_full/test-view/semantic_editable_assets`

### 6.3 参考对照

原始多视角真值：

- `/remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap/CoreView_377/21`
- `/remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap/CoreView_377/22`
- `/remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap/CoreView_377/23`

之前生成过的对照图：

- `/tmp/compare_377_v2_render/compare_c21_f000000.png`
- `/tmp/compare_377_v2_render/compare_c21_f000270.png`
- `/tmp/compare_377_v2_render/compare_c23_f000270.png`

---

## 7. 当前阶段的结论

到目前为止，可以下一个相对清楚的结论：

### 已经解决的层面

- Hulk 多视角语义数据已经打通。
- 多视角训练已经不再 OOM。
- 渲染分支已经不是“彻底坏掉”的状态。
- binding / compact semantic 输出已经比旧版规整。

### 还没有解决的层面

- 前景人体清晰度仍然不足。
- 细边界与局部细节仍然发糊。
- 后期训练会退化，最佳结果出现在早期。
- 当前结果还不足以支撑高质量、强可编辑的人体 Avatar 表示。

### 当前最准确的问题归纳

当前不是“语义标签接不进来”，也不是“render 分支单独崩掉”，而是：

**多视角 Hulk 语义已经接进来了，但当前训练配置过于保守，导致几何密度和纹理表达能力不足，前景人体仍然偏糊。**

---

## 8. 下一步建议

接下来更合理的方向不是继续纠结黑背景，而是直接围绕“前景更清楚、语义更稳”去改训练方案：

1. 开启一段安全 redensify，而不是完全冻结高斯拓扑。
2. 提升纹理头表达能力，不再继续使用过浅的 `shallow_mlp`。
3. 增强前景边界监督，让手、鞋、四肢边缘更锋利。
4. 保留已经打通的多视角 Hulk 对齐链路，不退回单视角弱监督。
5. 用 `best_ckpt` 逻辑继续选模，不再默认拿最终 checkpoint 当最佳结果。

如果后续继续按这条线推进，那么当前这一轮工作的意义可以概括为：

**已经把“多视角 RGB + 多视角 Hulk 标签 + 显式绑定语义监督”这条路线跑通；下一步要做的，是把它从“能用”推到“清楚、稳定、可解释”。**


---

## 9. 最近新增的问题定位与修复

### 9.1 `crop` 路线的训练崩溃问题已经被逐层定位并修复

在继续沿着多视角 Hulk 路线推进时，新增了一条更激进的 `person crop` Stage A 训练线，实验目录包括：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_multiview_recon_hq_crop_v1_bg`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_multiview_recon_hq_crop_nanfix_probe`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_multiview_recon_hq_crop_nanfix_probe_v2`

这条路线先后暴露出几类新问题：

1. 训练在约 `1900 iter` 附近出现 `Loss=nan` 并退出。
2. 修掉 `NaN` 后，训练又在约 `1950 iter` 附近因为 `AIAP / KNN` 逻辑崩溃。
3. 即便训练跑完，render 结果仍然是全黑。

针对这些问题，已经做过的修复包括：

- 在 `train.py` 中增加非有限值保护：
  - 非有限 `loss` 直接跳过该 iteration
  - 在 optimizer step 前将非有限梯度清零
- 在 `scene/gaussian_model.py` 中增加非有限高斯剔除：
  - `prune_nonfinite_points()`
  - densify 前先清掉坏点
- 在 `utils/loss_utils.py` 中补了 `AIAP` 的安全保护：
  - 当当前高斯点数过少时，不再强行 `KNN(K=5)`
  - 避免因点数太少导致 `ValueError: K must be positive`
- 在 `utils/general_utils.py` 中修复了前景 `SSIM / LPIPS` 裁剪 bug：
  - 避免 `Mean of empty slice`
  - 避免前景指标被空 crop 污染
- 在 `dataset/zjumocap.py` 中修了 direct parser sample filter 的未定义变量问题。
- 在 `tools/run_377_stageA_recon_hq.py` 中补了默认选项逻辑：
  - 当 `dataset=zjumocap_377_multiview_hq_crop` 时，默认自动叠加 `stageA_377_multiview_recon_hq_crop_safe_v1`
  - 避免用户以为在跑 safe 版，实际上还是旧激进参数。

这些修复的实际结果是：

- 原先 `1900` 左右的 `NaN` 崩溃点已经不再出现。
- 原先 `1950` 左右的 `AIAP/KNN` 崩溃点也已经被压住。
- 也就是说，`crop` 路线已经从“会在中途直接炸掉”变成了“能完整跑完”。

### 9.2 但 `crop` 路线虽然不再崩溃，仍然不是当前正确主线

进一步检查发现，`crop` 路线真正的问题并不是“Hulk 没接进来”，而是：

**模型在训练后期塌成了近乎透明的解，导致渲染结果全黑。**

已经确认的证据：

- `render_c21_f000000.png`、`render_c21_f000270.png`、`render_c22_f000000.png` 等图像像素统计均为：
  - `min=0`
  - `max=0`
  - `mean=0`
- 对应 checkpoint：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_multiview_recon_hq_crop_nanfix_probe_v2/best_ckpt.pth`
- 其中高斯点数只剩：
  - `2352`
- 原始 opacity 参数均值约：
  - `-3.35`
- 经过 sigmoid 后接近低透明度，导致渲染几乎不可见。

这说明：

- 问题不是 render 存图错了。
- 问题也不是 Hulk 标签本身把图像监督破坏掉了。
- 真正的问题是 `crop + 当前 pruning / opacity / mask dynamics` 把模型推向了“透明也安全”的错误解。

因此，这一轮一个非常关键的新判断是：

**当前阶段不能继续把“多视角 Hulk + crop + 更强约束”一起硬塞进主重建。否则会在主重建还未站稳时，先把高斯本体训练塌掉。**

### 9.3 当前已切换为新的阶段性策略：先回稳 Stage A，再重新引回 Hulk

在明确 `crop` 路线不适合作为当前主线后，已经做了策略调整：

- Stage A 回退到 baseline 风格：
  - 全图
  - 原始 `png mask`
  - 无 Hulk
  - 无 crop
- 但仍然保留当前项目最重要的表示层差异：
  - `rigid=explicit_binding`

这个切换的核心思想是：

1. 先把“能稳定渲染出清楚人体”这件事重新做扎实。
2. 再把 Hulk 语义监督作为后续 Stage B 轻量接回。
3. 最后再继续往 interpretability / semantic assets / editable assets 推进。

换句话说，当前并不是放弃 Hulk 主线，而是把它从“主重建前置约束”改成“稳定重建后的轻量语义增强”。

---

## 10. 当前已经重新站稳的 Stage A 主线

### 10.1 新的 baseline 风格 Stage A 已经真正训练完成

新增的关键文件包括：

- 配置：
  - `/remote-home/ming/3dgs-avatar-release-main/configs/option/stageA_377_baseline_explicit_mono_v1.yaml`
- 启动脚本：
  - `/remote-home/ming/3dgs-avatar-release-main/tools/run_377_stageA_baseline_explicit.py`
  - `/remote-home/ming/3dgs-avatar-release-main/tools/start_377_stageA_baseline_explicit.sh`

这条线的核心配置是：

- `dataset=zjumocap_377_mono`
- `train_views=['1']`
- `img_hw=[512, 512]`
- `dataset.parsing_prior.enable=false`
- `dataset.person_crop.enable=false`
- `rigid=explicit_binding`
- `non_rigid=hashgrid`
- `pose_correction=direct`
- `texture=shallow_mlp`

实验目录：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_v1`

当前已经训练完成，并生成：

- `best_ckpt.pth`
- `best_test_metrics.json`
- `ckpt15000.pth`
- `point_cloud/`

最优结果：

- best iteration: `6000`
- PSNR: `30.7498`
- FG_PSNR: `19.9192`
- SSIM: `0.9774`
- LPIPS: `0.0265`

和前面的 `crop` 黑图线相比，这一版的意义是非常明确的：

- 主重建已经重新站稳。
- 模型不再塌成透明解。
- 人体已经能稳定渲出来。

### 10.2 当前这版 Stage A 渲染结果已经恢复到“可用”水平

当前 best checkpoint 渲染目录：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_v1_render_best`
- renders：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_v1_render_best/test-view/renders`

渲染指标：

- PSNR: `29.0239`
- SSIM: `0.9732`
- LPIPS: `0.0312`

人工查看代表帧后，当前结论是：

- 人已经能完整渲染出来。
- 正面、背面、整体姿态都基本成立。
- 不再是前面那种全黑、全空、彻底塌掉的结果。
- 但仍然存在：
  - 脸部偏糊
  - 四肢边缘偏软
  - 鞋子偏虚
  - 颜色略发暗、发灰

因此，这一版可以被定义为：

**主重建已经回到稳定可用状态，但清晰度仍不足以作为最终“高质量可编辑 Avatar”底座。**

---

## 11. 当前正在推进的清晰度增强路线

### 11.1 已经启动基于当前 best checkpoint 的 sharpen finetune

为了在不破坏当前稳定主重建的前提下，继续提升脸部、鞋子、边缘等细节，已经新增一条 `sharp finetune` 路线。

新增文件包括：

- 高分辨率单视角数据集配置：
  - `/remote-home/ming/3dgs-avatar-release-main/configs/dataset/zjumocap_377_mono_hq.yaml`
- 提清晰度 option：
  - `/remote-home/ming/3dgs-avatar-release-main/configs/option/stageA_377_baseline_explicit_mono_sharp_v1.yaml`
- 启动脚本：
  - `/remote-home/ming/3dgs-avatar-release-main/tools/start_377_stageA_baseline_explicit_sharp.sh`

当前 sharpen 实验目录：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_sharp_v1`

其训练方式是：

- 从以下 checkpoint 继续：
  - `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_v1/best_ckpt.pth`
- 将训练分辨率从 `512` 提到 `768`
- 打开 `lanczos`
- 保持 `explicit_binding + shallow_mlp`
- 只做轻量 sharpen，不重新引回 Hulk

### 11.2 这一版 sharpen 的核心思路

当前 sharpen 不是在重新发明一条新主线，而是只围绕当前已稳定的 Stage A 做保守增强：

- 使用更高分辨率输入，让脸部和边缘拥有更高像素预算。
- 增加轻量 `dssim` 和更强 perceptual，推动局部纹理与结构更清晰。
- 将 mask loss 从 `l1` 改到 `bce`，加强边界约束。
- 下调 `AIAP` 权重，减少对细节长出的抑制。
- 开一个很短、很保守的 redensify 窗口，让后续细节区域有机会长出额外高斯，而不是继续完全冻结拓扑。

这个版本的目的不是直接解决所有问题，而是：

**先把当前“能渲人但偏糊”的 Stage A 往“更清楚”推进一小步，验证高分辨率单视角锐化是否有效。**

---

## 12. 当前还没有解决的问题

### 12.1 细节仍然不够锐

虽然新的 baseline 风格 Stage A 已经把主重建救回来了，但现在最主要的问题仍然是：

- 脸部不够清楚
- 手臂和腿部边缘偏软
- 鞋子区域偏虚
- 局部纹理层次不足

这说明当前“恢复主重建”已经完成，但“提升局部细节”还没有完成。

### 12.2 当前稳定版本仍然是单视角训练

目前最稳定的是：

- `dataset=zjumocap_377_mono`
- 即只用 `view 1`

这意味着：

- 它适合作为当前 Stage A 的稳态底座。
- 但它还不是最终想要的“多视角一致、可编辑”的终版形态。

所以当前阶段判断应当是：

- 单视角 Stage A：用来站稳主重建
- 多视角 Hulk Stage B：后续在稳定底座上重新接回

### 12.3 Hulk 语义当前尚未重新接回新的稳定主线

当前 newest stable 路线中，Hulk 是暂时关闭的：

- `dataset.parsing_prior.enable=false`

这不是方向放弃，而是阶段性后置。

因此，当前仍未完成的工作包括：

1. 在新的稳定 Stage A 上，做一版轻量 Hulk finetune。
2. 验证 Hulk 重新接回后，是否还能保持当前的人体清晰度。
3. 在此基础上继续做 semantic assets 与 interpretability 导出。

---

## 13. 当前阶段的最新结论

到这一轮为止，可以把结论更新为：

### 已经明确解决的问题

- 多视角 Hulk 路线中的 `OOM`、`NaN`、`AIAP/KNN` 崩溃等工程问题已经基本被定位并修过。
- `crop` 路线失败的根因已经被识别，不再把它误判成 Hulk 标签本身的问题。
- baseline 风格 Stage A 已经重新把人体主重建救回来了。
- 当前 `explicit_binding` 并没有阻止人体重建成立，说明显式绑定主线仍然可继续推进。

### 当前仍未完全解决的问题

- 当前最好的人体 still 偏糊，尤其是脸和局部边界。
- 当前最稳的主线还是单视角，不是最终目标的多视角语义可编辑版本。
- Hulk 语义还没有重新在新的稳定主线上接回并验证成功。

### 当前最准确的阶段归纳

当前项目已经从：

- “多视角 Hulk 训练能否跑通”

进一步推进到了：

- “已经确认主重建必须先站稳，当前新的 baseline 风格 Stage A 已经重新渲出可用人体；接下来要在这个底座上继续提清晰度，然后再把 Hulk 语义监督轻量接回。”

也就是说，当前最合理的路线不再是“继续把所有约束一次性压进训练”，而是：

1. 先稳定主重建。
2. 再提清晰度。
3. 再接 Hulk。
4. 最后再做可解释和可编辑资产闭环。

---

## 14. 2026-03-27：sharp_v1 渲染对比后的最新判断

这一轮已经把当前 `sharp_v1` 渲染、上一版稳定 `mono_v1` 渲染、以及原始 GT 图做了逐帧对比。

对比面板：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels/c02_000000_compare.png`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels/c02_000270_compare.png`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels/c12_000270_compare.png`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels/c23_000540_compare.png`

局部放大图：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels_zoom/c02_000000_zoom.png`
- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/_compare_panels_zoom/c12_000270_zoom.png`

当前 `sharp_v1` 渲染目录：

- `/remote-home/ming/3dgs-avatar-release-main/exp/stageA/377_baseline_explicit_mono_sharp_v1_render_best/test-view/renders`

其渲染已经完整结束，共 `418` 帧。

### 14.1 对比结论

这次对比后可以明确得出结论：

- `sharp_v1` 没有明显优于上一版 `mono_v1`
- 个别视角下，`sharp_v1` 的外轮廓略紧一点
- 但脸部、肩膀、手臂、鞋子等关键区域并没有真正变清楚
- 与 GT 相比，人物本体仍然存在明显的高频细节缺失
- 当前问题已经不是“渲不出来”，而是“新视角下人物能出来，但高频纹理与边缘仍然不够”

指标也支持这个判断：

- `377_baseline_explicit_mono_v1`
  - `PSNR = 30.7498`
  - `FG_PSNR = 19.9192`
  - `LPIPS = 0.0265`
  - `FG_LPIPS = 0.1459`
- `377_baseline_explicit_mono_sharp_v1`
  - `PSNR = 30.6756`
  - `FG_PSNR = 19.5546`
  - `LPIPS = 0.0317`
  - `FG_LPIPS = 0.1766`

也就是说，这次 `sharp finetune` 并没有把前景质量拉高，反而略有退化。

### 14.2 当前问题的更深层根因

这轮比对后，已经可以把当前“为什么还是糊”定位得更准确。

#### 根因 1：当前稳定主线本质上还是单视角训练

当前最稳定的 Stage A 是：

- `dataset=zjumocap_377_mono`
- `train_views=['1']`
- `val/test` 仍然看的是 `2~23`

这意味着：

- 当前训练只看过 `view 1`
- 但渲染对比时，我们主要在看 `c02 ~ c23` 的新视角
- 因此现在看到的“前景偏糊”，本质上包含了明显的 novel-view 泛化损失

换句话说，当前版本不是“把多视角都学清楚了，只是渲染发糊”，而是：

**单视角底座本来就不擅长把未见视角渲得接近 GT。**

#### 根因 2：`512 -> 768` 的 finetune 不能凭空补回高频细节

当前 `sharp_v1` 的做法是：

- 从 `512x512` 的 `mono_v1` best checkpoint 继续训练
- 再切到 `768x768`
- 保持 `shallow_mlp`

这条路线的问题是：

- 底层几何与纹理表达是在低分辨率单视角基础上长出来的
- 后续的 HQ finetune 只能做有限修补
- 它不能凭空恢复那些一开始就没有学到的多视角高频结构

所以它更像是“轻微锐化尝试”，而不是“真正重建更清楚的人体”。

#### 根因 3：这次 sharp 配置更偏保守 regularization，不是真正的重建升级

当前 `sharp_v1` 里改动包括：

- `lambda_dssim` 提高
- `lambda_perceptual` 提高
- `lambda_mask` 提高
- `lambda_aiap_xyz / cov` 降低
- 打开一个很短的 redensify 窗口

但这条路线没有解决两个根问题：

- 训练视角还是单视角
- 纹理分支仍然是较浅的 `shallow_mlp`

因此它更容易变成“让前景更稳、更干净一点”，而不是“真正把脸和边界变清楚”。

### 14.3 当前最可行的解决办法

结合目前所有实验结果，当前最可行的方案已经比较明确。

#### 方案 A：先做真正的多视角 RGB Stage A，不接 Hulk

这是当前优先级最高的方案。

具体应该是：

- 多视角 RGB
- full image
- 原始 PNG mask
- 不开 crop
- 不开 Hulk 语义监督
- 保留 `explicit_binding`

这一步的目标是：

- 先把“新视角也能渲清楚的人体”重建出来
- 让 Stage A 先从“单视角 sanity base”升级成“多视角稳定几何 base”

如果不先做这一步，那么后面继续在单视角底座上加锐化、加 Hulk，收益都会受限。

#### 方案 B：如果继续保留单视角，只能把它当作 sanity baseline

单视角 `mono_v1` 仍然有价值，但它的价值主要是：

- 验证 `explicit_binding` 不会阻止主重建
- 验证 full image + 原 mask + no Hulk 这条线是稳定的

它不适合被当作“最终新视角质量基线”。

因此如果继续看单视角路线，就要接受一个前提：

**它更多是在证明系统稳定，而不是在证明最终 novel-view 质量。**

#### 方案 C：真正想提高清晰度，应优先改训练范式，而不是继续小修 sharp_v1

当前更值得投入的，不是继续在 `sharp_v1` 上小修小补，而是：

1. 直接做多视角 RGB Stage A
2. 从头在 HQ 分辨率训练，而不是从 `512` 结果往 `768` 补
3. 等多视角 base 稳定后，再决定要不要升级 texture 分支容量

这条路线比继续调 `lambda_dssim / mask / perceptual` 更接近问题本源。

#### 方案 D：Hulk 应该在 Stage B 轻量接回，而不是现在继续前置

当前 Hulk 并不是第一阻塞点。

真正的顺序应该是：

1. 先得到稳定、清楚的多视角 RGB base
2. 再在这个 base 上做轻量 Hulk finetune
3. 再继续 interpretability 与 semantic editable assets

这样更符合当前项目“结构化、可解释、最终可编辑”的主线推进方式。

