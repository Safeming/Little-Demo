# CoreView_386 Multi-Subject Strict Semantic Experiment Design

日期：2026-07-19

## 1. 目标

将原始主体：

```text
/remote-home/ming/dataSet/CoreView_386
```

转换为与 CoreView_377 相同的项目数据布局，使用：

```text
/remote-home/ming/Hulk
```

生成全 23 相机的人体解析监督，然后训练独立的 CoreView_386 Gaussian Avatar base 和 semantic asset adapter，最终按 CoreView_377 已冻结的方案 A 参数完成 calibration、validation 和一次性 held-out test。

本实验用于多主体论文复现，禁止复用 CoreView_377 checkpoint、Gaussian geometry、adopted geometry 或主体特定 validation 参数。

## 2. 固定原则

### 2.1 主体隔离

CoreView_386 必须拥有独立的：

```text
base checkpoint
semantic checkpoint
part label bank
voting posterior bank
target/support calibration bank
validation/test assets
evaluation reports
```

禁止加载：

```text
assets/adopted_geometry/377/*
CoreView_377 base checkpoint
CoreView_377 semantic checkpoint
CoreView_377 Gaussian bank
```

### 2.2 全局语义参数冻结

继承 CoreView_377 方案 A 的全局参数：

```text
parts = face hair upper lower shoes skin
outer penalty power = 0.2
support penalty power = 0.2
target retention floor = 0.6
soft threshold = 0.5
support threshold = 0.3
boundary radius = 6
```

相邻支持关系固定为：

```text
face:hair
hair:face
upper:skin
upper:lower
lower:upper
lower:skin
shoes:lower
shoes:skin
skin:upper
skin:lower
```

CoreView_386 validation 只报告结果，不重新选择这些参数。test 只运行一次，运行后不得返回 calibration 或 validation 调参。

## 3. 数据转换

### 3.1 项目训练数据

使用现有 `tools/prepare_zju_from_raw.py`：

```text
raw input:
  /remote-home/ming/dataSet/CoreView_386

output:
  /remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap/CoreView_386
```

训练 foreground 使用原始 `mask`，不用 `mask_cihp`：

```text
--mask-source mask
```

RGB 和 foreground mask 使用绝对路径 symlink，避免复制约 6 GB 数据。SMPL `new_params/new_vertices` 转为项目需要的 `models/*.npz`，相机平移使用 `1e-3` 缩放，与现有 CoreView_377 项目数据一致。

转换完成后必须检查：

```text
23 camera directories
646 RGB frames per camera
646 foreground masks per camera
646 models/*.npz
cam_params.json camera_translation_scale = 0.001
```

### 3.2 Hulk parser 数据

Hulk 输入从转换后的项目 RGB 读取：

```text
Hulk/data/zju386_multiview_cihp/CoreView_386/cam1 ... cam23
```

使用与 377 相同的：

```text
Hulk_vit-B checkpoint
CIHP task index 18
CIHP palette
480x480 flip-test config
```

推理实验前缀固定为：

```text
zju386_mv_hulk_cam1 ... zju386_mv_hulk_cam23
```

收集到独立 parser root：

```text
data/parsers_from_hulk_multiview/CoreView_386/mask_cihp/Camera_B1 ... Camera_B23
```

不得覆盖原始数据集中的 `/remote-home/ming/dataSet/CoreView_386/mask_cihp`。收集后要求恰好 23 个相机目录、每目录 646 个 palette-index PNG，并检查标签值属于 CIHP 映射范围。

## 4. Base Reconstruction

### 4.1 数据划分

base reconstruction 沿用固定相机划分：

```text
train cameras = 1-20
held-out cameras = 21-23
train frames = 0-569
image size = 768x768
```

Hulk/parser 不进入 base RGB reconstruction，base 只使用 RGB 和原始 foreground mask。

### 4.2 训练

先运行独立 smoke 实验，验证：

```text
dataset load
camera calibration
SMPL models
forward/backward
checkpoint write
```

smoke 通过后启动独立正式实验：

```text
40,000 optimization steps
checkpoints at 10k / 20k / 30k / 40k
subject = CoreView_386
```

使用现有 StageA2 multiview explicit-binding recipe 和相同超参数，不使用 377 adopted geometry。按 CoreView_377 40k 实测，预计正式 base 训练约 8.5 小时。

## 5. Subject-Generic Semantic Training And Export

现有 `run_377_v338_semantic_train.sh` 和 signed-geometry export launcher 强制依赖 `assets/adopted_geometry/377`，不能用于 386。

新增 subject-generic launcher，职责限定为：

1. 从 CoreView_386 base `.hydra/config.yaml` 和 `ckpt40000.pth` 加载同一模型结构。
2. 冻结 Gaussian geometry、appearance、pose、rigid 和 non-rigid 参数。
3. 只训练 semantic logits adapter 和 semantic asset logits adapter。
4. 使用 Hulk parser root 和固定 compact mapping。
5. semantic train 只读取 camera 1-16、frame 0/120/240/360/480。
6. 训练 2,000 steps，输出 `ckpt42000.pth`。

随后用 subject-generic export launcher 分别导出：

```text
calibration: cameras 1-16, frames 0/120/240/360/480, 80 records
validation: cameras 17-20, frames 60/300, 8 records
test: cameras 21-23, frames 180/420/540, 9 records
```

export launcher 从 386 semantic checkpoint 直接渲染，不应用任何 377 formal preset 或 adopted geometry。

## 6. 方案 A 严格评估

### 6.1 Bank 构建

从 calibration assets 构建：

```text
raw trained semantic bank
multi-view voting posterior bank
```

最终方案 A 以 voting posterior bank 为输入执行 support-aware calibration，输出：

```text
edit_target_weights
edit_support_weights
```

### 6.2 固定配置物化

为 386 生成 frozen config 时，只替换：

```text
protocol fingerprint
checkpoint fingerprint
bank fingerprint
```

评估参数直接复制 377 方案 A 的：

```text
soft threshold 0.5
support threshold 0.3
boundary radius 6
```

不得调用 validation selector 进行主体特定选择。

### 6.3 Validation 和 test

validation 报告：

```text
mIoU
micro IoU
per-part IoU
boundary F1 / boundary IoU
matched-retention actionable leakage at 0.5 and 0.6
```

无论 validation 是否优于 B1，都不改变冻结参数。validation 资产和结果审计完成后，test 使用 frozen config 只运行一次。

## 7. 输出目录

统一输出到：

```text
exp/acceptdata/coreview386_multisubject_strict_20260719/
```

主要子目录：

```text
base_smoke/
base_train_40k/
semantic_train_strict/
assets/calibration/
assets/validation/
assets/test/
banks/raw_trained/
banks/multiview_voting/
banks/voting_evidence_target_support/
evaluation/validation/
evaluation/test/
audit/
```

## 8. 失败处理

按顺序设置停止条件：

1. 数据转换数量、相机或 SMPL 模型不完整时停止。
2. Hulk 任一相机缺帧、出现非 palette-index 输出或标签范围异常时停止。
3. base smoke 出现 NaN、CUDA OOM、点数为零或 checkpoint 未写出时停止。
4. 正式 base 训练失败时保留最后 checkpoint 和日志，不直接进入 semantic train。
5. semantic train 或 export provenance 读取了协议外记录时停止。
6. test 在 frozen config 生成前不得执行。

失败后只修复工程问题，不使用 386 test 指标调整语义超参数。

## 9. 验证与时间预估

预估耗时：

```text
数据转换和检查：10-20 分钟
Hulk 全 23 相机：35-60 分钟
base smoke：5-10 分钟
base 40k：约 8.5 小时
semantic 2k：约 20-30 分钟
export、bank、calibration、validation、test：约 30-60 分钟
总计：约 10-11 小时
```

正式 base 训练启动后，根据前 500-1000 steps 的实测速率重新计算结束训练的北京时间。训练结束时间指 base 40k 和 semantic 2k 均完成；后续评估结束时间单独记录。

验收命令必须覆盖：

```text
shell syntax checks
subject-generic launcher unit tests
protocol split tests
bank/evaluator regression tests
Hulk output count and label audit
checkpoint existence and iteration audit
validation/test provenance audit
```
