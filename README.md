# semantic-editable-avatar

## 运行项目需要准备的数据

这个项目的训练、测试和渲染主要依赖两类数据：人体多视角/单视角图像数据，以及 SMPL/骨骼参数数据。默认配置使用的是 ZJU-MoCap 格式的数据；代码里也保留了 People Snapshot 格式的数据加载器。

下面按实际代码会读取的文件来整理。

## 1. 默认运行需要的数据：ZJU-MoCap / ARAH 格式

默认配置在 `configs/config.yaml` 中指定：

```yaml
dataset: zjumocap_377_mono
```

对应的数据配置文件是 `configs/dataset/zjumocap_377_mono.yaml`，默认读取：

```text
data/ZJUMoCap/CoreView_377
```

如果运行其他 ZJU 配置，例如 `zjumocap_386_mono`、`zjumocap_393_mono` 或 `zjumocap_377_multiview_hq`，就需要把对应 subject 放在同一个根目录下，例如：

```text
data/ZJUMoCap/CoreView_386
data/ZJUMoCap/CoreView_387
data/ZJUMoCap/CoreView_392
data/ZJUMoCap/CoreView_393
data/ZJUMoCap/CoreView_394
```

### ZJU subject 目录结构

每个 `CoreView_xxx` 目录至少需要包含：

```text
data/ZJUMoCap/CoreView_377/
├── cam_params.json
├── models/
│   ├── 000000.npz
│   ├── 000001.npz
│   └── ...
├── 1/
│   ├── 000000.jpg
│   ├── 000000.png
│   ├── 000001.jpg
│   ├── 000001.png
│   └── ...
├── 2/
│   ├── 000000.jpg
│   ├── 000000.png
│   └── ...
└── ...
```

说明：

- `cam_params.json`：相机参数文件，必须包含 `all_cam_names`，并为每个相机保存 `K`、`D`、`R`、`T`。
- `1/`、`2/`、...：相机视角目录，目录名要和配置里的 `train_views`、`val_views`、`test_views` 对上。
- `*.jpg`：RGB 图像。
- `*.png`：前景 mask，和同帧 jpg 同名，例如 `000123.jpg` 对应 `000123.png`。
- `models/*.npz`：每帧人体 SMPL/骨骼参数，文件按帧号排序读取。

默认 `zjumocap_377_mono` 训练用 `CoreView_377/1`，验证和测试用 `2` 到 `23`。如果用多视角配置 `zjumocap_377_multiview_hq`，训练会读取 `1` 到 `20`，验证/测试读取 `21` 到 `23`。

### 每帧 `models/*.npz` 需要的字段

ZJU 数据加载器会从每个 npz 中读取这些字段：

```text
minimal_shape
betas
root_orient
pose_body
pose_hand
trans
bone_transforms
```

其中：

- `minimal_shape` 用来构建 canonical SMPL 顶点和初始化点云。
- `bone_transforms`、`trans` 用来把 canonical 人体驱动到当前帧姿态。
- `root_orient`、`pose_body`、`pose_hand` 用来生成姿态条件。
- `betas` 在 `dataset.train_smpl=true` 时也会被读取。

`cano_smpl.ply` 不是必须提前准备。代码会优先读取：

```text
data/ZJUMoCap/CoreView_377/cano_smpl.ply
```

如果没有这个文件，会根据 `minimal_shape` 和 SMPL faces 自动采样生成。

## 2. SMPL body model 数据

项目运行还需要 SMPL 模型和预处理后的 SMPL 辅助文件。代码默认从仓库内读取：

```text
body_models/
├── smpl/
│   ├── male/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
│   ├── female/basicModel_f_lbs_10_207_0_v1.0.0.pkl
│   └── neutral/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
└── misc/
    ├── faces.npz
    ├── J_regressors.npz
    ├── posedirs_all.npz
    ├── shapedirs_all.npz
    ├── skinning_weights_all.npz
    ├── v_templates.npz
    └── kintree_table.npy
```

`body_models/misc/` 下的文件是代码直接读取的运行时依赖。它们可以通过项目里的脚本从 SMPL pkl 提取：

```bash
python extract_smpl_parameters.py
```

当前仓库里已经有这些 `misc` 文件；如果换机器或重新下载项目，需要确认这些文件仍然存在。

## 3. People Snapshot 数据，可选

如果使用 `configs/dataset/ps_male_3.yaml`、`ps_male_4.yaml`、`ps_female_3.yaml` 或 `ps_female_4.yaml`，会走 People Snapshot 数据加载器。

默认根目录写在配置里：

```text
../../data/peoplesnapshot_arah-format/people_snapshot_public
```

这个路径是相对于你运行命令时的项目根目录解析的；如果数据不在这里，需要运行时覆盖：

```bash
python train.py dataset=ps_male_3 dataset.root_dir=/你的/people_snapshot_public/路径
```

每个 subject 至少需要：

```text
people_snapshot_public/male-3-casual/
├── camera.pkl
├── image/
│   ├── 000000.jpg
│   └── ...
├── mask/
│   ├── 000000.png
│   └── ...
└── animnerf_models/
    ├── 000000.npz
    └── ...
```

`animnerf_models/*.npz` 需要的字段和 ZJU 基本一致：

```text
minimal_shape
betas
root_orient
pose_body
pose_hand
trans
bone_transforms
```

## 4. 可选增强数据：语义 parsing prior

默认配置里 `dataset.parsing_prior.enable=false`，所以不准备这部分也能训练。

如果打开语义先验，ZJU 数据加载器支持两种格式。

### 4.1 直接使用 CIHP/Hulk parser 标签图

配置相关字段：

```yaml
dataset:
  parsing_prior:
    enable: true
    parser_root: data/parsers_from_hulk_multiview
    parser_layout: cihp_subject
    use_direct_parser_labels: true
```

需要的目录结构：

```text
data/parsers_from_hulk_multiview/
└── CoreView_377/
    └── mask_cihp/
        ├── Camera_B1/
        │   ├── 000000.png
        │   └── ...
        ├── Camera_B2/
        └── ...
```

代码内置会把部分 CIHP label 分成 body 和 cloth：

- body labels：`13, 14, 15, 16, 17`
- cloth labels：`5, 6, 7, 9, 10, 11, 12`

### 4.2 使用预先转换好的 body/cloth mask

配置相关字段：

```yaml
dataset:
  parsing_prior:
    enable: true
    root_dir: /你的/parsing_prior/路径
    body_dirname: body
    cloth_dirname: cloth
    valid_dirname: valid
    uncertain_dirname: uncertain
```

需要的目录结构：

```text
/你的/parsing_prior/路径/
└── CoreView_377/
    └── 1/
        ├── body/000000.png
        ├── cloth/000000.png
        ├── valid/000000.png
        └── uncertain/000000.png
```

`uncertain/` 是可选的；`body/`、`cloth/`、`valid/` 至少要能覆盖你启用语义监督的训练视角和帧。

如果配置了：

```yaml
dataset.parsing_prior.compact_mapping_file: xxx.json
```

还需要提供对应 JSON，用来把 parser label 合并成自定义语义组。代码期望里面有 `groups`，可选 `ignore_labels` 和 `class_names`。

## 5. 可选增强数据：soft mask / matte

默认只需要每个相机目录里的硬 mask，也就是 `000000.png` 这类文件。

如果配置里打开：

```yaml
dataset:
  soft_mask:
    enable: true
    root_dir: data/mattes_from_hulk_multiview
    dirname: alpha
    layout: cihp_subject
    suffix: .png
```

则需要：

```text
data/mattes_from_hulk_multiview/
└── CoreView_377/
    └── alpha/
        ├── Camera_B1/
        │   ├── 000000.png
        │   └── ...
        ├── Camera_B2/
        └── ...
```

soft mask 找不到时，代码会退回使用原始硬 mask。

## 6. 可选数据：predict / 新动作序列

普通训练和测试不需要这部分。

当 `mode=predict` 时，ZJU 数据加载器会根据 `dataset.predict_seq` 去 subject 目录下找额外动作序列：

```text
CoreView_377/gBR_sBM_cAll_d04_mBR1_ch05_view1/*.npz
CoreView_377/gBR_sBM_cAll_d04_mBR1_ch06_view1/*.npz
CoreView_377/MPI_Limits-03099-op8_poses_view1/*.npz
CoreView_377/canonical_pose_view1/*.npz
```

People Snapshot 的 predict 序列是：

```text
male-3-casual/rotating_models/*.npz
male-3-casual/gLO_sBM_cAll_d14_mLO1_ch05_view1/*.npz
```

这些 npz 同样需要包含姿态、位移和骨骼变换字段。predict 模式仍会读取一张已有图像和 mask 当 dummy GT，所以 subject 里基础的 `image/mask` 或 ZJU 相机 `1/000000.jpg`、`1/000000.png` 也要存在。

## 7. 训练断点和测试渲染

从头训练不需要预训练权重。训练时会在 `exp/${name}` 下保存高斯点云和 checkpoint。

如果测试或渲染已有模型，需要提供 checkpoint：

```bash
python render.py mode=test load_ckpt=/path/to/ckpt.pth
```

如果不显式指定 `load_ckpt`，`render.py` 默认会找：

```text
exp/${name}/ckpt${opt.iterations}.pth
```

如果从已有模型继续训练，需要：

```bash
python train.py start_checkpoint=/path/to/ckpt.pth
```

## 8. 最小数据检查清单

默认跑 `zjumocap_377_mono` 时，至少确认这些文件存在：

```text
data/ZJUMoCap/CoreView_377/cam_params.json
data/ZJUMoCap/CoreView_377/models/000000.npz
data/ZJUMoCap/CoreView_377/1/000000.jpg
data/ZJUMoCap/CoreView_377/1/000000.png
body_models/misc/faces.npz
body_models/misc/J_regressors.npz
body_models/misc/posedirs_all.npz
body_models/misc/skinning_weights_all.npz
body_models/misc/v_templates.npz
body_models/misc/kintree_table.npy
```

并且根据配置的 `train_frames`、`val_frames`、`test_frames`，对应帧范围内的 jpg、png、npz 都要齐全。
