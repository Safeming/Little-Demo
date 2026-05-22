# Formal 工具入口

当前正式入口放在：

```text
tools/formal/run_377_signed_geometry_render.sh
tools/formal/run_377_signed_geometry_export.sh
tools/formal/run_377_v338_mainline.sh
tools/formal/run_377_v338_raw_contour_gate.sh
tools/formal/run_377_v338_semantic_train.sh
```

`tools/` 根目录保留兼容 wrapper：

```text
tools/run_377_formal_signed_geometry_render.sh
tools/run_377_formal_signed_geometry_export.sh
```

这些入口统一使用：

```text
++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard
++render_export_refine=false
dataset.subject=CoreView_377
```

`v338_temporal_selector_grow_only_guard` 保留 v320/v307 formal component asset，
并叠加已采用的 v338 per-image signed point field。仅做历史 ablation 时使用
`FORMAL_PRESET=v320_v307_signed_geometry` 回到旧 formal preset。

`run_377_v338_semantic_train.sh` 只训练独立的 semantic asset logits adapter，不训练
color/SH/texture/converter 或几何参数来修边。legacy semantic logits adapter 保持给
texture/render 旧链路使用，训练 resume 会保留 boundary tags / binding state，避免
无关状态改变 raw RGB contour。训练出的 ckpt 用下面的 gate 验收：

```text
CANDIDATE_CKPT=/path/to/ckpt.pth tools/formal/run_377_v338_raw_contour_gate.sh
```

当前主线一键入口是：

```text
tools/formal/run_377_v338_mainline.sh
```

它会按顺序执行 semantic asset adapter train、candidate raw contour gate、gate 通过后的
interpretability + semantic editable assets export，并用
`tools/check_semantic_editable_assets.py` 校验导出资产结构。`TRAIN_STEPS` 表示从
base checkpoint 继续追加的本地训练步数，checkpoint 文件名仍使用 base iteration offset。

历史 StageB / explicit binding 实验脚本见：

```text
tools/archive/stageB_exp/MANIFEST.md
```

后续如果要继续物理精简 `tools/` 根目录，应先确认没有外部任务仍直接调用对应历史脚本。
