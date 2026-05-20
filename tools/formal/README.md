# Formal 工具入口

当前正式入口放在：

```text
tools/formal/run_377_signed_geometry_render.sh
tools/formal/run_377_signed_geometry_export.sh
```

`tools/` 根目录保留兼容 wrapper：

```text
tools/run_377_formal_signed_geometry_render.sh
tools/run_377_formal_signed_geometry_export.sh
```

这些入口统一使用：

```text
++explicit_binding_render_preset=v320_v307_signed_geometry
++render_export_refine=false
dataset.subject=CoreView_377
```

历史 StageB / explicit binding 实验脚本见：

```text
tools/archive/stageB_exp/MANIFEST.md
```

后续如果要继续物理精简 `tools/` 根目录，应先确认没有外部任务仍直接调用对应历史脚本。
