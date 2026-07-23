# 五主体 A5 LOSO 与正式统计设计

## 目标

对 CoreView 377、386、387、393、394 执行统一的 leave-one-subject-out 超参数选择：每个 held-out 主体只使用另外四个主体的 validation 结果选择 A5 阈值，再在 held-out test 上一次性评测冻结 A5 主表。随后生成主体级配对统计、bootstrap 95% CI、逐部位和逐视角统计。

## 协议边界

- 方法固定为 `configs/semantic/frozen_a5_main_method_v1.json`，主方法只能是 A5。
- validation 候选必须使用 A5 footprint bank 的 `soft_edit_weights`，不得复用历史 B5/A6 候选。
- 候选阈值固定为 `0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5`，support threshold 固定为 0.1，boundary radius 固定为 6。
- 每个 held-out 主体的候选选择只读取另外四个主体的 validation 目录。
- test 评测不允许命令行阈值覆盖，必须读取生成的 LOSO frozen config。
- 主表方法固定为 `B0 B1 B2 B3 B4 A5`；A6 不进入本轮主表。
- 现有 42k checkpoint、raw bank、voting bank、A6 evidence bank 和 A5 footprint bank全部复用，不进行训练。

## 数据流

1. 对五个主体和七个阈值分别运行 validation 评测，输出 `B1/A5` baseline、curve 和 matched-retention 文件。
2. A5 LOSO 选择器验证每个 donor 的 mIoU gap 不超过 0.02，且 retention 0.5/0.6 的 A5 actionable leakage 不高于 B1。
3. 在所有 donor 均通过的候选中，依次最小化 donor 平均 A5 leakage、最大化 A5 mIoU、最大化 boundary F1、最小化阈值。
4. 为每个 held-out 主体写入包含四个 donor、方法冻结指纹和输入指纹的 LOSO config。
5. 使用该 config 在 held-out test 上运行 `B0-B4+A5` 主表。
6. 汇总五主体结果并计算 A5-B1 配对差值、样本标准差、确定性 subject bootstrap 95% CI、win/tie/loss、逐部位差值与逐视角差值。

## 输出

输出根目录为 `exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/`：

- `validation_candidates/CoreView_<subject>/t*/`
- `CoreView_<subject>/loso_frozen_config.json`
- `CoreView_<subject>/main/`
- `aggregate/main_table.csv`
- `aggregate/matched_retention_table.csv`
- `aggregate/paired_subject_deltas.csv`
- `aggregate/paired_statistics.csv`
- `aggregate/per_part_deltas.csv`
- `aggregate/per_view_deltas.csv`
- `aggregate/statistics_summary.json`

## 失败处理

- 任一输入、donor 阈值目录或冻结指纹缺失时立即失败。
- 若没有阈值同时通过四个 donor gate，则不读取 test，队列失败并记录 held-out 主体。
- 汇总器拒绝主表中的 A6、缺失 A5/B1、非有限指标或 donor 数量不为四的配置。

## 验证

- 单元测试覆盖 A5 候选读取、四 donor 限制、候选排序、bootstrap 可复现性和 A6 主表拒绝。
- shell dry-run 检查五主体、七阈值、validation/test split、method freeze 和无训练命令。
- 正式运行后验证五个 config 的 donor 集合、冻结指纹、主表方法集合及全部统计值有限。
