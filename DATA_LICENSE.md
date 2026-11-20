# 数据许可与来源 / Data license and provenance

## 1. 本仓库中的数据

本仓库 `data/` 下的 6 个 JSONL 文件构成 **CAD-v2.1**（Civil Action Dataset v2.1），
由 AppealCase 数据集的一审文书派生而来：

- `data/train.jsonl` 7796 条、`data/valid_train.jsonl` 879 条、`data/fewshot_pool.jsonl` 154 条、
  `data/dev.jsonl` 231 条、`data/test.jsonl` 770 条、`data/fewshot10_folds.jsonl` 10 条（固定少样本示例）。
- 每条样例字段：`id, source_id, input_masked, gold_leaf, gold_level, gold_path, cause_raw, year`。
  `input_masked` 是案由名称已掩码（替换为 `[案由]`）的一审文书输入；`gold_path` 是映射到 2020 年修正版
  《民事案件案由规定》的四级案由路径。

## 2. 原始来源

- 数据集：**AppealCase**（ythuang02/AppealCase），作者 Yuantao Huang 等，2025。
  论文：Y. Huang et al., "AppealCase: A Dataset for Legal Judgment Prediction on Civil Appeals",
  arXiv:2505.16514, 2025。
- 原始地址：https://huggingface.co/datasets/ythuang02/AppealCase
  （文件 `AppealCase.json`、`AppealCase.metadata.json`）；代码仓库：https://github.com/ythuang02/AppealCase
- 原始数据集的许可为 **CC BY-NC 4.0**（见其数据集卡 `license: cc-by-nc-4.0` 与仓库 README 的
  "License: CC BY-NC 4.0"）。AppealCase 的文书本身来自公开裁判文书，数据集中保留了当事人姓名等
  原文信息；这些信息在原始数据集中已公开，本仓库不再另行处理，但**仅限非商业研究使用**。

因此，`data/` 下的派生数据沿用 **CC BY-NC 4.0**：允许在署名、非商业的前提下使用、修改与再分发；
不得用于商业目的。使用时请同时引用 AppealCase 与本仓库对应的论文。

## 3. 词表

`vocab/` 下的案由层级树由最高人民法院公布的《民事案件案由规定》（2020 年修正、2025 年修正）整理，
属公开的规范性文件文本。整理结果（层级树 JSON、节点表 JSONL、人工映射表）随本仓库以 CC BY-NC 4.0 提供，
与代码的 MIT 许可相互独立。

## 4. 代码

`scripts/`、`eval/` 等源代码以 **MIT** 许可发布，见 `LICENSE`。

## 5. English summary

The JSONL files under `data/` form CAD-v2.1, derived from the **AppealCase** dataset
(https://huggingface.co/datasets/ythuang02/AppealCase, CC BY-NC 4.0, Huang et al., arXiv:2505.16514).
Inputs are first-instance judgments whose cause-of-action names have been masked. Personal names
present in the original public judgments are kept as-is; the data are released for
**non-commercial research only**, under CC BY-NC 4.0. The code is released under the MIT license.