# data/ — CAD-v2.1

六份 JSONL，全部由 `scripts/build_cad_v2.py` 从 AppealCase 生成（2020 年修正版词表，927 个节点）。

| 文件 | 条数 | 用途 | sha256（前 16 位） |
| --- | --- | --- | --- |
| `train.jsonl` | 7796 | 训练池（监督基线用；本文方法不训练） | `fbf051b21ed6856f` |
| `valid_train.jsonl` | 879 | 监督基线早停 | `3f0d25461617c482` |
| `fewshot_pool.jsonl` | 154 | 少样本示例池（每案由 2 条，未进入其他划分） | `f82f24f16e7d8c7e` |
| `dev.jsonl` | 231 | 开发集（超参选择与门禁） | `2b94a3746a8820f1` |
| `test.jsonl` | 770 | 测试集（77 个叶案由 × 10 条） | `9e26002156540ff6` |
| `fewshot10_folds.jsonl` | 10 | 固定少样本示例（5 折 × 2 条） | `7eac07dac1cdad0d` |

合计 9840 条（= 9830 条划分 + 10 条示例；示例同时保留在 `fewshot_pool.jsonl` 中，故 154 = 144 + 10）。

字段：

- `id`：样本编号（与 `source_id` 相同，均为 AppealCase 的 `index`）；
- `source_id`：AppealCase 原始 `index`，可用于回溯原文书；
- `input_masked`：案由名称已掩码（`[案由]`）的一审文书输入，长度上限 4000 字（截断在运行时进行）；
- `gold_leaf` / `gold_path` / `gold_level`：映射到 2020 版词表后的金标叶案由、完整路径与层级；
- `cause_raw`：AppealCase metadata 中记录的一审案由原文；
- `year`：从一审案号提取的年份（缺失为空串）。

重建方式见 `README.md` 第 5 节；划分参数见 `configs/split_v2.yaml`（seed = 42）。