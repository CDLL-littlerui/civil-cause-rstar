# predictions/ — 论文表 4–表 6 的测试集预测

每行一条测试样本，字段：

- `id`：测试样本编号；
- `gold_path` / `gold_leaf`：金标路径与叶案由（冗余存储，便于脱离 `data/` 单独检查）；
- `raw_output`：模型原始输出（截断到 2000 字符）——评测脚本从中抽取“最终案由：…”之后的路径；
- `pred_path`：方法侧最终选定的路径（仅本文方法与消融运行提供；评测时以 `raw_output` 为准，二者一致）。

文件与论文表号的对应关系见仓库根目录 `README.md` 第 7 节与 `expected_metrics.csv`。

复算（无需 GPU）：

```bash
python scripts/reproduce_tables.py          # 打印表 4–表 6
python scripts/reproduce_tables.py --check  # 与 expected_metrics.csv 逐项比对（容差 1e-4）
```

说明：这些文件只保留复算所需的字段，`paths`、`votes`、`summary` 等中间产物未随仓库发布；
因此仅凭 `predictions/` 无法重跑鉴别器统计或案例轨迹（这两项需要重新运行方法本身）。