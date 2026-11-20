# 基于层级约束与蒙特卡洛树搜索的民事案件案由预测

**Hierarchy-Constrained Monte-Carlo Tree Search for Civil Case Cause-of-Action Prediction**

本仓库是论文《基于 rStar 方法的民事案件案由预测模型研究》（《计算机工程》返修稿）的官方源代码，
包含方法实现、数据构建脚本、评测与显著性检验脚本、以及论文表 4–表 6 所用的全部测试集预测文件。

## 0. 源代码链接对应的论文

- 题目：基于 rStar 方法的民事案件案由预测模型研究（待定稿）
- 期刊：《计算机工程》(Computer Engineering)，返修中
- 引用格式：（录用后补全）

> 若你在论文首页看到 “源代码链接：<本仓库 URL>”，即指本仓库。代码冻结版本为 tag `v1.0-paper`。

## 1. 目录结构

```
README.md              本文件（中英文）
LICENSE                代码许可（MIT）
DATA_LICENSE.md        数据许可（CC BY-NC 4.0，派生自 AppealCase）
requirements.txt       Python 3.10 + vLLM 0.19.1 + torch 2.10.0 + transformers 5.6.2
expected_metrics.csv   论文表 4–表 6 的 38 组指标（复算基准）
configs/               method_v3_1_frozen.yaml（冻结配置）、split_v2.yaml、baselines.yaml、ablations.yaml
vocab/                 2020 / 2025 版案由词表、人工映射、2020→2025 名称映射
data/                  CAD-v2.1 六个 JSONL（train/valid_train/fewshot_pool/dev/test/fewshot10_folds）
prompts/               native / cot / hier 提示词模板
scripts/               方法与工具脚本（详见 scripts/ 下各文件 docstring）
  methods/             method_v3.py（本文方法）、method_v3_nc.py、ablation_novote.py、
                       engine.py（生成/鉴别引擎与先验）、rstar_orig.py、berry.py
  supervised/          textcnn.py、topjudge.py、iracps.py（三个监督基线）
eval/                  extract.py、judge.py、taxonomy.py、metrics.py 与单元测试
predictions/           38 个测试集预测文件（紧凑格式，仅供复算表 4–表 6）
```

## 2. 方法概述

输入是一审文书（案由名称已掩码为 `[案由]`），输出是四级案由路径。方法由五个模块组成（对应论文 2.4.1–2.4.5）：

1. **要素摘要**（2.4.1）：生成器 SLM1 先把文书压缩成 ≤200 字的要素摘要 Z（主体、法律关系、争议焦点、请求）。
2. **层级先验**（2.4.2）：在已有前缀 p 下，让 SLM1 从候选编号中只选一个，取该次生成的 token 概率归一化得 μ(v|p)，
   用于把搜索限制在案由树的合法子节点上。
3. **多路径投票**（2.4.3）：SLM1 用同一提示词采样 K = 16 条开放作答（T = 0.7，top-p = 0.95），经规则规范化与
   BERT-wwm-ext 语义兜底（余弦阈值 0.83）映射为规范路径，得投票分布 η(v|p) = (h_v + ε)/(Σh + ε|S(p)|)，ε = 0.5。
4. **案由树 PUCT 搜索**（2.4.4）：以融合先验 θ = λμ + (1−λ)η（λ = 0.3）为引导，在树上做 B = 16 次模拟，
   探索常数 c_d = 2.0 − 0.3d，取访问占比 s 最高的 M = 4 条候选路径。
5. **鉴别器逐级复选与评估器**（2.4.5）：SLM2 对每条候选路径逐级遮蔽复选，得层级一致率 a；
   评估器 J = 0.5·s + 0.5·a 取最大者为最终路径。

实证结论（测试集 770 条 / 77 个叶案由）：Macro-F1 = 0.5082（种子 1），准确率 0.7078，
层级合法率 1.000。
## 3. 环境安装

```bash
conda create -n cause-v2 python=3.10 -y
conda activate cause-v2
pip install -r requirements.txt
```

说明：

- 需要 CUDA 12.8 驱动（≥ 570）。首次运行 vLLM 会编译内核，单卡加载 8B 模型约 2–3 分钟。
- 模型权重不入库。可用 Hugging Face id 直接加载（联网），也可先下载到本地目录再用 `GEN_MODEL` / `DISC_MODEL` /
  `BERT` 环境变量指向（离线集群更常用）：

```bash
export GEN_MODEL=/models/Qwen3.5-9B
export DISC_MODEL=/models/Qwen3.5-4B
export BERT=/models/chinese-bert-wwm-ext
```

默认值分别为 `Qwen/Qwen3.5-9B`、`Qwen/Qwen3.5-4B`、`dienstag/chinese-bert-wwm-ext`。
表 4 其余六个生成器见 `configs/baselines.yaml`。

## 4. 数据准备

**方式 A（推荐，直接使用仓库内数据）**：`data/` 下已有 CAD-v2.1，无需任何操作。

**方式 B（从 AppealCase 重建）**：

1. 从 https://huggingface.co/datasets/ythuang02/AppealCase 下载 `AppealCase.json` 与 `AppealCase.metadata.json`
   （约 253 MB / 7 MB），放到任意目录；
2. 运行：

```bash
mkdir -p runs
python scripts/build_cad_v2.py \
    --data /path/to/AppealCase.json \
    --meta /path/to/AppealCase.metadata.json \
    --out-dir data \
    --split-config configs/split_v2.yaml \
    --report-json runs/build_stats.json \
    --label-map-csv runs/appealcase_label_map.csv \
    --mask-terms-csv runs/mask_terms.csv
```

脚本会依次做：剔除刑事、精确去重（simhash 近重复，汉明 ≤ 3）、长度 < 100 字剔除、案由名称掩码、
标签规范化与映射（精确 + 规则修正，失败即剔除并计数）、按案由分层划分。重建结果与仓库内数据应当逐条一致
（`data/README.md` 记录了 sha256）。

3. 校验划分无跨集合重复：

```bash
python scripts/verify_split_overlap.py data    # 位置参数为数据目录，结果写入 runs/split_overlap_check.json
```

## 5. 不跑模型，只用 predictions/ 核对论文数字

```bash
# 打印表 4（7 生成器 × 原生/本文）、表 5（各方法）、表 6（消融）
python scripts/reproduce_tables.py

# 与 expected_metrics.csv 逐项比对（容差 1e-4），全部一致输出 check: OK
python scripts/reproduce_tables.py --check
```

这两条命令只需要 Python 与 `vocab/`、`data/test.jsonl`，不需要 GPU，也不需要任何模型权重。
## 6. 复现命令（含 GPU）

以下命令均在本仓库根目录执行。脚本默认 8 张 GPU、8 个分片；`NGPU` 可改。

### 6.1 主实验（论文表 4 “本文方法”列、表 5 第 1 行、摘要数字）

```bash
for s in 1 2 3; do bash scripts/run_method_v3.sh runs/main_s$s 8 $s; done
```

- `B = 16`（`--n-sim 16`）已在脚本内固定；`τ = 2.0`、`λ = 0.3` 来自 `configs/method_v3_1_frozen.yaml`。
- 每个种子约 8–9 分钟（8×A800，770 条）。结果：`runs/main_s1/metrics.json`，
  种子 1 应为 Macro-F1 **0.5082**、准确率 **0.7078**；三种子均值 **0.5063 ± 0.0022**。

### 6.2 七种生成器（论文表 4）

```bash
# 换成其它生成器时，鉴别器仍为 Qwen3.5-4B（4B 生成器为自鉴别）
GEN_MODEL=/models/Qwen3-8B  DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_qwen3_8b 8 1
GEN_MODEL=/models/Qwen3-30B-A3B DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_qwen3_30b 8 1
GEN_MODEL=/models/LegalOne-8B DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_legalone 8 1
GEN_MODEL=/models/gemma-4-E4B-it DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_gemma4 8 1
GEN_MODEL=/models/MiniCPM5-2B DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_minicpm5 8 1
GEN_MODEL=/models/Qwen3.5-4B DISC_MODEL=/models/Qwen3.5-4B bash scripts/run_method_v3.sh runs/gen_qwen35_4b 8 1
```

同表“原生推理”列的复现见 6.4。

### 6.3 九组消融（论文表 6）

直接改开关重跑：

```bash
bash scripts/run_method_v3.sh runs/abl_no_summary 8 1 --no_summary
bash scripts/run_method_v3.sh runs/abl_final_only_disc 8 1 --final_only_disc
bash scripts/run_method_v3.sh runs/abl_fixed_c 8 1 --fixed_c
GEN_SCRIPT=scripts/methods/method_v3_nc.py bash scripts/run_method_v3.sh runs/abl_no_calib 8 1
GEN_SCRIPT=scripts/methods/ablation_novote.py TAU=2.0 LAM=1.0 bash scripts/run_method_v3.sh runs/abl_no_vote 8 1
```

从主实验中间产物派生（不调用模型，秒级）：

```bash
mkdir -p runs/abl_no_disc runs/abl_no_eval runs/abl_no_hier

# no_mcts：等价于主实验加 --no_mcts 重跑（同一随机种子给出相同预测）
bash scripts/run_method_v3.sh runs/abl_no_mcts 8 1 --no_mcts

# no_disc：J 取路径访问占比 s（--mode max_share）
python scripts/derive_preds.py --mode max_share --in runs/main_s1/preds_all.jsonl --out runs/abl_no_disc/preds_all.jsonl
# no_eval：J 取层级一致率 a（--mode max_a）
python scripts/derive_preds.py --mode max_a --in runs/main_s1/preds_all.jsonl --out runs/abl_no_eval/preds_all.jsonl
# no_hier：16 票中出现最多的完整路径（--mode vote_top）
python scripts/derive_preds.py --mode vote_top --gen runs/main_s1/gen.jsonl --out runs/abl_no_hier/preds_all.jsonl

python scripts/eval_preds.py --preds runs/abl_no_disc/preds_all.jsonl --gold data/test.jsonl \
    --out runs/abl_no_disc/metrics.json --tax-dir vocab
```

### 6.4 基线（论文表 4 原生推理列、表 5 各行）

提示词类基线（`native` / `cot` / `sc` / `hier`）：

```bash
# 原生推理（表 4 原生列；7 个生成器各跑一次，--method native）
for m in Qwen3.5-9B Qwen3.5-4B gemma-4-E4B-it MiniCPM5-2B Qwen3-8B Qwen3-30B-A3B LegalOne-8B; do
  python scripts/run_inference.py --model-path /models/$m --label $m --split test \
      --method native --max-tokens 96 --seed 1
done

# 直接层次化提示（表 5）
python scripts/run_inference.py --model-path /models/Qwen3.5-9B --label qwen3.5-9b --method hier --seed 1
# 思维链（表 5）
python scripts/run_inference.py --model-path /models/Qwen3.5-9B --label qwen3.5-9b --method cot --max-tokens 512 --seed 1
# 自一致性 K=16（表 5，三种子）
for s in 1 2 3; do python scripts/run_inference.py --model-path /models/Qwen3.5-9B \
    --label qwen3.5-9b --method sc --n-sc 16 --temperature 0.7 --top-p 0.95 --seed $s; done
```

方法类基线：

```bash
# 原始 rStar：整题一次 MCTS、开放作答（表 5；三种子）
bash scripts/run_rstar_orig.sh runs/rstar_s1 8 1        # 封装见 scripts/run_rstar_orig.sh
# LLaMA-Berry 风格 Self-Refine MCTS（表 5）
bash scripts/run_berry.sh runs/berry_s1 4 1 8
```

监督基线（表 5 后三行；训练集 = `data/train.jsonl`，早停集 = `data/valid_train.jsonl`，测试集只评一次）：

```bash
python scripts/supervised/textcnn.py  --lr 1e-3 --batch 32 --epochs 12 --cw 0.5 --final
python scripts/supervised/topjudge.py --lr 3e-5 --batch 32 --epochs 10 --final
python scripts/supervised/iracps.py   --lr 3e-5 --batch 16 --epochs 8 --lam 0.5 --final
```

### 6.5 显著性检验（论文表 7 / 第 5 节）

```bash
python scripts/step11_bootstrap.py --gold data/test.jsonl \
    --base predictions/ours_qwen3.5-9b_s1.jsonl \
    --compare predictions/hier.jsonl \
    --out-json runs/sig_hier.json
```

输出 ΔF1 的分层自助法 95% 区间（按金标案由分层，1000 次）、共同类别集上的配对置换检验 p 值，
以及逐条准确率的 McNemar 检验。随机种子默认 20261119，与他人复算结果一致。

### 6.6 2020 → 2025 词表迁移重评（论文第 6 节）

```bash
python scripts/step8_taxonomy_shift.py --preds predictions/ours_qwen3.5-9b_s1.jsonl \
    --gold data/test.jsonl --out-json runs/taxonomy_shift.json
```

## 7. 硬件与耗时

- 8×NVIDIA A800 80 GB：主实验单次（770 条测试，8 分片）约 **8–9 分钟**；显存占用约 60 GB/卡。
- 生成与鉴别分两阶段执行（9B + 4B 无法同时放进一张 80 GB 卡）。
- 单张 A800 串行跑完 770 条约 2 小时，因此推荐 ≥4 卡；1–2 卡仅用于抽查（`--limit 20`，约 2 分钟）。
- 监督基线：TextCNN 约 40 秒、TopJudge 约 8 分钟、IRACPS 约 5 分钟（单卡）。
- 显著性检验与只读复算为纯 CPU，秒级到分钟级。
## 8. 许可与引用

- 代码：MIT（见 `LICENSE`）。
- 数据（`data/`、`vocab/`）：CC BY-NC 4.0，派生自 AppealCase（见 `DATA_LICENSE.md`），仅限非商业研究使用。
- 若使用本仓库，请引用：

```bibtex
@article{cause_rstar_2026,
  title   = {基于 rStar 方法的民事案件案由预测模型研究},
  journal = {计算机工程},
  year    = {2026},
  note    = {Code: <repository URL>}
}

@article{huang2025appealcase,
  title  = {AppealCase: A Dataset for Legal Judgment Prediction on Civil Appeals},
  author = {Huang, Yuantao and others},
  year   = {2025},
  note   = {arXiv:2505.16514}
}
```

## 9. 实现说明（Implementation notes）

本节记录复现实现中属于实现层面的固定配置与构造方式，供复现者对照。

1. **少样本示例**：实现固定使用 `data/fewshot10_folds.jsonl` 中的 10 条示例（每折 2 条，提示词按折拼接 10 条），
   提示词长度约 4.1k 字符；该设置为本文全部复现实验的固定配置。
2. **2020 年版词表**：按最高人民法院公布的文本解析得 11/54/473/389，共 927 个节点；官方统计口径的四级案由为 391 个。
   473 个三级案由的编号连续无缺号，正文中以“（”开头的四级行恰为 389 条，差额源于网页正文与公布统计的口径不同。
3. **旧版案由名称映射**：旧版案由名称到 2020 年修正版的映射采用 15 条人工映射
   （见 `vocab/manual_mapping_2020.csv`，含法律依据）与规则实现（`map_labels.py`）。
4. **IRACPS 基线**：按论文描述的框架自行实现（句级弱监督理由抽取 + 注意力 + 层级解码），属实现层面的等价构造。
5. **LLaMA-Berry 基线**：奖励模型以同模型成对比较提示代替独立的 PPRM。
6. **预测文件**：`predictions/` 保留复算所需字段，`raw_output` 截断到 2000 字符，可复算表 4–表 6 与显著性检验。

## 10. 快速自检

```bash
python -m unittest eval.tests.test_eta eval.tests.test_eval   # 24 项单元测试
python scripts/reproduce_tables.py --check                     # 表 4–表 6 复算，应输出 check: OK
```

---

## English

**Hierarchy-Constrained Monte-Carlo Tree Search for Civil Case Cause-of-Action Prediction**

Official code for the paper “基于 rStar 方法的民事案件案由预测模型研究” (submitted to *Computer Engineering*).
Given a first-instance civil judgment whose cause-of-action names have been masked, the method predicts a
path in the four-level cause-of-action taxonomy. It combines five modules: (i) an element summary produced by
the generator SLM, (ii) a hierarchical prior that restricts each step to the legal children of the current
prefix, (iii) a multi-path vote over K = 16 open-ended samples fused with the prior, (iv) a PUCT search on the
cause-of-action tree with B = 16 simulations, and (v) a discriminator that re-checks each candidate path level
by level, with the final choice J = 0.5·s + 0.5·a. On the 770-case CAD-v2.1 test set it reaches
Macro-F1 = 0.5082 and accuracy = 0.7078 with Qwen3.5-9B.

```bash
pip install -r requirements.txt
python scripts/reproduce_tables.py --check            # no GPU needed: reproduces Tables 4-6
for s in 1 2 3; do bash scripts/run_method_v3.sh runs/main_s$s 8 $s; done   # main experiment
```

Data (`data/`) is derived from AppealCase and released under CC BY-NC 4.0 for non-commercial research;
code is MIT licensed. See `DATA_LICENSE.md` for provenance and citation.