# -*- coding: utf-8 -*-
"""CAD-v2.1 监督式基线公共件：数据读取、标签空间（训练集叶路径）、层级解码辅助、
以及与主实验完全一致的指标（直接复用仓库 eval 包）。"""
import json, os, random, sys

import numpy as np

REPO = "."
DATA = os.path.join(REPO, "data")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from eval import Taxonomy, judge_one, compute_metrics  # noqa: E402

LEVELS = 4


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_split(name):
    return load_jsonl(os.path.join(DATA, name + ".jsonl"))


def set_seed(s=2024):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def truncate_chars(t, head, tail):
    """head+tail 截断（保头保尾），总长 head+tail。"""
    t = (t or "").replace("\x00", " ")
    if len(t) <= head + tail:
        return t
    return t[:head] + "\n" + t[-tail:]


class LabelSpace:
    """标签空间 = 训练集出现过的 gold_path 元组（146 个），并据此建层级树。"""

    def __init__(self, paths):
        self.classes = sorted({tuple(p) for p in paths})
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.names = [[]]
        self.name2idx = [{}]
        for lv in range(1, LEVELS + 1):
            names = sorted({c[lv - 1] for c in self.classes if len(c) >= lv})
            self.names.append(names)
            self.name2idx.append({n: i for i, n in enumerate(names)})
        self.parent = [[]]
        for lv in range(1, LEVELS + 1):
            par = []
            for n in self.names[lv]:
                if lv == 1:
                    par.append(-1)
                else:
                    p = None
                    for c in self.classes:
                        if len(c) >= lv and c[lv - 1] == n:
                            p = c[lv - 2]; break
                    par.append(self.name2idx[lv - 1][p])
            self.parent.append(par)
        self.children = [[]]
        for lv in range(1, LEVELS):
            ch = [[] for _ in self.names[lv]]
            for i, n in enumerate(self.names[lv]):
                for c in self.classes:
                    if len(c) > lv and c[lv - 1] == n:
                        j = self.name2idx[lv + 1][c[lv]]
                        if j not in ch[i]:
                            ch[i].append(j)
            self.children.append(ch)
        self.children.append([[] for _ in self.names[LEVELS]])
        self.cls_nodes = [[-1] * len(self.classes) for _ in range(LEVELS + 1)]
        for ci, c in enumerate(self.classes):
            for lv in range(1, len(c) + 1):
                self.cls_nodes[lv][ci] = self.name2idx[lv][c[lv - 1]]

    def path(self, ci):
        return self.classes[ci]

    def cls_node_global(self, lv, ci):
        """第 ci 个类在第 lv 级的节点索引（不存在返回 -1）。"""
        return self.cls_nodes[lv][ci]

    def stats(self):
        return dict(n_classes=len(self.classes),
                    depth=dict((d, sum(1 for c in self.classes if len(c) == d)) for d in range(1, 5)),
                    level_sizes={lv: len(self.names[lv]) for lv in range(1, LEVELS + 1)})


_TAX = None


def taxonomy():
    global _TAX
    if _TAX is None:
        _TAX = Taxonomy("2020", os.path.join(REPO, "vocab"))
    return _TAX


def metrics_for(pred_paths, gold_recs):
    """pred_paths 与 gold_recs 对齐；返回仓库 compute_metrics 的结果（与 eval_preds.py 同口径）。"""
    tax = taxonomy()
    rows = []
    for names, g in zip(pred_paths, gold_recs):
        nodes, ok, hier = tax.build_pred_path(names) if names else ([], [], False)
        j = judge_one(nodes, g["gold_path"], g["gold_leaf"])
        rows.append(dict(id=g["id"], gold_leaf=g["gold_leaf"], gold_path=g["gold_path"],
                         pred_nodes=nodes, hierarchy_ok=hier, parse_ok=bool(names),
                         resolved=bool(ok and ok[0]), pred_label=j["pred_label"],
                         correct=j["correct"], pred_depth=j["pred_depth"]))
    return compute_metrics(rows)


def write_preds(path, ids, pred_paths):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, p in zip(ids, pred_paths):
            f.write(json.dumps({"id": i, "raw_output": "最终案由：" + " > ".join(p)},
                               ensure_ascii=False) + "\n")


def brief(m):
    keys = ["macro_f1", "macro_f1_goldset", "macro_precision", "macro_recall", "accuracy",
            "hvr", "parse_failure_rate", "oov_rate", "invalid_rate", "n_classes"]
    return {k: m[k] for k in keys}
