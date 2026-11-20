# -*- coding: utf-8 -*-
"""指标：标准 Macro-F1（算术平均，类别集 = 金 ∪ 预测，含 __INVALID__）、宏P、宏R、准确率、HVR、分级准确率。"""
import collections

from .judge import INVALID


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def compute_metrics(rows):
    """rows: [{gold_leaf, gold_path, pred_label, pred_nodes, hierarchy_ok, parse_ok}]"""
    n = len(rows)
    gold = [r["gold_leaf"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    classes = sorted(set(gold) | set(pred))
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s, ps, rs = [], [], []
    per_class = {}
    for c in classes:
        p, r, f = _prf(tp[c], fp[c], fn[c])
        f1s.append(f); ps.append(p); rs.append(r)
        per_class[c] = dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
                            support=sum(1 for x in gold if x == c), predicted=sum(1 for x in pred if x == c))
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0
    hvr = sum(1 for r in rows if r.get("hierarchy_ok")) / n if n else 0.0
    parse_fail = sum(1 for r in rows if not r.get("parse_ok")) / n if n else 0.0
    oov = sum(1 for r in rows if r.get("parse_ok") and not r.get("resolved", True)) / n if n else 0.0
    level_acc = {}
    for k in range(1, 5):
        sub = [r for r in rows if len(r["gold_path"]) >= k]
        if not sub:
            level_acc[f"L{k}"] = dict(n=0, acc=None)
            continue
        ok = 0
        for r in sub:
            pn = r.get("pred_nodes") or []
            if len(pn) >= k and pn[k - 1] is not None and pn[k - 1]["name"] == r["gold_path"][k - 1]:
                ok += 1
        level_acc[f"L{k}"] = dict(n=len(sub), acc=round(ok / len(sub), 4))
    # 附录口径：类别集只取金标签（不含预测独有的类与 __INVALID__），仅用于附录对照，
    # 主指标仍为上面的并集 Macro-F1（类别集 = 金 ∪ 预测）。
    gold_classes = sorted(set(gold))
    f1s_gold = [per_class[c]["f1"] for c in gold_classes]
    invalid = sum(1 for p in pred if p == INVALID)
    return dict(n=n,
                macro_f1_goldset=round(sum(f1s_gold) / len(f1s_gold), 4) if f1s_gold else 0.0,
                macro_f1=round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
                macro_precision=round(sum(ps) / len(ps), 4) if ps else 0.0,
                macro_recall=round(sum(rs) / len(rs), 4) if rs else 0.0,
                accuracy=round(acc, 4),
                hvr=round(hvr, 4),
                parse_failure_rate=round(parse_fail, 4),
                oov_rate=round(oov, 4),
                invalid_rate=round(invalid / n, 4) if n else 0.0,
                n_classes=len(classes),
                n_classes_gold=len(gold_classes),
                level_acc=level_acc,
                per_class=per_class)
