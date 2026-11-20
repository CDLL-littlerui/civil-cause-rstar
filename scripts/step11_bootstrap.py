#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 11：修正配对 bootstrap 的置信区间 + 配对近似随机化检验。

问题：旧实现每次重采样都用"金 ∪ 预测"重新求类别集，类别集随样本变化；
预测类别多的运行（|C| 大）在重采样里类别集缩水更严重，宏 F1 被系统性抬高，
使 ΔF1 = base − compare 上偏（base 的 |C| 小、compare 的 |C| 大时最明显）。

修正：
 1) 分层重采样：按金标叶案由分层，每类内部有放回抽 n_c 条（77 类各 10 条）。
 2) 固定类别集：每个运行用全样本类别集 C = gold ∪ pred（与式 13—16 一致），
    重采样中不变；未出现的类别 F1 记 0。
 3) 百分位区间 + BCa 区间；4) 10,000 次配对随机化检验（每条以 0.5 概率交换预测）。

用法：
  python scripts/step11_bootstrap.py --gold data/test.jsonl \
     --base runs/method_v3_1/test/main_s1/preds.jsonl --compare A B C --out-json out.json
"""
import argparse, json, os, random, sys
from math import comb, sqrt, erf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import Taxonomy, extract_answer, parse_path, judge_one


def norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2)))


def load_preds(path, gold, tax):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        g = gold.get(p["id"])
        if g is None:
            continue
        ans, _ = extract_answer(p.get("raw_output", ""))
        names = parse_path(ans)
        nodes, ok, hier = tax.build_pred_path(names) if names else ([], [], False)
        j = judge_one(nodes, g["gold_path"], g["gold_leaf"])
        out[g["id"]] = dict(id=g["id"], gold=g["gold_leaf"], pred=j["pred_label"],
                            correct=bool(j["correct"]))
    return out


def counts(rows, classes):
    ci = {c: i for i, c in enumerate(classes)}
    tp = np.zeros(len(classes)); fp = np.zeros(len(classes)); fn = np.zeros(len(classes))
    for r in rows:
        g, p = r["gold"], r["pred"]
        if g == p:
            tp[ci[g]] += 1
        else:
            fp[ci[p]] += 1
            fn[ci[g]] += 1
    return tp, fp, fn


def macro_from_counts(tp, fp, fn):
    pp = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * pp * rr, pp + rr, out=np.zeros_like(tp), where=(pp + rr) > 0)
    return float(f.mean())


def macro_cells(a_cell, idx_g, idx_p, diag, nC):
    fp = np.bincount(idx_p, weights=a_cell, minlength=nC)
    fn = np.bincount(idx_g, weights=a_cell, minlength=nC)
    tp = np.bincount(idx_g, weights=a_cell * diag, minlength=nC)
    return macro_from_counts(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/test.jsonl")
    ap.add_argument("--base", required=True)
    ap.add_argument("--compare", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()

    tax = Taxonomy("2020", "vocab")
    gold = {}
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r

    base = load_preds(a.base, gold, tax)
    order = [i for i in gold if i in base]
    base_C = sorted({base[i]["gold"] for i in order} | {base[i]["pred"] for i in order})
    strata = {}
    for i in order:
        strata.setdefault(base[i]["gold"], []).append(i)
    strata = {k: v for k, v in sorted(strata.items())}

    rng = random.Random(a.seed)
    boot_idx = []
    for _ in range(a.n_boot):
        idx = []
        for c, ids in strata.items():
            idx.extend(rng.choices(ids, k=len(ids)))
        boot_idx.append(idx)

    out = dict(base=a.base, n=len(order), n_boot=a.n_boot, n_perm=a.n_perm, seed=a.seed,
               base_classes=len(base_C), comparisons=[])
    for cp in a.compare:
        if not os.path.exists(cp):
            out["comparisons"].append(dict(preds=cp, error="missing"))
            continue
        cm = load_preds(cp, gold, tax)
        ids = [i for i in order if i in cm]
        if len(ids) == len(order):
            bids = boot_idx
        else:
            st = {}
            for i in ids:
                st.setdefault(base[i]["gold"], []).append(i)
            r2 = random.Random(a.seed); bids = []
            for _ in range(a.n_boot):
                idx = []
                for c, v in sorted(st.items()):
                    idx.extend(r2.choices(v, k=len(v)))
                bids.append(idx)
        b_rows = [base[i] for i in ids]
        c_rows = [cm[i] for i in ids]
        cmp_C = sorted({r["gold"] for r in c_rows} | {r["pred"] for r in c_rows})

        f1b = macro_from_counts(*counts(b_rows, base_C))
        f1c = macro_from_counts(*counts(c_rows, cmp_C))
        delta = f1b - f1c

        # 旧口径复算：不分层 + 每次重算类别集（仅用于说明 |C| 均值）
        rr = random.Random(a.seed)
        old_d, old_Cb, old_Cc = [], [], []
        for _ in range(a.n_boot):
            smp = [ids[rr.randrange(len(ids))] for _ in range(len(ids))]
            br = [base[i] for i in smp]; cr = [cm[i] for i in smp]
            Cb = sorted({x["gold"] for x in br} | {x["pred"] for x in br})
            Cc = sorted({x["gold"] for x in cr} | {x["pred"] for x in cr})
            old_d.append(macro_from_counts(*counts(br, Cb)) - macro_from_counts(*counts(cr, Cc)))
            old_Cb.append(len(Cb)); old_Cc.append(len(Cc))
        old_d.sort()

        new_d = []
        for idx in bids:
            new_d.append(macro_from_counts(*counts([base[i] for i in idx], base_C))
                         - macro_from_counts(*counts([cm[i] for i in idx], cmp_C)))
        new_d = np.array(new_d)
        lo, hi = float(np.percentile(new_d, 2.5)), float(np.percentile(new_d, 97.5))

        # BCa（仅当百分位区间不含点估计时才算，省时）
        if lo <= delta <= hi:
            bca = [None, None]; z0 = None; acc = None
        else:
            prop = float((new_d < delta).mean())
            prop = min(max(prop, 1.0 / (a.n_boot + 1)), 1 - 1.0 / (a.n_boot + 1))
            z0 = norm_ppf(prop)
            jk = []
            for i in ids:
                jk.append(macro_from_counts(*counts([x for x in b_rows if x["id"] != i], base_C))
                          - macro_from_counts(*counts([x for x in c_rows if x["id"] != i], cmp_C)))
            jk = np.array(jk); jbar = jk.mean()
            num = float(((jbar - jk) ** 3).sum()); den = 6.0 * float(((jbar - jk) ** 2).sum()) ** 1.5
            acc = num / den if den > 0 else 0.0
            def adj(p):
                z = norm_ppf(p)
                return float(phi(z0 + (z0 + z) / (1 - acc * (z0 + z))))
            bca = [float(np.percentile(new_d, 100 * adj(0.025))), float(np.percentile(new_d, 100 * adj(0.975)))]

        # 配对近似随机化检验（联合类别集；观测与零分布同口径）
        joint = sorted(set(base_C) | set(cmp_C))
        jn = {c: i for i, c in enumerate(joint)}
        obs_j = macro_from_counts(*counts(b_rows, joint)) - macro_from_counts(*counts(c_rows, joint))
        pair = {}
        for rb, rc in zip(b_rows, c_rows):
            k = (rb["gold"], rb["pred"])
            pair.setdefault(k, [0, 0])[0] += 1
            k2 = (rc["gold"], rc["pred"])
            pair.setdefault(k2, [0, 0])[1] += 1
        keys = sorted(pair)
        cntB = np.array([pair[k][0] for k in keys], dtype=float)
        cntP = np.array([pair[k][1] for k in keys], dtype=float)
        idx_g = np.array([jn[k[0]] for k in keys])
        idx_p = np.array([jn[k[1]] for k in keys])
        diag = np.array([1.0 if k[0] == k[1] else 0.0 for k in keys])
        nC = len(joint)
        prng = np.random.RandomState(a.seed)
        perm = np.empty(a.n_perm)
        for k in range(a.n_perm):
            kept_b = prng.binomial(cntB.astype(int), 0.5).astype(float)
            kept_c = prng.binomial(cntP.astype(int), 0.5).astype(float)
            A = kept_b + (cntP - kept_c)
            B = kept_c + (cntB - kept_b)
            perm[k] = (macro_cells(A, idx_g, idx_p, diag, nC)
                       - macro_cells(B, idx_g, idx_p, diag, nC))
        p_perm = (1 + int((np.abs(perm) >= abs(obs_j) - 1e-12).sum())) / (a.n_perm + 1)

        bt = [base[i]["correct"] for i in ids]; ct = [cm[i]["correct"] for i in ids]
        b = sum(1 for x, y in zip(bt, ct) if x and not y)
        c = sum(1 for x, y in zip(bt, ct) if not x and y)
        n = b + c
        p_mc = 1.0 if n == 0 else min(1.0, sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n) * 2)

        out["comparisons"].append(dict(
            preds=cp, n=len(ids), macro_f1_cmp=round(f1c, 4),
            delta_f1=round(delta, 4), delta_joint=round(obs_j, 4),
            classes_base=len(base_C), classes_cmp=len(cmp_C),
            old=dict(ci95=[round(old_d[int(0.025 * len(old_d))], 4), round(old_d[int(0.975 * len(old_d)) - 1], 4)],
                     boot_mean=round(float(np.mean(old_d)), 4),
                     C_base_mean=round(float(np.mean(old_Cb)), 2),
                     C_cmp_mean=round(float(np.mean(old_Cc)), 2),
                     C_base_min=int(min(old_Cb)), C_cmp_min=int(min(old_Cc))),
            new=dict(ci95=[round(lo, 4), round(hi, 4)],
                     bca=None if bca[0] is None else [round(bca[0], 4), round(bca[1], 4)],
                     boot_mean=round(float(new_d.mean()), 4),
                     boot_median=round(float(np.median(new_d)), 4),
                     inside=bool(lo <= delta <= hi),
                     inside_bca=None if bca[0] is None else bool(bca[0] <= delta <= bca[1]),
                     z0=None if z0 is None else round(z0, 4),
                     accel=None if acc is None else round(acc, 6)),
            perm_p=round(float(p_perm), 6), perm_mean=round(float(perm.mean()), 4),
            perm_sd=round(float(perm.std()), 4),
            mcnemar=dict(b=b, c=c, p=round(p_mc, 6))))
        print(json.dumps(out["comparisons"][-1], ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
