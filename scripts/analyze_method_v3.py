#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 7 诊断：从 gen.jsonl / paths_*.jsonl / preds_*.jsonl 汇总各项诊断指标。

用法：
  python scripts/analyze_method_v3.py --gen runs/method_v3/dev/merged/gen.jsonl \
    --paths-dir runs/method_v3/dev/merged --preds-dir runs/method_v3/dev/eval \
    --gold data/dev.jsonl --selected t1_l0.5 \
    --extra hier=runs/qwen3.5-9b/dev/hier/preds.jsonl \
    --out-json reports/step7_diag.json --out-md reports/step7_diag.md
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import Taxonomy, judge_one, compute_metrics, extract_answer, parse_path


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_rows(preds_path, gold, tax):
    rows = []
    for p in read_jsonl(preds_path):
        g = gold.get(p["id"])
        if g is None:
            continue
        ans, _ = extract_answer(p.get("raw_output", ""))
        names = parse_path(ans)
        nodes, ok, hier = tax.build_pred_path(names) if names else ([], [], False)
        j = judge_one(nodes, g["gold_path"], g["gold_leaf"])
        rows.append(dict(id=p["id"], gold_leaf=g["gold_leaf"], gold_path=g["gold_path"],
                         pred_names=names, pred_nodes=nodes, hierarchy_ok=hier,
                         parse_ok=bool(names), resolved=bool(ok and ok[0]),
                         pred_label=j["pred_label"], correct=j["correct"],
                         pred_depth=j["pred_depth"]))
    return rows


def trunc_correct(path_names, gold_path, gold_leaf, tax):
    nodes, ok, hier = tax.build_pred_path(path_names) if path_names else ([], [], False)
    return judge_one(nodes, gold_path, gold_leaf)["correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--paths-dir", required=True)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--taxonomy", default="2020")
    ap.add_argument("--tax-dir", default="vocab")
    ap.add_argument("--selected", default="")
    ap.add_argument("--extra", default="", help="name=preds.jsonl,...")
    ap.add_argument("--disc-from-preds", action="store_true",
                    help="鉴别器统计读 preds 的 paths（含 a/J），默认读 gen 的 paths_*.jsonl（无 a/J）")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    a = ap.parse_args()

    tax = Taxonomy(a.taxonomy, a.tax_dir)
    gold = {str(r["id"]): r for r in read_jsonl(a.gold)}
    gen = {str(r["id"]): r for r in read_jsonl(a.gen)}
    out = {"n": len(gen), "selected": a.selected, "combos": {}, "extra": {},
           "l1_disagree": {}, "cost": {}}

    # ---- 成本
    gs = list(gen.values())
    out["cost"] = {
        "calls_mean": round(sum(r["calls"] for r in gs) / len(gs), 2),
        "prior_calls_mean": round(sum(r["prior_calls"] for r in gs) / len(gs), 2),
        "tokens_in_mean": round(sum(r["tokens_in"] for r in gs) / len(gs)),
        "tokens_out_mean": round(sum(r["tokens_out"] for r in gs) / len(gs)),
        "seconds_mean": round(sum(r["seconds"] for r in gs) / len(gs), 2),
        "votes_ok_mean": round(sum(r["n_votes"] for r in gs) / len(gs), 2),
        "votes_fail": sum(1 for r in gs if r["n_votes"] == 0),
        "summary_chars_mean": round(sum(len(r["summary"]) for r in gs) / len(gs)),
    }

    # ---- L1 不一致
    dis = [r for r in gs if r["l1_prior_top"] != r["l1_vote_top"]]
    out["l1_disagree"] = {"n": len(dis), "rate": round(len(dis) / len(gs), 4),
                          "ids": [r["id"] for r in dis]}
    acc_on = {}
    for name, p in [("v3", os.path.join(a.preds_dir, "preds_%s.jsonl" % a.selected))] + \
            [tuple(x.split("=", 1)) for x in a.extra.split(",") if "=" in x]:
        if not os.path.exists(p):
            continue
        rows = {r["id"]: r for r in build_rows(p, gold, tax)}
        sub = [rows[i] for i in out["l1_disagree"]["ids"] if i in rows]
        acc_on[name] = {"n": len(sub),
                        "acc": round(sum(1 for r in sub if r["correct"]) / max(1, len(sub)), 4)}
    out["l1_disagree"]["acc"] = acc_on

    # ---- 各组合
    tags = []
    if os.path.isdir(a.paths_dir):
        for f in sorted(os.listdir(a.paths_dir)):
            if f.startswith("paths_") and f.endswith(".jsonl") and "nomcts" not in f and "s16" not in f:
                tags.append(f[len("paths_"):-len(".jsonl")])
    for tag in tags:
        pf = os.path.join(a.paths_dir, "paths_%s.jsonl" % tag)
        pp = os.path.join(a.preds_dir, "preds_%s.jsonl" % tag)
        if not os.path.exists(pp):
            continue
        if a.disc_from_preds:
            precs = {str(r["id"]): r for r in read_jsonl(pp) if r.get("paths")}
        else:
            precs = {str(r["id"]): r for r in read_jsonl(pf)}
        d = {"paths": {}, "oracle": {}, "disc": {}}
        rows = build_rows(pp, gold, tax)
        m = compute_metrics(rows)
        d["metrics"] = {k: m[k] for k in ["n", "macro_f1", "macro_f1_goldset", "macro_precision",
                                          "macro_recall", "accuracy", "hvr", "n_classes",
                                          "parse_failure_rate", "oov_rate", "invalid_rate"]}
        d["metrics"]["level_acc"] = {k: m["level_acc"][k]["acc"] for k in m["level_acc"]}
        # 路径数分布 / share
        npaths = collections.Counter()
        shares = []
        orc_t, orc_l = 0, 0
        a_gold, a_non = [], []
        jflip = {"changed": 0, "fixed": 0, "broken": 0}
        for rid, rec in precs.items():
            g = gold.get(rid)
            if g is None:
                continue
            ps = rec["paths"]
            npaths[len(ps)] += 1
            if ps:
                shares.append(ps[0]["share"])
            ok_t = ok_l = False
            for p in ps:
                if trunc_correct(p["path"], g["gold_path"], g["gold_leaf"], tax):
                    ok_t = True
                if p["path"] == g["gold_path"]:
                    ok_l = True
                if p.get("a") is not None:
                    (a_gold if p["path"] == g["gold_path"] else a_non).append(p["a"])
            orc_t += int(ok_t)
            orc_l += int(ok_l)
            if len(ps) > 1:
                by_v = max(ps, key=lambda p: p["visits"])
                by_j = max(ps, key=lambda p: (p.get("J", 0.0), p["visits"]))
                if by_v["path"] != by_j["path"]:
                    jflip["changed"] += 1
                    cv = trunc_correct(by_v["path"], g["gold_path"], g["gold_leaf"], tax)
                    cj = trunc_correct(by_j["path"], g["gold_path"], g["gold_leaf"], tax)
                    if cj and not cv:
                        jflip["fixed"] += 1
                    elif cv and not cj:
                        jflip["broken"] += 1
        d["paths"] = {"dist": {str(k): v for k, v in sorted(npaths.items())},
                      "share_mean": round(sum(shares) / max(1, len(shares)), 4),
                      "share_top1_over_90": round(sum(1 for s in shares if s >= 0.9) / max(1, len(shares)), 4)}
        d["oracle"] = {"truncated": round(orc_t / max(1, len(precs)), 4),
                       "leaf_exact": round(orc_l / max(1, len(precs)), 4)}
        d["disc"] = {"a_gold_mean": round(sum(a_gold) / max(1, len(a_gold)), 4),
                     "a_nongold_mean": round(sum(a_non) / max(1, len(a_non)), 4),
                     "n_paths_gold": len(a_gold), "n_paths_nongold": len(a_non),
                     "jflip": jflip}
        out["combos"][tag] = d

    # ---- 额外对照（no_mcts / s16 / vote / hier / sc）
    for name, p in [tuple(x.split("=", 1)) for x in a.extra.split(",") if "=" in x]:
        if not os.path.exists(p):
            continue
        rows = build_rows(p, gold, tax)
        m = compute_metrics(rows)
        out["extra"][name] = {k: m[k] for k in ["n", "macro_f1", "macro_f1_goldset", "accuracy",
                                                "hvr", "n_classes", "parse_failure_rate",
                                                "oov_rate", "invalid_rate"]}
        out["extra"][name]["level_acc"] = {k: m["level_acc"][k]["acc"] for k in m["level_acc"]}

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"n": out["n"], "cost": out["cost"],
                      "l1_disagree_n": out["l1_disagree"]["n"],
                      "combos": {t: out["combos"][t]["metrics"]["macro_f1"] for t in out["combos"]}},
                     ensure_ascii=False, indent=1))

    if a.out_md:
        L = ["# 步骤 7 诊断（method_v3，dev）", ""]
        c = out["cost"]
        L += ["## 成本（每样本均值）", "",
              "| 调用数 | 先验调用 | 输入 token | 输出 token | 秒 | 有效票 | 摘要字数 |",
              "| --- | --- | --- | --- | --- | --- | --- |",
              "| %.2f | %.2f | %d | %d | %.2f | %.2f | %d |" % (
                  c["calls_mean"], c["prior_calls_mean"], c["tokens_in_mean"], c["tokens_out_mean"],
                  c["seconds_mean"], c["votes_ok_mean"], c["summary_chars_mean"]), ""]
        L += ["## 各组合", "",
              "| tag | F1 | goldset | acc | HVR | 路径数分布 | share均值 | Oracle截断 | Oracle叶精确 | a(金) | a(非金) | J改选/改对/改错 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t, d in out["combos"].items():
            m, p, o, dc = d["metrics"], d["paths"], d["oracle"], d["disc"]
            L.append("| %s | %.4f | %.4f | %.4f | %.4f | %s | %.3f | %.4f | %.4f | %.3f | %.3f | %d/%d/%d |" % (
                t, m["macro_f1"], m["macro_f1_goldset"], m["accuracy"], m["hvr"],
                " / ".join("%s:%d" % (k, v) for k, v in p["dist"].items()),
                p["share_mean"], o["truncated"], o["leaf_exact"],
                dc["a_gold_mean"], dc["a_nongold_mean"],
                dc["jflip"]["changed"], dc["jflip"]["fixed"], dc["jflip"]["broken"]))
        L += ["", "## 逐级准确率", "",
              "| tag | L1 | L2 | L3 | L4 |", "| --- | --- | --- | --- | --- |"]
        for t, d in out["combos"].items():
            la = d["metrics"]["level_acc"]
            L.append("| %s | %s | %s | %s | %s |" % (t, la.get("L1"), la.get("L2"), la.get("L3"), la.get("L4")))
        if out["extra"]:
            L += ["", "## 对照", "", "| 配置 | F1 | acc | HVR | L1 | L2 | L3 | L4 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for t, m in out["extra"].items():
                la = m["level_acc"]
                L.append("| %s | %.4f | %.4f | %.4f | %s | %s | %s | %s |" % (
                    t, m["macro_f1"], m["accuracy"], m["hvr"], la.get("L1"), la.get("L2"), la.get("L3"), la.get("L4")))
        L += ["", "## L1 层级先验与投票不一致", "",
              "样本数 %d（%.1f%%）；子集准确率：%s" % (
                  out["l1_disagree"]["n"], 100 * out["l1_disagree"]["rate"],
                  "; ".join("%s %.4f (n=%d)" % (k, v["acc"], v["n"]) for k, v in out["l1_disagree"]["acc"].items())), ""]
        open(a.out_md, "w", encoding="utf-8").write("\n".join(L))
        print("wrote", a.out_md)


if __name__ == "__main__":
    main()
