#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一评测入口（步骤 4-B）。

用法:
  python scripts/eval_preds.py --preds runs/qwen9b_native/preds.jsonl \
      --gold data/test.jsonl --out runs/qwen9b_native/metrics.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import Taxonomy, extract_answer, parse_path, judge_one, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taxonomy", default="2020")
    ap.add_argument("--tax-dir", default="vocab")
    a = ap.parse_args()

    tax = Taxonomy(a.taxonomy, a.tax_dir)
    gold = {}
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        gold[r["id"]] = r
    preds = [json.loads(l) for l in open(a.preds, encoding="utf-8") if l.strip()]

    rows, missing = [], []
    for p in preds:
        g = gold.get(p["id"])
        if g is None:
            missing.append(p["id"])
            continue
        ans, method = extract_answer(p.get("raw_output", ""))
        names = parse_path(ans)
        nodes, ok, hier = tax.build_pred_path(names) if names else ([], [], False)
        j = judge_one(nodes, g["gold_path"], g["gold_leaf"])
        rows.append(dict(id=g["id"], gold_leaf=g["gold_leaf"], gold_path=g["gold_path"],
                         answer=ans, extract_method=method, pred_names=names,
                         pred_nodes=nodes, hierarchy_ok=hier,
                         parse_ok=bool(names),
                         resolved=bool(ok and ok[0]),
                         pred_label=j["pred_label"], correct=j["correct"],
                         pred_depth=j["pred_depth"]))

    m = compute_metrics(rows)
    m["preds_file"] = a.preds
    m["gold_file"] = a.gold
    m["n_preds"] = len(preds)
    m["missing_gold"] = missing[:20]
    m["n_missing_gold"] = len(missing)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(m, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: m[k] for k in ["n", "macro_f1", "macro_f1_goldset", "macro_precision", "macro_recall",
                                        "accuracy", "hvr", "parse_failure_rate", "oov_rate", "invalid_rate",
                                        "n_classes", "n_classes_gold", "level_acc"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
