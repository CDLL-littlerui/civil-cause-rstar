#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute every number in Tables 4-6 from predictions/ only (no GPU, no model).

Usage:
  python scripts/reproduce_tables.py                 # print Tables 4-6
  python scripts/reproduce_tables.py --check         # also diff against expected_metrics.csv
  python scripts/reproduce_tables.py --write-expected# regenerate expected_metrics.csv
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval import Taxonomy, extract_answer, parse_path, judge_one, compute_metrics  # noqa: E402

# file -> (table, system, method)
SPEC = [
    ("ours_qwen3.5-9b_s1", "4", "Qwen3.5-9B", "Ours (seed 1)"),
    ("ours_qwen3.5-9b_s2", "4", "Qwen3.5-9B", "Ours (seed 2)"),
    ("ours_qwen3.5-9b_s3", "4", "Qwen3.5-9B", "Ours (seed 3)"),
    ("ours_qwen3.5-4b", "4", "Qwen3.5-4B", "Ours"),
    ("ours_gemma-4-e4b", "4", "Gemma-4-E4B-it", "Ours"),
    ("ours_minicpm5-2b", "4", "MiniCPM5-2B", "Ours"),
    ("ours_qwen3-8b", "4", "Qwen3-8B", "Ours"),
    ("ours_qwen3-30b-a3b", "4", "Qwen3-30B-A3B", "Ours"),
    ("ours_legalone-8b", "4", "LegalOne-8B", "Ours"),
    ("native_qwen3.5-9b", "4", "Qwen3.5-9B", "Native"),
    ("native_qwen3.5-4b", "4", "Qwen3.5-4B", "Native"),
    ("native_gemma-4-e4b", "4", "Gemma-4-E4B-it", "Native"),
    ("native_minicpm5-2b", "4", "MiniCPM5-2B", "Native"),
    ("native_qwen3-8b", "4", "Qwen3-8B", "Native"),
    ("native_qwen3-30b-a3b", "4", "Qwen3-30B-A3B", "Native"),
    ("native_legalone-8b", "4", "LegalOne-8B", "Native"),
    ("hier_qwen3.5-9b", "5", "Qwen3.5-9B", "Hierarchical prompting"),
    ("native_qwen3.5-9b", "5", "Qwen3.5-9B", "Native"),
    ("cot_qwen3.5-9b", "5", "Qwen3.5-9B", "CoT"),
    ("sc_qwen3.5-9b_s1", "5", "Qwen3.5-9B", "Self-consistency (seed 1)"),
    ("sc_qwen3.5-9b_s2", "5", "Qwen3.5-9B", "Self-consistency (seed 2)"),
    ("sc_qwen3.5-9b_s3", "5", "Qwen3.5-9B", "Self-consistency (seed 3)"),
    ("rstar_s1", "5", "Qwen3.5-9B + Qwen3.5-4B", "rStar (seed 1)"),
    ("rstar_s2", "5", "Qwen3.5-9B + Qwen3.5-4B", "rStar (seed 2)"),
    ("rstar_s3", "5", "Qwen3.5-9B + Qwen3.5-4B", "rStar (seed 3)"),
    ("berry", "5", "Qwen3.5-9B", "LLaMA-Berry"),
    ("native_r1-distill-7b", "5", "R1-Distill-Qwen-7B", "Native"),
    ("sup_textcnn", "5", "TextCNN", "Supervised baseline"),
    ("sup_topjudge", "5", "TopJudge (BERT-wwm-ext)", "Supervised baseline"),
    ("sup_iracps", "5", "IRACPS (BERT-wwm-ext)", "Supervised baseline"),
    ("abl_no_mcts", "6", "Qwen3.5-9B", "w/o MCTS"),
    ("abl_no_disc", "6", "Qwen3.5-9B", "w/o discriminator"),
    ("abl_no_eval", "6", "Qwen3.5-9B", "w/o evaluator (J = a)"),
    ("abl_no_hier", "6", "Qwen3.5-9B", "w/o hierarchy"),
    ("abl_final_only_disc", "6", "Qwen3.5-9B", "discriminator, final level only"),
    ("abl_no_calib", "6", "Qwen3.5-9B", "w/o BERT calibration"),
    ("abl_no_summary", "6", "Qwen3.5-9B", "w/o element summary"),
    ("abl_fixed_c", "6", "Qwen3.5-9B", "fixed c = 2.0"),
    ("abl_no_vote", "6", "Qwen3.5-9B", "w/o vote prior (pure P_h)"),
]
COLS = ["macro_f1", "macro_precision", "macro_recall", "accuracy", "hvr", "n_classes", "invalid_rate"]


def load_gold(path):
    gold = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r
    return gold


def score(preds_path, gold, tax):
    rows, missing = [], 0
    for line in open(preds_path, encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        g = gold.get(p["id"])
        if g is None:
            missing += 1
            continue
        ans, _ = extract_answer(p.get("raw_output", ""))
        names = parse_path(ans)
        nodes, ok, hier = tax.build_pred_path(names) if names else ([], [], False)
        j = judge_one(nodes, g["gold_path"], g["gold_leaf"])
        rows.append(dict(id=g["id"], gold_leaf=g["gold_leaf"], gold_path=g["gold_path"],
                         pred_nodes=nodes, hierarchy_ok=hier, parse_ok=bool(names),
                         resolved=bool(ok and ok[0]), pred_label=j["pred_label"]))
    m = compute_metrics(rows)
    m["n_missing_gold"] = missing
    return m


def fmt(v):
    return ("%.4f" % v) if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(ROOT, "data", "test.jsonl"))
    ap.add_argument("--preds-dir", default=os.path.join(ROOT, "predictions"))
    ap.add_argument("--tax-dir", default=os.path.join(ROOT, "vocab"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-expected", action="store_true")
    a = ap.parse_args()

    tax = Taxonomy("2020", a.tax_dir)
    gold = load_gold(a.gold)
    res = {}
    for name, table, system, method in SPEC:
        path = os.path.join(a.preds_dir, name + ".jsonl")
        if not os.path.exists(path):
            print("MISSING", path, file=sys.stderr)
            continue
        res[name] = score(path, gold, tax)

    for table in ["4", "5", "6"]:
        print("\n### Table %s\n" % table)
        print("| system | method | Macro-F1 | Macro-P | Macro-R | Acc | HVR | |C| | invalid |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, t, system, method in SPEC:
            if t != table or name not in res:
                continue
            m = res[name]
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                system, method, fmt(m["macro_f1"]), fmt(m["macro_precision"]), fmt(m["macro_recall"]),
                fmt(m["accuracy"]), fmt(m["hvr"]), m["n_classes"], fmt(m["invalid_rate"])))

    if a.write_expected:
        out = os.path.join(ROOT, "expected_metrics.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "table", "system", "method", "n"] + COLS)
            for name, table, system, method in SPEC:
                if name not in res:
                    continue
                m = res[name]
                w.writerow([name + ".jsonl", table, system, method, m["n"]] + [m[c] for c in COLS])
        print("\nwrote", out)

    if a.check:
        exp_path = os.path.join(ROOT, "expected_metrics.csv")
        with open(exp_path, encoding="utf-8") as f:
            exp = {r["file"]: r for r in csv.DictReader(f)}
        bad = 0
        for name, table, system, method in SPEC:
            if name not in res:
                continue
            m = res[name]
            e = exp.get(name + ".jsonl")
            if e is None:
                print("NO EXP", name)
                bad += 1
                continue
            for c in COLS:
                if abs(float(e[c]) - float(m[c])) > 1e-4:
                    print("DIFF %-24s %-18s expected %s got %s" % (name, c, e[c], fmt(m[c])))
                    bad += 1
        print("\ncheck: %s" % ("OK, all within 1e-4" if bad == 0 else "%d diffs" % bad))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

