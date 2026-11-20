#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 method_v3 的中间产物派生消融预测（不调用大模型、不改动冻结代码）。

模式：
  nomcts     读 paths_nomcts_<tag>.jsonl（单路径）→ preds
  max_share  --no_disc：J 换成 share，按 share 选路径
  max_a      --final_only_disc：J 换成 a，按 a 选路径
  max_visits --no_eval：按访问次数选最高叶
  vote_top   --no_hier：开放投票的多数答案（16 票中出现最多的完整路径）

用法：
  python scripts/derive_preds.py --mode max_share --in runs/.../preds_t2_l0.3.jsonl --out .../preds_nodisc.jsonl
  python scripts/derive_preds.py --mode vote_top --gen runs/.../gen.jsonl --out .../preds_nohier.jsonl
"""
import argparse
import collections
import json


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pick(recs, key):
    out = []
    for r in recs:
        paths = r.get("paths") or []
        if not paths:
            out.append({"id": str(r["id"]), "pred_path": [], "raw_output": ""})
            continue
        best = max(paths, key=key)
        out.append({"id": str(r["id"]), "pred_path": best["path"],
                    "raw_output": "最终案由：" + " > ".join(best["path"])})
    return out


def vote_top(recs):
    out = []
    for r in recs:
        votes = r.get("votes") or []
        if not votes:
            out.append({"id": str(r["id"]), "pred_path": [], "raw_output": ""})
            continue
        cnt = collections.Counter(tuple(v) for v in votes)
        # 票数最多者胜；并列时取票数最大的第一条（Counter 保序）
        best, _ = max(cnt.items(), key=lambda kv: kv[1])
        out.append({"id": str(r["id"]), "pred_path": list(best),
                    "raw_output": "最终案由：" + " > ".join(best)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["nomcts", "max_share", "max_a", "max_visits", "vote_top"])
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--gen", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "vote_top":
        recs = read_jsonl(a.gen)
        rows = vote_top(recs)
    elif a.mode == "nomcts":
        recs = read_jsonl(a.inp)
        rows = []
        for r in recs:
            paths = r.get("paths") or []
            p = paths[0]["path"] if paths else []
            rows.append({"id": str(r["id"]), "pred_path": p,
                         "raw_output": ("最终案由：" + " > ".join(p)) if p else ""})
    else:
        key = {"max_share": lambda p: (p.get("share", 0.0), p.get("visits", 0)),
               "max_a": lambda p: (p.get("a", 0.0), p.get("share", 0.0)),
               "max_visits": lambda p: (p.get("visits", 0), p.get("share", 0.0))}[a.mode]
        rows = pick(read_jsonl(a.inp), key)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("mode=%s wrote %d -> %s" % (a.mode, len(rows), a.out))


if __name__ == "__main__":
    main()
