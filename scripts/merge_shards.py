#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把分片（shard）输出的 jsonl 按原始划分顺序合并成一个文件。

用法：
  python scripts/merge_shards.py --split dev --data-dir data \
      --out runs/method_main/dev/k8_merged/paths.jsonl \
      --inputs runs/method_main/dev/k8/paths.jsonl,runs/method_main/dev/k8_s1/paths.jsonl
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--inputs", required=True, help="逗号分隔")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    order = {}
    for i, line in enumerate(open(os.path.join(a.data_dir, a.split + ".jsonl"), encoding="utf-8")):
        if line.strip():
            order[str(json.loads(line)["id"])] = i

    recs = {}
    dup = 0
    for p in a.inputs.split(","):
        if not os.path.exists(p):
            print("MISSING", p)
            continue
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = str(r["id"])
            if k in recs:
                dup += 1
                continue
            recs[k] = r
    missing = [k for k in order if k not in recs]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for k in sorted(recs, key=lambda x: order.get(x, 10 ** 9)):
            f.write(json.dumps(recs[k], ensure_ascii=False) + "\n")
    print(json.dumps({"out": a.out, "n": len(recs), "dup": dup, "missing": len(missing),
                      "missing_ids": missing[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
