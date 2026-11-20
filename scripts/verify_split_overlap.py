#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-C 校验：各划分之间是否存在精确重复 / 近重复（simhash 汉明 <=3）。"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 与构建脚本共用同一套 simhash 定义，避免“两套哈希各说各话”
from build_cad_v2 import simhash64, hamming

def norm_ws(s):
    return "".join((s or "").split())

d = sys.argv[1] if len(sys.argv) > 1 else "data"
names = ["train", "valid_train", "fewshot_pool", "dev", "test"]
rows = {}
for k in names:
    rows[k] = [json.loads(l) for l in open(os.path.join(d, k + ".jsonl"), encoding="utf-8")]

# 1) 精确
seen = collections.defaultdict(list)
for k in names:
    for r in rows[k]:
        seen[norm_ws(r["input_masked"])].append((k, r["id"]))
exact_cross = {h: v for h, v in seen.items() if len({x[0] for x in v}) > 1}
print("exact cross-split groups:", len(exact_cross))

# 2) 近重复：跨划分两两比较
sig = {}
for k in names:
    for r in rows[k]:
        sig[(k, r["id"])] = simhash64(norm_ws(r["input_masked"]))
buckets = collections.defaultdict(list)
for key, s in sig.items():
    for b in range(4):
        buckets[(b, (s >> (b * 16)) & 0xFFFF)].append(key)
pairs = set()
for _, keys in buckets.items():
    if len(keys) < 2:
        continue
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b_ = keys[i], keys[j]
            if a[0] == b_[0]:
                continue
            if hamming(sig[a], sig[b_]) <= 3:
                pairs.add((min(a, b_), max(a, b_)))
print("near cross-split pairs:", len(pairs))
by_pair = collections.Counter((p[0][0], p[1][0]) for p in pairs)
for k, v in by_pair.items():
    print(" ", k, v)
if pairs:
    for p in list(pairs)[:5]:
        print("  e.g.", p)
out = {"exact_cross_groups": len(exact_cross), "near_cross_pairs": len(pairs),
       "by_pair": {str(k): v for k, v in by_pair.items()}}
os.makedirs("runs", exist_ok=True)
json.dump(out, open(os.path.join("runs", "split_overlap_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
