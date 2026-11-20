#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-B/C: 由 AppealCase 构建并冻结 CAD-v2。

流程: 读取 -> 排除刑事 -> 组输入(claim + fact_description) -> 精确/近重复去重
      -> 长度过滤(<100 字) -> 案由映射(2020 词表) -> 掩码 -> 分层划分 -> 冻结

用法:
  python scripts/build_cad_v2.py --out-dir data \
      --split-config configs/split_v2.yaml --report-json runs/build_stats.json
"""
import argparse, json, re, hashlib, csv, collections, os, sys, zlib
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_labels as ml

TAX = "vocab"
MASK = "[案由]"

# 步骤 4 A0：这 15 条改由 vocab/manual_mapping_2020.csv（人工核定，优先级最高）处理，
# 不再走别名表，也不再标“需人工确认”。
APPEAL_ALIAS = {}
APPEAL_ALIAS_CONFIRM = set()

# 掩码禁用词（单独出现会抹掉通用法律词，禁止进入掩码词清单）
FORBIDDEN_MASK = {"合同", "侵权", "物权", "纠纷", "责任", "权属", "损害", "赔偿",
                  "劳动", "婚姻", "继承", "借款", "租赁", "买卖"}
CORE_MIN_LEN = 4  # 叶节点核心词（去掉“纠纷”后）长度下限

CRIMINAL_HINTS = ["刑事"]


def norm_ws(s):
    return re.sub(r"\s+", "", s or "")


def simhash64(text, ngram=4):
    """字符 n-gram simhash（numpy 向量化）。返回 64bit 整数。"""
    import numpy as np
    if not text:
        return 0
    if len(text) < ngram:
        text = (text + " ") * ngram
    grams = [text[i:i + ngram] for i in range(len(text) - ngram + 1)]
    lo = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
    hi = np.fromiter((zlib.crc32(g[::-1].encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
    hs = lo | (hi << np.uint64(32))
    ar = np.arange(64, dtype=np.uint64)
    bits = ((hs[:, None] >> ar) & np.uint64(1)).astype(np.int32)
    v = bits.sum(axis=0) * 2 - len(grams)
    out = 0
    pos = np.nonzero(v > 0)[0]
    for b in pos:
        out |= (1 << int(b))
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def dedup_groups(items, key_text, max_dist=3, bands=4):
    """返回 (keep_indices, dup_pairs); items 为 (idx, text). 精确 + simhash 近重复。"""
    exact = {}
    keep, dup = [], []
    norm = {}
    for idx, text in items:
        n = norm_ws(text)
        norm[idx] = n
        if n in exact:
            dup.append((exact[n], idx, "exact"))
            continue
        exact[n] = idx
        keep.append(idx)
    # 近重复: 4 段 16bit 分桶
    sig = {idx: simhash64(norm[idx]) for idx in keep}
    buckets = collections.defaultdict(list)
    removed = set()
    for idx in keep:
        s = sig[idx]
        cands = set()
        for b in range(bands):
            key = (b, (s >> (b * 16)) & 0xFFFF)
            for other in buckets[key]:
                cands.add(other)
            buckets[key].append(idx)
        hit = None
        for other in cands:
            if other in removed:
                continue
            if hamming(s, sig[other]) <= max_dist:
                hit = other
                break
        if hit is not None:
            removed.add(idx)
            dup.append((hit, idx, "near"))
    keep = [i for i in keep if i not in removed]
    return keep, dup


def _core(name):
    return name[:-2] if name.endswith("纠纷") else name


def mask_terms_for(path, cause_raw=""):
    """返回 [(term, source)]。

    步骤 4 A1 规则：
      - 祖先节点只掩码完整名称（不再掩码去“纠纷”的核心词）；
      - 叶节点完整名称始终掩码；叶的核心词仅当长度 ≥ CORE_MIN_LEN(4) 时掩码；
      - 原始标签名与原始核心词（≥4）同样掩码；
      - 禁止 FORBIDDEN_MASK 中的通用词单独进入清单。
    """
    out, seen = [], set()

    def add(t, src):
        if not t or t in seen or t in FORBIDDEN_MASK:
            return
        seen.add(t)
        out.append((t, src))

    for i, name in enumerate(path):
        if i == len(path) - 1:
            add(name, "叶")
        else:
            add(name, "祖先")
    if path:
        core = _core(path[-1])
        if len(core) >= CORE_MIN_LEN:
            add(core, "核心词")
    if cause_raw:
        add(cause_raw, "原名")
        rc = _core(cause_raw)
        if len(rc) >= CORE_MIN_LEN:
            add(rc, "原名核心词")
    return out


def apply_mask(text, terms, counter):
    for t in sorted(terms, key=len, reverse=True):
        if t and t in text:
            counter[t] += text.count(t)
            text = text.replace(t, MASK)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/appealcase/AppealCase.json")
    ap.add_argument("--meta", default="data/raw/appealcase/AppealCase.metadata.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-per-cause", type=int, default=10)
    ap.add_argument("--dev-per-cause", type=int, default=3)
    ap.add_argument("--fewshot-per-cause", type=int, default=2)
    ap.add_argument("--min-per-cause", type=int, default=15)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--fewshot10-size", type=int, default=10)
    ap.add_argument("--fewshot10-folds", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--report-json", default="runs/build_stats.json")
    ap.add_argument("--label-map-csv", default="runs/appealcase_label_map.csv")
    ap.add_argument("--split-config", default="configs/split_v2.yaml")
    ap.add_argument("--mask-terms-csv", default="runs/mask_terms.csv")
    a = ap.parse_args()
    for _p in (a.report_json, a.label_map_csv, a.mask_terms_csv):
        os.makedirs(os.path.dirname(os.path.abspath(_p)), exist_ok=True)

    # 划分配置（唯一事实来源；命令行参数可覆盖，便于消融）
    if a.split_config and os.path.exists(a.split_config):
        cfg = yaml.safe_load(open(a.split_config, encoding="utf-8")) or {}
        for key, val in cfg.items():
            if hasattr(a, key):
                setattr(a, key, val)
        print("loaded split config:", a.split_config, cfg)

    nodes20 = ml.load_nodes("2020")
    idx20 = ml.build_index(nodes20)
    manual = ml.load_manual()

    data = json.load(open(a.data, encoding="utf-8"))
    meta = json.load(open(a.meta, encoding="utf-8"))
    assert len(data) == len(meta)

    funnel = collections.Counter()
    funnel["raw"] = len(data)

    recs = []
    for rec, m in zip(data, meta):
        fi, si = rec["first-instance"], rec["second-instance"]
        types = m["first-instance"].get("type") or []
        if any(h in t for t in types for h in CRIMINAL_HINTS):
            funnel["drop_criminal"] += 1
            continue
        claim = (fi.get("claim") or "").strip()
        fact = (fi.get("fact_description") or "").strip()
        if not claim or not fact:
            funnel["drop_missing_section"] += 1
            continue
        cause_raw = (m["first-instance"].get("cause_of_action") or "").strip()
        no = (m["first-instance"].get("case") or {}).get("no") or ""
        ym = re.search(r"[（(](\d{4})[）)]", no)
        year = int(ym.group(1)) if ym else None
        recs.append(dict(
            source_id=str(rec["index"]),
            claim=claim, fact=fact,
            input_raw=claim + "\n" + fact,
            cause_raw=cause_raw,
            cause_second=(m["second-instance"].get("cause_of_action") or "").strip(),
            year=year, case_no=no,
        ))
    funnel["after_section"] = len(recs)

    # 去重
    items = [(i, r["input_raw"]) for i, r in enumerate(recs)]
    keep, dups = dedup_groups(items, None)
    funnel["dup_exact"] = sum(1 for d in dups if d[2] == "exact")
    funnel["dup_near"] = sum(1 for d in dups if d[2] == "near")
    recs = [recs[i] for i in keep]
    funnel["after_dedup"] = len(recs)

    # 长度过滤
    before = len(recs)
    recs = [r for r in recs if len(norm_ws(r["input_raw"])) >= a.min_len]
    funnel["drop_short"] = before - len(recs)
    funnel["after_len"] = len(recs)

    # 标签映射
    label_rows = collections.Counter(r["cause_raw"] for r in recs)
    map_cache = {}
    mapped, fail = [], collections.Counter()
    for r in recs:
        cause = r["cause_raw"]
        if cause not in map_cache:
            map_cache[cause] = ml.normalize(cause, idx20, manual)
        mp = map_cache[cause]
        if not mp["规范名称"] or ";" in mp["规范名称"]:
            fail[cause] += 1
            continue
        r["gold_leaf"] = mp["规范名称"]
        r["method"] = mp["方法"]
        r["need_confirm"] = mp["需人工确认"]
        if cause in APPEAL_ALIAS_CONFIRM:
            r["need_confirm"] = "是"
        r["gold_level"] = mp["层级"]
        r["gold_path"] = mp["完整路径"].split("/")
        mapped.append(r)
    funnel["drop_unmapped"] = len(recs) - len(mapped)
    recs = mapped
    funnel["after_map"] = len(recs)

    # 标签映射表
    with open(a.label_map_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["原名", "频次", "规范名", "层级", "完整路径", "方法", "需人工确认"])
        for name, f in label_rows.most_common():
            mp = map_cache.get(name) or ml.normalize(name, idx20, manual)
            conf = mp["需人工确认"]
            if name in APPEAL_ALIAS_CONFIRM:
                conf = "是"
            w.writerow([name, f, mp["规范名称"] or "（无法映射）", mp["层级"],
                        mp["完整路径"], mp["方法"], conf])

    # 掩码（步骤 4 A1 新规则）＋ 掩码词清单
    hit = collections.Counter()
    mask_stat = collections.Counter()
    term_hits = collections.Counter()   # (gold_leaf, term, source) -> 全库出现次数
    term_src = {}
    for r in recs:
        pairs = mask_terms_for(r["gold_path"], (r.get("cause_raw") or "").strip())
        terms = {t for t, _ in pairs}
        for t, src in pairs:
            term_src[(r["gold_leaf"], t)] = src
            term_hits[(r["gold_leaf"], t)] += r["input_raw"].count(t)
        txt = r["input_raw"]
        before_len = len(txt)
        txt = apply_mask(txt, terms, hit)
        r["input_masked"] = txt
        if txt != r["input_raw"]:
            mask_stat["any"] += 1
        # 各级祖先命中（按 path 位置）
        for lvl, name in enumerate(r["gold_path"]):
            if name in r["input_raw"]:
                mask_stat[f"path_pos{lvl}"] += 1
        if r["gold_leaf"] in r["input_raw"]:
            mask_stat["leaf_name"] += 1
        core = r["gold_leaf"][:-2] if r["gold_leaf"].endswith("纠纷") else r["gold_leaf"]
        if core and core != r["gold_leaf"] and core in r["input_raw"]:
            mask_stat["core_name"] += 1

    # 掩码后再去重：掩码会把不同案由名替换成同一个 [案由]，可能把两条原本不同的文本
    # 变成近重复（实测跨划分有 9 对）。这里在 input_masked 上再做一次精确 + 近重复去重，
    # 保证划分之间同时满足“精确无重复”和“近重复（汉明 ≤3）无重复”。
    items2 = [(i, r["input_masked"]) for i, r in enumerate(recs)]
    keep2, dups2 = dedup_groups(items2, None)
    funnel["dup_exact_after_mask"] = sum(1 for d in dups2 if d[2] == "exact")
    funnel["dup_near_after_mask"] = sum(1 for d in dups2 if d[2] == "near")
    recs = [recs[i] for i in keep2]
    funnel["after_mask_dedup"] = len(recs)

    # 掩码统计重算（分母改为去重后的最终样本集；input_raw / input_masked 均已就绪）
    hit = collections.Counter()
    mask_stat = collections.Counter()
    for r in recs:
        if r["input_masked"] != r["input_raw"]:
            mask_stat["any"] += 1
        for lvl, name in enumerate(r["gold_path"]):
            if name in r["input_raw"]:
                mask_stat[f"path_pos{lvl}"] += 1
        if r["gold_leaf"] in r["input_raw"]:
            mask_stat["leaf_name"] += 1
        core = r["gold_leaf"][:-2] if r["gold_leaf"].endswith("纠纷") else r["gold_leaf"]
        if core and core != r["gold_leaf"] and core in r["input_raw"]:
            mask_stat["core_name"] += 1

    # 掩码词清单（最终样本集口径；全库命中次数 = 该案由所有样本 input_raw 中出现次数之和）
    term_hits, term_src = collections.Counter(), {}
    for r in recs:
        for t, src in mask_terms_for(r["gold_path"], (r.get("cause_raw") or "").strip()):
            term_src[(r["gold_leaf"], t)] = src
            term_hits[(r["gold_leaf"], t)] += r["input_raw"].count(t)
    os.makedirs(os.path.dirname(a.mask_terms_csv), exist_ok=True)
    with open(a.mask_terms_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["案由", "掩码词", "来源", "全库命中次数"])
        for (leaf, t), c in sorted(term_hits.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
            w.writerow([leaf, t, term_src[(leaf, t)], c])
    bad_forbidden = sorted({t for (_, t) in term_hits if t in FORBIDDEN_MASK})

    # 划分
    import random
    def cause_rng(cause):
        """按案由独立播种：样本池增删只影响该案由自身，跨版本身份稳定。"""
        return random.Random("%d:%s" % (a.seed, cause))
    by_cause = collections.defaultdict(list)
    for r in recs:
        by_cause[r["gold_leaf"]].append(r)
    splits = {"train": [], "valid_train": [], "fewshot_pool": [], "dev": [], "test": []}
    excluded = []
    fewshot10_pool = []
    for cause, group in sorted(by_cause.items()):
        g = sorted(group, key=lambda x: x["source_id"])
        cause_rng(cause).shuffle(g)
        if len(g) < a.min_per_cause:
            excluded.append((cause, len(g)))
            splits["train"].extend(g)
            continue
        splits["test"].extend(g[:a.test_per_cause])
        g = g[a.test_per_cause:]
        splits["dev"].extend(g[:a.dev_per_cause])
        g = g[a.dev_per_cause:]
        splits["fewshot_pool"].extend(g[:a.fewshot_per_cause])
        rest = g[a.fewshot_per_cause:]
        splits["train"].extend(rest)

    # 固定 10 条 few-shot 示例：按 L1 大类分层，从训练池每类取 1 条（不足 10 类时按序补足）
    l1_pool = collections.defaultdict(list)
    for r in splits["train"]:
        l1_pool[r["gold_path"][0]].append(r)
    picked = []
    for l1 in sorted(l1_pool):
        if len(picked) >= a.fewshot10_size:
            break
        picked.append(sorted(l1_pool[l1], key=lambda x: x["source_id"])[0])
    if len(picked) < a.fewshot10_size:
        extra = [r for r in sorted(splits["train"], key=lambda x: x["source_id"]) if r not in picked]
        picked.extend(extra[:a.fewshot10_size - len(picked)])
    picked_ids = {r["source_id"] for r in picked}
    splits["train"] = [r for r in splits["train"] if r["source_id"] not in picked_ids]

    # valid_train: 从 train 中按案由分层取 10%
    train_by_cause = collections.defaultdict(list)
    for r in splits["train"]:
        train_by_cause[r["gold_leaf"]].append(r)
    valid, train_final = [], []
    for cause, g in sorted(train_by_cause.items()):
        g = sorted(g, key=lambda x: x["source_id"])
        cause_rng(cause).shuffle(g)
        k = max(1, int(round(len(g) * a.valid_ratio))) if len(g) >= 2 else 0
        valid.extend(g[:k])
        train_final.extend(g[k:])
    splits["valid_train"] = valid
    splits["train"] = train_final

    os.makedirs(a.out_dir, exist_ok=True)
    def dump(name, rows):
        p = os.path.join(a.out_dir, name + ".jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "id": r["source_id"],
                    "source_id": r["source_id"],
                    "input_masked": r["input_masked"],
                    "gold_leaf": r["gold_leaf"],
                    "gold_level": r["gold_level"],
                    "gold_path": r["gold_path"],
                    "cause_raw": r["cause_raw"],
                    "year": r["year"],
                }, ensure_ascii=False) + "\n")
        return p

    paths = {k: dump(k, v) for k, v in splits.items()}
    # fewshot10: 5 折 × 2
    per_fold = max(1, -(-len(picked) // a.fewshot10_folds))
    folds = [picked[i:i + per_fold] for i in range(0, len(picked), per_fold)]
    few10 = os.path.join(a.out_dir, "fewshot10_folds.jsonl")
    with open(few10, "w", encoding="utf-8") as fh:
        for fi, fold in enumerate(folds):
            for r in fold:
                fh.write(json.dumps({
                    "id": r["source_id"], "source_id": r["source_id"],
                    "fold": fi, "input_masked": r["input_masked"],
                    "gold_leaf": r["gold_leaf"], "gold_level": r["gold_level"],
                    "gold_path": r["gold_path"], "cause_raw": r["cause_raw"], "year": r["year"],
                }, ensure_ascii=False) + "\n")
    paths["fewshot10_folds"] = few10

    # 跨划分重复校验
    all_ids = collections.defaultdict(list)
    for name, rows in splits.items():
        for r in rows:
            all_ids[r["source_id"]].append(name)
    cross = {k: v for k, v in all_ids.items() if len(v) > 1}

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    stats = {
        "funnel": dict(funnel),
        "splits": {k: len(v) for k, v in splits.items()},
        "n_causes_all": len(by_cause),
        "n_causes_test": len({r["gold_leaf"] for r in splits["test"]}),
        "excluded_causes_lt_min": sorted(excluded, key=lambda x: -x[1]),
        "mask": {"records_any_mask": mask_stat["any"],
                 "leaf_name_hit": mask_stat["leaf_name"],
                 "core_name_hit": mask_stat["core_name"],
                 "path_pos_hits": {k: v for k, v in mask_stat.items() if k.startswith("path_pos")},
                 "top_terms": hit.most_common(20),
                 "core_min_len": CORE_MIN_LEN,
                 "forbidden_in_terms": bad_forbidden,
                 "n_mask_terms": len(term_hits),
                 "mask_terms_csv": a.mask_terms_csv},
        "cross_split_duplicates": cross,
        "sha256": {k: sha(p) for k, p in paths.items()},
        "fewshot10": [{"id": r["source_id"], "gold_leaf": r["gold_leaf"]} for r in picked],
        "label_map_fail": fail.most_common(),
    }
    json.dump(stats, open(a.report_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in ["funnel", "splits", "n_causes_all", "n_causes_test", "mask", "cross_split_duplicates"]}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
