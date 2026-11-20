#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把数据集的案由名称映射到 2020 版词表（精确 / 规则 / 失败），禁止嵌入相似度。

用法:
  python3 scripts/map_labels.py --input data/raw/claimgen/ClaimGen-CN.json \
      --dataset claimgen --cause-field cause --out data/processed/claimgen_label_map.csv

输入支持 JSON 数组或 JSONL；cause 字段可为字符串或字符串列表。
输出 CSV 列: 原名,频次,规范名称,层级,完整路径,方法,是否需人工确认
"""
import argparse, csv, json, re
from collections import Counter

TAX = "vocab"


def load_nodes(version):
    return [json.loads(l) for l in open(f"{TAX}/cause_taxonomy_{version}_nodes.jsonl", encoding="utf-8")]


# 人工核定的别名词表（只收已确认的写法差异，不做相似度推断）
ALIAS = {
    "医疗损害赔责任纠纷": "医疗损害责任纠纷",
    "执行异议纠纷": "执行程序中的异议之诉",
    "民间借贷合同纠纷": "民间借贷纠纷",
    "劳动争议纠纷": "劳动争议",
    "人事争议纠纷": "人事争议",
    # 民法典术语替换（2020 修正版删去旧名）
    "合伙协议纠纷": "合伙合同纠纷",
    "居间合同纠纷": "中介合同纠纷",
}
CONJ = {
    "民间借贷及追偿权纠纷": ["民间借贷纠纷", "追偿权纠纷"],
}

# 人工核定映射（步骤 4 A0）：vocab/manual_mapping_2020.csv，优先级最高，命中即不再标“需人工确认”
MANUAL_PATH = "vocab/manual_mapping_2020.csv"


def load_manual(path=None):
    path = path or MANUAL_PATH
    out = {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                out[r["原标签"].strip()] = r
    except FileNotFoundError:
        pass
    return out


def split_components(name):
    core = re.sub(r"纠纷$", "", name)
    return [x for x in core.split("、") if x]


def build_index(nodes):
    """name -> [nodes]; 同名时按 level 从深到浅排序"""
    idx = {}
    for n in nodes:
        idx.setdefault(n["name"], []).append(n)
    for v in idx.values():
        v.sort(key=lambda x: -x["level"])
    return idx


def pick(name, idx):
    """同名节点取最深（最具体）的一个，返回 (node, 说明)"""
    cands = idx.get(name)
    if not cands:
        return None, ""
    if len(cands) == 1:
        return cands[0], ""
    n = cands[0]
    return n, f"同名节点{len(cands)}个(层级{','.join('L%d'%c['level'] for c in cands)})，取L{n['level']}"


def apply_rules(name, idx):
    """返回 (规范名, 说明) 或 None"""
    # R1 连续重复片段（租赁合同合同纠纷 -> 租赁合同纠纷）
    for size in range(2, 8):
        for i in range(len(name) - 2 * size + 1):
            frag = name[i:i + size]
            if name[i + size:i + 2 * size] == frag:
                cand = name[:i + size] + name[i + 2 * size:]
                if cand in idx:
                    return cand, "R1重复片段"
    # R2 去尾部“纠纷”后命中（劳动争议纠纷 -> 劳动争议）
    if name.endswith("纠纷") and name[:-2] in idx:
        return name[:-2], "R2去尾纠纷"
    # R3 钉死的错字/别名
    if name in ALIAS and ALIAS[name] in idx:
        return ALIAS[name], "R3别名表"
    # R4 顿号成分重排
    comps = split_components(name)
    if len(comps) >= 2:
        key = sorted(comps)
        for v in idx:
            vc = split_components(v)
            if len(vc) >= 2 and sorted(vc) == key:
                return v, "R4顿号重排"
    # R5 并列写法拆分（A及B）
    if name in CONJ:
        return ";".join(CONJ[name]), "R5并列拆分"
    # R6 名称缺“纠纷”
    if not name.endswith("纠纷") and (name + "纠纷") in idx:
        return name + "纠纷", "R6补纠纷"
    # R7 “异议纠纷” -> “异议之诉”
    if name.endswith("异议纠纷") and (name[:-4] + "异议之诉") in idx:
        return name[:-4] + "异议之诉", "R7异议之诉"
    return None


def normalize(name, idx, manual=None):
    """返回 dict(规范名称, 层级, 完整路径, 方法, 需人工确认, 备注)"""
    name = (name or "").strip()
    if not name:
        return dict(规范名称="", 层级="", 完整路径="", 方法="无法映射", 需人工确认="是", 备注="空名称")
    if manual is None:
        manual = load_manual()
    if name in manual:
        m = manual[name]
        n, _ = pick(m["规范名"], idx)
        if n is not None and "L%d" % n["level"] == m["层级"].strip():
            return dict(规范名称=n["name"], 层级="L%d" % n["level"], 完整路径="/".join(n["full_path"]),
                        方法="人工映射", 需人工确认="否", 备注="%s｜%s" % (m["类型"], m.get("法律依据") or m.get("依据", "")))
    if name in idx:
        n, note = pick(name, idx)
        return dict(规范名称=n["name"], 层级="L%d" % n["level"],
                    完整路径="/".join(n["full_path"]), 方法="精确匹配",
                    需人工确认="是" if note else "否", 备注=note)
    r = apply_rules(name, idx)
    if r:
        std, why = r
        parts = std.split(";")
        paths, levels, notes = [], [], [why]
        for p in parts:
            n, note = pick(p, idx)
            if n is None:
                return dict(规范名称=std, 层级="", 完整路径="", 方法="规则修正", 需人工确认="是", 备注=why + ";拆分项未命中")
            paths.append("/".join(n["full_path"]))
            levels.append("L%d" % n["level"])
            if note:
                notes.append(note)
        return dict(规范名称=std, 层级=";".join(levels), 完整路径=";".join(paths),
                    方法="规则修正", 需人工确认="是", 备注=";".join(notes))
    return dict(规范名称="", 层级="", 完整路径="", 方法="无法映射", 需人工确认="是", 备注="")


def iter_causes(rec, field):
    cur = rec
    for k in field.split("."):
        if not isinstance(cur, dict):
            return []
        cur = cur.get(k)
    if cur is None:
        return []
    if isinstance(cur, str):
        return [cur]
    if isinstance(cur, list):
        return [c for c in cur if isinstance(c, str)]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cause-field", default="cause")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-json", default="")
    ap.add_argument("--format", default="auto", choices=["auto", "json", "jsonl"])
    a = ap.parse_args()

    nodes20 = load_nodes("2020")
    nodes25 = load_nodes("2025")
    idx20, idx25 = build_index(nodes20), build_index(nodes25)
    only25 = sorted(set(idx25) - set(idx20))

    fmt = a.format
    if fmt == "auto":
        head = open(a.input, encoding="utf-8").read(1)
        fmt = "json" if head == "[" else "jsonl"

    cnt = Counter()
    n_rec = 0
    if fmt == "json":
        data = json.load(open(a.input, encoding="utf-8"))
        for rec in data:
            n_rec += 1
            for c in iter_causes(rec, a.cause_field):
                leaf = c.strip().split("/")[-1]
                if leaf:
                    cnt[leaf] += 1
    else:
        for line in open(a.input, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_rec += 1
            for c in iter_causes(rec, a.cause_field):
                leaf = c.strip().split("/")[-1]
                if leaf:
                    cnt[leaf] += 1

    rows = []
    by_count, by_name = Counter(), Counter()
    for name, f in cnt.most_common():
        r = normalize(name, idx20)
        rows.append(dict(原名=name, 频次=f, 规范名称=r["规范名称"], 层级=r["层级"],
                         完整路径=r["完整路径"], 方法=r["方法"], 需人工确认=r["需人工确认"]))
        by_count[r["方法"]] += f
        by_name[r["方法"]] += 1

    with open(a.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["原名", "频次", "规范名称", "层级", "完整路径", "方法", "需人工确认"])
        w.writeheader()
        w.writerows(rows)

    total = sum(by_count.values())
    ok = by_count["精确匹配"] + by_count["规则修正"]
    stat = dict(dataset=a.dataset, samples=n_rec, distinct_causes=len(rows), total_cause_labels=total,
                by_count=dict(by_count), by_name=dict(by_name),
                coverage_by_count=round(100.0 * ok / total, 2) if total else 0.0,
                need_manual=sum(1 for r in rows if r["需人工确认"] == "是"),
                mapped_to_2020=sum(1 for r in rows if r["规范名称"]),
                only_in_2025=only25)
    print(json.dumps({k: v for k, v in stat.items() if k != "only_in_2025"}, ensure_ascii=False, indent=1))
    print("2025-only names:", len(only25))
    if a.stats_json:
        json.dump(stat, open(a.stats_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
