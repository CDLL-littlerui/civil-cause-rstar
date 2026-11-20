#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 8-D：案由规定 2020 → 2025 的影响分析。

1) 测试集 77 个案由的 2020 路径映射到 2025：不变/改名/升降级/拆分/删除
2) 把 v3(9B, seed1) 的预测路径按同一映射转换，在 2025 标签下重评
3) 2025 新增、2020 无法输出的案由数

用法：python scripts/step8_taxonomy_shift.py --preds runs/method_v3/test/s32/preds.jsonl \
        --gold data/test.jsonl --out-json runs/taxonomy_shift.json
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import Taxonomy, extract_answer, parse_path, judge_one, compute_metrics


def load_nodes(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


class Shift:
    def __init__(self, d="vocab"):
        self.n20 = load_nodes(os.path.join(d, "cause_taxonomy_2020_nodes.jsonl"))
        self.n25 = load_nodes(os.path.join(d, "cause_taxonomy_2025_nodes.jsonl"))
        self.key25 = {(n["level"], n["parent"], n["name"]): n for n in self.n25}
        self.byname25 = collections.defaultdict(list)
        for n in self.n25:
            self.byname25[n["name"]].append(n)
        self.kids25 = collections.defaultdict(list)
        self.kids20 = collections.defaultdict(list)
        for n in self.n25:
            if n["parent"]:
                self.kids25[(n["parent"], n["level"] - 1)].append(n["name"])
        for n in self.n20:
            if n["parent"]:
                self.kids20[(n["parent"], n["level"] - 1)].append(n["name"])
        self.names20 = {n["name"] for n in self.n20}

    def map_path(self, names):
        """2020 路径（名称列表）→ (2025 节点列表, notes)；失败返回 (None, [原因])。"""
        out, notes, parent = [], [], None
        for lvl, name in enumerate(names, start=1):
            node = self.key25.get((lvl, parent, name))
            if node is None:
                cands = [c for c in self.n25 if c["level"] == lvl and c["parent"] == parent
                         and (c["name"].startswith(name) or name.startswith(c["name"]))]
                if len(cands) == 1:
                    node = cands[0]
                    notes.append("改名:%s→%s" % (name, node["name"]))
            if node is None:
                for c in self.byname25.get(name, []):
                    if [x["name"] for x in out] == c["full_path"][:-1] and c["level"] != lvl:
                        node = c
                        notes.append("升降级:%s L%d→L%d" % (name, lvl, c["level"]))
                        break
            if node is None:
                return None, ["missing:%s(L%d)" % (name, lvl)]
            out.append(node)
            parent = node["name"]
        return out, notes

    def has_children(self, name, level):
        return bool(self.kids25.get((name, level)))


def classify(sh, path20):
    nodes, notes = sh.map_path(list(path20))
    if nodes is None:
        return "删除", notes[0], None, []
    names25 = [n["name"] for n in nodes]
    leaf20 = path20[-1]
    was_leaf = not sh.kids20.get((leaf20, len(path20)))
    now_leaf = not sh.has_children(nodes[-1]["name"], nodes[-1]["level"])
    if any(n.startswith("升降级") for n in notes):
        cls = "升降级"
    elif any(n.startswith("改名") for n in notes):
        cls = "改名"
    elif was_leaf and not now_leaf:
        cls = "拆分"
    elif was_leaf and now_leaf:
        cls = "不变"
    else:
        cls = "不变(非叶)"
    return cls, ";".join(notes) or "路径一致", names25, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/method_v3/test/s32/preds.jsonl")
    ap.add_argument("--gold", default="data/test.jsonl")
    ap.add_argument("--out-json", default="runs/taxonomy_shift.json")
    a = ap.parse_args()
    sh = Shift()
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    cause_cnt = collections.Counter(tuple(g["gold_path"]) for g in gold)

    rows = []
    agg, samples = collections.Counter(), collections.Counter()
    for path20, cnt in cause_cnt.items():
        cls, why, names25, notes = classify(sh, path20)
        rows.append(dict(path2020=list(path20), n_samples=cnt, cls=cls, why=why,
                         notes=notes, path2025=names25))
        agg[cls] += 1
        samples[cls] += cnt
    rows.sort(key=lambda r: (-r["n_samples"], r["path2020"]))

    new25 = [n for n in sh.n25 if n["name"] not in sh.names20]
    new25_leaves = [n for n in new25 if not sh.has_children(n["name"], n["level"])]

    out = dict(cause_counts=dict(agg), sample_counts=dict(samples), n_causes=len(cause_cnt),
               n_samples=len(gold), new_names_2025=len(new25),
               new_leaf_names_2025=len(set(n["name"] for n in new25_leaves)), causes=rows)

    if os.path.exists(a.preds):
        tax20 = Taxonomy("2020", "vocab")
        goldmap = {g["id"]: g for g in gold}
        gold25 = {}
        for g in gold:
            nodes, _ = sh.map_path(list(g["gold_path"]))
            gold25[g["id"]] = [n["name"] for n in nodes] if nodes else list(g["gold_path"])
        r20, r25 = [], []
        unmapped = 0
        for line in open(a.preds, encoding="utf-8"):
            if not line.strip():
                continue
            p = json.loads(line)
            g = goldmap.get(p["id"])
            if g is None:
                continue
            ans, _ = extract_answer(p.get("raw_output", ""))
            names = parse_path(ans)
            nodes20, ok20, hier20 = tax20.build_pred_path(names) if names else ([], [], False)
            j20 = judge_one(nodes20, g["gold_path"], g["gold_leaf"])
            r20.append(dict(id=g["id"], gold_leaf=g["gold_leaf"], gold_path=g["gold_path"],
                            pred_nodes=nodes20, hierarchy_ok=hier20, parse_ok=bool(names),
                            resolved=bool(ok20 and ok20[0]), pred_label=j20["pred_label"],
                            correct=j20["correct"]))
            g25 = gold25[g["id"]]
            pn25, hier25 = [], False
            if names:
                mapped, notes = sh.map_path(names)
                if mapped:
                    pn25 = mapped
                    hier25 = "hierarchy_ok" in notes or True
                else:
                    unmapped += 1
            j25 = judge_one(pn25, g25, g25[-1])
            r25.append(dict(id=g["id"], gold_leaf=g25[-1], gold_path=g25, pred_nodes=pn25,
                            hierarchy_ok=hier25, parse_ok=bool(names),
                            resolved=bool(hier25), pred_label=j25["pred_label"],
                            correct=j25["correct"]))
        m20, m25 = compute_metrics(r20), compute_metrics(r25)
        out["preds"] = a.preds
        keys = ["macro_f1", "macro_precision", "macro_recall", "accuracy", "hvr",
                "invalid_rate", "n_classes", "level_acc"]
        out["eval_2020"] = {k: m20[k] for k in keys}
        out["eval_2025"] = {k: m25[k] for k in keys}
        out["pred_unmapped_to_2025"] = unmapped
        out["pred_label_changed"] = sum(1 for x, y in zip(r20, r25) if x["pred_label"] != y["pred_label"])
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in out if k != "causes"}, ensure_ascii=False, indent=1)[:3000])
    print("wrote", a.out_json)


if __name__ == "__main__":
    main()
