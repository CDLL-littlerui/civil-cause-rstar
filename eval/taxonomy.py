# -*- coding: utf-8 -*-
"""2020 版词表：查找、规则规范化（复用 map_labels 的精确/规则/同名取最深），禁止嵌入相似度。"""
import json, os, sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import map_labels as ml


class Taxonomy:
    def __init__(self, version="2020", tax_dir="vocab"):
        self.nodes = [json.loads(l) for l in open(os.path.join(tax_dir, f"cause_taxonomy_{version}_nodes.jsonl"), encoding="utf-8")]
        self.idx = ml.build_index(self.nodes)
        self.manual = ml.load_manual(os.path.join(tax_dir, "manual_mapping_2020.csv"))
        self.by_path = {}
        self.by_name = {}
        for n in self.nodes:
            self.by_path.setdefault(tuple(n["full_path"]), n)
            self.by_name.setdefault(n["name"], []).append(n)

    def _canon(self, name):
        r = ml.normalize(name, self.idx, self.manual)
        if not r["规范名称"] or ";" in r["规范名称"]:
            return None
        return r["完整路径"].split("/")[-1]

    def _ancestors_of(self, node):
        return [self.by_path[tuple(node["full_path"][:i + 1])] for i in range(node["level"])]

    def _chain(self, names):
        """把模型给出的若干级名称对齐成一条从 L1 到末级的连续路径。

        若首级不是 L1（模型省略了上级），按词表补全前缀；中间跳级则整条链判失败。
        """
        head = self._canon(names[0])
        if head is None:
            return None
        best = None
        for node in self.by_name.get(head, []):
            path = self._ancestors_of(node)
            i = 1
            while i < len(names):
                cn = self._canon(names[i])
                if cn is None:
                    break
                nxt = self.by_path.get(tuple(path[-1]["full_path"]) + (cn,))
                if nxt is None or nxt["level"] != path[-1]["level"] + 1:
                    break
                path.append(nxt)
                i += 1
            if i == len(names) and (best is None or len(path) > len(best)):
                best = path
        return best

    def resolve(self, name):
        """name -> (node|None, method, note)。method ∈ 精确匹配/规则修正/人工映射/";" """
        r = ml.normalize(name, self.idx, self.manual)
        if not r["规范名称"] or ";" in r["规范名称"]:
            return None, r["方法"], r["备注"]
        n = self.by_path.get(tuple(r["完整路径"].split("/")))
        return n, r["方法"], r["备注"]

    def build_pred_path(self, names):
        """构造预测路径。

        - 只给一个名称：按词表反查其完整祖先链（"只给叶名的，按词表反查路径"）；
        - 给多级：逐级校验，要求 level 序号与位置一致、相邻为直接父子。
        """
        if len(names) == 1:
            n, _, _ = self.resolve(names[0])
            if n is None:
                return [None], [False], False
            chain = [self.by_path[tuple(n["full_path"][:i + 1])] for i in range(n["level"])]
            return chain, [True] * len(chain), True
        chain = self._chain(names)
        if chain is not None:
            return chain, [True] * len(chain), True
        # 模型给出的多级路径与词表不一致（名称写错/跳级）：退化为“取末级名称反查路径”，HVR 记失败
        n, _, _ = self.resolve(names[-1])
        if n is None:
            nodes, ok, _ = self.resolve_path(names)
            return nodes, ok, False
        chain = [self.by_path[tuple(n["full_path"][:i + 1])] for i in range(n["level"])]
        return chain, [True] * len(chain), False

    def resolve_path(self, names):
        """逐级解析。返回 (nodes, ok_levels, hierarchy_ok)。

        ok_levels[i] 表示第 i 级解析成功且其 level == i+1。
        hierarchy_ok: 全部层级可解析、层级序号正确、且相邻两级是直接父子。
        """
        nodes, ok = [], []
        for i, nm in enumerate(names):
            n, _, _ = self.resolve(nm)
            nodes.append(n)
            ok.append(bool(n) and n["level"] == i + 1)
        hier = all(ok)
        if hier:
            for a, b in zip(nodes, nodes[1:]):
                if b.get("parent") != a["name"]:
                    hier = False
                    break
        return nodes, ok, hier
