# -*- coding: utf-8 -*-
"""Shared engine utilities used by the released method (method_v3): vLLM engine wrapper,
candidate prior, element summary, and the level-wise discriminator stage.

Extracted from the development code base; the earlier method variant (method_v2) is not part
of this release, only these reusable pieces are kept.
"""
import argparse
import gc
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MASK = "[案由]"
LOGPROBS = 20  # vLLM 0.17.1 上限 20

SUMMARY_PROMPT = """你是民事案件案由预测助手。请阅读案情，用不超过 200 字写一段结构化摘要，依次覆盖：当事人关系；法律关系性质；争议焦点；诉讼请求要点。不得出现任何案由名称，不要解释。

案情：
{input}

摘要："""

OPEN_PROMPT = """你是民事案件案由预测助手。请阅读案情，写出本案的完整案由路径（从一级到最末级，用 > 分隔）。

{summary_block}案情：
{input}

最后一行严格写成：案由路径：一级 > 二级 > …
"""


def load_mask_terms(path):
    import csv
    terms = set()
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            w = (r.get("掩码词") or "").strip()
            if len(w) >= 2:
                terms.add(w)
    return sorted(terms, key=len, reverse=True)


def mask_summary(text, terms):
    hits = 0
    for t in terms:
        if t in text:
            hits += text.count(t)
            text = text.replace(t, MASK)
    return text, hits


def labels_for(n):
    """候选编号：<=9 用单个数字，>9 用大写字母（已验分词为单 token）。"""
    if n <= 9:
        return [str(i + 1) for i in range(n)], "num"
    return [chr(ord("A") + i) for i in range(n)], "alpha"


def children(tax, prefix):
    return C.children_of(tax, prefix)


def cand_lines(cands, labels):
    return "\n".join("%s. %s" % (labels[i], n["name"]) for i, n in enumerate(cands))


class Engine:
    """vLLM 生成封装：单条 1-token 先验、批量生成、token 统计。"""

    def __init__(self, model, gpu_mem, max_len=8192, seed=0, eager=False):
        from vllm import LLM
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=max_len,
                       dtype="bfloat16", enable_prefix_caching=True, disable_log_stats=True,
                       enforce_eager=eager)
        C.setup_chat(self.llm, enable_thinking=False)
        self.tok = self.llm.get_tokenizer()
        self.seed = seed
        self.n_calls = 0
        self.tok_in = 0
        self.tok_out = 0

    def _run(self, prompts, **sp_kw):
        from vllm import SamplingParams
        sp = SamplingParams(seed=self.seed, **sp_kw)
        outs = self.llm.generate(C._apply_template(self.llm, prompts), sp, use_tqdm=False)
        res = []
        for o in outs:
            self.n_calls += 1
            self.tok_in += len(o.prompt_token_ids)
            self.tok_out += len(o.outputs[0].token_ids)
            res.append(o)
        return res

    def gen(self, prompt, max_tokens=8, logprobs=None, temperature=0.0):
        kw = dict(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
        if logprobs:
            kw["logprobs"] = logprobs
        o = self._run([prompt], **kw)[0]
        lp = None
        if logprobs and o.outputs[0].logprobs:
            lp = o.outputs[0].logprobs[0]
        return o.outputs[0].text, lp

    def gen_batch(self, prompts, max_tokens=8, temperature=0.0):
        outs = self._run(prompts, max_tokens=max_tokens, temperature=temperature, top_p=1.0)
        return [o.outputs[0].text for o in outs]


def lp_map(entry):
    """把 vLLM 的 logprobs 结构统一成 {token_id: logprob}。"""
    d = {}
    if entry is None:
        return d
    items = entry.items() if isinstance(entry, dict) else enumerate(entry)
    for k, v in items:
        try:
            if isinstance(k, str) and not k.isdigit():
                continue
            tid = int(k)
        except Exception:
            continue
        lp = getattr(v, "logprob", v)
        try:
            d[tid] = float(lp)
        except Exception:
            continue
    return d


def softmax_over(d, ids):
    vals = [d.get(i, None) for i in ids]
    known = [v for v in vals if v is not None]
    floor = (min(known) - 8.0) if known else -20.0
    vals = [v if v is not None else floor for v in vals]
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [x / s for x in e]


class Prior:
    """候选先验：每个前缀一次 1-token 调用，缓存；解析失败用 BERT 兜底。"""

    def __init__(self, eng, tax, ctx, summary_on, calib, no_calib):
        self.eng = eng
        self.tax = tax
        self.ctx = ctx          # dict: fact, summary
        self.summary_on = summary_on
        self.calib = calib
        self.no_calib = no_calib
        self.cache = {}
        self.n_calls = 0
        self.n_fallback = 0

    def body(self, prefix, cands, labels, mode):
        sblock = ""
        if self.summary_on and self.ctx.get("summary"):
            sblock = "案情摘要：\n%s\n\n" % self.ctx["summary"]
        ask = "编号" if mode == "num" else "字母"
        lines = ["你是民事案件案由预测助手。请逐级选择案由。", ""]
        if sblock:
            lines.append(sblock.rstrip())
        lines.append("案情：")
        lines.append(self.ctx["fact"])
        lines.append("")
        lines.append("已确定案由路径：" + (" > ".join(prefix) if prefix else "（尚未确定）"))
        lines.append("")
        lines.append("第 %d 级候选：" % (len(prefix) + 1))
        lines.append(cand_lines(cands, labels))
        lines.append("")
        lines.append("只回答候选的%s，不要解释，不要输出其他内容。" % ask)
        lines.append("答案：")
        return "\n".join(lines)

    def get(self, prefix):
        if prefix in self.cache:
            return self.cache[prefix]
        cands = children(self.tax, prefix)
        if not cands:
            self.cache[prefix] = ([], [], "num")
            return self.cache[prefix]
        labels, mode = labels_for(len(cands))
        ids = []
        for lb in labels:
            t = self.eng.tok.encode(lb, add_special_tokens=False)
            ids.append(t[0] if len(t) == 1 else None)
        probs = None
        if all(i is not None for i in ids):
            text, lp = self.eng.gen(self.body(prefix, cands, labels, mode), max_tokens=1,
                                    logprobs=LOGPROBS, temperature=0.0)
            self.n_calls += 1
            d = lp_map(lp)
            if d and any(i in d for i in ids):
                probs = softmax_over(d, ids)
        if probs is None:
            # 兜底：生成短文本 -> 编号/名称 -> BERT 校准
            self.n_fallback += 1
            text, _ = self.eng.gen(self.body(prefix, cands, labels, mode), max_tokens=6, temperature=0.0)
            self.n_calls += 1
            pick = None
            t = (text or "").strip()
            for i, lb in enumerate(labels):
                if t.startswith(lb):
                    pick = i
                    break
            if pick is None:
                m = re.search(r"\d+", t)
                if m and 1 <= int(m.group()) <= len(cands):
                    pick = int(m.group()) - 1
            probs = [0.02 / max(1, len(cands))] * len(cands)
            if pick is not None:
                probs[pick] = 0.98
            elif self.calib is not None and not self.no_calib and t:
                i, sim = self.calib.best(t[:40], [{"name": c["name"]} for c in cands])
                if sim >= self.calib.threshold:
                    probs = [0.02 / max(1, len(cands))] * len(cands)
                    probs[i] = 0.98
        self.cache[prefix] = (cands, probs, mode)
        return self.cache[prefix]


class Node:
    __slots__ = ("prefix", "parent", "P", "N", "Q", "kids", "expanded")

    def __init__(self, prefix, parent=None, P=1.0):
        self.prefix = prefix
        self.parent = parent
        self.P = P
        self.N = 0
        self.Q = 0.0
        self.kids = []
        self.expanded = False


def c_of(depth, fixed_c):
    return 2.0 if fixed_c else max(0.2, 2.0 - 0.3 * depth)


def expand(prior, node, nodes):
    cands, probs, _ = prior.get(node.prefix)
    for i, cn in enumerate(cands):
        pref = tuple(node.prefix) + (cn["name"],)
        if pref not in nodes:
            nodes[pref] = Node(pref, node, probs[i])
        node.kids.append(nodes[pref])
    node.expanded = True
    return bool(node.kids)


def ancestors(n):
    out = []
    while n is not None:
        out.append(n)
        n = n.parent
    return out


def mcts(prior, tax, n_sim, max_depth=5, fixed_c=False):
    root = Node(())
    nodes = {(): root}
    n_expand = 0
    for _ in range(n_sim):
        node, depth = root, 0
        # 选择（PUCT）
        while node.expanded and node.kids and depth < max_depth:
            c = c_of(depth, fixed_c)
            best, best_u = None, -1e18
            for ch in node.kids:
                q = ch.Q / ch.N if ch.N else 0.0
                u = q + c * ch.P * math.sqrt(max(1, node.N)) / (1 + ch.N)
                if u > best_u:
                    best, best_u = ch, u
            node, depth = best, depth + 1
        # 扩展
        if not node.expanded and len(node.prefix) < max_depth:
            if expand(prior, node, nodes):
                n_expand += 1
                node = max(node.kids, key=lambda x: x.P)
                depth += 1
        # 模拟：按先验贪心下探到叶
        cur = node
        while len(cur.prefix) < max_depth:
            if not cur.expanded:
                if not expand(prior, cur, nodes):
                    break
            if not cur.kids:
                break
            cur = max(cur.kids, key=lambda x: x.P)
        # 奖励：根到叶路径上各级先验的几何平均
        chain = [n for n in ancestors(cur)][::-1]
        ps = [n.P for n in chain if n.P > 0]
        r = math.exp(sum(math.log(p) for p in ps) / len(ps)) if ps else 0.0
        for n in chain:
            n.N += 1
            n.Q += r
    leaves = [n for n in nodes.values() if not n.kids and n.N > 0]
    leaves.sort(key=lambda n: (-n.N, len(n.prefix)))
    return leaves, nodes, n_expand


def summarize(eng, fact, mask_terms, on=True):
    if not on:
        return "", 0
    text, _ = eng.gen(SUMMARY_PROMPT.format(input=fact), max_tokens=260, temperature=0.0)
    text = (text or "").strip().split("\n\n")[0].strip()
    if len(text) > 220:
        text = text[:220]
    masked, hits = mask_summary(text, mask_terms)
    return masked, hits


def gen_one(eng, prior, tax, row, args, mask_terms):
    t0 = time.time()
    c0, i0, o0 = eng.n_calls, eng.tok_in, eng.tok_out
    pc0, pf0 = prior.n_calls, prior.n_fallback
    fact = row["input_masked"][:args.input_chars]
    summary, hits = summarize(eng, fact, mask_terms, on=not args.no_summary)
    prior.ctx = {"fact": fact, "summary": summary}
    prior.cache = {}
    rec = {"id": str(row["id"]), "gold_path": row["gold_path"], "gold_leaf": row["gold_leaf"],
           "summary": summary, "summary_mask_hits": hits, "n_sim": args.n_sim}
    if args.no_hier:
        body = OPEN_PROMPT.format(summary_block=("案情摘要：\n%s\n\n" % summary) if summary else "", input=fact)
        text, _ = eng.gen(body, max_tokens=96, temperature=0.0)
        ans = C.clean_ans(C.extract_answer(text)[0] if hasattr(C, "extract_answer") else text)
        rec.update({"paths": [], "open_answer": ans, "nodes_seen": 0, "n_expand": 0})
    elif args.no_mcts:
        prefix = []
        while True:
            cands, probs, _ = prior.get(tuple(prefix))
            if not cands:
                break
            i = int(np.argmax(probs))
            prefix.append(cands[i]["name"])
        rec.update({"paths": [{"path": prefix, "visits": args.n_sim, "share": 1.0,
                               "prior_gm": 1.0}], "nodes_seen": 1, "n_expand": len(prefix)})
    else:
        leaves, nodes, n_expand = mcts(prior, tax, args.n_sim, args.max_depth, args.fixed_c)
        paths = []
        for n in leaves[: args.max_paths]:
            ps = []
            cur = n
            while cur.parent is not None:
                ps.append(cur.P)
                cur = cur.parent
            gm = math.exp(sum(math.log(p) for p in ps if p > 0) / max(1, len([p for p in ps if p > 0])))
            paths.append({"path": list(n.prefix), "visits": n.N, "share": n.N / max(1, args.n_sim),
                          "prior_gm": gm})
        rec.update({"paths": paths, "nodes_seen": len(nodes), "n_expand": n_expand})
    rec["calls"] = eng.n_calls - c0
    rec["tokens_in"] = eng.tok_in - i0
    rec["tokens_out"] = eng.tok_out - o0
    rec["prior_calls"] = prior.n_calls - pc0
    rec["prior_fallback"] = prior.n_fallback - pf0
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def disc_prompt(ctx, path, k, cands, labels, mode):
    lines = ["你是民事案件案由预测助手。请逐级核对案由。", ""]
    if ctx.get("summary"):
        lines += ["案情摘要：", ctx["summary"], ""]
    lines += ["案情：", ctx["fact"], ""]
    if k > 0:
        lines.append("已确定：" + " > ".join(path[:k]))
    lines.append("第 %d 级候选：" % (k + 1))
    lines.append(cand_lines(cands, labels))
    lines.append("")
    lines.append("只回答候选的%s，不要解释。" % ("编号" if mode == "num" else "字母"))
    lines.append("答案：")
    return "\n".join(lines)


def disc_stage(eng, tax, rows_ctx, args):
    out = []
    for rec in rows_ctx:
        ctx = {"fact": rec["fact"], "summary": rec.get("summary", "")}
        paths = rec.get("paths", [])
        if args.no_hier:
            ans = rec.get("open_answer", "")
            node, _, _ = tax.resolve(ans)
            g = 1.0 if node is not None else 0.0
            h = g
            conf = 1.0
            if node is None and rec.get("calib_conf") is not None:
                conf = rec["calib_conf"]
            J = 0.5 * (g + h) / 2 + 0.5 * conf
            out.append({**rec, "pred_path": [node["name"]] if node else [ans], "a": None, "J": J,
                        "g": g, "h": h, "raw_output": "最终案由：" + (ans or "")})
            continue
        for p in paths:
            path = p["path"]
            hit, tot = 0, 0
            levels = [len(path) - 1] if args.final_only_disc else list(range(len(path)))
            for k in levels:
                cands = children(tax, tuple(path[:k]))
                if len(cands) <= 1:
                    hit += 1
                    tot += 1
                    continue
                labels, mode = labels_for(len(cands))
                body = disc_prompt(ctx, path, k, cands, labels, mode)
                text, _ = eng.gen(body, max_tokens=2, temperature=0.0)
                pick = None
                t = (text or "").strip()
                for i, lb in enumerate(labels):
                    if t.startswith(lb):
                        pick = i
                        break
                if pick is None:
                    m = re.search(r"\d+", t)
                    if m and 1 <= int(m.group()) <= len(cands):
                        pick = int(m.group()) - 1
                tot += 1
                if pick is not None and cands[pick]["name"] == path[k]:
                    hit += 1
            a = hit / max(1, tot) if args.final_only_disc else hit / max(1, len(path))
            p["a"] = a
            if args.no_disc:
                p["J"] = p["share"]
            else:
                p["J"] = 0.5 * p["share"] + 0.5 * a
        if not paths:
            out.append({**rec, "pred_path": [], "a": None, "J": 0.0, "raw_output": ""})
            continue
        if args.no_eval:
            best = max(paths, key=lambda p: p["visits"])
        else:
            best = max(paths, key=lambda p: (p["J"], p["visits"]))
        out.append({**rec, "pred_path": best["path"], "a": best.get("a"), "J": best["J"],
                    "raw_output": "最终案由：" + " > ".join(best["path"])})
    return out
