# -*- coding: utf-8 -*-
"""步骤 7：method_v3 = 开放生成投票（rStar 候选轨迹）+ 层级先验的案由树搜索。

每样本流程：
 1. 要素摘要（不做掩码：输入已掩码，摘要是模型自身推理）
 2. 复用步骤 4 冻结的自一致性 prompt（native 模板 + 10 条少样本，每条截 300 字），
    前置要素摘要；K=16 采样（T=0.7, top_p=0.95）。每条答案按统一规范化规则映射为
    词表路径；OOV 时用 BERT-wwm-ext 余弦（阈值 0.83）校准，仍失败则弃票。
 3. 投票分布 V(c|p) = (经过 c 的票数 + eps) / (经过 p 的票数 + eps*|S(p)|)，eps=0.5
 4. 层级先验 P_h 加温度 tau：P_h ~ exp(logprob/tau)；混合 P' = lam*P_h + (1-lam)*V
 5. PUCT 树搜索（c_d = 2.0-0.3d），奖励 = 路径上各级 P' 的几何平均，N_sim 次
 6. 取叶访问次数前 M=4 条路径；鉴别器（Qwen3.5-4B）逐级软验证一致率 a；
    J = 0.5*visit_share + 0.5*a（复用 engine 的 disc 阶段，口径完全一致）

一次 gen 运行同时产出 6 组网格（tau x lam）的 N_sim=32 与 N_sim=16 路径，
以及 --no_mcts（按 P' 贪心，M=1）与 sc_vote_path（按 V 逐级 argmax）对照。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.methods.common as C            # noqa: E402
from eval import Taxonomy, extract_answer, parse_path  # noqa: E402
from scripts.methods import engine as V2   # noqa: E402

EPS = 0.5
GRID = [(1.0, 0.3), (1.0, 0.5), (1.0, 0.7), (2.0, 0.3), (2.0, 0.5), (2.0, 0.7)]


def tag_of(tau, lam):
    return "t%s_l%s" % (("%g" % tau), ("%g" % lam))


# ---------------------------------------------------------------- 投票

class VoteMapper:
    """开放式答案 -> 词表完整路径；OOV 时 BERT 余弦（阈值 0.83）。"""

    def __init__(self, tax, calib=None):
        self.tax = tax
        self.calib = calib
        self.names = [n["name"] for n in tax.nodes]
        self.mat = calib.embed(self.names) if calib is not None else None
        self.th = getattr(calib, "threshold", 0.83) if calib is not None else 0.83
        self.stats = {"exact": 0, "path": 0, "bert": 0, "fail": 0}

    def _chain(self, name):
        n, _, _ = self.tax.resolve(name)
        if n is None:
            return None
        return [self.tax.by_path[tuple(n["full_path"][:i + 1])]["name"] for i in range(n["level"])]

    def _bert(self, text):
        e = self.calib.embed([text[:40]])
        sims = (e @ self.mat.T)[0].tolist()
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def to_path(self, text):
        ans, _ = extract_answer(text or "")
        names = parse_path(ans)
        if not names:
            self.stats["fail"] += 1
            return None
        if len(names) > 1:
            nodes, ok, hier = self.tax.resolve_path(names)
            if hier and all(nodes):
                self.stats["path"] += 1
                return [n["name"] for n in nodes]
        ch = self._chain(names[-1])
        if ch:
            self.stats["exact"] += 1
            return ch
        if self.mat is not None and ans:
            i, sim = self._bert(ans)
            if sim >= self.th:
                ch = self._chain(self.names[i])
                if ch:
                    self.stats["bert"] += 1
                    return ch
        self.stats["fail"] += 1
        return None


def vote_counts(vpaths):
    """{prefix tuple: 经过该前缀的票数}"""
    cnt = {}
    for p in vpaths:
        for k in range(1, len(p) + 1):
            kk = tuple(p[:k])
            cnt[kk] = cnt.get(kk, 0) + 1
    return cnt


def vdist(cnt, prefix, cands):
    """η(v|p) = (n(p⊕v)+ε) / (Σ_{u∈S(p)} n(p⊕u) + ε·|S(p)|) —— 按子节点票数之和归一化。

    分母只取直接子节点的票数：停在 p 的票不计入 (v3.1 修正，见步骤 10A)。
    空候选表返回 []；全部票都校准失败时退化为均匀分布。
    """
    if not cands:
        return []
    prefix = tuple(prefix)
    h = [cnt.get(prefix + (c["name"],), 0) for c in cands]
    den = sum(h) + EPS * len(cands)
    return [(x + EPS) / den for x in h]


def ph_tempered(probs, tau):
    if abs(tau - 1.0) < 1e-9:
        return list(probs)
    q = [max(p, 1e-12) ** (1.0 / tau) for p in probs]
    s = sum(q)
    return [x / s for x in q]


def blended(prior, cnt, prefix, tau, lam):
    cands, probs, _ = prior.get(tuple(prefix))
    if not cands:
        return [], []
    ph = ph_tempered(probs, tau)
    pv = vdist(cnt, tuple(prefix), cands)
    return cands, [lam * ph[i] + (1.0 - lam) * pv[i] for i in range(len(cands))]


# ---------------------------------------------------------------- 搜索

def expand_v3(node, cands, P, nodes):
    for i, cn in enumerate(cands):
        pref = tuple(node.prefix) + (cn["name"],)
        if pref not in nodes:
            nodes[pref] = V2.Node(pref, node, P[i])
        node.kids.append(nodes[pref])
    node.expanded = True
    return bool(node.kids)


def mcts_v3(prior, cnt, tau, lam, n_sim, max_depth=5, fixed_c=False):
    root = V2.Node(())
    nodes = {(): root}
    n_expand = 0
    for _ in range(n_sim):
        node, depth = root, 0
        while node.expanded and node.kids and depth < max_depth:
            c = V2.c_of(depth, fixed_c)
            best, best_u = None, -1e18
            for ch in node.kids:
                q = ch.Q / ch.N if ch.N else 0.0
                u = q + c * ch.P * math.sqrt(max(1, node.N)) / (1 + ch.N)
                if u > best_u:
                    best, best_u = ch, u
            node, depth = best, depth + 1
        if not node.expanded and len(node.prefix) < max_depth:
            cands, P = blended(prior, cnt, node.prefix, tau, lam)
            if cands and expand_v3(node, cands, P, nodes):
                n_expand += 1
                node = max(node.kids, key=lambda x: x.P)
                depth += 1
        cur = node
        while len(cur.prefix) < max_depth:
            if not cur.expanded:
                cands, P = blended(prior, cnt, cur.prefix, tau, lam)
                if not cands or not expand_v3(cur, cands, P, nodes):
                    break
            if not cur.kids:
                break
            cur = max(cur.kids, key=lambda x: x.P)
        chain = V2.ancestors(cur)[::-1]
        ps = [n.P for n in chain if n.P > 0]
        r = math.exp(sum(math.log(p) for p in ps) / len(ps)) if ps else 0.0
        for n in chain:
            n.N += 1
            n.Q += r
    leaves = [n for n in nodes.values() if not n.kids and n.N > 0]
    leaves.sort(key=lambda n: (-n.N, len(n.prefix)))
    return leaves, nodes, n_expand


def paths_from_leaves(leaves, n_sim, max_paths):
    out = []
    for n in leaves[:max_paths]:
        ps = []
        cur = n
        while cur.parent is not None:
            ps.append(cur.P)
            cur = cur.parent
        ps = [p for p in ps if p > 0]
        gm = math.exp(sum(math.log(p) for p in ps) / len(ps)) if ps else 0.0
        out.append({"path": list(n.prefix), "visits": n.N, "share": n.N / max(1, n_sim),
                    "prior_gm": gm})
    return out


def greedy_blend(prior, cnt, tau, lam, max_depth=5):
    prefix = []
    while len(prefix) < max_depth:
        cands, P = blended(prior, cnt, tuple(prefix), tau, lam)
        if not cands:
            break
        prefix.append(cands[int(np.argmax(P))]["name"])
    return prefix


def greedy_vote(prior, cnt, max_depth=5):
    """sc_vote_path：只按投票分布 V 逐级 argmax；某层无票时退回层级先验。"""
    prefix = []
    while len(prefix) < max_depth:
        cands = C.children_of(prior.tax, tuple(prefix))
        if not cands:
            break
        if sum(cnt.get(tuple(prefix) + (c["name"],), 0) for c in cands) > 0:
            v = vdist(cnt, tuple(prefix), cands)
            i = int(np.argmax(v))
        else:
            _, probs, _ = prior.get(tuple(prefix))
            i = int(np.argmax(probs))
        prefix.append(cands[i]["name"])
    return prefix


# ---------------------------------------------------------------- 生成

def summarize_nomask(eng, fact, on=True):
    if not on:
        return ""
    text, _ = eng.gen(V2.SUMMARY_PROMPT.format(input=fact), max_tokens=260, temperature=0.0)
    text = (text or "").strip().split("\n\n")[0].strip()
    return text[:220]


def build_examples_native(rows, limit=C.FEWSHOT_CHARS):
    blocks = []
    for i, r in enumerate(rows, 1):
        text = r["input_masked"][:limit].replace("\r", "")
        blocks.append("<示例 %d>\n案情：%s\n最终案由：%s\n</示例 %d>" % (i, text, " > ".join(r["gold_path"]), i))
    return "\n\n".join(blocks)


def gen_sc(eng, prompt, n, max_tokens, temperature, top_p, seed):
    from vllm import SamplingParams
    sp = SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    outs = eng.llm.generate(C._apply_template(eng.llm, [prompt]), sp, use_tqdm=False)
    o = outs[0]
    eng.n_calls += len(o.outputs)
    eng.tok_in += len(o.prompt_token_ids) * len(o.outputs)
    eng.tok_out += sum(len(x.token_ids) for x in o.outputs)
    return [x.text for x in o.outputs]


def gen_one_v3(eng, prior, mapper, row, args, few, native_tpl):
    t0 = time.time()
    c0, i0, o0 = eng.n_calls, eng.tok_in, eng.tok_out
    pc0, pf0 = prior.n_calls, prior.n_fallback
    fact = row["input_masked"][: args.input_chars]
    summary = summarize_nomask(eng, fact, on=not args.no_summary)
    prior.ctx = {"fact": fact, "summary": summary}
    prior.cache = {}
    head = ("案情摘要：\n%s\n\n" % summary) if summary else ""
    prompt = head + native_tpl.format(n=len(few), examples=build_examples_native(few), input=fact)
    texts = gen_sc(eng, prompt, args.k_votes, args.vote_max_tokens, args.vote_temp,
                   args.vote_top_p, args.seed)
    vpaths = []
    for t in texts:
        p = mapper.to_path(t)
        if p:
            vpaths.append(p)
    cnt = vote_counts(vpaths)

    cands0, probs0, _ = prior.get(())
    ph0 = ph_tempered(probs0, 1.0)
    pv0 = vdist(cnt, (), cands0) if cands0 else []
    rec = {"id": str(row["id"]), "gold_path": row["gold_path"], "gold_leaf": row["gold_leaf"],
           "summary": summary, "prompt_chars": len(prompt), "fact": fact,
           "n_votes": len(vpaths), "votes": vpaths,
           "l1_prior_top": cands0[int(np.argmax(ph0))]["name"] if cands0 else "",
           "l1_vote_top": cands0[int(np.argmax(pv0))]["name"] if pv0 else "",
           "l1_prior_p": round(float(max(ph0)), 4) if ph0 else 0.0,
           "l1_vote_share": round(float(max(pv0)), 4) if pv0 else 0.0,
           "combos": {}}
    for tau, lam in GRID:
        tag = tag_of(tau, lam)
        leaves, nodes, n_expand = mcts_v3(prior, cnt, tau, lam, args.n_sim, args.max_depth, args.fixed_c)
        leaves_lo, _, _ = mcts_v3(prior, cnt, tau, lam, args.n_sim_lo, args.max_depth, args.fixed_c)
        rec["combos"][tag] = {
            "paths": paths_from_leaves(leaves, args.n_sim, args.max_paths),
            "paths_lo": paths_from_leaves(leaves_lo, args.n_sim_lo, args.max_paths),
            "nomcts": greedy_blend(prior, cnt, tau, lam, args.max_depth),
            "nodes_seen": len(nodes), "n_expand": n_expand}
    rec["vote_path"] = greedy_vote(prior, cnt, args.max_depth)
    rec["calls"] = eng.n_calls - c0
    rec["tokens_in"] = eng.tok_in - i0
    rec["tokens_out"] = eng.tok_out - o0
    rec["prior_calls"] = prior.n_calls - pc0
    rec["prior_fallback"] = prior.n_fallback - pf0
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------- 主流程

def openw(cache, path):
    if path not in cache:
        cache[path] = open(path, "a", encoding="utf-8")
    return cache[path]


def run_gen(a, rows, tax, native_tpl, few):
    eng = V2.Engine(a.gen_model, a.gpu_gen, max_len=a.max_model_len, eager=a.enforce_eager)
    calib = None
    try:
        calib = C.Calibrator(a.bert, threshold=0.83, device=a.bert_device)
    except Exception as e:  # 校准器不可用时仍可运行（OOV 票直接丢弃）
        print("[v3] calibrator 不可用:", e, flush=True)
    mapper = VoteMapper(tax, None if a.no_calib else calib)
    prior = V2.Prior(eng, tax, {}, True, None, a.no_calib)
    master = os.path.join(a.out, "gen.jsonl")
    done = set()
    if os.path.exists(master):
        for line in open(master, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    writers = {}
    t0 = time.time()
    n = 0
    for i, r in enumerate(rows):
        if str(r["id"]) in done:
            continue
        rec = gen_one_v3(eng, prior, mapper, r, a, few, native_tpl)
        base = {"id": rec["id"], "gold_path": rec["gold_path"], "gold_leaf": rec["gold_leaf"],
                "fact": rec["fact"], "summary": rec["summary"], "n_votes": rec["n_votes"]}
        for tag, cb in rec["combos"].items():
            openw(writers, os.path.join(a.out, "paths_%s.jsonl" % tag)).write(json.dumps(
                {**base, "n_sim": a.n_sim, "paths": cb["paths"], "nodes_seen": cb["nodes_seen"],
                 "n_expand": cb["n_expand"]}, ensure_ascii=False) + "\n")
            openw(writers, os.path.join(a.out, "paths16_%s.jsonl" % tag)).write(json.dumps(
                {**base, "n_sim": a.n_sim_lo, "paths": cb["paths_lo"],
                 "nodes_seen": cb["nodes_seen"], "n_expand": cb["n_expand"]}, ensure_ascii=False) + "\n")
            openw(writers, os.path.join(a.out, "paths_nomcts_%s.jsonl" % tag)).write(json.dumps(
                {**base, "n_sim": a.n_sim, "paths": [{"path": cb["nomcts"], "visits": a.n_sim,
                                                     "share": 1.0, "prior_gm": 1.0}]},
                ensure_ascii=False) + "\n")
        openw(writers, os.path.join(a.out, "paths_vote.jsonl")).write(json.dumps(
            {**base, "n_sim": 1, "paths": [{"path": rec["vote_path"], "visits": 1, "share": 1.0,
                                           "prior_gm": 1.0}]}, ensure_ascii=False) + "\n")
        openw(writers, master).write(json.dumps(rec, ensure_ascii=False) + "\n")
        for f in writers.values():
            f.flush()
        n += 1
        if n % 10 == 0:
            print("[gen] %d/%d  %.1fs  calls=%d  votes_ok=%d/16" % (
                n, len(rows), time.time() - t0, eng.n_calls, rec["n_votes"]), flush=True)
    for f in writers.values():
        f.close()
    print("[gen] done n=%d wall=%.1fs calls=%d tok_in=%d tok_out=%d mapper=%s" % (
        n, time.time() - t0, eng.n_calls, eng.tok_in, eng.tok_out, mapper.stats), flush=True)
    C.write_run_log(os.path.join(a.out, "run_log_gen.json"), n=n, n_sim=a.n_sim, n_sim_lo=a.n_sim_lo,
                    k_votes=a.k_votes, vote_temp=a.vote_temp, vote_top_p=a.vote_top_p,
                    calls=eng.n_calls, tokens_in=eng.tok_in, tokens_out=eng.tok_out,
                    seconds=round(time.time() - t0, 1), machine=C.machine(), commit=C.git_commit(),
                    mapper=mapper.stats, grid=[tag_of(t, l) for t, l in GRID])


def run_disc(a, tax):
    eng = V2.Engine(a.disc_model, a.gpu_disc, max_len=a.max_model_len, eager=a.enforce_eager)
    split_rows = C.load_jsonl(os.path.join(a.data_dir, a.split + ".jsonl"))
    pos = {str(r["id"]): i for i, r in enumerate(split_rows)}
    t0 = time.time()
    for name in a.inputs.split(","):
        name = name.strip()
        if not name:
            continue
        p = name if os.path.isabs(name) else os.path.join(a.out, name)
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if a.shard_count > 1:
            before = len(recs)
            recs = [r for r in recs if pos.get(str(r["id"]), -1) % a.shard_count == a.shard_index]
            print("[disc] shard %d/%d %s: %d -> %d" % (a.shard_index, a.shard_count,
                                                       os.path.basename(p), before, len(recs)), flush=True)
        if a.limit:
            recs = recs[: a.limit]
        outs = V2.disc_stage(eng, tax, recs, a)
        stem = os.path.basename(p)
        stem = stem[:-6] if stem.endswith(".jsonl") else stem
        stem = stem.replace("paths_", "").replace("paths16_", "s16_").replace("paths_nomcts_", "nomcts_")
        op = os.path.join(a.out, "preds_%s.jsonl" % stem)
        with open(op, "w", encoding="utf-8") as f:
            for o in outs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print("[disc] %s -> %s n=%d calls=%d wall=%.1fs" % (
            os.path.basename(p), os.path.basename(op), len(outs), eng.n_calls,
            time.time() - t0), flush=True)
    C.write_run_log(os.path.join(a.out, "run_log_disc.json"), calls=eng.n_calls,
                    tokens_in=eng.tok_in, tokens_out=eng.tok_out,
                    seconds=round(time.time() - t0, 1), machine=C.machine(),
                    commit=C.git_commit(), inputs=a.inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="gen", choices=["gen", "disc"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="", help="disc 阶段：逗号分隔的 paths 文件名")
    ap.add_argument("--n-sim", type=int, default=32)
    ap.add_argument("--n-sim-lo", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--max-paths", type=int, default=4)
    ap.add_argument("--k-votes", type=int, default=16)
    ap.add_argument("--vote-temp", type=float, default=0.7)
    ap.add_argument("--vote-top-p", type=float, default=0.95)
    ap.add_argument("--vote-max-tokens", type=int, default=96)
    ap.add_argument("--input-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--disc-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--bert", default="dienstag/chinese-bert-wwm-ext")
    ap.add_argument("--bert-device", default="cpu")
    ap.add_argument("--gpu-gen", type=float, default=0.58)
    ap.add_argument("--gpu-disc", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--no_summary", action="store_true")
    ap.add_argument("--no_calib", action="store_true")
    ap.add_argument("--no_disc", action="store_true")
    ap.add_argument("--final_only_disc", action="store_true")
    ap.add_argument("--fixed_c", action="store_true")
    ap.add_argument("--tau", type=float, default=None, help="restrict the hyper-parameter grid to this tau (with --lam)")
    ap.add_argument("--lam", type=float, default=None, help="restrict the hyper-parameter grid to this lambda (with --tau)")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_hier", action="store_true")  # 仅为复用 disc_stage 的签名
    a = ap.parse_args()
    if a.tau is not None and a.lam is not None:
        global GRID
        GRID = [(a.tau, a.lam)]
        print("[v3] grid restricted to tau=%g lam=%g" % (a.tau, a.lam), flush=True)

    rows = C.load_jsonl(os.path.join(a.data_dir, a.split + ".jsonl"))
    if a.shard_count > 1:
        rows = [r for i, r in enumerate(rows) if i % a.shard_count == a.shard_index]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    print("[v3] split=%s n=%d stage=%s n_sim=%d/%d K=%d T=%.2f gpu=%.2f" % (
        a.split, len(rows), a.stage, a.n_sim, a.n_sim_lo, a.k_votes, a.vote_temp, a.gpu_gen), flush=True)
    tax = C.Taxonomy("2020")
    if a.stage == "gen":
        native_tpl = open(os.path.join(ROOT, "prompts", "native_template.txt"), encoding="utf-8").read()
        few = C.load_jsonl(os.path.join(a.data_dir, "fewshot10_folds.jsonl"))
        run_gen(a, rows, tax, native_tpl, few)
    else:
        run_disc(a, tax)


if __name__ == "__main__":
    main()
