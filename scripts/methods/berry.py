# -*- coding: utf-8 -*-
"""LLaMA-Berry 基线（berry）：Self-Refine MCTS + 成对比较奖励（无 PPRM 的简化版）。

论文如实写明：本题没有训练好的成对偏好奖励模型（PPRM），因此用同一模型（Qwen3.5-9B）
以成对比较提示词替代，结果按 Borda 计数汇总；预算与 method_main 的单样本生成调用数对齐。

用法：
  python scripts/methods/berry.py --split dev --budget 8 --out runs/berry/dev/b8
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

INIT_Q = ("你是民事案件案由预测助手。\n\n【案情】\n%s\n\n"
          "请分析本案的法律关系与争议焦点，最后一行只写：最终案由：<名称>（不要编号，不要解释）")

REFINE_Q = ("你是民事案件案由预测助手。\n\n【案情】\n%s\n\n【当前作答】\n%s\n\n"
            "请先指出当前作答可能的问题（批评），再给出改写后的完整作答，"
            "最后一行只写：最终案由：<名称>（不要编号，不要解释，不要列表）")

JUDGE_Q = ("下面是同一案件的两份案由预测作答，请判断哪一份更可能正确（依据法律关系认定与说理充分性）。\n\n"
           "【案情（节选）】\n%s\n\n【作答 A】\n%s\n\n【作答 B】\n%s\n\n"
           "只输出一行：胜者：A 或 胜者：B 或 胜者：平局")


TAX = None


def pick_answer(text):
    """从模型输出里抽出案由名（纯字符串匹配，不用嵌入）。

    顺序：① 标记行右侧的文本恰好等于词表节点名；② 任一行恰好等于词表节点名；
    ③ 全文中最靠后的、作为子串出现的词表节点名（长者优先）。
    """
    if TAX is None:
        return C.clean_ans(C.extract_answer(text)[0])

    def norm(s):
        s = s.strip()
        for pre in ("最终案由：", "最终案由:", "案由：", "案由:"):
            if s.startswith(pre):
                s = s[len(pre):].strip()
        return s.strip("。，,.;；:：*# \t")

    # ① 标记行右侧
    for line in reversed(text.splitlines()):
        s = line.strip()
        for pre in ("最终案由：", "最终案由:", "案由：", "案由:"):
            if s.startswith(pre):
                cand = norm(s)
                if cand in TAX.by_name:
                    return cand
                for name in TAX.by_name:
                    if cand.startswith(name):
                        return name
    # ② 整行等于节点名
    for line in reversed(text.splitlines()):
        s = norm(line)
        if s in TAX.by_name:
            return s
    # ③ 全文最靠后的子串命中
    best = None
    for name in TAX.by_name:
        pos = text.rfind(name)
        if pos < 0:
            continue
        key = (pos, len(name))
        if best is None or key > best[0]:
            best = (key, name)
    if best:
        return best[1]
    return C.clean_ans(C.extract_answer(text)[0])


class Node:
    __slots__ = ("answer", "text", "depth", "parent", "children", "N", "Q")

    def __init__(self, answer, text, depth=0, parent=None):
        self.answer = answer
        self.text = text
        self.depth = depth
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0.0

    def uct(self, c):
        if self.N == 0:
            return 1e9
        return self.Q / self.N + c * np.sqrt(np.log(max(1, self.parent.N)) / self.N)


def run(llm, row, args):
    case = C.case_text(row)
    t0 = time.time()
    tin = tout = calls = 0
    outs = C.chat(llm, [INIT_Q % case], max_tokens=512, temperature=0.7, top_p=0.95, seed=args.seed)
    text, ni, no = outs[0]
    calls += 1
    tin += ni
    tout += no
    ans = pick_answer(text)
    root = Node(ans, text)
    nodes = [root]
    rng = random.Random(args.seed)

    for it in range(args.budget):
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.uct(args.c))
        outs = C.chat(llm, [REFINE_Q % (case, node.text[-1200:])], max_tokens=512,
                      temperature=0.7, top_p=0.95, seed=args.seed + it + 1)
        text, ni, no = outs[0]
        calls += 1
        tin += ni
        tout += no
        ans = pick_answer(text)
        child = Node(ans, text, depth=node.depth + 1, parent=node)
        node.children.append(child)
        nodes.append(child)
        # 成对比较：与随机 2 个已有节点比，Borda 计分
        others = [n for n in nodes if n is not child]
        rivals = rng.sample(others, min(2, len(others))) if others else []
        prompts = [JUDGE_Q % (case, child.text[-1200:], r.text[-1200:]) for r in rivals]
        if prompts:
            outs = C.chat(llm, prompts, max_tokens=12, temperature=0.0, seed=args.seed + it)
            calls += len(prompts)
            score = 0.0
            for (jt, _ni, _no), _r in zip(outs, rivals):
                tin += _ni
                tout += _no
                if "胜者：A" in jt:
                    score += 1.0
                elif "平局" in jt:
                    score += 0.5
            child.Q = score / len(rivals)
        child.N = 1
        node.N += 1
        node.Q += child.Q

    best = max(nodes, key=lambda n: (n.Q / n.N, n.depth))
    return dict(id=row["id"], raw_output="最终案由：" + (best.answer or ""),
                parsed_path=[best.answer] if best.answer else [],
                gold_path=row["gold_path"], gold_leaf=row["gold_leaf"],
                tokens_in=tin, tokens_out=tout, latency=round(time.time() - t0, 2),
                n_calls=calls, n_levels=1, calib_hits=0,
                levels=[dict(level=1, n_candidates=0, n_sims=args.budget, n_calls=calls,
                             tokens_in=tin, tokens_out=tout, sec=round(time.time() - t0, 2),
                             n_terminal=len(nodes), conf=round(best.Q / best.N, 4), alt_conf=0.0,
                             n_calib=0, n_reask=0)],
                paths=[dict(path=[best.answer] if best.answer else [], delta=None, mask_level=None,
                            J=round(best.Q / best.N, 4), confs=[round(best.Q / best.N, 4)])],
                kept=1, fallback=0, n_unique_answers=len({n.answer for n in nodes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--c", type=float, default=1.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--gpu-gen", type=float, default=0.62)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()
    from vllm import LLM
    global TAX
    TAX = C.Taxonomy("2020", "vocab")
    os.makedirs(args.out, exist_ok=True)
    rows = {str(r["id"]): r for r in C.load_jsonl(os.path.join(args.data_dir, args.split + ".jsonl"))}
    if args.limit:
        rows = {k: rows[k] for k in list(rows)[:args.limit]}
    preds = os.path.join(args.out, "preds.jsonl")
    done = set()
    if args.resume and os.path.exists(preds):
        done = {str(json.loads(l)["id"]) for l in open(preds) if l.strip()}
    todo = [r for k, r in rows.items() if k not in done]
    if args.shard_count > 1:
        pos = {k: i for i, k in enumerate(rows)}
        todo = [r for r in todo if pos[str(r["id"])] % args.shard_count == args.shard_index]
    t0 = time.time()
    llm = LLM(model=args.gen_model, gpu_memory_utilization=args.gpu_gen, max_model_len=8192,
              dtype="bfloat16", enable_prefix_caching=True, disable_log_stats=True)
    C.setup_chat(llm)
    load_s = time.time() - t0
    t1 = time.time()
    with open(preds, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo):
            rec = run(llm, row, args)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                el = time.time() - t1
                print("%d/%d %.1fs/sample" % (i + 1, len(todo), el / (i + 1)), flush=True)
    C.write_run_log(os.path.join(args.out, "run_log.json"), label="berry", method="berry",
                    split=args.split, budget=args.budget, seed=args.seed, n_rows=len(todo),
                    commit=C.git_commit(), machine=C.machine(),
                    gpus=os.environ.get("CUDA_VISIBLE_DEVICES", ""), gen_model=args.gen_model,
                    load_s=round(load_s, 1), elapsed_s=round(time.time() - t1, 1))
    print("DONE", len(todo))


if __name__ == "__main__":
    main()
