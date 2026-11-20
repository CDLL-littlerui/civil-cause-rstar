# -*- coding: utf-8 -*-
"""原始 rStar（rstar_orig）基线：整题一次 MCTS + 开放式作答 + 互一致性验证。

忠实复现 rStar（Qi et al., 2024）在本题上的形态：
- 搜索空间不做层级约束、不给候选，动作仍为 A1–A5；
- 终端奖励 = 最终答案的多数投票置信度；
- 互一致性验证：随机取一条轨迹的中间步骤，由鉴别器补全，比较最终答案；
- 结果选择 = 终端奖励 × 互一致性通过率。

用法（两阶段同 method_main）：
  python scripts/methods/rstar_orig.py --stage gen --split dev --k 8 --out runs/rstar_orig/dev/k8
  python scripts/methods/rstar_orig.py --stage disc --split dev --out runs/rstar_orig/dev/k8
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

ACTIONS = {
    "A1": "先做一步思考：分析本案的法律关系性质与争议焦点，不超过 3 句。",
    "A2": "写出剩余推理，并在最后一行给出最终案由，格式严格为：最终案由：<名称>",
    "A3": "提出一个有助于确定案由的子问题，并简要回答它。",
    "A4": "重新回答上一步提出的子问题，给出修正后的结论。",
    "A5": "先重述案件的关键事实与争议焦点，不超过 4 句。",
}
ORDER = ["A2", "A1", "A3", "A4", "A5"]


class Node:
    __slots__ = ("text", "depth", "children", "answer", "N", "Q", "parent")

    def __init__(self, text="", depth=0, parent=None):
        self.text = text
        self.depth = depth
        self.children = {}
        self.answer = None
        self.N = 0
        self.Q = 0.0
        self.parent = parent

    def uct(self, c):
        if self.N == 0:
            return 1e9
        return self.Q / self.N + c * np.sqrt(np.log(max(1, self.parent.N)) / self.N)


def clean_ans(ans):
    """把开放式输出收敛成一个案由名：取最后一行、去前缀、过长则视为无答案。"""
    if not ans:
        return ""
    lines = [l.strip() for l in str(ans).splitlines() if l.strip()]
    if not lines:
        return ""
    s = lines[-1]
    for pre in ("最终案由：", "最终案由:", "案由：", "案由:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
            break
    s = s.strip(" \t。.，,；;\"“”")
    return s if 0 < len(s) <= 30 else ""


def build_prompt(case, thoughts, action_instr):
    parts = [C.SYSTEM, "", "【待预测案件】", case]
    if thoughts:
        parts += ["", "【已有推演】"] + thoughts
    parts += ["", "【当前任务】" + action_instr]
    return "\n".join(parts)


def select_expand(root, max_depth, c):
    node = root
    while True:
        if node.answer is not None or node.depth >= max_depth:
            return None
        untried = [a for a in ORDER if a not in node.children and (a != "A5" or node.depth == 0)]
        if untried:
            return node, untried[0]
        kids = list(node.children.values())
        if not kids:
            return None
        node = max(kids, key=lambda x: x.uct(c))


def gen_sample(llm, row, args, examples_block):
    case = C.case_text(row)
    root = Node()
    sims = [root] * args.k
    paths = [[] for _ in range(args.k)]
    finished = [None] * args.k
    n_calls = tin = tout = 0
    t0 = time.time()
    for _ in range(args.max_depth + 1):
        batch, metas = [], []
        for i, node in enumerate(sims):
            if finished[i] is not None:
                continue
            pick = select_expand(node, args.max_depth, args.c)
            if pick is None:
                finished[i] = node
                continue
            parent, action = pick
            thoughts = ["- " + n.text.strip() for n in paths[i] if n.text.strip()]
            batch.append(examples_block + "\n\n" + build_prompt(case, thoughts, ACTIONS[action]))
            metas.append((i, parent, action))
        if not batch:
            break
        outs = C.chat(llm, batch, max_tokens=512, temperature=0.7, top_p=0.95, seed=args.seed + len(batch))
        n_calls += 1
        for (i, parent, action), (text, ni, no) in zip(metas, outs):
            tin += ni
            tout += no
            child = Node(text=text, depth=parent.depth + 1, parent=parent)
            ans = clean_ans(C.extract_answer(text)[0])
            if action == "A2" and ans and ans != "无法判断":
                child.answer = ans
            parent.children[action] = child
            paths[i].append(child)
            sims[i] = child
            if child.answer is not None:
                finished[i] = child
    counts = {}
    for leaf in finished:
        if leaf is not None and leaf.answer:
            counts[leaf.answer] = counts.get(leaf.answer, 0) + 1
    n_term = sum(counts.values())
    conf = {k: v / n_term for k, v in counts.items()} if n_term else {}
    for leaf in finished:
        if leaf is None or not leaf.answer:
            continue
        r = conf[leaf.answer]
        node = leaf.parent
        while node is not None:
            node.N += 1
            node.Q += r
            node = node.parent
    best = max(conf.items(), key=lambda x: x[1])[0] if conf else ""
    cands = sorted(conf.items(), key=lambda x: -x[1])[:5]
    return dict(id=row["id"], best=best, n_calls=n_calls, tokens_in=tin, tokens_out=tout,
                latency=round(time.time() - t0, 2), n_levels=1, calib_hits=0,
                levels=[dict(level=1, n_candidates=0, n_sims=args.k, n_calls=n_calls,
                             tokens_in=tin, tokens_out=tout, sec=round(time.time() - t0, 2),
                             n_terminal=n_term, conf=conf.get(best, 0.0), alt_conf=0.0,
                             n_calib=0, n_reask=0)],
                paths=[dict(path=[p], confs=[conf.get(best, 0.0)]) for p in [best] if p],
                candidates=cands,
                chains=[[{"text": n.text, "depth": n.depth} for n in path] for path in paths if path])


def disc_prompt(case, step_text, upto):
    return ("你是民事案件案由预测助手。下面是一段未完成的说理推演，请续写并给出最终案由。\n\n"
            "【案情】\n" + case + "\n\n【已有推演】\n" + "\n".join(
                "- " + t for t in step_text[:upto]) + "\n\n请在最后一行严格写成：最终案由：<名称>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="gen", choices=["gen", "disc"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--disc-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--gpu-gen", type=float, default=0.62)
    ap.add_argument("--gpu-disc", type=float, default=0.55)
    ap.add_argument("--resume", type=int, default=1)
    # 仅并行执行用的行过滤（不改变单样本行为）；disc 的遮蔽 RNG 因此按分片独立
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = {str(r["id"]): r for r in C.load_jsonl(os.path.join(args.data_dir, args.split + ".jsonl"))}
    if args.limit:
        rows = {k: rows[k] for k in list(rows)[:args.limit]}
    if args.shard_count > 1:
        keys = list(rows.keys())
        rows = {k: rows[k] for i, k in enumerate(keys) if i % args.shard_count == args.shard_index}

    if args.stage == "gen":
        from vllm import LLM
        preds = os.path.join(args.out, "paths.jsonl")
        done = set()
        if args.resume and os.path.exists(preds):
            done = {str(json.loads(l)["id"]) for l in open(preds) if l.strip()}
        todo = [r for k, r in rows.items() if k not in done]
        t0 = time.time()
        llm = LLM(model=args.gen_model, gpu_memory_utilization=args.gpu_gen, max_model_len=8192,
                  dtype="bfloat16", enable_prefix_caching=True, disable_log_stats=True)
        C.setup_chat(llm)
        load_s = time.time() - t0
        t1 = time.time()
        with open(preds, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo):
                rec = gen_sample(llm, row, args, "")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    el = time.time() - t1
                    print("%d/%d %.1fs/sample" % (i + 1, len(todo), el / (i + 1)), flush=True)
        C.write_run_log(os.path.join(args.out, "run_log_gen.json"), label="rstar_orig", method="rstar_orig",
                        stage="gen", k=args.k, split=args.split, seed=args.seed, n_rows=len(todo),
                        commit=C.git_commit(), machine=C.machine(),
                        gpus=os.environ.get("CUDA_VISIBLE_DEVICES", ""), gen_model=args.gen_model,
                        load_s=round(load_s, 1), elapsed_s=round(time.time() - t1, 1))
        print("GEN_DONE", len(todo))
        return

    from vllm import LLM
    data = C.load_jsonl(os.path.join(args.out, "paths.jsonl"))
    llm4b = LLM(model=args.disc_model, gpu_memory_utilization=args.gpu_disc, max_model_len=8192,
                dtype="bfloat16", enable_prefix_caching=True, disable_log_stats=True)
    C.setup_chat(llm4b)
    rng = random.Random(args.seed)
    prompts, metas = [], []
    for rec in data:
        if str(rec["id"]) not in rows:   # 分片：只处理本片的样本
            continue
        if not rec["chains"]:
            continue
        case = C.case_text(rows[str(rec["id"])])
        for ci, chain in enumerate(rec["chains"][:3]):
            texts = [n["text"] for n in chain[:-1]]
            if not texts:
                continue
            k = rng.randint(1, len(texts))
            prompts.append(disc_prompt(case, texts, k))
            metas.append((rec["id"], ci))
    outs = C.chat(llm4b, prompts, max_tokens=512, temperature=0.0)
    per = {}
    for (rid, ci), (text, ni, no) in zip(metas, outs):
        ans = clean_ans(C.extract_answer(text)[0])
        per.setdefault(rid, []).append(ans)

    preds = os.path.join(args.out, "preds.jsonl")
    with open(preds, "w", encoding="utf-8") as f:
        for rec in data:
            row = rows[str(rec["id"])]
            answers = per.get(rec["id"], [])
            best = rec["best"]
            consist = (sum(1 for a in answers if a == best) / len(answers)) if answers else 1.0
            score = rec["levels"][0]["conf"] * consist
            f.write(json.dumps(dict(id=rec["id"], raw_output="最终案由：" + best,
                                    parsed_path=[best] if best else [], gold_path=row["gold_path"],
                                    gold_leaf=row["gold_leaf"], tokens_in=rec["tokens_in"],
                                    tokens_out=rec["tokens_out"], latency=rec["latency"],
                                    n_calls=rec["n_calls"], n_levels=1, calib_hits=0,
                                    levels=rec["levels"], consistency=round(consist, 4),
                                    score=round(score, 4),
                                    paths=[dict(path=[best] if best else [], delta=None, mask_level=None,
                                                J=round(score, 4), confs=[rec["levels"][0]["conf"]])],
                                    kept=1, fallback=0), ensure_ascii=False) + "\n")
    C.write_run_log(os.path.join(args.out, "run_log_disc.json"), label="rstar_orig", method="rstar_orig",
                    stage="disc", split=args.split, n_rows=len(data), commit=C.git_commit(),
                    machine=C.machine(), gpus=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    disc_model=args.disc_model)
    print("DISC_DONE", len(data))


if __name__ == "__main__":
    main()
