#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤4 D：统一推理入口。

方法：
  native  少样本 10 条，直接给最终案由
  cot     少样本 10 条（带 2-3 句推理），先推理再给结论
  sc      同 cot，采样 K=16，temperature=0.7/top_p=0.95，多数投票
  hier    直接层次化提示，逐级给候选编号/全名，无少样本

输出 runs/<label>/<method>[_seed<seed>]/preds.jsonl + run_log.json
"""
import argparse, hashlib, json, os, random, re, socket, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

COT_REASONING = {
    "2376": "本案因机动车商业保险理赔引发，原告主张保险公司在车辆损失险限额内赔付修理费，争议为财产保险合同的履行。",
    "369": "原告在劳务作业中受伤，请求赔偿医疗费、误工费、残疾赔偿金等，争议为身体权、健康权受侵害的损害赔偿。",
    "223": "原告诉请赔偿被告强行带走家具、拆除装修造成的损失，属一般侵权损害赔偿，未归入具体特殊侵权类型。",
    "2321": "原告主张拖欠工资及未签订书面劳动合同的双倍工资差额，争议发生于劳动关系存续期间。",
    "0": "双方签订存量房买卖合同，原告请求交付房屋并支付逾期交房违约金，争议源于房屋买卖合同的履行。",
    "1160": "双方已离婚，原告请求分割婚姻关系存续期间的共同财产，争议为离婚后的财产分割。",
    "100": "被告借住房屋后拒不腾退，原告请求返还房屋并支付占有使用费，属对物权行使的排除妨害。",
    "3786": "原告作为案外人主张对被执行房屋享有权利并请求停止执行，系对执行标的提出的异议之诉。",
    "8478": "双方签订检测技术服务合同，原告请求支付检测费及利息，争议为技术服务合同的履行。",
    "8069": "原告申请追加被执行人以实现生效调解书确定的债权，涉及生效法律文书的强制执行。",
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_path(p):
    return " > ".join(p)


def build_examples(rows, mode, limit=300):
    blocks = []
    for i, r in enumerate(rows, 1):
        text = r["input_masked"][:limit].replace("\r", "")
        ans = fmt_path(r["gold_path"])
        if mode == "cot":
            reason = COT_REASONING.get(str(r["id"]), "本案争议的法律关系应归入%s。" % r["gold_path"][-1])
            blocks.append("<示例 %d>\n案情：%s\n推理：%s\n最终案由：%s\n</示例 %d>" % (i, text, reason, ans, i))
        else:
            blocks.append("<示例 %d>\n案情：%s\n最终案由：%s\n</示例 %d>" % (i, text, ans, i))
    return "\n\n".join(blocks)


def load_taxonomy(tax_dir):
    nodes = [json.loads(l) for l in open(os.path.join(tax_dir, "cause_taxonomy_2020_nodes.jsonl"), encoding="utf-8")]
    by_path = {}
    for n in nodes:
        by_path[tuple(n["full_path"])] = n
    l1 = [n["name"] for n in nodes if n["level"] == 1]
    return nodes, by_path, l1


def children_of(by_path, prefix):
    prefix = tuple(prefix)
    out = []
    for p, n in by_path.items():
        if len(p) == len(prefix) + 1 and p[:len(prefix)] == prefix:
            out.append(n)
    out.sort(key=lambda n: n["full_path"])
    return out


def parse_choice(text, cands):
    """选择题解析：先按开头整数（须有边界，避免 "12" 被当成 "1"），再按名称（长名优先）。"""
    t = (text or "").strip().replace(" ", "")
    m = re.match(r"^(\d+)(?=$|[.、．,，)）:：])", t)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(cands):
            return cands[i]
    for c in sorted(cands, key=len, reverse=True):
        if c.replace(" ", "") in t:
            return c
    m2 = re.fullmatch(r"(\d+)", t)
    if m2:
        i = int(m2.group(1)) - 1
        if 0 <= i < len(cands):
            return cands[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "valid_train", "fewshot_pool"])
    ap.add_argument("--method", default="native", choices=["native", "cot", "sc", "hier"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tax-dir", default="vocab")
    ap.add_argument("--prompt-dir", default="prompts")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--n-sc", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--input-chars", type=int, default=4000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enable-thinking", default="0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(os.path.join(args.data_dir, args.split + ".jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    few = read_jsonl(os.path.join(args.data_dir, "fewshot10_folds.jsonl"))

    run_name = (args.method if args.method != "sc" else "sc") + ("" if args.method == "native" else "_seed%d" % args.seed)
    if args.tag:
        run_name += "_" + args.tag
    out_dir = os.path.join(args.out_root, args.label, args.split, run_name)
    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, "preds.jsonl")

    done = {}
    if args.resume and os.path.exists(preds_path):
        for r in read_jsonl(preds_path):
            done[str(r["id"])] = r
    rows = [r for r in rows if str(r["id"]) not in done]

    native_tpl = open(os.path.join(args.prompt_dir, "native_template.txt"), encoding="utf-8").read()
    cot_tpl = open(os.path.join(args.prompt_dir, "cot_template.txt"), encoding="utf-8").read()
    hier_tpl = open(os.path.join(args.prompt_dir, "hier_template.txt"), encoding="utf-8").read()
    tpl_sha = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in
               [("native", native_tpl), ("cot", cot_tpl), ("hier", hier_tpl)]}

    SYSTEM = "你是民事案件案由预测助手。"
    ex_native = build_examples(few, "native")
    ex_cot = build_examples(few, "cot")

    prompts = []
    for r in rows:
        text = r["input_masked"][:args.input_chars]
        if args.method in ("native", "cot", "sc"):
            tpl = native_tpl if args.method == "native" else cot_tpl
            ex = ex_native if args.method == "native" else ex_cot
            prompts.append(tpl.format(n=len(few), examples=ex, input=text))
        else:
            prompts.append(None)

    nodes, by_path, l1 = load_taxonomy(args.tax_dir)
    l1_sorted = sorted(l1)

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, dtype="bfloat16", enable_prefix_caching=True,
              trust_remote_code=True, disable_log_stats=True)
    load_s = time.time() - t0

    tok = llm.get_tokenizer()

    def chat(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=bool(int(args.enable_thinking)))
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    results = dict(done)
    n_out_per_req = args.n_sc if args.method == "sc" else 1
    if args.method == "sc":
        sp = SamplingParams(n=args.n_sc, temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    else:
        sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    if args.method == "hier":
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=24, seed=args.seed)

    def gen(batch_prompts):
        t = time.time()
        outs = llm.generate(batch_prompts, sp, use_tqdm=False)
        return outs, time.time() - t

    written = 0
    with open(preds_path, "a", encoding="utf-8") as f:
        def emit(rec):
            nonlocal written
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            results[str(rec["id"])] = rec
            written += 1

        if args.method != "hier":
            for i in range(0, len(rows), args.chunk):
                chunk = rows[i:i + args.chunk]
                msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompts[i + j]}] for j in range(len(chunk))]
                batch = [chat(m) for m in msgs]
                outs, dt = gen(batch)
                per = dt / max(1, len(chunk))
                for r, o in zip(chunk, outs):
                    toks_in = len(o.prompt_token_ids)
                    if args.method == "sc":
                        cands = []
                        for c in o.outputs:
                            cands.append({"text": c.text, "tokens_out": len(c.token_ids)})
                        picks = []
                        for c in cands:
                            leaf = ""
                            m = re.findall(r"(?:最终案由|案由路径)[：:]\s*(.+)", c["text"])
                            if m:
                                leaf = re.split(r"\s*(?:>|＞|→|->)\s*", m[-1].strip())[-1].strip()
                            picks.append(leaf)
                        counts = {}
                        best = ""
                        for p in picks:
                            counts[p] = counts.get(p, 0) + 1
                        bestn = max(counts.values()) if counts else 0
                        for p in picks:
                            if counts[p] == bestn:
                                best = p
                                break
                        raw = next((c["text"] for c in cands if picks[cands.index(c)] == best), cands[0]["text"])
                        emit({"id": str(r["id"]), "raw_output": raw, "samples": cands,
                              "parsed_answer": best, "gold_path": r["gold_path"], "gold_leaf": r["gold_leaf"],
                              "tokens_in": toks_in, "tokens_out": sum(c["tokens_out"] for c in cands),
                              "latency": per, "method": args.method, "seed": args.seed})
                    else:
                        emit({"id": str(r["id"]), "raw_output": o.outputs[0].text,
                              "gold_path": r["gold_path"], "gold_leaf": r["gold_leaf"],
                              "tokens_in": toks_in, "tokens_out": len(o.outputs[0].token_ids),
                              "latency": per, "method": args.method, "seed": args.seed})
                print("[%s] %d/%d  %.1fs" % (args.method, min(i + args.chunk, len(rows)), len(rows), dt), flush=True)
        else:
            # 逐级批处理
            state = {str(r["id"]): {"row": r, "prefix": [], "path": [], "done": False, "tokens_in": 0, "tokens_out": 0, "lat": 0.0} for r in rows}
            for level in range(1, 5):
                active = [s for s in state.values() if not s["done"] and len(s["prefix"]) == level - 1]
                if not active:
                    break
                cand_map = {}
                batches = []
                for s in active:
                    cands = l1_sorted if level == 1 else [n["name"] for n in children_of(by_path, s["prefix"])]
                    if not cands:
                        s["done"] = True
                        continue
                    cand_map[str(s["row"]["id"])] = cands
                    body = hier_tpl.format(input=s["row"]["input_masked"][:args.input_chars],
                                           level=level,
                                           candidates="\n".join("%d. %s" % (k + 1, c) for k, c in enumerate(cands)))
                    batches.append((s, chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}])))
                if not batches:
                    break
                for i in range(0, len(batches), args.chunk):
                    sub = batches[i:i + args.chunk]
                    outs, dt = gen([b for _, b in sub])
                    for (s, _), o in zip(sub, outs):
                        s["tokens_in"] += len(o.prompt_token_ids)
                        s["tokens_out"] += len(o.outputs[0].token_ids)
                        s["lat"] += dt / max(1, len(sub))
                        pick = parse_choice(o.outputs[0].text, cand_map[str(s["row"]["id"])])
                        if pick is None:
                            s["done"] = True
                            s["failed"] = True
                        else:
                            s["prefix"] = s["prefix"] + [pick] if level > 1 else [pick]
                            # 校验前缀与词表路径一致
                            n = by_path.get(tuple(s["prefix"]))
                            if n is None:
                                s["done"] = True
                                s["failed"] = True
                            else:
                                s["path"] = list(n["full_path"])
                                if not children_of(by_path, s["path"]):
                                    s["done"] = True
                print("[hier L%d] done batch %d" % (level, len(active)), flush=True)
            for s in state.values():
                r = s["row"]
                raw = "案由路径：" + " > ".join(s["path"]) if s["path"] else ""
                emit({"id": str(r["id"]), "raw_output": raw, "gold_path": r["gold_path"], "gold_leaf": r["gold_leaf"],
                      "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
                      "latency": s["lat"], "method": "hier", "seed": args.seed, "failed": s.get("failed", False)})

    log = {
        "label": args.label, "model_path": args.model_path, "method": args.method, "seed": args.seed,
        "split": args.split, "n_rows": len(rows), "n_written": written, "tag": args.tag,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "machine": socket.gethostname(), "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_mem": args.gpu_mem, "max_model_len": args.max_model_len, "max_tokens": args.max_tokens,
        "temperature": 0.7 if args.method == "sc" else args.temperature,
        "top_p": 0.95 if args.method == "sc" else args.top_p,
        "n_sc": args.n_sc if args.method == "sc" else 1,
        "prompt_sha256": tpl_sha, "prompt_dir": args.prompt_dir,
        "load_s": round(load_s, 1), "elapsed_s": round(time.time() - t0 - load_s, 1),
        "input_chars": args.input_chars, "enable_thinking": args.enable_thinking,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(log, open(os.path.join(out_dir, "run_log.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
