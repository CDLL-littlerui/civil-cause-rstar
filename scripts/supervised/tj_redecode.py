# -*- coding: utf-8 -*-
"""TopJudge 解码端长度先验重扫（不重训）。

动机：trie 上的联合概率是逐级概率的乘积，深路径多乘几项 <1 的概率，
即使各级都"基本正确"，146 类 argmax 也会系统性偏向浅路径 —— 官方评测把
"预测级数 < gold 级数" 记 __INVALID__（算错），因此深度偏差是可直接量化的失分项。
做法：score_c = log p(c|x) - beta * depth(c)，beta 只在 valid_train 上扫，
选 valid Macro-F1 最优的 beta，再一次前向应用到 test（test 全程不参与选择）。
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import bertdata as BD
from topjudge import TopJudge, Trie, MODEL_DIR


def logits_of(model, data, trie, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for bi in BD.batches(len(data), bs):
            t = BD.collate([data[i] for i in bi])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(t["input_ids"].to(trie.dev), t["attention_mask"].to(trie.dev))
            out.append(trie.class_logprob([x.float() for x in lg]).float().cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sup/topjudge/final_lr3e-5_bs32")
    ap.add_argument("--betas", default="-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.run, "summary.json")))["config"]

    tr, va, te = C.load_split("train"), C.load_split("valid_train"), C.load_split("test")
    lab = C.LabelSpace([x["gold_path"] for x in tr])
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    data = BD.BertData(tok, cfg["maxlen"], cfg["head"], cfg["tail"])
    Dva, Dte = data.encode_all(va), data.encode_all(te)

    model = TopJudge(AutoModel.from_pretrained(MODEL_DIR), lab, cfg["drop"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev))
    trie = Trie(lab, dev)

    lp_va = logits_of(model, Dva, trie, a.bs)
    lp_te = logits_of(model, Dte, trie, a.bs)
    depth = torch.tensor([len(p) for p in lab.classes], dtype=torch.float32)

    rows, best = [], None
    for beta in [float(x) for x in a.betas.split(",")]:
        score = lp_va - beta * depth.unsqueeze(0)
        paths = [lab.path(int(c)) for c in score.argmax(1)]
        m = C.metrics_for(paths, va)
        rec = dict(beta=beta, valid_f1=m["macro_f1"], valid_acc=m["accuracy"],
                   valid_invalid=m["invalid_rate"])
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        key = (m["macro_f1"], m["accuracy"])
        if best is None or key > best[0]:
            best = (key, beta)
    beta = best[1]
    print("BEST beta", beta, flush=True)

    score = lp_te - beta * depth.unsqueeze(0)
    paths = [lab.path(int(c)) for c in score.argmax(1)]
    m = C.metrics_for(paths, te)
    print("TEST " + json.dumps(C.brief(m), ensure_ascii=False), flush=True)
    C.write_preds(os.path.join(a.run, "preds_beta.jsonl"), [x["id"] for x in te], paths)
    json.dump(dict(beta=beta, sweep=rows, test=C.brief(m)),
              open(os.path.join(a.run, "beta_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
