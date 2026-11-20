# -*- coding: utf-8 -*-
"""IRACPS：理由抽取 + 预测 的弱监督复现。

原论文把判决预测拆成「抽取裁判理由(Reason) → 基于理由预测结论」两段。本数据没有理由/法条标注，
按原文的弱监督设置近似：
  - 弱标签（句子级）：句子含 [案由] 掩码（案由名曾被提及处）或含 gold 案由名的核心词，记为正；
    都没有时退化到法条/裁判常用词命中句；仍没有则整篇为正（见 --weak report）。无句级人工标注。
  - 抽取：BERT 每 token 打一个「理由」分数，句分数 = 句内 token 分数均值，句级 BCE 弱监督。
  - 预测：表示 = [CLS] ⊕ Σ_j α_j · s_j（α 为句分数的 softmax，s_j 为句向量），
    接 L1–L4 逐级头（含 STOP，复用 TopJudge 的 Trie 与 product 解码），使两者只差「理由抽取」这一项。
损失 = 层级 CE + λ · 句级 BCE。
"""
import argparse, bisect, json, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import bertdata as BD
from topjudge import Trie, loss_fn, decode

MODEL_DIR = os.environ.get("BERT_MODEL", "dienstag/chinese-bert-wwm-ext")
SPLIT_RE = re.compile(r"[。！？；\n]+")
CUE = ["合同", "借款", "赔偿", "离婚", "劳动", "工资", "事故", "责任", "履行", "支付", "欠", "伤",
       "房屋", "买卖", "租赁", "继承", "赡养", "侵权", "工程", "保险", "公司", "股权", "票据"]
SUFFIX = ["纠纷", "争议", "责任纠纷", "案", "案件"]


def sent_spans(text):
    spans, s = [], 0
    for m in SPLIT_RE.finditer(text):
        e = m.end()
        if e - s >= 2:
            spans.append((s, e))
        s = e
    if s < len(text):
        spans.append((s, len(text)))
    return spans or [(0, len(text))]


def cores_of(rec):
    out = set()
    names = list(rec["gold_path"]) + [rec.get("gold_leaf", ""), rec.get("cause_raw", "")]
    for n in names:
        if not n:
            continue
        for sfx in SUFFIX:
            if n.endswith(sfx) and len(n) - len(sfx) >= 2:
                out.add(n[:-len(sfx)])
        if len(n) >= 3:
            out.add(n)
    return {c for c in out if len(c) >= 2}


def weak_labels(rec, text, spans, cores):
    labs, kind = [], "core"
    for (a, b) in spans:
        seg = text[a:b]
        labs.append(1 if ("[案由]" in seg or any(c in seg for c in cores)) else 0)
    if sum(labs) == 0:
        kind = "cue"
        labs = [1 if any(c in text[a:b] for c in CUE) else 0 for (a, b) in spans]
    if sum(labs) == 0:
        kind = "all"
        labs = [1] * len(spans)
    if all(l == 1 for l in labs) and kind == "core":
        kind = "all"
    return labs, kind


def build_items(recs, data, max_sent=48):
    items, stats = [], {"core": 0, "cue": 0, "all": 0}
    for r in recs:
        text = (r["input_masked"] or "").replace("\x00", " ")
        spans = sent_spans(text)
        if len(spans) > max_sent:                      # 句子过多则按块合并，保证 S 有界
            k = int(np.ceil(len(spans) / max_sent))
            spans = [(spans[i][0], spans[min(i + k, len(spans)) - 1][1]) for i in range(0, len(spans), k)]
        labs, kind = weak_labels(r, text, spans, cores_of(r))
        stats[kind] += 1
        enc = data.encode(text)
        starts = [s for s, _ in spans]
        sid = []
        for (a, b) in enc["offsets"]:
            if a is None or a < 0 or b <= a:
                sid.append(-1)
            else:
                j = bisect.bisect_right(starts, a) - 1
                sid.append(j if 0 <= j < len(spans) else -1)
        items.append(dict(input_ids=enc["input_ids"], sent_id=sid, sent_labels=labs))
    return items, stats


def collate(items):
    B = len(items)
    L = max(len(it["input_ids"]) for it in items)
    S = max(len(it["sent_labels"]) for it in items)
    ids = np.zeros((B, L), dtype=np.int64); am = np.zeros((B, L), dtype=np.int64)
    sid = np.full((B, L), -1, dtype=np.int64)
    lab = np.zeros((B, S), dtype=np.float32); smask = np.zeros((B, S), dtype=np.float32)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        ids[i, :n] = it["input_ids"]; am[i, :n] = 1
        sid[i, :n] = it["sent_id"]
        m = len(it["sent_labels"])
        lab[i, :m] = it["sent_labels"]; smask[i, :m] = 1.0
    return dict(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(am),
                sent_id=torch.from_numpy(sid), sent_labels=torch.from_numpy(lab),
                sent_mask=torch.from_numpy(smask))


class IRACPS(nn.Module):
    def __init__(self, bert, lab, drop=0.1):
        super().__init__()
        self.bert = bert
        self.drop = nn.Dropout(drop)
        self.scorer = nn.Linear(768, 1)
        self.fuse = nn.Sequential(nn.Linear(1536, 768), nn.GELU(), nn.Dropout(drop))
        self.heads = nn.ModuleList([nn.Linear(768, len(lab.names[lv]) + 1) for lv in range(1, 5)])

    def forward(self, ids, am, sent_id=None, smask=None):
        h = self.bert(input_ids=ids, attention_mask=am).last_hidden_state
        ts = self.scorer(self.drop(h)).squeeze(-1)                       # [B,L]
        if sent_id is None:
            return [hd(h[:, 0]) for hd in self.heads], None, None
        B, L = ids.shape
        S = int(sent_id.max().item()) + 1 if int(sent_id.max().item()) >= 0 else 1
        M = torch.zeros(B, L, S, device=ids.device)
        M.scatter_(2, sent_id.clamp(min=0).unsqueeze(-1), 1.0)
        M = M * sent_id.ge(0).unsqueeze(-1).float()
        cnt = M.sum(1).clamp(min=1.0)                                    # [B,S]
        sent_score = torch.bmm(M.transpose(1, 2), ts.unsqueeze(-1)).squeeze(-1) / cnt   # [B,S]
        sent_vec = torch.bmm(M.transpose(1, 2), h) / cnt.unsqueeze(-1)   # [B,S,768]
        logits_s = sent_score.masked_fill(smask[:, :S] <= 0, -1e4)
        alpha = torch.softmax(logits_s, dim=1) * smask[:, :S]
        reason = torch.bmm(alpha.unsqueeze(1), sent_vec).squeeze(1)      # [B,768]
        rep = self.fuse(torch.cat([h[:, 0], reason], dim=-1))
        return [hd(self.drop(rep)) for hd in self.heads], sent_score[:, :S], smask[:, :S]


def evaluate(model, items, recs, trie, bs=32):
    model.eval()
    paths = []
    with torch.no_grad():
        for bi in BD.batches(len(items), bs):
            t = collate([items[i] for i in bi])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = model(t["input_ids"].to(trie.dev), t["attention_mask"].to(trie.dev),
                                 t["sent_id"].to(trie.dev), t["sent_mask"].to(trie.dev))
            paths.extend(decode([x.float() for x in lg], trie, "product"))
    return paths, C.metrics_for(paths, recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--head", type=int, default=384)
    ap.add_argument("--tail", type=int, default=126)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="./runs/sup/iracps")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    C.set_seed(a.seed)
    tag = a.tag or "lr%s_bs%d_lam%s" % (a.lr, a.batch, a.lam)
    outdir = os.path.join(a.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device("cuda")

    from transformers import AutoTokenizer, AutoModel
    tr, va, te = C.load_split("train"), C.load_split("valid_train"), C.load_split("test")
    lab = C.LabelSpace([x["gold_path"] for x in tr])
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    data = BD.BertData(tok, a.maxlen, a.head, a.tail, with_offsets=True)
    Itr, str_stats = build_items(tr, data)
    Iva, sva_stats = build_items(va, data)
    Ite, ste_stats = build_items(te, data)
    ytr = [lab.cls2idx[tuple(x["gold_path"])] for x in tr]

    model = IRACPS(AutoModel.from_pretrained(MODEL_DIR), lab, a.drop).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    trie = Trie(lab, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = (len(tr) + a.batch - 1) // a.batch
    total = steps * a.epochs
    warm = max(int(total * a.warmup), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(total - warm, 1)))
    print(json.dumps(dict(nparam=nparam, weak_train=str_stats, weak_valid=sva_stats, weak_test=ste_stats,
                          config=vars(a), steps=steps, label=lab.stats()), ensure_ascii=False), flush=True)

    rng = np.random.RandomState(a.seed)
    best = dict(f1=-1.0, epoch=-1, state=None)
    curve, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nce, naux, nb = 0.0, 0.0, 0.0, 0
        for bi in BD.batches(len(tr), a.batch, shuffle=True, rng=rng):
            t = collate([Itr[i] for i in bi])
            ids = t["input_ids"].to(dev); am = t["attention_mask"].to(dev)
            sd = t["sent_id"].to(dev); sm = t["sent_mask"].to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, sscore, smask2 = model(ids, am, sd, sm)
            lg = [x.float() for x in lg]
            main_loss = loss_fn(lg, trie, torch.tensor([ytr[i] for i in bi], dtype=torch.long, device=dev))
            aux = F.binary_cross_entropy_with_logits(sscore.float(), t["sent_labels"].to(dev),
                                                     reduction="none")
            aux = (aux * smask2.float()).sum() / smask2.float().sum().clamp(min=1.0)
            loss = main_loss + a.lam * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); nce += float(main_loss.detach()); naux += float(aux.detach()); nb += 1
        _, m = evaluate(model, Iva, va, trie)
        rec = dict(tag=tag, epoch=ep, loss=round(tot / nb, 4), ce=round(nce / nb, 4),
                   ext=round(naux / nb, 4), valid_acc=m["accuracy"], valid_f1=m["macro_f1"],
                   valid_p=m["macro_precision"], valid_r=m["macro_recall"], sec=round(time.time() - t0, 1))
        curve.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
        if m["macro_f1"] > best["f1"]:
            best = dict(f1=m["macro_f1"], epoch=ep, acc=m["accuracy"],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        elif ep - best["epoch"] >= a.patience:
            break
    model.load_state_dict(best["state"])
    pv, mva = evaluate(model, Iva, va, trie)
    pt, mte = evaluate(model, Ite, te, trie)
    with open(os.path.join(outdir, "curve.jsonl"), "w", encoding="utf-8") as f:
        for r in curve:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = dict(tag=tag, config=vars(a), nparam=nparam, best_epoch=best["epoch"],
                   best_valid_f1=best["f1"], best_valid_acc=best["acc"],
                   wall_sec=round(time.time() - t0, 1), weak_train=str_stats)
    if a.final:
        C.write_preds(os.path.join(outdir, "preds.jsonl"), [x["id"] for x in te], pt)
        C.write_preds(os.path.join(outdir, "preds_valid.jsonl"), [x["id"] for x in va], pv)
        torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        summary["test"] = C.brief(mte)
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
