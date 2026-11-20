# -*- coding: utf-8 -*-
"""TopJudge（Yang et al., EMNLP 2018, "Legal Judgment Prediction via Topological Learning"）
改造为 L1->L2->L3->L4 的 DAG 多任务层级预测，主干 BERT-wwm-ext。

与原论文的对应与改造
--------------------
原论文按 DAG 拓扑序预测多个子任务，后序任务注入前序输出。本数据只有一条严格链 L1->L2->L3->L4：
  - 子任务 = 案由的 4 个层级；共享 BERT [CLS] 表示，每级一个分类头；
  - 依赖注入 = 父节点条件归一化：第 l 级 softmax 只在该级父节点的子节点集合上归一化（树上的层次化 softmax）。
    训练用 gold 父节点（teacher forcing），推理用预测父节点；
  - 29 个标签同时是内部节点，故每级增加 STOP 选项，使 146 个类上的概率构成 trie 上的真分布；
  - 解码：对 146 个类做联合概率 argmax（--decode product，默认），另实现逐级贪心（greedy）。
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import bertdata as BD

MODEL_DIR = os.environ.get("BERT_MODEL", "dienstag/chinese-bert-wwm-ext")


class TopJudge(nn.Module):
    def __init__(self, bert, lab, drop=0.1):
        super().__init__()
        self.bert = bert
        self.drop = nn.Dropout(drop)
        self.heads = nn.ModuleList([nn.Linear(768, len(lab.names[lv]) + 1) for lv in range(1, 5)])

    def forward(self, ids, am):
        h = self.drop(self.bert(input_ids=ids, attention_mask=am).last_hidden_state[:, 0])
        return [head(h) for head in self.heads]


class Trie:
    """标签树上的分组表 + 预计算索引（全部向量化，避免逐样本 python 循环）。"""

    def __init__(self, lab, dev):
        self.lab, self.dev = lab, dev
        self.C = Cc = len(lab.classes)
        self.n_l = [len(lab.names[lv]) for lv in range(1, 5)]
        self.depth = torch.tensor([len(c) for c in lab.classes], dtype=torch.long, device=dev)
        self.idx, self.valid_l, self.par, self.pos, self.stop_pos, self.mg = [], [], [], [], [], []
        for lv in range(1, 5):
            n = self.n_l[lv - 1]
            if lv == 1:
                groups, par = [list(range(n))], [0] * n
                stop_ok = [False]
            else:
                groups = [list(map(int, g)) for g in lab.children[lv - 1]]
                par = list(lab.parent[lv])
                stop_ok = self._is_class_node(lv - 1)
            mg = max([len(g) for g in groups] + [1])
            if lv > 1 and any(stop_ok):
                mg = max(mg, max(len(g) + 1 for g, ok in zip(groups, stop_ok) if ok))
            idx = torch.zeros(len(groups), mg, dtype=torch.long, device=dev)
            val = torch.zeros(len(groups), mg, dtype=torch.bool, device=dev)
            spos = torch.full((len(groups),), -1, dtype=torch.long, device=dev)
            for p, g in enumerate(groups):
                for k, j in enumerate(g):
                    idx[p, k] = j; val[p, k] = True
                if lv > 1 and stop_ok[p]:
                    idx[p, len(g)] = n          # STOP 列 = 第 n 个 logit
                    val[p, len(g)] = True
                    spos[p] = len(g)
            pos = torch.zeros(n, dtype=torch.long, device=dev)
            for p, g in enumerate(groups):
                for k, j in enumerate(g):
                    pos[j] = k
            self.idx.append(idx); self.valid_l.append(val); self.mg.append(mg)
            self.par.append(torch.tensor(par, dtype=torch.long, device=dev))
            self.pos.append(pos); self.stop_pos.append(spos)
        # 每个类在各级的节点索引（不存在则 -1）
        self.cls_node = [torch.tensor([lab.cls_nodes[lv][ci] for ci in range(Cc)],
                                      dtype=torch.long, device=dev) for lv in range(1, 5)]
        self._build_tables()

    def _is_class_node(self, lv):
        out = [False] * self.n_l[lv - 1]
        for c in self.lab.classes:
            if len(c) == lv:
                out[self.lab.name2idx[lv][c[lv - 1]]] = True
        return out

    def _build_tables(self):
        """每级：参与该级监督的类集合 K、其父组 par、组内目标位置 pos、有效掩码 mask。"""
        self.par_cls = []
        for lv in range(1, 5):
            pars = [0 if lv == 1 else max(self.lab.cls_node_global(lv - 1, c), 0) for c in range(self.C)]
            self.par_cls.append(torch.tensor(pars, dtype=torch.long, device=self.dev))
        self.parc, self.posc, self.maskc = [], [], []
        depth = [len(c) for c in self.lab.classes]
        for lv in range(1, 5):
            par, pos, mask = [-1] * self.C, [0] * self.C, [0] * self.C
            for c in range(self.C):
                d = depth[c]
                if d >= lv:
                    par[c] = 0 if lv == 1 else self.lab.cls_node_global(lv - 1, c)
                    pos[c] = int(self.pos[lv - 1][self.lab.cls_node_global(lv, c)]); mask[c] = 1
                elif d == lv - 1 and lv > 1:
                    p = self.lab.cls_node_global(lv - 1, c)
                    par[c] = p; pos[c] = int(self.stop_pos[lv - 1][p]); mask[c] = 1
            self.parc.append(torch.tensor(par, dtype=torch.long, device=self.dev))
            self.posc.append(torch.tensor(pos, dtype=torch.long, device=self.dev))
            self.maskc.append(torch.tensor(mask, dtype=torch.float32, device=self.dev))

    def pad_logits(self, logits, lv):
        pad = logits[:, self.idx[lv - 1]]
        return pad.masked_fill(~self.valid_l[lv - 1], -1e4)

    def level_ce(self, logits, lv, cidx):
        """第 lv 级、每个样本按其自身类别的父组做交叉熵（含 STOP）。"""
        pad = self.pad_logits(logits, lv)                      # [B,P,mg]
        B, mg = pad.size(0), pad.size(2)
        par = self.parc[lv - 1][cidx].clamp(min=0)
        sel = pad.gather(1, par.view(B, 1, 1).expand(-1, -1, mg)).squeeze(1)   # [B,mg]
        ce = F.cross_entropy(sel, self.posc[lv - 1][cidx], reduction="none")
        m = self.maskc[lv - 1][cidx].float()
        return (ce * m).sum() / m.sum().clamp(min=1.0)

    def class_logprob(self, logits_l):
        """[B,146] 每个类的 log 联合概率（trie 上的层次化 softmax 乘积）。"""
        B = logits_l[0].size(0)
        tot = torch.zeros(B, self.C, device=self.dev)
        for lv in range(1, 5):
            pad = self.pad_logits(logits_l[lv - 1], lv)
            lse = torch.logsumexp(pad, dim=2)                  # [B,P]
            par_of_cls = self.par_cls[lv - 1]
            node = self.cls_node[lv - 1]
            lp = logits_l[lv - 1][:, node.clamp(min=0)] - lse.gather(1, par_of_cls.unsqueeze(0).expand(B, -1))
            tot = tot + torch.where((node >= 0).unsqueeze(0), lp, torch.zeros_like(lp))
            if lv >= 2:
                sc = self._stop_classes(lv)          # 深度 = lv-1 的类：在本级选 STOP
                if sc:
                    par = torch.tensor([self.lab.cls_node_global(lv - 1, c) for c in sc],
                                       dtype=torch.long, device=self.dev)
                    stop_lp = logits_l[lv - 1][:, self.n_l[lv - 1]].unsqueeze(1) - \
                        lse.gather(1, par.unsqueeze(0).expand(B, -1))     # [B,K]
                    full = torch.zeros(B, self.C, device=self.dev)
                    full[:, sc] = stop_lp
                    tot = tot + full
        return tot

    def _stop_classes(self, lv):
        if not hasattr(self, "_sc"):
            self._sc = {}
        if lv not in self._sc:
            self._sc[lv] = [c for c in range(self.C) if len(self.lab.classes[c]) == lv - 1]
        return self._sc[lv]


def loss_fn(logits_l, trie, cidx):
    if not torch.is_tensor(cidx):
        cidx = torch.tensor(cidx, dtype=torch.long, device=trie.dev)
    return sum(trie.level_ce(logits_l[lv - 1], lv, cidx) for lv in range(1, 5)) / 4


def decode(logits_l, trie, mode="product"):
    if mode == "product":
        ci = trie.class_logprob(logits_l).argmax(1)
        return [trie.lab.path(int(c)) for c in ci]
    B = logits_l[0].size(0)
    active, paths, lvl = list(range(B)), [[] for _ in range(B)], 0
    while active and lvl < 4:
        pad = trie.pad_logits(logits_l[lvl], lvl + 1)
        nxt = []
        for b in active:
            p = 0 if lvl == 0 else trie.lab.name2idx[lvl][paths[b][-1]]
            gidx = int(trie.idx[lvl][p, int(pad[b, p].argmax())])
            if lvl > 0 and gidx == trie.n_l[lvl]:
                continue
            paths[b].append(trie.lab.names[lvl][gidx]); nxt.append(b)
        active, lvl = nxt, lvl + 1
    return paths


def evaluate(model, data, recs, trie, bs=64, decode_mode="product"):
    model.eval()
    paths = []
    with torch.no_grad():
        for bi in BD.batches(len(data), bs):
            t = BD.collate([data[i] for i in bi])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(t["input_ids"].to(trie.dev), t["attention_mask"].to(trie.dev))
            paths.extend(decode([x.float() for x in lg], trie, decode_mode))
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
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--head", type=int, default=384)
    ap.add_argument("--tail", type=int, default=126)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--decode", default="product", choices=["product", "greedy"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="./runs/sup/topjudge")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    C.set_seed(a.seed)
    tag = a.tag or "lr%s_bs%d" % (a.lr, a.batch)
    outdir = os.path.join(a.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device("cuda")

    from transformers import AutoTokenizer, AutoModel
    tr, va, te = C.load_split("train"), C.load_split("valid_train"), C.load_split("test")
    lab = C.LabelSpace([x["gold_path"] for x in tr])
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    data = BD.BertData(tok, a.maxlen, a.head, a.tail)
    Dtr, Dva, Dte = data.encode_all(tr), data.encode_all(va), data.encode_all(te)
    ytr = [lab.cls2idx[tuple(x["gold_path"])] for x in tr]

    model = TopJudge(AutoModel.from_pretrained(MODEL_DIR), lab, a.drop).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    trie = Trie(lab, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = (len(tr) + a.batch - 1) // a.batch
    total = steps * a.epochs
    warm = max(int(total * a.warmup), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(total - warm, 1)))
    print(json.dumps(dict(n_train=len(tr), n_valid=len(va), n_test=len(te), nparam=nparam,
                          label=lab.stats(), config=vars(a), steps=steps), ensure_ascii=False), flush=True)

    rng = np.random.RandomState(a.seed)
    best = dict(f1=-1.0, epoch=-1, state=None)
    curve, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bi in BD.batches(len(tr), a.batch, shuffle=True, rng=rng):
            t = BD.collate([Dtr[i] for i in bi])
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(t["input_ids"].to(dev), t["attention_mask"].to(dev))
            loss = loss_fn([x.float() for x in lg], trie,
                           torch.tensor([ytr[i] for i in bi], dtype=torch.long, device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        _, m = evaluate(model, Dva, va, trie, decode_mode=a.decode)
        rec = dict(tag=tag, epoch=ep, train_loss=round(tot / max(nb, 1), 4), valid_acc=m["accuracy"],
                   valid_f1=m["macro_f1"], valid_p=m["macro_precision"], valid_r=m["macro_recall"],
                   sec=round(time.time() - t0, 1))
        curve.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if m["macro_f1"] > best["f1"]:
            best = dict(f1=m["macro_f1"], epoch=ep, acc=m["accuracy"],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        elif ep - best["epoch"] >= a.patience:
            break
    model.load_state_dict(best["state"])
    pv, mva = evaluate(model, Dva, va, trie, decode_mode=a.decode)
    pt, mte = evaluate(model, Dte, te, trie, decode_mode=a.decode)
    with open(os.path.join(outdir, "curve.jsonl"), "w", encoding="utf-8") as f:
        for r in curve:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = dict(tag=tag, config=vars(a), nparam=nparam, best_epoch=best["epoch"],
                   best_valid_f1=best["f1"], best_valid_acc=best["acc"],
                   wall_sec=round(time.time() - t0, 1), decode=a.decode)
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
