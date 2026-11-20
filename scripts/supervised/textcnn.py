# -*- coding: utf-8 -*-
"""TextCNN（Kim 2014, EMNLP）字级基线，适配 CAD-v2.1 民事案由预测。

- 输入：input_masked 字符序列，head+tail 截断到 2048 字；
- embedding 128 维随机初始化（不预训练），conv kernel 3/4/5 各 100，max-over-time pooling，
  dropout 0.5，交叉熵；Adam。
- 标签空间 = 训练集 146 个叶路径；预测时直接输出该类的完整路径。
"""
import argparse, json, os, time, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HEAD, TAIL = 1536, 512
MAXLEN = HEAD + TAIL


class Vocab:
    def __init__(self, texts, min_freq=2):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t)
        self.itos = ["<pad>", "<unk>"] + [c for c, n in cnt.most_common() if n >= min_freq]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text):
        ids = [self.stoi.get(ch, 1) for ch in C.truncate_chars(text, HEAD, TAIL)]
        return ids[:MAXLEN]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_class, emb=128, filters=100, kernels=(3, 4, 5), drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(filters * len(kernels), n_class)

    def forward(self, x):
        h = self.drop(self.emb(x)).transpose(1, 2)      # B,E,L
        hs = [torch.relu(c(h)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(hs, dim=1)))


def make_batches(idx, bs, rng=None):
    if rng is not None:
        idx = list(idx); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def pad_batch(seqs):
    m = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), m), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.from_numpy(out)


def run_once(args, tr, va, te, lab, vocab, tag):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = lambda d: [vocab.encode(x["input_masked"]) for x in d]
    Xtr, Xva, Xte = enc(tr), enc(va), enc(te)
    ytr = np.array([lab.cls2idx[tuple(x["gold_path"])] for x in tr])
    yva = np.array([lab.cls2idx[tuple(x["gold_path"])] for x in va])

    model = TextCNN(len(vocab.itos), len(lab.classes)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.cw > 0:
        cnt = np.bincount(ytr, minlength=len(lab.classes)).astype(np.float64)
        w = (cnt.sum() / (cnt + 1e-9)) ** args.cw
        w = w / w.mean()
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    else:
        crit = nn.CrossEntropyLoss()
    rng = np.random.RandomState(args.seed)
    best = dict(f1=-1.0, epoch=-1)
    curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for b in make_batches(np.arange(len(tr)), args.batch, rng):
            x = pad_batch([Xtr[i] for i in b]).to(dev)
            y = torch.from_numpy(ytr[b]).to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        # valid
        model.eval()
        preds = []
        with torch.no_grad():
            for b in make_batches(np.arange(len(va)), 128):
                x = pad_batch([Xva[i] for i in b]).to(dev)
                preds.append(model(x).argmax(1).cpu().numpy())
        pv = np.concatenate(preds)
        pp = [lab.path(int(c)) for c in pv]
        m = C.metrics_for(pp, va)
        rec = dict(tag=tag, epoch=ep, train_loss=round(tot / max(nb, 1), 4),
                   valid_acc=m["accuracy"], valid_f1=m["macro_f1"],
                   valid_p=m["macro_precision"], valid_r=m["macro_recall"], sec=round(time.time() - t0, 1))
        curve.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if m["macro_f1"] > best["f1"]:
            best = dict(f1=m["macro_f1"], epoch=ep, acc=m["accuracy"], state={k: v.cpu().clone() for k, v in model.state_dict().items()})
        elif ep - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    out = {}
    for nm, X, recs in [("valid", Xva, va), ("test", Xte, te)]:
        preds = []
        with torch.no_grad():
            for b in make_batches(np.arange(len(X)), 128):
                x = pad_batch([X[i] for i in b]).to(dev)
                preds.append(model(x).argmax(1).cpu().numpy())
        pv = np.concatenate(preds)
        pp = [lab.path(int(c)) for c in pv]
        out[nm] = (pp, C.metrics_for(pp, recs) if nm == "valid" else None)
    wall = time.time() - t0
    return dict(model=model, best=best, curve=curve, wall=wall, nparam=nparam, out=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--cw", type=float, default=0.0, help="class weight exponent, 0=off")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="./runs/sup/textcnn")
    ap.add_argument("--final", action="store_true", help="最优配置：写 preds/metrics/ckpt")
    a = ap.parse_args()
    C.set_seed(a.seed)
    tag = a.tag or "lr%s_bs%d_cw%s" % (a.lr, a.batch, a.cw)
    outdir = os.path.join(a.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    tr, va, te = C.load_split("train"), C.load_split("valid_train"), C.load_split("test")
    lab = C.LabelSpace([x["gold_path"] for x in tr])
    vocab = Vocab([x["input_masked"] for x in tr])
    print(json.dumps(dict(n_train=len(tr), n_valid=len(va), n_test=len(te), vocab=len(vocab.itos),
                          label=lab.stats(), config=vars(a)), ensure_ascii=False), flush=True)

    res = run_once(a, tr, va, te, lab, vocab, tag)
    with open(os.path.join(outdir, "curve.jsonl"), "w", encoding="utf-8") as f:
        for r in res["curve"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = dict(tag=tag, config=dict(lr=a.lr, batch=a.batch, cw=a.cw, seed=a.seed,
                                        max_len=MAXLEN, emb=128, filters=100, kernels=[3, 4, 5], drop=0.5),
                   nparam=res["nparam"], best_epoch=res["best"]["epoch"], best_valid_f1=res["best"]["f1"],
                   best_valid_acc=res["best"]["acc"], wall_sec=round(res["wall"], 1), vocab=len(vocab.itos))
    if a.final:
        pp_test, _ = res["out"]["test"]
        pp_valid = res["out"]["valid"][0]
        C.write_preds(os.path.join(outdir, "preds.jsonl"), [x["id"] for x in te], pp_test)
        C.write_preds(os.path.join(outdir, "preds_valid.jsonl"), [x["id"] for x in va], pp_valid)
        summary["test"] = C.brief(C.metrics_for(pp_test, te))
        torch.save(res["model"].state_dict(), os.path.join(outdir, "model.pt"))
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
