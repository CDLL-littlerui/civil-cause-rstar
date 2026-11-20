# -*- coding: utf-8 -*-
"""Trie / 层次化解码自检：_uniform 概率和为 1；每个类的 one-hot 前向 loss≈0 且解码回自身。"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from topjudge import Trie, loss_fn, decode

tr = C.load_split("train")
lab = C.LabelSpace([x["gold_path"] for x in tr])
tri = Trie(lab, torch.device("cpu"))
n = len(lab.classes)
print("classes", n, "level sizes", tri.n_l)

# 1) 均匀 logits：146 个类的概率和应为 1
uni = [torch.zeros(1, tri.n_l[lv - 1] + 1) for lv in range(1, 5)]
lp = tri.class_logprob(uni)
print("uniform prob sum %.4f  (log space max-min %.3f)" % (lp.exp().sum().item(), (lp.max() - lp.min()).item()))

# 2) one-hot 正确前向
bad = []
for c in range(n):
    path = lab.classes[c]
    lg = []
    for lv in range(1, 5):
        v = torch.zeros(1, tri.n_l[lv - 1] + 1)
        if lv <= len(path):
            v[0, lab.cls_node_global(lv, c)] = 10.0
        if lv == len(path) + 1 and len(path) < 4:
            v[0, tri.n_l[lv - 1]] = 10.0
        lg.append(v)
    lp = tri.class_logprob(lg)
    L = float(loss_fn(lg, tri, torch.tensor([c])))
    ok_arg = int(lp.argmax()) == c
    if not ok_arg or L > 1e-3:
        bad.append((c, path, ok_arg, int(lp.argmax()), round(L, 5)))
print("bad", len(bad), bad[:5])

# 3) 随机 logits 下，loss 对参数的梯度非零、解码深度分布
torch.manual_seed(0)
lg = [torch.randn(4, tri.n_l[lv - 1] + 1, requires_grad=True) for lv in range(1, 5)]
L = loss_fn(lg, tri, torch.tensor([0, 50, 120, 145]))
L.backward()
print("rand loss %.4f  grad norms %s" % (float(L), [round(float(x.grad.norm()), 3) for x in lg]))
print("decode depth dist", sorted(set(len(p) for p in decode(lg, tri, "product"))))
