# -*- coding: utf-8 -*-
"""BERT 系列基线的数据管线：char->token 截断（head+tail），可带 char offset 供弱监督对齐。"""
import numpy as np

import common as C


class BertData:
    def __init__(self, tok, max_len=512, head=384, tail=126, with_offsets=False):
        self.tok = tok
        self.max_len = max_len
        self.head = head
        self.tail = tail
        self.with_offsets = with_offsets

    def encode(self, text):
        t = (text or "").replace("\x00", " ")
        enc = self.tok(t, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        off = enc["offset_mapping"]
        if len(ids) > self.max_len - 2:
            h, tl = self.head, self.tail
            sel = list(range(h)) + list(range(len(ids) - tl, len(ids)))
        else:
            sel = list(range(len(ids)))
        ids = [ids[i] for i in sel]
        out = dict(input_ids=[self.tok.cls_token_id] + ids + [self.tok.sep_token_id])
        if self.with_offsets:
            out["offsets"] = [(-1, -1)] + [off[i] for i in sel] + [(-1, -1)]
        return out

    def encode_all(self, recs):
        return [self.encode(r["input_masked"]) for r in recs]


def collate(batch, pad_id=0, with_offsets=False):
    import torch
    L = max(len(b["input_ids"]) for b in batch)
    ids = np.zeros((len(batch), L), dtype=np.int64)
    am = np.zeros((len(batch), L), dtype=np.int64)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        am[i, :n] = 1
    out = dict(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(am),
               length=torch.tensor([len(b["input_ids"]) for b in batch]))
    if with_offsets:
        offs = np.full((len(batch), L, 2), -1, dtype=np.int64)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            offs[i, :n] = np.array(b["offsets"], dtype=np.int64)
        out["offsets"] = torch.from_numpy(offs)
    return out


def batches(n, bs, shuffle=False, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [idx[i:i + bs] for i in range(0, n, bs)]
