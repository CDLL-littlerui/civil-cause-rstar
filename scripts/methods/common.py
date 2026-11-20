# -*- coding: utf-8 -*-
"""步骤 5 公共件：词表遍历、提示词、vLLM 批量生成、BERT 校准器、统计记录。

所有方法（method_main / rstar_orig / berry）共用本模块，保证提示词风格、
截断规则（4000 字）与评测口径一致。
"""
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import Taxonomy, extract_answer, parse_path  # noqa: E402

SYSTEM = "你是民事案件案由预测助手。"
INPUT_CHARS = 4000
FEWSHOT_CHARS = 300

ACTIONS = {
    "A1": "先做一步思考：分析本案的法律关系性质与争议焦点，不超过 3 句。",
    "A2": "写出剩余推理，并在最后一行给出候选编号，格式严格为：候选编号：<编号>",
    "A3": "提出一个有助于确定本级案由的子问题，并简要回答它。",
    "A4": "重新回答上一步提出的子问题，给出修正后的结论。",
    "A5": "先重述案件的关键事实与争议焦点，不超过 4 句。",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def level_nodes(tax, level):
    return [n for n in tax.nodes if n["level"] == level]


def children_of(tax, prefix):
    """返回 prefix（节点名列表）对应节点的直接下级节点列表。空 prefix=根。"""
    if not prefix:
        return [n for n in tax.nodes if n["level"] == 1]
    node = tax.by_path.get(tuple(prefix))
    if node is None:
        return []
    return [n for n in tax.nodes if n["level"] == node["level"] + 1 and n["parent"] == node["name"]
            and tuple(n["full_path"][:-1]) == tuple(prefix)]


def cand_block(cands):
    return "\n".join("%d. %s" % (i + 1, n["name"]) for i, n in enumerate(cands))


def build_examples(tax, fewshot, n=2):
    """2 条少样本示例（取 fewshot10 第 1 折），只保留输入前 300 字。"""
    l1 = level_nodes(tax, 1)
    block = ["【示例】"]
    for ex in fewshot[:n]:
        gold1 = ex["gold_path"][0]
        idx = next((i + 1 for i, n in enumerate(l1) if n["name"] == gold1), 1)
        block.append("案情（节选）：" + ex["input_masked"][:FEWSHOT_CHARS].replace("\n", " "))
        block.append("第 1 级候选：\n" + cand_block(l1))
        block.append("分析：先看当事人之间是合同关系还是法定关系，再看标的与请求权的性质，"
                     "本案的请求权基础落在%s这一类。" % gold1)
        block.append("候选编号：%d" % idx)
        block.append("")
    return "\n".join(block)


def build_prompt(case, prefix, cands, thoughts, action_instr):
    parts = [SYSTEM, ""]
    parts.append("【已确定案由路径】" + (" > ".join(prefix) if prefix else "（尚未确定）"))
    parts.append("")
    parts.append("【第 %d 级候选】" % (len(prefix) + 1))
    parts.append(cand_block(cands))
    parts.append("")
    parts.append("【待预测案件】")
    parts.append(case)
    if thoughts:
        parts.append("")
        parts.append("【已有推演】")
        parts.extend(thoughts)
    parts.append("")
    parts.append("【当前任务】" + action_instr)
    return "\n".join(parts)


def setup_chat(llm, enable_thinking=False):
    """给 vLLM 的 LLM 对象挂上 chat 模板（与步骤 4 基线一致）。"""
    llm._tok = llm.get_tokenizer()
    llm._enable_thinking = enable_thinking
    return llm


def _apply_template(llm, prompts):
    tok = getattr(llm, "_tok", None)
    if tok is None:
        return prompts
    et = getattr(llm, "_enable_thinking", False)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    try:
        return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=et) for m in msgs]
    except TypeError:
        return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]


def chat(llm, prompts, max_tokens=96, temperature=0.0, top_p=1.0, seed=None):
    """批量生成，返回 [(text, n_in, n_out)]。"""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    outs = llm.generate(_apply_template(llm, prompts), sp, use_tqdm=False)
    res = []
    for o in outs:
        text = o.outputs[0].text
        res.append((text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)))
    return res


RE_CHOICE = re.compile(r"候选编号\s*[：:]\s*(\d+)")


def parse_choice(text, n_cands):
    """从推演文本里取候选编号（1-based），失败返回 None。"""
    ms = RE_CHOICE.findall(text or "")
    if not ms:
        return None
    idx = int(ms[-1]) - 1
    if 0 <= idx < n_cands:
        return idx
    return None


class Calibrator:
    """BERT-wwm-ext 余弦相似度校准器，只在编号/名称无法直接对应候选时触发。"""

    def __init__(self, model_dir, threshold=0.83, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.threshold = threshold
        self.cache = {}

    def embed(self, texts):
        import torch
        with torch.no_grad():
            batch = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            out = self.model(**batch).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(out, dim=-1)

    def best(self, text, cands):
        key = (text, tuple(c["name"] for c in cands))
        if key in self.cache:
            return self.cache[key]
        embs = self.embed([text] + [c["name"] for c in cands])
        sims = (embs[0:1] @ embs[1:].T)[0].tolist()
        i = int(np.argmax(sims))
        r = (i, float(sims[i]))
        self.cache[key] = r
        return r


def case_text(row, limit=INPUT_CHARS):
    return row["input_masked"][:limit]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def machine():
    """运行机器标识（hostname + 首个 IP 尾段），替代硬编码。"""
    import socket
    try:
        h = socket.gethostname()
        ip = socket.gethostbyname(h)
        return "%s/%s" % (h, ip)
    except Exception:
        return "unknown"


def write_run_log(path, **kw):
    kw.setdefault("finished_at", now())
    json.dump(kw, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return ""


def clean_ans(ans, max_len=30):
    """把开放式输出收敛成一个案由名：取最后一行、去前缀、过长视为无答案。"""
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
    return s if 0 < len(s) <= max_len else ""
