# -*- coding: utf-8 -*-
"""答案抽取：推理模型去 think 块 -> 取最后一个“最终案由：/案由路径：”之后的内容。"""
import re

THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*$", re.S | re.I)

MARKERS = ["最终案由", "案由路径"]
SEP_RE = re.compile(r"\s*(?:>|＞|→|->|—>|/)\s*")


def strip_think(text):
    """去掉 <think>...</think>；未闭合的 think 块整段丢弃（推理模型被截断时常见）。"""
    if not text:
        return ""
    t = THINK_RE.sub("", text)
    if "<think" in t.lower() and "</think" not in t.lower():
        t = THINK_OPEN_RE.sub("", t)
    return t


def _clean(s):
    s = s.strip()
    s = s.strip("\u3000 \t\r\n")
    s = s.strip("“”\"'‘’《》【】[]()（）")
    return s.strip()


def extract_answer(text):
    """返回 (answer, method)。method ∈ {marker, last_line, empty}"""
    t = _clean(strip_think(text))
    if not t:
        return "", "empty"
    best = None
    for m in re.finditer(r"(最终案由|案由路径)\s*[:：]", t):
        best = m
    if best is not None:
        rest = t[best.end():]
        line = rest.split("\n")[0].strip()
        if not line:
            tail = [x.strip() for x in rest.split("\n") if x.strip()]
            line = tail[0] if tail else ""
        return _clean(line), "marker"
    lines = [x.strip() for x in t.split("\n") if x.strip()]
    return _clean(lines[-1] if lines else ""), "last_line"


def parse_path(answer):
    """把答案切成层级列表。单名返回 [name]；带 > 或 / 的按分隔符切。"""
    a = _clean(answer)
    if not a:
        return []
    a = re.sub(r"^(最终案由|案由路径)\s*[:：]\s*", "", a)
    parts = [p for p in (_clean(x) for x in SEP_RE.split(a)) if p]
    return parts or [a]
