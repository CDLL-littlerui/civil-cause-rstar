# -*- coding: utf-8 -*-
"""判定：金标签深度 d，取预测路径第 d 级与 gold_leaf 比较；不足 d 级或不可解析记为 __INVALID__。"""
INVALID = "__INVALID__"


def judge_one(pred_nodes, gold_path, gold_leaf):
    """返回 dict(pred_label, correct, depth_ok, pred_depth)。"""
    d = len(gold_path)
    if pred_nodes is None or len(pred_nodes) < d:
        return dict(pred_label=INVALID, correct=False, depth_ok=False, pred_depth=len(pred_nodes or []))
    node = pred_nodes[d - 1]
    if node is None:
        return dict(pred_label=INVALID, correct=False, depth_ok=False, pred_depth=len(pred_nodes))
    return dict(pred_label=node["name"], correct=(node["name"] == gold_leaf), depth_ok=True, pred_depth=len(pred_nodes))
