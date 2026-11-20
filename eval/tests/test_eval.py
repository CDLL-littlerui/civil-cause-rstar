# -*- coding: utf-8 -*-
"""10 条边界用例（python -m unittest eval.tests.test_eval）。"""
import os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from eval import Taxonomy, extract_answer, parse_path, judge_one, compute_metrics
from eval.extract import strip_think
from eval.judge import INVALID

TAX = Taxonomy("2020", os.path.join(ROOT, "vocab"))


def judge(ans, gold_path):
    names = parse_path(ans)
    nodes, ok, hier = TAX.build_pred_path(names)
    j = judge_one(nodes, gold_path, gold_path[-1])
    return j, nodes, ok, hier


class TestExtract(unittest.TestCase):
    def test_01_marker_last_wins(self):
        t = "先猜一个：合同纠纷\n最终案由：买卖合同纠纷"
        a, m = extract_answer(t)
        self.assertEqual(a, "买卖合同纠纷")
        self.assertEqual(m, "marker")

    def test_02_think_block_removed(self):
        t = "<think>让我想想…最终案由：侵权责任纠纷</think>最终案由：买卖合同纠纷"
        a, m = extract_answer(t)
        self.assertEqual(a, "买卖合同纠纷")
        self.assertEqual(m, "marker")

    def test_03_unclosed_think_dropped(self):
        t = "最终案由：租赁合同纠纷\n<think>然后继续胡言乱语"
        a, _ = extract_answer(t)
        self.assertEqual(a, "租赁合同纠纷")

    def test_04_path_marker_and_separators(self):
        t = "分析结束。\n案由路径：合同、准合同纠纷 > 合同纠纷 > 买卖合同纠纷"
        a, m = extract_answer(t)
        self.assertEqual(m, "marker")
        self.assertEqual(parse_path(a), ["合同、准合同纠纷", "合同纠纷", "买卖合同纠纷"])

    def test_05_no_marker_fallback_last_line(self):
        t = "这是一段推理。\n买卖合同纠纷"
        a, m = extract_answer(t)
        self.assertEqual((a, m), ("买卖合同纠纷", "last_line"))


class TestJudge(unittest.TestCase):
    GOLD = ["合同、准合同纠纷", "合同纠纷", "买卖合同纠纷"]

    def test_06_exact_leaf(self):
        j, nodes, ok, hier = judge("最终案由：买卖合同纠纷", self.GOLD)
        self.assertTrue(j["correct"])
        self.assertEqual(j["pred_label"], "买卖合同纠纷")
        self.assertTrue(hier)

    def test_07_truncated_compare_at_gold_depth(self):
        # 预测比金标更深：按金标深度 d=3 截断比较，应判对
        j, nodes, ok, hier = judge("案由路径：合同、准合同纠纷 > 合同纠纷 > 买卖合同纠纷 > 商品房买卖合同纠纷", self.GOLD)
        self.assertTrue(j["correct"])
        self.assertEqual(j["pred_label"], "买卖合同纠纷")

    def test_08_shallower_pred_is_invalid(self):
        j, nodes, ok, hier = judge("案由路径：合同、准合同纠纷 > 合同纠纷", self.GOLD)
        self.assertFalse(j["correct"])
        self.assertEqual(j["pred_label"], INVALID)

    def test_09_out_of_vocab(self):
        j, nodes, ok, hier = judge("最终案由：外星人纠纷", self.GOLD)
        self.assertEqual(j["pred_label"], INVALID)
        self.assertFalse(hier)

    def test_10_same_name_picks_deepest(self):
        # “不当得利纠纷”同名 L2/L3，取最深 L3
        gold = ["合同、准合同纠纷", "不当得利纠纷", "不当得利纠纷"]
        j, nodes, ok, hier = judge("最终案由：不当得利纠纷", gold)
        self.assertEqual(len(nodes), 3)          # 反查补全祖先链到 L3
        self.assertEqual(nodes[-1]["name"], "不当得利纠纷")
        self.assertEqual(j["pred_label"], "不当得利纠纷")
        self.assertTrue(j["correct"])

    def test_11_hierarchy_skip_marked_by_hvr(self):
        # 跳过一级：L1 直接到 L3 -> HVR 记失败，但末级名称可反查，仍按金标深度取标签
        gold = ["合同、准合同纠纷", "合同纠纷", "买卖合同纠纷"]
        j, nodes, ok, hier = judge("案由路径：合同、准合同纠纷 > 买卖合同纠纷", gold)
        self.assertFalse(hier)
        self.assertEqual(j["pred_label"], "买卖合同纠纷")
        self.assertTrue(j["correct"])

    def test_13_partial_path_prefix_completed(self):
        # 模型省略 L1：从 L2 起给出连续路径 -> 补全前缀，HVR 通过
        gold = ["合同、准合同纠纷", "合同纠纷", "房屋买卖合同纠纷"]
        j, nodes, ok, hier = judge("最终案由：合同纠纷 > 房屋买卖合同纠纷", gold)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0]["name"], "合同、准合同纠纷")
        self.assertTrue(hier)
        self.assertTrue(j["correct"])

    def test_14_same_name_chain(self):
        # 同名节点（侵权责任纠纷 L1/L2）：多级路径按父子关系对齐
        gold = ["侵权责任纠纷", "侵权责任纠纷", "提供劳务者受害责任纠纷"]
        j, nodes, ok, hier = judge("最终案由：侵权责任纠纷 > 侵权责任纠纷 > 提供劳务者受害责任纠纷", gold)
        self.assertEqual([n["name"] for n in nodes], gold)
        self.assertTrue(hier)
        self.assertTrue(j["correct"])

    def test_15_wrong_ancestor_falls_back_to_leaf(self):
        # 首级名称写错（多一个“与…有关”），退化为末级反查，HVR 失败但标签可用
        gold = ["合同、准合同纠纷", "合同纠纷", "买卖合同纠纷"]
        j, nodes, ok, hier = judge("案由路径：经济合同纠纷 > 合同纠纷 > 买卖合同纠纷", gold)
        self.assertIsNotNone(nodes[-1])
        self.assertFalse(hier)


class TestMetrics(unittest.TestCase):
    def test_12_macro_f1_standard(self):
        rows = [
            dict(gold_leaf="A", gold_path=["A"], pred_label="A", pred_nodes=[{"name": "A"}], hierarchy_ok=True, parse_ok=True),
            dict(gold_leaf="A", gold_path=["A"], pred_label="A", pred_nodes=[{"name": "A"}], hierarchy_ok=True, parse_ok=True),
            dict(gold_leaf="B", gold_path=["B"], pred_label="A", pred_nodes=[{"name": "A"}], hierarchy_ok=True, parse_ok=True),
            dict(gold_leaf="C", gold_path=["C"], pred_label=INVALID, pred_nodes=[], hierarchy_ok=False, parse_ok=False),
        ]
        m = compute_metrics(rows)
        # A: tp2 fp1 -> P=2/3 R=1 F1=0.8; B: P0 R0 F0; C: 0; INVALID: tp0 fp1 fn0 -> 0
        # classes = {A,B,C,__INVALID__} -> (0.8+0+0+0)/4 = 0.2
        self.assertAlmostEqual(m["macro_f1"], 0.2, places=4)
        self.assertAlmostEqual(m["accuracy"], 0.5, places=4)
        self.assertAlmostEqual(m["parse_failure_rate"], 0.25, places=4)
        self.assertEqual(m["n_classes"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
