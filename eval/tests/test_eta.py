# -*- coding: utf-8 -*-
"""η 归一化单元测试（v3.1，步骤 10A）：python -m unittest eval.tests.test_eta。"""
import os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.methods import method_v3 as V3

EPS = V3.EPS
TOL = 1e-9
C = lambda *n: [{"name": x} for x in n]


def cnts(paths):
    return V3.vote_counts([tuple(p) for p in paths])


class FakeTax:
    def __init__(self, cands_by_prefix):
        self.m = cands_by_prefix
        self.nodes = [{"name": c["name"], "level": 1} for c in cands_by_prefix.get((), [])]

    def children_of(self, prefix):
        return self.m.get(tuple(prefix), [])


class FakePrior:
    """最小的先验桩：给定每层候选与均匀先验。"""

    def __init__(self, cands_by_prefix):
        self.m = cands_by_prefix
        self.cache = {}
        self.tax = FakeTax(cands_by_prefix)

    def get(self, prefix):
        cands = self.m.get(tuple(prefix), [])
        n = max(1, len(cands))
        return cands, [1.0 / n] * len(cands), None


class TestRoot(unittest.TestCase):
    def test_01_root_normalized(self):
        # 11 个一级案由，16 票全部落在同一个一级案由上（9088 号案例的根层）
        cands = C(*["c%d" % i for i in range(10)], "劳动争议、人事争议")
        cnt = cnts([["劳动争议、人事争议", "劳动争议"]] * 12 +
                   [["劳动争议、人事争议", "劳动争议", "劳动合同纠纷", "确认劳动关系纠纷"]] * 4)
        eta = V3.vdist(cnt, (), cands)
        self.assertAlmostEqual(sum(eta), 1.0, delta=TOL)
        self.assertAlmostEqual(eta[-1], (16 + EPS) / (16 + EPS * 11), delta=1e-12)
        self.assertAlmostEqual(eta[-1], 0.7674418604651163, delta=1e-12)

    def test_02_root_all_votes_fail(self):
        cands = C("a", "b", "c")
        eta = V3.vdist(cnts([]), (), cands)
        self.assertAlmostEqual(sum(eta), 1.0, delta=TOL)
        for x in eta:
            self.assertAlmostEqual(x, 1.0 / 3, delta=1e-12)


class TestMidStop(unittest.TestCase):
    def test_03_votes_stop_mid_level(self):
        # 12 票停在 a>b，4 票走到 a>b>c（9088 号案例的第 3 级）
        cands = C("c", "d", "e")
        cnt = cnts([["a", "b"]] * 12 + [["a", "b", "c", "f"]] * 4)
        eta = V3.vdist(cnt, ("a", "b"), cands)
        self.assertAlmostEqual(sum(eta), 1.0, delta=TOL)
        self.assertAlmostEqual(eta[0], (4 + EPS) / (4 + EPS * 3), delta=1e-12)
        self.assertAlmostEqual(eta[0], 0.8181818181818182, delta=1e-12)
        self.assertAlmostEqual(eta[1], EPS / (4 + EPS * 3), delta=1e-12)

    def test_04_deep_prefix_stop(self):
        # 更深一层：票停在 a>b>c，其子节点分母应只含 d 的票
        cnt = cnts([["a", "b", "c"]] * 5 + [["a", "b", "c", "d"]] * 2)
        cands = C("d", "e")
        eta = V3.vdist(cnt, ("a", "b", "c"), cands)
        self.assertAlmostEqual(sum(eta), 1.0, delta=TOL)
        self.assertAlmostEqual(eta[0], (2 + EPS) / (2 + EPS * 2), delta=1e-12)


class TestAllFail(unittest.TestCase):
    def test_05_all_votes_failed_uniform(self):
        cands = C(*["x%d" % i for i in range(4)])
        eta = V3.vdist(cnts([]), ("a",), cands)
        self.assertAlmostEqual(sum(eta), 1.0, delta=TOL)
        self.assertAlmostEqual(eta[0], 0.25, delta=1e-12)

    def test_06_empty_cands(self):
        self.assertEqual(V3.vdist(cnts([["a"]]), (), []), [])


class TestTheta(unittest.TestCase):
    def _check(self, cands_by_prefix, paths, prefix, tau, lam):
        prior = FakePrior(cands_by_prefix)
        cnt = cnts(paths)
        cands, theta = V3.blended(prior, cnt, tuple(prefix), tau, lam)
        self.assertAlmostEqual(sum(theta), 1.0, delta=TOL)
        mu = V3.ph_tempered(prior.get(tuple(prefix))[1], tau)
        eta = V3.vdist(cnt, tuple(prefix), cands)
        for i in range(len(cands)):
            self.assertAlmostEqual(theta[i], lam * mu[i] + (1 - lam) * eta[i], delta=TOL)
        return cands, theta

    def test_07_theta_root(self):
        cands = C("劳动争议、人事争议", "合同、准合同纠纷", "知识产权与竞争纠纷")
        paths = [["劳动争议、人事争议", "劳动争议"]] * 4 + [["合同、准合同纠纷"]]
        for tau in (1.0, 2.0):
            for lam in (0.3, 0.5, 0.7):
                self._check({(): cands}, paths, (), tau, lam)

    def test_08_theta_mid_level(self):
        root = C("a")
        lv2 = C("b1", "b2")
        lv3 = C("c1", "c2", "c3")
        paths = [["a", "b1"]] * 3 + [["a", "b1", "c1"]] * 2
        for tau in (1.0, 2.0):
            for lam in (0.3, 0.5, 0.7):
                self._check({(): root, ("a",): lv2, ("a", "b1"): lv3}, paths, ("a", "b1"), tau, lam)


class TestGreedyVote(unittest.TestCase):
    def test_09_no_child_votes_falls_back_to_prior(self):
        cands = C("p1", "p2", "p3")
        prior = FakePrior({(): cands})
        prior.get = lambda pre: (cands, [0.1, 0.7, 0.2], None)
        # 全部票校准失败（cnt 为空）→ 子节点票数和为 0，应走先验（选 p2），
        # 不能被均匀 η 拉到第一个候选 p1
        v = V3.greedy_vote(prior, cnts([]), max_depth=1)
        self.assertEqual(v, ["p2"])


if __name__ == "__main__":
    unittest.main()
