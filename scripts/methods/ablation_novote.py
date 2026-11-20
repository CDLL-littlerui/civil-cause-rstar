#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 8-A2 消融 --no_vote：λ=1（只用层级先验，投票分布不参与 P'）。

做法：复用冻结的 method_v3（不改其文件），把网格改为 τ=2、λ=1.0 后调用其 main()。
其余组件（摘要、16 票、PUCT、M=4、4B 软验证、J）与主实验一致；票仍生成，
以保证 share/visits 与鉴别器口径不变，只是不再进入 P'。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import method_v3 as V3

V3.GRID = [(2.0, 1.0)]

if __name__ == "__main__":
    V3.main()
