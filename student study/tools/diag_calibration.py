# -*- coding: utf-8 -*-
"""
diag_calibration.py —— 诊断"校准后固定阈值 0.5 失效"的原因（诊断用，可保留）

【要回答的问题】
A2 校准实验出现了一个反直觉结果：
    校准后 Brier/ECE 大幅改善，但固定阈值 0.5 下 precision/recall 变成 0。
这不是 bug，而是概率校准的固有副作用，本脚本用数据把原因讲清楚：
    Platt（sigmoid）校准把概率**压向类别基准率**，
    因此校准后的概率上限可能显著低于 0.5，0.5 这个阈值就再也不合适了。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
import config as cfg          # noqa: E402
import evaluate as ev         # noqa: E402


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC 的成对定义（不依赖 sklearn），对单调变换不变。"""
    pos = s[y == 1]
    neg = s[y == 0]
    cmp = pos[:, None] > neg[None, :]
    tie = pos[:, None] == neg[None, :]
    return float(cmp.mean() + 0.5 * tie.mean())


def main() -> int:
    path = os.path.join(cfg.METRICS_DIR, "calibration.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    y = np.array(d["y_true"], dtype=int)
    pr = np.array(d["proba_raw"], dtype=float)
    pc = np.array(d["proba_calibrated"], dtype=float)

    print("=" * 74)
    print("【诊断】校准后为什么固定阈值 0.5 会失效")
    print("=" * 74)
    print("原始概率   ：取值 %s，最大值 %.4f" % (sorted(np.unique(pr).tolist()), pr.max()))
    print("校准后概率 ：最大值 %.4f，均值 %.4f" % (pc.max(), pc.mean()))
    print("            分位数 50%%=%.3f  90%%=%.3f  99%%=%.3f"
          % (np.percentile(pc, 50), np.percentile(pc, 90), np.percentile(pc, 99)))
    print("真实高风险基准率 = %.4f" % y.mean())
    print("→ Platt 校准把概率压向基准率，因此上限低于 0.5；"
          "0.5 是二分类的默认阈值，不再是校准概率的合适工作点。")
    print()

    print("校准后在各阈值下的表现：")
    print("  %-8s%-9s%-9s%-9s%s" % ("阈值", "P", "R", "F1", "名单人数"))
    best = (0.0, -1.0)
    for th in (0.05, 0.10, 0.135, 0.15, 0.163, 0.20, 0.25, 0.30, 0.50):
        m = ev.evaluate_binary(y, (pc >= th).astype(int))
        print("  %-8.3f%-9.4f%-9.4f%-9.4f%d"
              % (th, m["precision"], m["recall"], m["f1"], m["TP"] + m["FP"]))
        if m["f1"] > best[1]:
            best = (th, m["f1"], m["precision"], m["recall"])
    print()
    print("校准后 F1 最优阈值 ≈ %.3f（F1=%.4f，P=%.4f，R=%.4f）"
          % (best[0], best[1], best[2], best[3]))
    print("原始概率的 F1 最优阈值 = 0.5（此时只有 0/1 两值，任何 (0,1) 内阈值都等价）")
    print()

    print("排序能力对比（AUC 对单调变换不变，因此校准前后应几乎相同）：")
    a_raw, a_cal = auc(y, pr), auc(y, pc)
    print("  AUC(原始)   = %.4f" % a_raw)
    print("  AUC(校准后) = %.4f   ← 差值 %.4f" % (a_cal, a_cal - a_raw))
    print()
    print("结论：")
    print("  1) 校准**不改变排序能力**（AUC 几乎不变），所以它不会让模型'更准'；")
    print("  2) 它改变的是概率的数值刻度：Brier %.4f→%.4f，ECE %.4f→%.4f，"
          % (d["brier"]["raw"], d["brier"]["calibrated"],
             d["ece"]["raw"], d["ece"]["calibrated"]))
    print("     概率取值数 %d→%d（原本 k=1 只有 2 个取值）；"
          % (d["n_unique_proba"]["raw"], d["n_unique_proba"]["calibrated"]))
    print("  3) **使用校准概率时必须同时重新选工作点**，否则 0.5 这种默认阈值会失效；")
    print("  4) 三档等级阈值（0.3 / 0.7）同样需要重新标定 —— "
          "这也解释了为何校准后'中'档恢复、'高'档消失。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
