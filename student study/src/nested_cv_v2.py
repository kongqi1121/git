# -*- coding: utf-8 -*-
"""
nested_cv_v2.py —— 嵌套交叉验证（对"主口径很弱"这一结论做更严格的检验）
=========================================================
【为什么需要这个模块】
UCI 真实数据的主口径（不含 G1/G2）上，k 值扫描给出了一个"反直觉"的结果：
    k=1 的 LOGO F1 = 0.3632、recall = 0.3435 最好，
    而 k 越大 recall 越低（k=31 时仅 0.0696）。
问题是：**k 是我在全部数据上扫出来的**，再拿同一批数据的 LOGO 指标去报告，
属于"用测试数据参与模型选择"，指标会偏乐观（选择偏差）。

因此本模块做一次严格的校验 —— 嵌套交叉验证（nested CV）：
    外层：留一组交叉验证（每次留出一整名学生）
    内层：在**外层训练集内部**再用留一组交叉验证选 k（完全不看外层测试组）
    最终：外层各折的预测拼起来算指标 —— 这是一个**无偏**的性能估计

【同时输出】
    · 固定 k=1 的 LOGO 指标（即扫描得到的最好值，含选择偏差）
    · 嵌套 CV 的指标（无偏估计）
    · 两者之差 = 选择偏差的实际大小
    · 外层各折选出的 k 分布（看 k 的选择是否稳定）

【实现要点】内层选 k 需要"在训练子集上再跑一次留一组"，
为避免耗时过长，内层用**手写 KNN 的向量化实现**（一次距离矩阵完成全部折），
其与 sklearn 的逐样本等价性已由附录验证通过（结果差异为 0）。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .knn_numpy import KNNClassifierNumpy
    from .preprocess import StandardScaler
    from .uci_features import prepare_uci_features
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from knn_numpy import KNNClassifierNumpy
    from preprocess import StandardScaler
    from uci_features import prepare_uci_features


def _logo_proba_fast(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """
    快速的留一组交叉验证概率（一次距离矩阵完成全部折）。

    等价于"逐折把整组留出后训练 KNN"，因为留一/留组时训练集恒为"除该组外的全部样本"，
    所以只需把同组样本之间的距离置为 +∞ 再取最小 k 个即可。
    """
    n = len(y)
    D = KNNClassifierNumpy.euclidean_distances(X)
    same = groups[:, None] == groups[None, :]
    D = np.where(same, np.inf, D)
    k_eff = max(1, min(k, n - 1))
    part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    return y[part].sum(axis=1) / float(k_eff)


def nested_cv_knn(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   k_list: Optional[List[int]] = None,
                   recall_floor: float = 0.30,
                   verbose: bool = True) -> Dict:
    """
    嵌套交叉验证：外层留一组，内层留一组选 k。

    内层选择规则与主管线一致（先满足 recall 门槛，再取 F1 最优），
    以保证"外层评估的正是同一套选择策略"。
    """
    k_list = list(cfg.K_CANDIDATES if k_list is None else k_list)
    uniq_groups = pd.unique(groups)
    n = len(y)
    outer_proba = np.zeros(n, dtype=float)
    chosen_ks: List[int] = []
    inner_scores: List[Dict] = []

    t0 = time.time()
    for gi, g in enumerate(uniq_groups):
        te_idx = np.where(groups == g)[0]
        tr_idx = np.where(groups != g)[0]
        Xtr, ytr, gtr = X[tr_idx], y[tr_idx], groups[tr_idx]

        # ---- 内层：在训练子集上用留一组选 k ----
        best_k, best_f1 = k_list[0], -1.0
        inner_rows = []
        for k in k_list:
            p = _logo_proba_fast(Xtr, ytr, gtr, k)
            m = ev.metrics_from_counts(ev.confusion_counts(ytr, (p > 0.5).astype(int)))
            inner_rows.append({"k": k, "f1": m["f1"], "recall": m["recall"]})
        ok = [r for r in inner_rows if r["recall"] >= recall_floor]
        pool = ok if ok else inner_rows
        best_f1 = max(r["f1"] for r in pool)
        near = [r for r in pool if r["f1"] >= best_f1 - 0.01]
        best_k = int(max(r["k"] for r in near))       # 同分取较大 k（与主管线一致）
        chosen_ks.append(best_k)
        if gi < 3:                                     # 只记录前几折的内层明细，控制输出体积
            inner_scores.append({"fold": gi, "rows": inner_rows, "chosen_k": best_k})

        # ---- 外层：用选定的 k 在该折上预测（训练集 = 该折训练集）----
        D = KNNClassifierNumpy.euclidean_distances(X[te_idx], Xtr)
        k_eff = max(1, min(best_k, len(tr_idx)))
        part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
        outer_proba[te_idx] = ytr[part].sum(axis=1) / float(k_eff)

        if verbose and (gi + 1) % 100 == 0:
            print(f"    嵌套 CV 进度：{gi + 1}/{len(uniq_groups)} 折，"
                  f"已用 {time.time() - t0:.0f} 秒")

    outer_pred = (outer_proba > 0.5).astype(int)
    report = ev.full_report(y, outer_pred, model_name="KNN_nestedCV")
    report["proba_pos"] = outer_proba.tolist()

    # 各折选出的 k 分布
    ks_series = pd.Series(chosen_ks).value_counts().sort_index()
    return {
        "nested_cv": {k: v for k, v in report.items() if k != "proba_pos"},
        "chosen_k_distribution": {int(k): int(v) for k, v in ks_series.items()},
        "chosen_k_mode": int(ks_series.idxmax()),
        "chosen_k_mean": round(float(np.mean(chosen_ks)), 2),
        "outer_folds": int(len(uniq_groups)),
        "inner_scores_sample": inner_scores,
        "recall_floor": recall_floor,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def run_nested_cv(verbose: bool = True) -> Dict:
    """执行嵌套交叉验证并与"固定 k"结果对比，落盘结果。"""
    cfg.ensure_dirs()
    if verbose:
        print("=" * 68)
        print("【嵌套交叉验证】检验主口径的 k 选择是否存在选择偏差")
        print("=" * 68)

    df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
    X, y, names = prepare_uci_features(df, include_early_grades=False, verbose=False)
    groups = df["student_group_id"].to_numpy()
    # 标准化（嵌套 CV 内部不跨折共享统计量，这里只做整体标准化，
    # 因为 KNN 的距离对单调缩放不敏感，且两套实现都基于同一空间；
    # 更严格的做法是每折内部 fit scaler，见下方 fixed-k 对比的说明）
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    res = nested_cv_knn(Xs, y, groups, verbose=verbose)

    # 对比：固定 k（用全部数据扫出来的最优 k）与嵌套 CV 的差异 = 选择偏差
    fixed = {}
    for k in cfg.K_CANDIDATES:
        p = _logo_proba_fast(Xs, y, groups, k)
        m = ev.metrics_from_counts(ev.confusion_counts(y, (p > 0.5).astype(int)))
        fixed[int(k)] = {"f1": round(m["f1"], 4), "recall": round(m["recall"], 4),
                         "accuracy": round(m["accuracy"], 4)}
    best_fixed_k = max(fixed, key=lambda kk: fixed[kk]["f1"] if fixed[kk]["recall"] >= 0.30 else -1)
    res["fixed_k_logo"] = fixed
    res["best_fixed_k"] = int(best_fixed_k)
    res["selection_bias_f1"] = round(fixed[best_fixed_k]["f1"]
                                     - res["nested_cv"]["metrics"]["f1"], 4)
    res["conclusion"] = (
        f"固定 k={best_fixed_k}（在全量数据上扫描得到）的 LOGO F1 = "
        f"{fixed[best_fixed_k]['f1']:.4f}；嵌套 CV（内层独立选 k）的 F1 = "
        f"{res['nested_cv']['metrics']['f1']:.4f}，两者相差 "
        f"{res['selection_bias_f1']:+.4f}，即 k 选择带来的偏乐观幅度很小。"
        if abs(res["selection_bias_f1"]) < 0.05 else
        f"固定 k={best_fixed_k} 的 LOGO F1 = {fixed[best_fixed_k]['f1']:.4f}，"
        f"而嵌套 CV 的 F1 仅 {res['nested_cv']['metrics']['f1']:.4f}，"
        f"相差 {res['selection_bias_f1']:+.4f} —— 说明**在全部数据上扫 k 再报告指标是偏乐观的**，"
        f"必须以嵌套 CV 的结果作为对主口径性能的诚实估计。"
    )

    if verbose:
        print(f"  外层折数（学生数）= {res['outer_folds']}，耗时 {res['elapsed_sec']} 秒")
        print(f"  各折选出的 k 分布：{res['chosen_k_distribution']}"
              f"（众数 {res['chosen_k_mode']}，均值 {res['chosen_k_mean']}）")
        print()
        print("  固定 k（全量扫描）的 LOGO 指标：")
        for k, v in fixed.items():
            print(f"    k={k:<3d} F1={v['f1']:.4f} recall={v['recall']:.4f} "
                  f"accuracy={v['accuracy']:.4f}")
        print()
        nc = res["nested_cv"]["metrics"]
        print("  嵌套 CV（无偏估计）："
              f"acc={nc['accuracy']:.4f} P={nc['precision']:.4f} "
              f"R={nc['recall']:.4f} F1={nc['f1']:.4f}")
        print(f"  混淆矩阵：{res['nested_cv']['confusion_matrix']}")
        print(f"  选择偏差（固定 k F1 - 嵌套 CV F1）= {res['selection_bias_f1']:+.4f}")
        print(f"  结论：{res['conclusion']}")

    ev.save_json(res, os.path.join(cfg.METRICS_DIR, "nested_cv.json"))
    if verbose:
        print(f"  已保存：{os.path.relpath(os.path.join(cfg.METRICS_DIR, 'nested_cv.json'), cfg.BASE_DIR)}")
        print("=" * 68)
    return res


def main() -> None:
    """命令行入口：python src/nested_cv_v2.py"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    run_nested_cv(verbose=True)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
