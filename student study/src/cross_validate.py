# -*- coding: utf-8 -*-
"""
cross_validate.py —— 交叉验证与 k 值选择模块
=========================================================
本项目要求**双口径评估**，两套口径各有各的价值，缺一不可：

    口径 A：留一法（Leave-One-Out, LOO）
        n 个样本做 n 折，每次留 1 个样本做测试、其余全部训练。
        - 优点：几乎用满数据，结果**确定性**（不依赖随机划分），适合写进报告做主口径；
        - 缺点：n 次预测并非相互独立，方差被低估；且计算量为 n×(n-1) 距离。
        - 本项目的关键优化：留一法的训练集恒为"除自己外全部样本"，
          因此只需算**一次 n×n 距离矩阵**，把对角线置为 +∞ 即等价于留一法，
          复杂度从 O(n^2) 次距离计算降到 O(n^2) 一次矩阵乘，400 条数据瞬间完成。

    口径 B：10 次随机划分（7:3）
        每次用不同 seed 随机划分，报 accuracy/precision/recall/F1 的**均值 ± 标准差**。
        - 优点：反映"换一批学生会怎样"，标准差能暴露模型稳定性；
        - 意义：避免"挑一次最好看的划分写进报告"这种选择性报告。
        - 注意：划分前**重新 fit 标准化器**（只用训练集的均值/方差），
          否则测试集信息会泄漏进训练过程，指标虚高。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .knn_numpy import KNNClassifierNumpy, loo_predict, loo_predict_proba
    from .preprocess import Dataset, MedianImputer, StandardScaler, random_split
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from knn_numpy import KNNClassifierNumpy, loo_predict, loo_predict_proba
    from preprocess import Dataset, MedianImputer, StandardScaler, random_split


# ======================================================================
# 一、留一法
# ======================================================================
def leave_one_out(X: np.ndarray, y: np.ndarray, k: int) -> Dict:
    """
    对一个固定 k 做留一法评估。

    返回 full_report 结构 + 逐样本概率（供阈值分析与错误样本分析使用）。
    """
    p = loo_predict_proba(X, y, k)      # 每个样本"被留出时"的高风险概率
    y_pred = (p > 0.5).astype(int)      # 与 KNNClassifierNumpy.predict 平票口径一致
    report = ev.full_report(y, y_pred, model_name=f"KNN_LOO_k{k}")
    report["k"] = int(k)
    report["proba_pos"] = p.tolist()    # 便于复算与绘图
    return report


def leave_one_out_multi_k(X: np.ndarray, y: np.ndarray,
                          k_list: Optional[List[int]] = None) -> Dict[int, Dict]:
    """
    多个 k 值的留一法扫描。

    性能关键：所有 k 共用同一个 n×n 距离矩阵，只有"取每行最小的 k 个"这一步不同，
    因此扫描 9 个 k 值的总耗时 ≈ 算一次距离矩阵的耗时。
    """
    k_list = list(cfg.K_CANDIDATES if k_list is None else k_list)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel().astype(int)
    n = len(X)

    D = KNNClassifierNumpy.euclidean_distances(X)
    D[np.arange(n), np.arange(n)] = np.inf   # 排除自身 → 等价于留一法

    results: Dict[int, Dict] = {}
    for k in k_list:
        if k < 1:
            continue
        k_eff = max(1, min(int(k), n - 1))
        if k_eff == n - 1:
            idx = np.argsort(D, axis=1, kind="stable")[:, :k_eff]
        else:
            idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
        p = y[idx].sum(axis=1) / float(k_eff)       # 近邻中高风险样本占比
        y_pred = (p > 0.5).astype(int)
        rep = ev.full_report(y, y_pred, model_name=f"KNN_LOO_k{k}")
        rep["k"] = int(k)
        rep["k_effective"] = int(k_eff)
        rep["proba_pos"] = p.tolist()
        results[int(k)] = rep
    return results


# ======================================================================
# 二、10 次随机划分
# ======================================================================
def repeated_random_split(ds: Dataset, k: int,
                          n_splits: int = cfg.N_SPLITS,
                          test_size: float = cfg.TEST_SIZE,
                          base_seed: int = cfg.RANDOM_SEED,
                          model_factory=None) -> Tuple[List[Dict], List[Dict]]:
    """
    重复 n_splits 次随机划分，每次独立训练+预测并记录明细。

    重要细节：每次划分都用"该次划分自己的训练集"重新 fit 标准化器。
    若沿用全量数据的均值/方差，测试集信息就渗进了训练流程（数据泄漏），
    指标会明显偏乐观，这是教学项目最容易踩的坑之一。

    参数
    ----
    model_factory : 可调用对象 k -> 模型实例；默认构造 KNNClassifierNumpy。
                    传入 lambda: LogisticRegressionNumpy(...) 即可复用本函数评估对照模型。

    返回
    ----
    (reports, details)
        reports : 每次划分的 full_report 列表
        details : 每次划分的元信息（seed、样本数、指标），用于写 split_details.csv
    """
    reports: List[Dict] = []
    details: List[Dict] = []
    n = len(ds)

    for i in range(n_splits):
        seed = base_seed + i
        # 用独立随机种子做划分 → 第 i 次划分可被单独复现
        tr_raw_idx, te_raw_idx = _split_indices(n, test_size, seed)
        Xtr_raw = ds.raw[cfg.FEATURE_COLUMNS].to_numpy(dtype=float)[tr_raw_idx]
        Xte_raw = ds.raw[cfg.FEATURE_COLUMNS].to_numpy(dtype=float)[te_raw_idx]
        ytr, yte = ds.y[tr_raw_idx], ds.y[te_raw_idx]

        # ① 缺失值填补：中位数只用训练集统计
        imp = MedianImputer().fit(Xtr_raw)
        # ② 标准化：均值/标准差只用训练集统计
        sc = StandardScaler().fit(imp.transform(Xtr_raw))
        Xtr = sc.transform(imp.transform(Xtr_raw))
        Xte = sc.transform(imp.transform(Xte_raw))

        # ③ 训练 + 预测
        clf = (model_factory(k) if model_factory is not None
               else KNNClassifierNumpy(k=k))
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]

        rep = ev.full_report(yte, y_pred, model_name=f"KNN_split{i+1}_k{k}")
        rep["k"] = int(k)
        rep["seed"] = int(seed)
        rep["n_train"] = int(len(tr_raw_idx))
        rep["n_test"] = int(len(te_raw_idx))
        rep["proba_pos"] = proba.tolist()
        reports.append(rep)
        details.append({
            "split": i + 1, "seed": seed, "k": int(k),
            "n_train": len(tr_raw_idx), "n_test": len(te_raw_idx),
            "test_risk_ratio": round(float(yte.mean()), 4),
            **{key: round(float(rep["metrics"][key]), 6) for key in
               ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")},
            **rep["counts"],
        })
    return reports, details


def _split_indices(n: int, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """按 seed 生成划分索引（与 preprocess.random_split 保持同一套随机逻辑）。"""
    n_test = max(1, min(n - 1, int(round(n * test_size))))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return perm[n_test:], perm[:n_test]


# ======================================================================
# 三、k 值扫描与主 k 选择
# ======================================================================
def k_scan(ds: Dataset,
           k_list: Optional[List[int]] = None,
           n_splits: int = cfg.N_SPLITS,
           test_size: float = cfg.TEST_SIZE,
           base_seed: int = cfg.RANDOM_SEED,
           seed: Optional[int] = None,   # 别名，便于调用方统一用 seed 传参
           verbose: bool = True) -> Tuple[pd.DataFrame, Dict[int, Dict], Dict[int, List[Dict]]]:
    """
    对每个候选 k 同时做"留一法"和"10 次随机划分"，输出对比表。

    返回 (df_scan, loo_results, split_results)
    """
    if seed is not None:      # 两个名字等效，避免调用方踩坑
        base_seed = int(seed)
    k_list = list(cfg.K_CANDIDATES if k_list is None else k_list)
    if verbose:
        print(f"[k 值扫描] 候选 k = {k_list}，留一法 + {n_splits} 次随机划分（7:3）")

    loo_results = leave_one_out_multi_k(ds.X, ds.y, k_list)   # 共用一次距离矩阵，很快

    rows = []
    split_results: Dict[int, List[Dict]] = {}
    for k in k_list:
        if k not in loo_results:
            continue
        reps, _ = repeated_random_split(ds, k, n_splits=n_splits,
                                       test_size=test_size, base_seed=base_seed)
        split_results[k] = reps      # 供上层写 split_details.csv 与画稳定性曲线
        agg = ev.aggregate_folds(reps)
        loo = loo_results[k]
        rows.append({
            "k": k,
            "loo_accuracy": loo["metrics"]["accuracy"],
            "loo_precision": loo["metrics"]["precision"],
            "loo_recall": loo["metrics"]["recall"],
            "loo_f1": loo["metrics"]["f1"],
            "split_acc_mean": agg["accuracy"]["mean"],
            "split_acc_std": agg["accuracy"]["std"],
            "split_prec_mean": agg["precision"]["mean"],
            "split_prec_std": agg["precision"]["std"],
            "split_rec_mean": agg["recall"]["mean"],
            "split_rec_std": agg["recall"]["std"],
            "split_f1_mean": agg["f1"]["mean"],
            "split_f1_std": agg["f1"]["std"],
        })
        if verbose:
            print(f"  k={k:<3d} LOO: acc={loo['metrics']['accuracy']:.4f} "
                  f"P={loo['metrics']['precision']:.4f} R={loo['metrics']['recall']:.4f} "
                  f"F1={loo['metrics']['f1']:.4f} | "
                  f"划分均值±标准差: F1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")

    df_scan = pd.DataFrame(rows)
    return df_scan, loo_results, split_results


def select_best_k(df_scan: pd.DataFrame, min_k: int = 3,
                  tolerance: float = 0.006) -> Dict:
    """
    选择主 k，并给出**可答辩的选择理由**（这也是验收要求"说明理由"）。

    选择规则（按优先级）：
      1. 排除 k=1：k=1 在留一法下几乎无泛化意义（只看最近的一个同学），
         且对噪声样本极敏感，留一法分数常常虚高；
      2. 在可用 k 中找 LOO F1 最高者；
      3. 若若干 k 的 F1 差距在 tolerance 内（统计上难以区分），
         则选 **划分标准差更小** 的那个（更稳定）；
      4. 若仍并列，选 k 更大者（投票人数多 → 抗噪声，预测概率分辨率也更细）。
    最终返回包含选择过程与理由的结构体，直接写进 experiments.md。
    """
    df = df_scan.copy()
    cand = df[df["k"] >= min_k].copy()
    if cand.empty:
        cand = df.copy()
    best_f1 = float(cand["loo_f1"].max())
    near = cand[cand["loo_f1"] >= best_f1 - tolerance].copy()
    # 依次按 标准差升序、k 降序排序
    near = near.sort_values(by=["split_f1_std", "k"], ascending=[True, False])
    chosen = int(near.iloc[0]["k"])

    k1_row = df[df["k"] == 1]
    reason_parts = [
        f"以留一法 F1 为主排序依据，最高值为 {best_f1:.4f}",
        f"在 F1 相差不超过 {tolerance} 的候选 {near['k'].tolist()} 中，"
        f"选择 10 次划分 F1 标准差最小者（稳定性优先），最终 k = {chosen}",
    ]
    if not k1_row.empty:
        reason_parts.append(
            f"k=1 的 LOO 指标为 acc={float(k1_row.iloc[0]['loo_accuracy']):.4f}、"
            f"F1={float(k1_row.iloc[0]['loo_f1']):.4f}，"
            "但 k=1 只参考单个最近邻，对噪声样本与异常值几乎没有抵抗能力，"
            "在真实预警场景会给出极端概率（非 0 即 1），因此不作为主模型取值"
        )

    return {
        "chosen_k": chosen,
        "best_loo_f1": round(best_f1, 6),
        "near_candidates": near["k"].tolist(),
        "candidate_detail": near[["k", "loo_f1", "split_f1_mean", "split_f1_std"]]
        .round(6).to_dict(orient="records"),
        "reason": "；".join(reason_parts),
    }


# ======================================================================
# 四、自检：留一法向量化实现 vs 逐样本实现
# ======================================================================
def _self_test() -> None:
    """
    关键自检：证明"一次距离矩阵 + 掩对角线"的留一法，
    与"逐个样本训练 n-1 个样本再预测"的朴素留一法结果完全一致。
    这是本项目最重要的正确性证据之一，答辩时可直接展示该断言通过。
    """
    print("=" * 68)
    print("【自检】留一法高效实现 vs 朴素实现一致性")
    print("=" * 68)
    rng = np.random.RandomState(7)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.8, size=60) > 0).astype(int)

    for k in (1, 3, 5, 9):
        fast = loo_predict(X, y, k)
        slow = []
        for i in range(len(X)):
            mask = np.ones(len(X), dtype=bool)
            mask[i] = False
            clf = KNNClassifierNumpy(k=k).fit(X[mask], y[mask])
            slow.append(int(clf.predict(X[i:i + 1])[0]))
        slow = np.array(slow)
        same = bool((fast == slow).all())
        print(f"  k={k}: 高效实现与朴素实现预测完全一致 = {same}"
              f"（不一致数 {int((fast != slow).sum())}）")
        assert same, f"k={k} 时留一法实现与朴素实现不一致，存在 bug"
    print("自检通过 ✅ 留一法无需真的训练 n 次")
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    _self_test()
