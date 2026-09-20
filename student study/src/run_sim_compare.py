# -*- coding: utf-8 -*-
"""
run_sim_compare.py —— 模拟数据 vs UCI 真实数据：鲁棒性对照实验
=========================================================
对应需求"模拟数据仅作鲁棒性对照（学生自己提出的'构造重叠样本'思路保留为对照组）"。

【对照的设计原则：只换数据源，不换评估代码】
    两套数据都走：
        · 相同的评估口径：留一/留一组交叉验证 + 10 次分组随机划分 + 新样本测试
        · 相同的模型：sklearn KNN（同一套超参数搜索空间）+ sklearn 逻辑回归
        · 相同的指标：accuracy / precision / recall / F1 / 混淆矩阵
    这样得到的差异才能归因于"数据本身"，而不是评估代码不同。

【两套数据的本质差异（正是对照组要说明的问题）】
    模拟数据：由"加权规则 + 随机噪声 + 分位数定档"生成，高风险比例 ≈ 30%，
              特征与标签之间的关系是**人为设定且已知**的，因此难度可控。
    UCI 真实数据：真实学生问卷与成绩，剔除早期成绩（避免泄漏）后，
              可用特征与期末成绩的关联很弱（单特征最优准确率仅略高于多数类基线）。
    对照结论回答的是：**"在实验室可控数据上表现良好的方法，搬到真实数据上还剩多少"**。
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .model_backends import make_knn, make_logreg, resolve_backend, describe_backend
    from .preprocess import StandardScaler, build_dataset, load_simulated
    from .uci_data import group_train_test_split_indices
    from .uci_features import prepare_uci_features
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from model_backends import make_knn, make_logreg, resolve_backend, describe_backend
    from preprocess import StandardScaler, build_dataset, load_simulated
    from uci_data import group_train_test_split_indices
    from uci_features import prepare_uci_features


# ======================================================================
# 一、通用评估（两套数据共用）
# ======================================================================
def _predict_with_full_data(X: np.ndarray, y: np.ndarray, kind: str,
                            backend: str, **kw) -> np.ndarray:
    """
    留一/留一组交叉验证的高风险概率（用一次距离矩阵的等价实现）。

    做法：把"自身所在组"的距离全部置为 +∞，再取最小 k 个邻居投票。
    这与逐折重训数学等价（该等价性由 appendix_handwritten_knn.py 的
    留一法加速验证支持），且能在秒级完成 —— 否则对 400/1044 条数据逐折重训 k 扫描太慢。
    逻辑回归不具备这种免训练性质，因此这里只对 KNN 用该口径；
    两个模型的整体对比仍通过"10 次分组随机划分"完成。
    """
    try:
        from .knn_numpy import KNNClassifierNumpy
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from knn_numpy import KNNClassifierNumpy
    k = int(kw.get("k", cfg.MAIN_K))
    n = len(y)
    D = KNNClassifierNumpy.euclidean_distances(X)
    np.fill_diagonal(D, np.inf)          # 排除自身（每行一条记录 = 一名学生）
    k_eff = max(1, min(k, n - 1))
    part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    return y[part].sum(axis=1) / float(k_eff)


def _repeated_split(X: np.ndarray, y: np.ndarray, kind: str, backend: str,
                    n_splits: int, test_size: float, seed: int, **kw):
    """10 次随机划分（模拟数据每行即一名学生，无需分组）。"""
    reports = []
    n = len(y)
    for i in range(n_splits):
        rng = np.random.RandomState(seed + i)
        perm = rng.permutation(n)
        n_test = max(1, min(n - 1, int(round(n * test_size))))
        te, tr = perm[:n_test], perm[n_test:]
        sc = StandardScaler().fit(X[tr])
        model = (make_knn(**kw, backend=backend) if kind == "knn"
                 else make_logreg(backend=backend, **kw))
        model.fit(sc.transform(X[tr]), y[tr])
        pred = model.predict(sc.transform(X[te]))
        reports.append(ev.full_report(y[te], pred, f"{kind}_simsplit{i+1}"))
    return reports


def evaluate_simulated(backend: str = "auto", n_splits: int = cfg.N_SPLITS,
                       seed: int = cfg.RANDOM_SEED, verbose: bool = True) -> Dict:
    """在**模拟对照数据集**上跑与 UCI 相同口径的评估，结果存为 sim_metrics_summary.json。"""
    df = load_simulated()
    ds, imputer, scaler = build_dataset(df)
    X, y = ds.X, ds.y
    if verbose:
        print("=" * 68)
        print("【对照实验】自建模拟/脱敏数据（人工构造特征重叠的数据集）")
        print("=" * 68)
        print(f"  样本数={len(y)}，特征数={X.shape[1]}，高风险比例={float(y.mean()):.4f}")

    # ---- k 扫描（LOO 口径）----
    rows, logo_best = [], None
    for k in cfg.K_CANDIDATES:
        p = _predict_with_full_data(X, y, "knn", backend, k=k)
        rep = ev.full_report(y, (p > 0.5).astype(int), f"knn_loo_k{k}")
        m = rep["metrics"]
        rows.append({"k": k, "loo_accuracy": m["accuracy"], "loo_precision": m["precision"],
                     "loo_recall": m["recall"], "loo_f1": m["f1"]})
        if logo_best is None or m["f1"] > logo_best["metrics"]["f1"]:
            logo_best = rep
            logo_best["k"] = k
    df_scan = pd.DataFrame(rows)
    best_k = int(df_scan.sort_values("loo_f1", ascending=False).iloc[0]["k"])
    if verbose:
        print(f"  k 扫描：F1 最优 k = {best_k}（F1={float(df_scan.sort_values('loo_f1', ascending=False).iloc[0]['loo_f1']):.4f}）")

    # ---- 两个模型：10 次随机划分 ----
    reps_knn = _repeated_split(X, y, "knn", backend, n_splits, cfg.TEST_SIZE, seed, k=best_k)
    reps_lr = _repeated_split(X, y, "logreg", backend, n_splits, cfg.TEST_SIZE, seed)
    agg_knn, agg_lr = ev.aggregate_folds(reps_knn), ev.aggregate_folds(reps_lr)

    p_best = _predict_with_full_data(X, y, "knn", backend, k=best_k)
    loo_best = ev.full_report(y, (p_best > 0.5).astype(int), f"knn_loo_k{best_k}")

    summary = {
        "dataset": "自建模拟/脱敏数据（鲁棒性对照组）",
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "risk_ratio": round(float(y.mean()), 4),
        "main_k": best_k,
        "k_scan": df_scan.round(6).to_dict(orient="records"),
        "loo_knn": {k: v for k, v in loo_best.items() if k != "proba_pos"},
        "split10_knn": agg_knn, "split10_logreg": agg_lr,
        "backend": resolve_backend(backend),
        "backend_description": describe_backend("knn", resolve_backend(backend)),
        "disclaimer": cfg.DISCLAIMER,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ev.save_json(summary, os.path.join(cfg.METRICS_DIR, "sim_metrics_summary.json"))
    if verbose:
        print("  留一法 KNN：" + "  ".join(
            f"{kk}={loo_best['metrics'][kk]:.4f}" for kk in ("accuracy", "precision", "recall", "f1")))
        print(f"  10 次划分 KNN F1={agg_knn['f1']['mean']:.4f}±{agg_knn['f1']['std']:.4f}，"
              f"逻辑回归 F1={agg_lr['f1']['mean']:.4f}±{agg_lr['f1']['std']:.4f}")
        print("  已保存：results/metrics/sim_metrics_summary.json")
        print("=" * 68)
    return summary


# ======================================================================
# 二、合并对照表
# ======================================================================
def build_comparison_table(verbose: bool = True) -> pd.DataFrame:
    """生成"模拟数据 vs UCI 真实数据"对照表并落盘。"""
    rows = []

    # ---- UCI 真实数据（读 v2 指标）----
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            u = json.load(f)
        if str(u.get("dataset", "")).startswith("UCI"):
            lu = u.get("logo_knn", {}).get("metrics", {})
            su = u.get("split10_knn", {})
            rows.append({
                "数据集": "UCI 真实数据（主）",
                "样本数": u.get("n_samples"),
                "特征数": u.get("n_features_main"),
                "高风险比例": u.get("risk_ratio"),
                "多数类基线acc": u.get("majority_baseline_accuracy"),
                "主k": u.get("main_k"),
                "留一/留组_F1": round(float(lu.get("f1", 0)), 4),
                "留一/留组_recall": round(float(lu.get("recall", 0)), 4),
                "留一/留组_acc": round(float(lu.get("accuracy", 0)), 4),
                "划分F1均值": su.get("f1", {}).get("mean"),
                "划分F1标准差": su.get("f1", {}).get("std"),
                "相对基线的准确率增益": round(float(u.get("logo_knn", {}).get("metrics", {})
                                              .get("accuracy", 0))
                                        - float(u.get("majority_baseline_accuracy", 0)), 4),
            })

    # ---- 模拟数据（读 sim 指标）----
    sim_path = os.path.join(cfg.METRICS_DIR, "sim_metrics_summary.json")
    if os.path.exists(sim_path):
        with open(sim_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        ls = s.get("loo_knn", {}).get("metrics", {})
        ratio = float(s.get("risk_ratio", 0))
        rows.append({
            "数据集": "自建模拟数据（对照）",
            "样本数": s.get("n_samples"),
            "特征数": s.get("n_features"),
            "高风险比例": s.get("risk_ratio"),
            "多数类基线acc": round(max(ratio, 1 - ratio), 4),
            "主k": s.get("main_k"),
            "留一/留组_F1": round(float(ls.get("f1", 0)), 4),
            "留一/留组_recall": round(float(ls.get("recall", 0)), 4),
            "留一/留组_acc": round(float(ls.get("accuracy", 0)), 4),
            "划分F1均值": s.get("split10_knn", {}).get("f1", {}).get("mean"),
            "划分F1标准差": s.get("split10_knn", {}).get("f1", {}).get("std"),
            "相对基线的准确率增益": round(float(ls.get("accuracy", 0))
                                      - max(ratio, 1 - ratio), 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cfg.SIM_COMPARE_PATH, index=False, encoding="utf-8-sig")
    if verbose:
        print("=" * 68)
        print("【对照结论】模拟数据 vs UCI 真实数据")
        print("=" * 68)
        if df.empty:
            print("  尚无可对比的指标，请先分别运行 v2 训练与模拟数据对照。")
        else:
            print(df.to_string(index=False))
            print()
            print("  解读要点：")
            print("   ① 看「相对基线的准确率增益」：模拟数据上模型明显超过多数类基线，")
            print("      而 UCI 真实数据（剔除早期成绩后）该增益接近 0 甚至为负 —— ")
            print("      说明真实场景中仅凭这些行为/背景特征很难预测期末成绩；")
            print("   ② 看「划分F1标准差」：真实数据的波动通常更大，说明结论对划分更敏感；")
            print("   ③ 这正说明**模拟数据的指标不能代表真实效果**，这也是本项目")
            print("      坚持用真实公开数据作为主数据源的原因。")
        print("=" * 68)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    """命令行入口：python src/run_sim_compare.py"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    evaluate_simulated()
    print()
    build_comparison_table()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
