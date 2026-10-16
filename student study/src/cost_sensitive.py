# -*- coding: utf-8 -*-
"""
cost_sensitive.py —— 代价敏感评估与代价曲线（建议 C2）
=========================================================
【要解决的问题】
现有评估只回答"模型准不准"，但辅导员真正要问的是：
    "我每周只能约谈 N 个人，应该把判定阈值设成多少？"
这需要把 precision/recall 的兑换关系换算成**可比较的代价**。

【代价模型（标准做法）】
    总代价 = C_FN × FN + C_FP × FP
其中 FN = 漏报（真需要关注却判成正常），FP = 误报（真正常却判成需关注）。
令 C_FN = 1 作为基准，C_FP = r 为"误报相对漏报的代价比"：
    r 越小 → 越不怕误报、越怕漏报 → 最优阈值应越低。

【必须声明的两个纪律（本项目对"严谨性"的硬要求）】
    ① **代价比 r 是场景假设，不是测量值**。
       "漏报一名学生的真实代价"属于教育决策问题，数据集里不存在，也不能编造。
       因此本模块把 r 全部显式列出，并允许调用方替换，绝不声称它是"算出来的"。
    ② **工作点必须由训练侧选出**。
       曲线可以画在留出数据上用于展示，但"选哪个阈值"这件事只能在训练集内部做，
       否则就是把测试集当验证集（本项目 train_eval_v2 的
       `_tune_threshold_within_training` 已按此实现）。
       本模块额外提供"训练侧选点、测试侧评估"的口径，把两种结果并排列出，便于对比。

【产物】results/metrics/cost_sensitive.csv、cost_sensitive.json
【命令】python src/cost_sensitive.py
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .preprocess import StandardScaler
    from .uci_data import group_train_test_split_indices, load_saved_dataset
    from .uci_features import prepare_uci_features
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from preprocess import StandardScaler
    from uci_data import group_train_test_split_indices, load_saved_dataset
    from uci_features import prepare_uci_features


# 默认代价比集合：覆盖"很怕漏报"到"误报与漏报等价"四种情形
DEFAULT_COST_RATIOS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def cost_of_threshold(tp: int, fp: int, fn: int, tn: int,
                      c_fn: float = 1.0, c_fp: float = 0.2) -> float:
    """按定义计算总代价（便于单测与复用）。"""
    return c_fn * fn + c_fp * fp


def sweep_with_counts(y_true: np.ndarray, proba: np.ndarray,
                      thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """
    阈值扫描，但**保留混淆矩阵四个计数**（现有 threshold_sweep 只留指标）。

    为什么必须保留计数：代价 = C_FN·FN + C_FP·FP 需要 FN 与 FP 的实际个数，
    只靠 precision/recall 无法唯一还原（还要知道正类总数）。这是本模块存在的直接原因。
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    proba = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for th in thresholds:
        pred = (proba >= th).astype(int)
        m = ev.evaluate_binary(y_true, pred)
        rows.append({
            "threshold": float(th),
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "n_predicted_risk": int(m["TP"] + m["FP"]),
        })
    return pd.DataFrame(rows)


def best_threshold_by_cost(table: pd.DataFrame, c_fp: float = 0.2,
                           c_fn: float = 1.0) -> Dict:
    """对给定代价比，从扫描表里选出总代价最小的阈值。"""
    cost = c_fn * table["FN"].to_numpy(dtype=float) + c_fp * table["FP"].to_numpy(dtype=float)
    k = int(np.argmin(cost))
    row = table.iloc[k]
    return {
        "c_fp": float(c_fp), "c_fn": float(c_fn),
        "threshold": float(row["threshold"]),
        "total_cost": round(float(cost[k]), 2),
        "FN": int(row["FN"]), "FP": int(row["FP"]),
        "TP": int(row["TP"]), "TN": int(row["TN"]),
        "recall": float(row["recall"]), "precision": float(row["precision"]),
        "n_predicted_risk": int(row["n_predicted_risk"]),
    }


def run_cost_sensitive(backend: str = "auto", verbose: bool = True) -> Dict:
    """
    正式流程：在**训练侧**用分组交叉验证选工作点，在**留出侧**评估，全程不泄漏。

    做法：
      对 10 次分组划分中的每一次：
        ① 用该次训练集训练 KNN（k 取主 k）；
        ② 在训练集内部再切一刀（3 折分组 CV），得到**训练侧的概率**；
        ③ 在训练侧概率上按各代价比选阈值；
        ④ 把选出的阈值应用到该次测试集，累计各代价比下的测试侧指标。
    这样"阈值选择"只见过训练数据，测试集标签全程不参与选点。
    """
    cfg.ensure_dirs()
    df = load_saved_dataset()
    X, y, _ = prepare_uci_features(df, include_early_grades=False, verbose=False)
    groups = df["student_group_id"].to_numpy()
    main_k = int(cfg.MAIN_K)
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if s.get("dataset", "").startswith("UCI"):
            main_k = int(s.get("main_k", main_k))

    try:
        from .model_backends import make_knn
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from model_backends import make_knn
    per_ratio: Dict[float, List[Dict]] = {r: [] for r in DEFAULT_COST_RATIOS}
    last_table = None

    for i in range(cfg.N_SPLITS):
        seed = cfg.RANDOM_SEED + i
        tr, te = group_train_test_split_indices(df, cfg.TEST_SIZE, seed)
        sc = StandardScaler().fit(X[tr])
        model = make_knn(k=main_k, backend=backend).fit(sc.transform(X[tr]), y[tr])

        # ---- 训练侧概率：在训练集内部再做 3 折分组 CV ----
        p_in = np.zeros(len(tr), dtype=float)
        df_tr = df.iloc[tr].reset_index(drop=True)
        Xtr = X[tr]
        for j in range(3):
            itr, ite = group_train_test_split_indices(df_tr, 0.34, seed * 10 + j)
            sc2 = StandardScaler().fit(Xtr[itr])
            m2 = make_knn(k=main_k, backend=backend).fit(sc2.transform(Xtr[itr]), y[tr][itr])
            p_in[ite] = m2.predict_proba(sc2.transform(Xtr[ite]))[:, 1]

        table_in = sweep_with_counts(y[tr], p_in)
        last_table = table_in

        # ---- 测试侧概率与评估 ----
        p_te = model.predict_proba(sc.transform(X[te]))[:, 1]
        for r in DEFAULT_COST_RATIOS:
            pick = best_threshold_by_cost(table_in, c_fp=r)
            pred = (p_te >= pick["threshold"]).astype(int)
            m = ev.evaluate_binary(y[te], pred)
            per_ratio[r].append({
                "seed": seed, "threshold": pick["threshold"],
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
                "n_predicted_risk": int(m["TP"] + m["FP"]),
                "cost": cost_of_threshold(m["TP"], m["FP"], m["FN"], m["TN"], 1.0, r),
            })

    # 汇总
    summary = []
    detail = {}
    for r, runs in per_ratio.items():
        arr = pd.DataFrame(runs)
        detail[str(r)] = runs
        summary.append({
            "c_fp": r, "c_fn": 1.0,
            "threshold_mean": round(float(arr["threshold"].mean()), 3),
            "precision_mean": round(float(arr["precision"].mean()), 4),
            "recall_mean": round(float(arr["recall"].mean()), 4),
            "f1_mean": round(float(arr["f1"].mean()), 4),
            "cost_mean": round(float(arr["cost"].mean()), 2),
            "n_predicted_risk_mean": round(float(arr["n_predicted_risk"].mean()), 1),
            "FN_mean": round(float(arr["FN"].mean()), 1),
            "FP_mean": round(float(arr["FP"].mean()), 1),
        })

    # 作为对照：把"在测试侧直接选最优阈值"的结果也列出来（这是**错误做法**，仅用于说明差距）
    naive = []
    for r in DEFAULT_COST_RATIOS:
        costs = []
        for i in range(cfg.N_SPLITS):
            seed = cfg.RANDOM_SEED + i
            tr, te = group_train_test_split_indices(df, cfg.TEST_SIZE, seed)
            sc = StandardScaler().fit(X[tr])
            model = make_knn(k=main_k, backend=backend).fit(sc.transform(X[tr]), y[tr])
            p_te = model.predict_proba(sc.transform(X[te]))[:, 1]
            tbl = sweep_with_counts(y[te], p_te)
            costs.append(best_threshold_by_cost(tbl, c_fp=r)["total_cost"])
        naive.append({"c_fp": r, "cost_mean_if_tuned_on_test": round(float(np.mean(costs)), 2)})

    # 落盘：扫描表（供图表使用）
    if last_table is not None:
        last_table.to_csv(os.path.join(cfg.METRICS_DIR, "cost_sensitive.csv"),
                          index=False, encoding="utf-8-sig")

    # 完整扫描（全量数据的 LOGO 概率）用于画曲线
    if os.path.exists(cfg.PREDICTIONS_PATH):
        pdf = pd.read_csv(cfg.PREDICTIONS_PATH, encoding="utf-8-sig")
        full = sweep_with_counts(pdf[cfg.LABEL_COLUMN].to_numpy(), pdf["logo_proba_risk"].to_numpy())
        full.to_csv(os.path.join(cfg.METRICS_DIR, "cost_sensitive.csv"),
                    index=False, encoding="utf-8-sig")

    out = {
        "method": "阈值在训练集内部（3 折分组 CV）按给定代价比选出，测试集仅用于评估",
        "cost_model": "总代价 = C_FN × FN + C_FP × FP，C_FN 固定为 1",
        "assumption_note": "代价比 C_FP/C_FN 是**场景假设**，不是数据测量值；"
                           "真实代价取决于辅导员时间与学生发展后果，只能由使用者设定",
        "main_k": main_k,
        "by_ratio": summary,
        "naive_reference": naive,
        "detail_by_ratio": detail,
        "disclaimer": cfg.DISCLAIMER,
    }
    ev.save_json(out, os.path.join(cfg.METRICS_DIR, "cost_sensitive.json"))

    if verbose:
        print("=" * 78)
        print("【代价敏感评估】阈值只在训练侧选，测试侧仅评估")
        print("=" * 78)
        print(f"主 k = {main_k}，划分次数 = {cfg.N_SPLITS}（按学生分组）")
        print()
        print(f"  {'C_FP/C_FN':>10}{'选定阈值':>10}{'precision':>11}{'recall':>9}{'F1':>8}"
              f"{'名单人数':>10}{'漏报FN':>8}{'误报FP':>8}{'总代价':>9}")
        for row in summary:
            print(f"  {row['c_fp']:>10.2f}{row['threshold_mean']:>10.2f}"
                  f"{row['precision_mean']:>11.4f}{row['recall_mean']:>9.4f}"
                  f"{row['f1_mean']:>8.4f}{row['n_predicted_risk_mean']:>10.1f}"
                  f"{row['FN_mean']:>8.1f}{row['FP_mean']:>8.1f}{row['cost_mean']:>9.1f}")
        print()
        print("  读法：C_FP 越小（越怕漏报）→ 选定阈值越低 → recall 上升、名单变长、误报增多。")
        print("        例如 C_FP=0.05 相当于『漏报 1 人 ≈ 误报 20 人』，"
              "适合人力充足、宁可多报的期末排查场景。")
        print()
        print("  对照（错误做法，仅用于说明差距）：若直接在测试集上挑最优阈值，"
              "得到的代价会系统性偏低：")
        for row in naive:
            fixed = next((x for x in summary if x["c_fp"] == row["c_fp"]), None)
            if fixed:
                print(f"    C_FP={row['c_fp']:<5} 测试侧挑点={row['cost_mean_if_tuned_on_test']:>8.1f}"
                      f"   训练侧选点={fixed['cost_mean']:>8.1f}"
                      f"   （低 {row['cost_mean_if_tuned_on_test'] - fixed['cost_mean']:+.1f}）")
        print()
        print(f"  已保存：results/metrics/cost_sensitive.csv 与 cost_sensitive.json")
        print("=" * 78)
    return out


def main() -> None:
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    run_cost_sensitive()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
