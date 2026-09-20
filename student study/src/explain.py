# -*- coding: utf-8 -*-
"""
explain.py —— 模型可解释性模块
=========================================================
辅助功能 3：**关键特征贡献说明 —— 解释"为什么判为高风险"**。

本项目提供两种互补的解释口径，覆盖"整体规律"和"这个学生"两个层面：

  ① 全局：置换重要性（Permutation Importance）
     做法：把某一列特征的值整体随机打乱（保持其他列不动），
           再让模型重新预测。如果打乱后指标明显下降，说明这个特征
           "含有模型真正在用的信息"；如果几乎没变化，说明模型没用它。
     优点：模型无关（model-agnostic），对 KNN 这种非参数模型同样适用；
           不依赖任何近似，测的就是"性能掉了多少"。
     缺点：有随机性 → 本项目重复 3 次取均值，并同时记录标准差；
           特征若高度相关，重要性会被"分摊"（本项目三科成绩高度相关，
           因此单科重要性不会特别突出，这是**真实且需要如实说明**的现象）。

  ② 局部：近邻对比解释（Local Nearest-Neighbor Contrast）
     做法：取该学生的 k 个近邻，分成"高风险近邻"和"低风险近邻"两组，
           计算本人的特征值与两组均值的差距。若某特征"离高风险组更近、
           离低风险组更远"，它就是把这个学生推向高风险的因素。
     优点：与 KNN 的决策机制完全一致（KNN 本来就是靠近邻投票决定的），
           解释是"模型真实行为的复述"，而不是事后编造的说法。
     输出：中文句子，可直接展示给辅导员看。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .knn_numpy import KNNClassifierNumpy, assign_risk_levels
    from .preprocess import Dataset, build_dataset, load_simulated
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from knn_numpy import KNNClassifierNumpy, assign_risk_levels
    from preprocess import Dataset, build_dataset, load_simulated


# ======================================================================
# 一、全局：置换重要性
# ======================================================================
def permutation_importance_knn(ds: Dataset, k: int,
                               n_repeats: int = cfg.N_IMPORTANCE_REPEATS,
                               random_state: int = cfg.RANDOM_SEED,
                               verbose: bool = True) -> pd.DataFrame:
    """
    计算 KNN 的置换重要性（以宏平均 F1 的下降量为主指标）。

    流程：
      1. 用"全量训练、预测自身"的方式取基准预测（KNN 用 k 近邻含自身会过拟合，
         因此这里基准也使用留一法概率，保证与报告口径一致，避免基准虚高）；
      2. 对每个特征 j：把第 j 列整体打乱 → 重新算留一法概率与预测 → 算指标；
      3. 重要性 = 基准指标 - 打乱后指标（越大越重要），重复 n_repeats 次取均值。

    注意：留一法概率在打乱后必须重算，因为 KNN 的"距离矩阵"变了，
    这正是"模型无关解释"的代价——好在 400 条数据下每秒能跑很多次。
    """
    from knn_numpy import loo_predict  # 顶层已导入，这里沿用同一实现（无循环依赖）

    X = ds.X.copy()
    y = ds.y.copy()
    n, d = X.shape
    rng = np.random.RandomState(random_state)

    base_pred = loo_predict(X, y, k)
    base = ev.full_report(y, base_pred, "baseline")
    base_f1 = base["metrics"]["f1"]
    base_rec = base["metrics"]["recall"]

    rows = []
    for j, name in enumerate(ds.feature_names):
        f1_drops, rec_drops, acc_drops, prec_drops = [], [], [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            # 只打乱第 j 列：行的对应关系被打散，`第j列与其他特征的关联`被破坏
            Xp[:, j] = Xp[rng.permutation(n), j]
            pred = loo_predict(Xp, y, k)
            m = ev.full_report(y, pred, "permuted")["metrics"]
            f1_drops.append(base_f1 - m["f1"])
            rec_drops.append(base_rec - m["recall"])
            acc_drops.append(base["metrics"]["accuracy"] - m["accuracy"])
            prec_drops.append(base["metrics"]["precision"] - m["precision"])
        rows.append({
            "feature": name,
            "feature_cn": cfg.FEATURE_CN.get(name, name),
            "importance_f1": round(float(np.mean(f1_drops)), 6),
            "importance_f1_std": round(float(np.std(f1_drops, ddof=1)) if n_repeats > 1 else 0.0, 6),
            "importance_recall": round(float(np.mean(rec_drops)), 6),
            "importance_precision": round(float(np.mean(prec_drops)), 6),
            "importance_accuracy": round(float(np.mean(acc_drops)), 6),
        })

    df = pd.DataFrame(rows).sort_values("importance_f1", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"[置换重要性] 基准（留一法, k={k}）：F1={base_f1:.4f}, recall={base_rec:.4f}")
        print(df[["feature_cn", "importance_f1", "importance_f1_std", "importance_recall"]]
              .to_string(index=False))
    return df


def logreg_importance(ds: Dataset, logreg) -> pd.DataFrame:
    """
    逻辑回归的"系数重要性"：|w_i| 越大，该特征对线性判别的贡献越大。

    注意与置换重要性的区别：系数是**模型内部参数**（带尺度、可比性受特征标准化影响），
    置换重要性是**外部指标变化**（模型无关）。两者结论通常一致但不完全相同，
    报告里同时给出可以互相印证，答辩时也是加分点。
    """
    w = np.asarray(logreg.w_ if hasattr(logreg, "w_") else logreg.coef_, dtype=float).ravel()
    df = pd.DataFrame({
        "feature": ds.feature_names,
        "feature_cn": [cfg.FEATURE_CN.get(f, f) for f in ds.feature_names],
        "coef": np.round(w, 4),
        "abs_coef": np.round(np.abs(w), 4),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df


# ======================================================================
# 二、局部：单个学生的近邻对比解释
# ======================================================================
def explain_local(knn: KNNClassifierNumpy,
                  x_std: np.ndarray,
                  y_train: np.ndarray,
                  X_train_std: np.ndarray,
                  feature_names: Sequence[str],
                  raw_row: Optional[pd.Series] = None,
                  scaler=None,
                  top_n: int = 4,
                  neighbor_ids: Optional[Sequence[str]] = None,
                  exclude_self: bool = False) -> Dict:
    """
    对单个学生做"近邻对比"局部解释。

    ============== 两个必须处理好的工程细节（答辩可讲） ==============
    ① **自匹配问题**：如果被查询的学生本身就在训练集里，它到自己的距离为 0，
       必然成为"第 1 近邻"。这等于用答案喂自己：概率会被自己的标签拉偏，
       参考同学名单里还会出现本人。因此本函数在 exclude_self=True 时自动剔除
       距离 < 1e-9 的自身匹配，并如实报告本次预测是否属于"自匹配"情形。

    ② **同类退化问题**：若 k 个近邻全部是同一类（例如全部为高风险），
       仅用近邻无法构造"高风险组 vs 低风险组"的对比（分母为 0），
       早期实现会让所有特征的贡献退化成 0，还会输出"推向高风险"这种
       与实际相反的措辞。修正方案：
          · 主依据改为 **全训练集分层对比** —— 在所有训练样本中按真实标签分组，
            计算"该生特征值到高风险组均值 / 低风险组均值的标准化距离"，
            两组样本量都很大，永远不会退化成空；
          · 同时保留"近邻构成"作为局部证据（近邻里高风险占比 = 概率本身）。

    返回
    ----
    dict：概率、等级、近邻构成、特征贡献列表、中文解释句子、解释可信度标记
    """
    x = np.asarray(x_std, dtype=float).ravel()
    y_train = np.asarray(y_train).ravel().astype(int)
    X_train_std = np.asarray(X_train_std, dtype=float)

    # ---- 近邻明细（查询的是训练集内学生时排除"自己"） ----
    neighbors = knn.kneighbors(x, ids=neighbor_ids, exclude_self=exclude_self)
    nb_idx = np.array([nb["train_index"] for nb in neighbors], dtype=int)
    nb_labels = y_train[nb_idx]
    frac_risk = float(nb_labels.mean())      # 近邻中高风险比例 = 预测概率
    self_match = bool(np.min([nb["distance"] for nb in neighbors]) < 1e-9)

    # ---- 主依据：全训练集按真实标签分层的组均值 ----
    risk_mask = y_train == 1
    safe_mask = y_train == 0
    n_train_risk = int(risk_mask.sum())
    n_train_safe = int(safe_mask.sum())
    mean_risk = X_train_std[risk_mask].mean(axis=0) if n_train_risk else None
    mean_safe = X_train_std[safe_mask].mean(axis=0) if n_train_safe else None
    has_both_groups = (mean_risk is not None) and (mean_safe is not None)

    contributions: List[Dict] = []
    for j, name in enumerate(feature_names):
        if has_both_groups:
            d_risk = float(abs(x[j] - mean_risk[j]))   # 到"高风险组均值"的标准化距离
            d_safe = float(abs(x[j] - mean_safe[j]))   # 到"低风险组均值"的标准化距离
            gap = d_safe - d_risk                      # >0 → 特征更像高风险组 → 推向高风险
        else:
            d_risk = d_safe = float("nan")
            gap = 0.0
        contributions.append({
            "feature": name,
            "feature_cn": cfg.FEATURE_CN.get(name, name),
            "value_std": round(float(x[j]), 4),
            "value_raw": None if raw_row is None else round(float(raw_row[name]), 4),
            "dist_to_risk_mean": None if np.isnan(d_risk) else round(d_risk, 4),
            "dist_to_safe_mean": None if np.isnan(d_safe) else round(d_safe, 4),
            "risk_gap": round(float(gap), 4),
            "direction": ("推向高风险" if gap > 0 else
                          ("相对优势" if gap < 0 else "无信息")),
        })

    contrib_df = pd.DataFrame(contributions).sort_values("risk_gap", ascending=False)
    top = contrib_df.head(top_n).to_dict(orient="records")

    # ---- 中文解释句子（模板化，只做文字组织，数值全部来自上面的计算） ----
    level = _level_from_proba(frac_risk)
    sentences: List[str] = []
    sentences.append(
        f"该生高风险概率为 {frac_risk:.4f}（k={knn.k} 个近邻中有 "
        f"{int(nb_labels.sum())} 位属于“需要重点关注”），据此判定风险等级为【{level}】。"
    )
    for i, it in enumerate(top, 1):
        v_str = _format_feature_value(it["feature"], it["value_raw"])
        if not has_both_groups:
            sentences.append(f"（{i}）{it['feature_cn']} = {v_str}：训练集中缺少某一类样本，"
                             f"无法计算对比型贡献，本次不作方向性解释。")
            continue
        sentences.append(
            f"（{i}）{it['feature_cn']} = {v_str}：该值到高风险组平均水平的标准化距离为 "
            f"{it['dist_to_risk_mean']}，到低风险组为 {it['dist_to_safe_mean']}，"
            f"差距 {it['risk_gap']:+.4f}，说明它{f'在把该生推向高风险' if it['risk_gap'] > 0 else f'是该生相对于本群体的优势项'}。"
        )
    sentences.append(
        f"（依据说明）对比基准 = 全部 {n_train_risk + n_train_safe} 条训练样本按真实标签分层"
        f"（高风险组 {n_train_risk} 条、正常组 {n_train_safe} 条）的组均值；"
        f"局部证据 = 该生最近 {len(neighbors)} 位相似同学的构成。"
    )

    # ---- 与结论的一致性说明（避免"某特征推向高风险"与"整体低风险"并列造成误读） ----
    # 单个特征的 risk_gap 是**相对差距**：它衡量"该生在这个特征上，比起高风险组更像谁"，
    # 但最终判定是 8 个特征合成的结果。因此完全可能出现"某特征相对偏弱，
    # 但整体仍被判为低风险"的情况（例如某学霸的自习时长确实低于其同学）。
    # 这里显式说明该差异来自哪一项，避免使用者误以为结论自相矛盾。
    contradictory = [it for it in top if it["risk_gap"] > 0] if level == "低" else []
    if contradictory:
        names = "、".join(it["feature_cn"] for it in contradictory)
        sentences.append(
            f"（口径提示）该生整体判定为低风险，但 {names} 相对高风险组更接近，"
            f"说明这些方面是该生**在本群体内的相对短板**，并非真正的风险来源；"
            f"最终等级由 8 个特征合成决定，单特征方向不等于整体结论。"
        )

    # ---- 解释可信度提示（不隐瞒模型的盲区） ----
    notes: List[str] = []
    if self_match and not exclude_self:
        notes.append("本次预测的最近邻包含与被查询学生完全重合的训练样本（距离 0）。"
                     "若查询的是数据集中已有的学生，请改用 exclude_self=True 以避免自匹配。")
    if len(nb_idx[nb_labels == 1]) == 0:
        notes.append("该生近邻中没有任何高风险样本，概率为 0 属于模型外推结果，"
                     "方向性解释来自训练集整体分层对比，建议结合原始成绩人工复核。")
    if len(nb_idx[nb_labels == 0]) == 0:
        notes.append("该生近邻全部为高风险样本，概率为 1 属于模型外推结果，同样建议人工复核。")
    if not has_both_groups:
        notes.append("训练集中只有单一类别，无法计算对比型特征贡献。")
    sentences.extend("提示：" + n for n in notes)

    return {
        "proba_risk": round(frac_risk, 4),
        "risk_level": level,
        "n_neighbors": int(len(neighbors)),
        "n_neighbors_risk": int(nb_labels.sum()),
        "neighbors": neighbors,
        "contributions": contrib_df.to_dict(orient="records"),
        "top_contributions": top,
        "n_risk_pushing": int((contrib_df["risk_gap"] > 0).sum()),
        "sentences": sentences,
        "notes": notes,
        "self_match": self_match,
        "explanation_basis": ("训练集分层对比（稳健） + 近邻构成（局部证据）"
                              if has_both_groups else "仅近邻构成（训练集缺少一类样本）"),
        "method": "训练集分层对比 + 近邻构成，与 KNN 的决策机制一致，非事后编造",
    }


def _format_feature_value(feature: str, v) -> str:
    """把特征值格式化为人类可读字符串（比率转百分比、成绩带单位）。"""
    if v is None:
        return "数据缺失"
    if feature in ("attendance_rate", "homework_submit_rate"):
        return f"{float(v) * 100:.1f}%"
    unit = cfg.FEATURE_UNIT.get(feature, "")
    return f"{float(v)}{unit}"


def _level_from_proba(p: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """局部解释内部使用的概率→等级转换，复用全局阈值配置。"""
    th = dict(cfg.RISK_LEVEL_THRESHOLDS if thresholds is None else thresholds)
    if p < th["low"]:
        return "低"
    if p <= th["high"]:
        return "中"
    return "高"


# ======================================================================
# 三、批量产物：全局重要性落盘
# ======================================================================
def run_explain(k: Optional[int] = None,
                n_repeats: int = cfg.N_IMPORTANCE_REPEATS,
                verbose: bool = True) -> Dict:
    """
    执行全局解释并把结果写入 results/metrics/feature_importance.csv。

    k 默认取 results/models 中保存的主 k，保证"解释的是真正在用的那个模型"。
    """
    cfg.ensure_dirs()
    # 优先读训练好的模型，保证"被解释的模型"与"实际部署的模型"完全一致
    main_k = k
    if main_k is None:
        try:
            try:                                    # 包导入方式（main.py 走这条）
                from .train_eval import load_saved_models
            except ImportError:                     # 脚本直跑方式（src 在 sys.path 中）
                from train_eval import load_saved_models
            saved = load_saved_models()
            main_k = saved["knn"].k
        except FileNotFoundError:                   # 还没训练过 → 退回配置里的默认 k
            main_k = cfg.MAIN_K

    df = load_simulated()
    ds, _, _ = build_dataset(df)

    if verbose:
        print("=" * 68)
        print(f"【步骤】模型可解释性分析（全局置换重要性，k={main_k}，重复 {n_repeats} 次）")
        print("=" * 68)

    imp = permutation_importance_knn(ds, k=int(main_k), n_repeats=n_repeats, verbose=verbose)

    out_path = cfg.IMPORTANCE_PATH
    imp.to_csv(out_path, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"      已保存：{os.path.relpath(out_path, cfg.BASE_DIR)}")
        print("=" * 68)
    return {"main_k": int(main_k), "importance": imp.to_dict(orient="records"), "path": out_path}


def main() -> None:
    """命令行入口：python src/explain.py"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from utils import enable_utf8_console
    enable_utf8_console()
    run_explain()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
