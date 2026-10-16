# -*- coding: utf-8 -*-
"""
dashboard_v3.py —— 看板 v3：新增图表（建议 A5、C2、A2、A1）

本模块只做**新增**图表，不改动已有图（11~15 保持不变）：
    16_uci_correlation_heatmap.png   A5 特征相关性热力图（识别冗余/共线结构）
    17_cost_curve.png                C2 代价敏感曲线（按人力选工作点）
    18_calibration.png               A2 概率校准（可靠性曲线 + Brier + 分档人数恢复）
    19_model_comparison.png          A1 多模型对比矩阵

设计原则：
  · 与已有图风格一致（中文字体、脱敏声明、基线显式标注）；
  · 图注必须写清"该怎么读、不能怎么读"，避免误读（尤其热力图的因果解读与代价比的假设性质）；
  · 数据来源全部是 results/metrics 下的产物，缺失时给出明确提示而不是崩溃。
"""
from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import config as cfg
    from . import evaluate as ev
    from .dashboard import _t, setup_chinese_font, _save
    from .uci_features import feature_names, feature_name_cn
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from dashboard import _t, setup_chinese_font, _save
    from uci_features import feature_names, feature_name_cn


# ======================================================================
# A5：特征相关性热力图
# ======================================================================
def fig_correlation_heatmap(df: Optional[pd.DataFrame] = None,
                            out_name: str = "16_uci_correlation_heatmap.png") -> str:
    """
    特征相关性热力图（默认 Pearson）。

    读图要点（写在图注里，避免误读）：
      · 大量分类字段的**互斥指示列**天然负相关（如各职业指示列），会在图上呈块状结构 ——
        这说明 one-hot 编码引入了结构性冗余，是"固定 schema 编码"的已知代价；
      · **相关系数高 ≠ 模型会失效**：KNN 对共线不敏感，逻辑回归有 L2 正则也基本吃得消；
      · 本项目**不做因果解释**：饮酒（Dalc/Walc）与成绩的相关**不能**读作"饮酒导致成绩差"。
    """
    if df is None:
        df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
    names = feature_names(include_early_grades=False)
    X = np.zeros((len(df), len(names)), dtype=float)

    try:
        from .uci_features import (CATEGORICAL_LEVELS, CATEGORICAL_REFERENCE,
                                   NUM_FEATURES, YN_FEATURES)
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from uci_features import (CATEGORICAL_LEVELS, CATEGORICAL_REFERENCE,
                                  NUM_FEATURES, YN_FEATURES)
    for c in NUM_FEATURES:
        X[:, names.index(c)] = df[c].to_numpy(dtype=float)
    for col, levels in CATEGORICAL_LEVELS.items():
        vals = df[col].astype(str).to_numpy()
        for lv in levels:
            if lv == CATEGORICAL_REFERENCE[col]:
                continue
            X[:, names.index("%s_%s" % (col, lv))] = (vals == lv).astype(float)
    for c in YN_FEATURES:
        X[:, names.index(c)] = (df[c].astype(str).str.lower() == "yes").astype(float)

    # Pearson 与 Spearman 各算一份：age/Medu/studytime 是**有序等级量表**，
    # 严格说 Spearman 更合适，因此两张并排展示，避免只用 Pearson 被质疑。
    C_pear = np.corrcoef(X, rowvar=False)
    # Spearman = 对秩做 Pearson
    Xr = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, X)
    C_spear = np.corrcoef(Xr, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    short = [feature_name_cn(n)[:14] for n in names]
    for ax, C, title in (
            (axes[0], C_pear, _t("Pearson 相关系数（等距假设）", "Pearson")),
            (axes[1], C_spear, _t("Spearman 秩相关（适合等级量表）", "Spearman"))):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(short)), short, rotation=90, fontsize=6)
        ax.set_yticks(range(len(short)), short, fontsize=6)
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label=_t("相关系数", "correlation"))

    n_hi = int((np.abs(C_pear) > 0.7).sum() - len(names)) // 2
    fig.suptitle(
        _t("图16 特征相关性热力图（红=正相关，蓝=负相关）\n"
           f"|r|>0.7 的特征对约 {max(n_hi, 0)} 组，主要来自分类字段互斥指示列的结构性冗余；"
           "共线不等于模型失效，且本项目不做因果解释",
           "Fig.16 Feature correlation"),
        fontsize=11)
    return _save(fig, out_name)


# ======================================================================
# C2：代价敏感曲线
# ======================================================================
def fig_cost_curve(threshold_table: Optional[pd.DataFrame] = None,
                   costs_ratio: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0),
                   out_name: str = "17_cost_curve.png") -> str:
    """
    代价敏感曲线：把 precision/recall 的兑换关系换算成"总代价 vs 判定阈值"。

    【数学定义】总代价 = C_FN × FN + C_FP × FP。
    令 C_FN = 1（漏报一名学生的代价设为基准），C_FP = r ∈ {0.1, 0.2, 0.5, 1}，
    则 r 越小表示"宁可多报不可漏报"，最优阈值应越小。

    【必须在图注里声明的两件事】
      ① 代价比 r 是**场景假设**，不是从数据里算出来的测量值
         （"漏报一名学生的真实代价"属于教育决策问题，数据集里没有）；
      ② 图上曲线可用于展示，但**工作点必须由训练侧选出**，
         不能用测试集挑阈值（本项目 `_tune_threshold_within_training` 即按此实现）。
    """
    p = cfg.METRICS_DIR + os.sep + "cost_sensitive.csv"
    if threshold_table is None:
        if os.path.exists(p):
            threshold_table = pd.read_csv(p, encoding="utf-8-sig")
        elif os.path.exists(cfg.METRICS_DIR + os.sep + "threshold_sweep.csv"):
            threshold_table = pd.read_csv(cfg.METRICS_DIR + os.sep + "threshold_sweep.csv",
                                          encoding="utf-8-sig")
        else:
            raise FileNotFoundError("缺少阈值扫描结果，请先运行 python main.py --step train")

    # 统一列名：cost_sensitive.csv 用 threshold/TP/FP/FN；
    # threshold_sweep.csv 只有指标 → 无法还原代价，必须用前者。
    # 注意 df 索引：`t["threshold"].iloc[k]` 与 `t.iloc[k]["threshold"]` 都可用，
    # 但 `t["threshold"].to_numpy()[k]` 才是 ndarray 下标，避免混用导致
    # AttributeError: 'numpy.ndarray' object has no attribute 'iloc'。
    if {"TP", "FP", "FN"}.issubset(threshold_table.columns):
        t = threshold_table.reset_index(drop=True)
    else:
        raise ValueError("阈值表缺少 TP/FP/FN 列，请改用 cost_sensitive.py 生成的 cost_sensitive.csv")
    ths = t["threshold"].to_numpy(dtype=float)
    fn_arr = t["FN"].to_numpy(dtype=float)
    fp_arr = t["FP"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # 左：不同代价比下的总代价曲线
    ax = axes[0]
    for r in costs_ratio:
        cost = fn_arr * 1.0 + fp_arr * r
        ax.plot(ths, cost, "o-", markersize=4, linewidth=1.6,
                label=_t(f"C_FP/C_FN = {r}", f"r={r}"))
        k = int(np.argmin(cost))
        ax.plot(ths[k], cost[k], "*", markersize=14, color="black", zorder=5)
        ax.annotate(f"{ths[k]:.2f}", (ths[k], cost[k]),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel(_t("判定阈值", "threshold"), fontsize=10)
    ax.set_ylabel(_t("总代价（C_FN=1 归一化）", "total cost"), fontsize=10)
    ax.set_title(_t("总代价随阈值变化（★ 为各代价比下的最优工作点）",
                    "Cost vs threshold"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    # 右：代价比 → 最优阈值 的映射（给辅导员读的结论）
    ax = axes[1]
    best = []
    for r in costs_ratio:
        cost = fn_arr * 1.0 + fp_arr * r
        k = int(np.argmin(cost))
        best.append((r, float(ths[k]), int(fn_arr[k]), int(fp_arr[k])))
    xs = np.arange(len(best))
    ax.bar(xs, [b[1] for b in best], color="#2E75B6", edgecolor="white", width=0.55)
    for i, b in enumerate(best):
        ax.text(i, b[1] + 0.015, f"阈值 {b[1]:.2f}\n漏报 {b[2]} / 误报 {b[3]}",
                ha="center", fontsize=8)
    ax.set_xticks(xs, [f"{b[0]}" for b in best], fontsize=9)
    ax.set_xlabel(_t("误报代价 C_FP（漏报代价固定为 1）", "C_FP (C_FN=1)"), fontsize=10)
    ax.set_ylabel(_t("最优判定阈值", "optimal threshold"), fontsize=10)
    ax.set_title(_t("人力越充足（越怕漏报）→ 阈值应越低", "Cost ratio → threshold"), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    fig.suptitle(
        _t("图17 代价敏感分析：把 precision/recall 的兑换关系换算成工作量\n"
           "代价比为场景假设（非数据测量值）；工作点必须由训练侧选出，不能用测试集挑阈值",
           "Fig.17 Cost-sensitive analysis"),
        fontsize=11)
    return _save(fig, out_name)


# ======================================================================
# A2：概率校准
# ======================================================================
def _reliability_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """计算可靠性曲线：分箱内的"预测均值 vs 实际正类率"。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y_true[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig_calibration(out_name: str = "18_calibration.png") -> str:
    """
    概率校准图：校准前 vs 校准后。

    含三块：
      ① 可靠性曲线（预测概率 vs 实际正类率，越贴近对角线越好）
      ② 概率分布直方图（校准前 k=1 时只会在 0/1 两端有柱子 → 直观展示"概率退化"）
      ③ 分档人数对比（校准前只有两档、校准后恢复三档）
    """
    path = os.path.join(cfg.METRICS_DIR, "calibration.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}，请先运行 python src/calibration.py")
    with open(path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    y = np.asarray(cal["y_true"], dtype=int)
    p_raw = np.asarray(cal["proba_raw"], dtype=float)
    p_cal = np.asarray(cal["proba_calibrated"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # ① 可靠性曲线
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label=_t("理想校准", "perfect"))
    for p, name, color in ((p_raw, _t("校准前（k=1）", "raw"), "#C00000"),
                           (p_cal, _t("校准后", "calibrated"), "#2E75B6")):
        xs, ys, ns = _reliability_curve(y, p)
        if len(xs):
            ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=6,
                    label=f"{name}（Brier={cal['brier'][('raw' if color == '#C00000' else 'calibrated')]:.4f}）")
    ax.set_xlabel(_t("预测概率（分箱均值）", "predicted"), fontsize=10)
    ax.set_ylabel(_t("实际高风险比例", "observed"), fontsize=10)
    ax.set_title(_t("可靠性曲线", "Reliability"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    # ② 概率分布
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(p_raw, bins=bins, alpha=0.65, color="#C00000",
            label=_t("校准前", "raw"), edgecolor="white")
    ax.hist(p_cal, bins=bins, alpha=0.65, color="#2E75B6",
            label=_t("校准后", "calibrated"), edgecolor="white")
    ax.axvline(cfg.RISK_LEVEL_THRESHOLDS["low"], color="orange", linestyle="--",
               linewidth=1.5, label=_t(f"低/中阈值 {cfg.RISK_LEVEL_THRESHOLDS['low']}", "th_low"))
    ax.axvline(cfg.RISK_LEVEL_THRESHOLDS["high"], color="darkred", linestyle="--",
               linewidth=1.5, label=_t(f"中/高阈值 {cfg.RISK_LEVEL_THRESHOLDS['high']}", "th_high"))
    ax.set_xlabel(_t("高风险概率", "probability"), fontsize=10)
    ax.set_ylabel(_t("样本数", "count"), fontsize=10)
    ax.set_title(_t("概率分布：校准前只挤在 0/1 两端", "Probability distribution"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # ③ 分档人数
    ax = axes[2]
    levels = cfg.RISK_LEVELS
    raw_counts = cal["level_counts_raw"]
    # 校准后的分档要用**重标定后的阈值**结果：默认 0.3/0.7 是为未校准概率设的，
    # 校准把概率压缩到基准率附近后，若不重标定会让"高"档变成 0 人（看起来像失败）
    cal_counts = cal.get("level_counts_calibrated_retuned") or \
        cal.get("level_counts_calibrated", {})
    xs = np.arange(len(levels))
    ax.bar(xs - 0.2, [raw_counts.get(l, 0) for l in levels], 0.4,
           color="#C00000", label=_t("校准前", "raw"))
    ax.bar(xs + 0.2, [cal_counts.get(l, 0) for l in levels], 0.4,
           color="#2E75B6", label=_t("校准后（阈值重标定）", "calibrated (retuned)"))
    for i, l in enumerate(levels):
        ax.text(i - 0.2, raw_counts.get(l, 0), str(raw_counts.get(l, 0)),
                ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, cal_counts.get(l, 0), str(cal_counts.get(l, 0)),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs, levels, fontsize=10)
    ax.set_ylabel(_t("人数", "count"), fontsize=10)
    ax.set_title(_t("分档人数：校准后必须重标定阈值，三档才可用", "Level counts"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    fig.suptitle(
        _t("图18 概率校准（Platt/sigmoid）。校准不改善排序能力，"
           "因此 F1/recall 不会本质变化；它改善的是概率的数值意义，让三档阈值真正可用",
           "Fig.18 Probability calibration"),
        fontsize=11)
    return _save(fig, out_name)


# ======================================================================
# A1：多模型对比
# ======================================================================
def fig_model_comparison(out_name: str = "19_model_comparison.png") -> str:
    """多模型对比矩阵：accuracy / precision / recall / F1 四项并排 + 相对基线。"""
    path = os.path.join(cfg.METRICS_DIR, "model_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}，请先运行 python src/model_comparison.py")
    df = pd.read_csv(path, encoding="utf-8-sig")

    with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    baseline = float(s.get("majority_baseline_accuracy", 0.0))

    metrics = ["accuracy", "precision", "recall", "f1"]
    models = df["model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))

    # 左：四项指标分组柱状图
    ax = axes[0]
    width = 0.8 / max(len(models), 1)
    xs = np.arange(len(metrics))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 2)))
    for i, (m, c) in enumerate(zip(models, colors)):
        sub = df[df["model"] == m].iloc[0]
        vals = [float(sub[k]) for k in metrics]
        ax.bar(xs + (i - (len(models) - 1) / 2) * width, vals, width,
               label=m, color=c, edgecolor="white")
    ax.axhline(baseline, color="black", linestyle=":", linewidth=1.2,
               label=_t(f"多数类基线 accuracy {baseline:.3f}", "baseline"))
    ax.set_xticks(xs, metrics, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
    ax.set_title(_t("各模型在同一口径下的对比（LOGO 留一组交叉验证）",
                    "Model comparison"), fontsize=11)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # 右：以"主模型 KNN"为基准的差值
    ax = axes[1]
    if "KNN(k=1)" in models:
        base_row = df[df["model"] == "KNN(k=1)"].iloc[0]
    else:
        base_row = df.iloc[0]
    labels, deltas = [], []
    for _, row in df.iterrows():
        if row["model"] == base_row["model"]:
            continue
        labels.append(row["model"])
        deltas.append(float(row["f1"]) - float(base_row["f1"]))
    ys = np.arange(len(labels))
    ax.barh(ys, deltas, color=["#2E75B6" if d >= 0 else "#C00000" for d in deltas],
            edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_xlabel(_t(f"F1 差值（相对 {base_row['model']}）", "F1 delta"), fontsize=10)
    ax.set_title(_t("与主模型的 F1 差距（>0 表示优于主模型）", "F1 vs main model"), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", axis="x")

    fig.suptitle(
        _t("图19 模型对比矩阵（用于论证主模型选择，而非仅凭'项目要求'）",
           "Fig.19 Model comparison matrix"),
        fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 总入口
# ======================================================================
def build_all_v3(verbose: bool = True) -> Dict[str, str]:
    """生成 v3 新增图表（16~19），缺失依赖的图会被跳过并给出提示。"""
    cfg.ensure_dirs()
    setup_chinese_font(verbose=verbose)
    if verbose:
        print("=" * 70)
        print("【看板 v3】新增图表（相关性热力图 / 代价曲线 / 概率校准 / 模型对比）")
        print("=" * 70)

    out: Dict[str, str] = {}
    tasks = [
        ("16_uci_correlation_heatmap", fig_correlation_heatmap),
        ("17_cost_curve", fig_cost_curve),
        ("18_calibration", fig_calibration),
        ("19_model_comparison", fig_model_comparison),
    ]
    for name, fn in tasks:
        try:
            p = fn()
            out[name] = p
            if verbose:
                print("  ✔ %s" % os.path.basename(p))
        except FileNotFoundError as e:
            if verbose:
                print("  – 跳过 %s（%s）" % (name, e))
        except Exception as e:  # 单张图失败不应中断整个看板
            if verbose:
                print("  ✗ %s 生成失败：%s: %s" % (name, type(e).__name__, e))
    if verbose:
        print("  共生成 %d 张 → %s" % (len(out), os.path.relpath(cfg.FIGURES_DIR, cfg.BASE_DIR)))
        print("=" * 70)
    return out


def main() -> None:
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    build_all_v3()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
