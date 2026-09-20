# -*- coding: utf-8 -*-
"""
dashboard.py —— 可视化看板（辅助功能 1）
=========================================================
用 matplotlib 生成 6 张图，全部保存到 results/figures/：

    01_score_distribution.png   三科成绩分布（按真实风险标签分层叠加）+ 行为特征分布
    02_risk_level_distribution.png  风险等级分布、高风险比例、概率直方图与风险名单构成
    03_confusion_matrix.png     混淆矩阵（KNN 留一法 / 新样本 / 逻辑回归新样本，三联图）
    04_k_selection.png          k 值对比（留一法 F1/recall + 10 次划分均值±标准差）
    05_feature_importance.png   特征重要性（置换重要性 + 逻辑回归系数）
    06_prediction_analysis.png  预测质量分析（概率分布、错分样本画像、阈值-指标权衡）

【字体说明】中文图表必须指定中文字体，否则会显示成方框。
本模块按 config.CN_FONT_CANDIDATES 逐个尝试，并显式设置 axes.unicode_minus=False
（否则负号也会显示成方框）。找不到中文字体时自动回退英文标签，保证脚本不崩。
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")            # 无界面后端：只出图不弹窗，适合脚本/服务器运行
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from . import config as cfg
    from . import evaluate as ev
    from .preprocess import load_simulated, build_dataset
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from preprocess import load_simulated, build_dataset


# ======================================================================
# 一、中文字体配置
# ======================================================================
def setup_chinese_font(verbose: bool = False) -> str:
    """
    配置 matplotlib 中文字体。

    返回实际使用的字体名；若一个都没找到，返回空字符串，
    此时所有标签会自动切换为英文（见 _t 函数），脚本仍可正常运行。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in cfg.CN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
            if verbose:
                print(f"[字体] 使用中文字体：{name}")
            return name
    plt.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("[字体] 未找到中文字体，图表标签自动切换为英文（不影响数据内容）")
    return ""


_FONT_OK: Optional[bool] = None


def _cn() -> bool:
    """惰性判断当前是否可用中文标签（只判断一次，避免重复扫描字体表）。"""
    global _FONT_OK
    if _FONT_OK is None:
        _FONT_OK = bool(setup_chinese_font(verbose=False))
    return _FONT_OK


def _t(cn: str, en: str) -> str:
    """按字体可用性在中英文标签之间切换。"""
    return cn if _cn() else en


def _save(fig, name: str) -> str:
    """统一保存图片：150 dpi、紧凑布局，便于插入实验报告。"""
    cfg.ensure_dirs()
    path = os.path.join(cfg.FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ======================================================================
# 二、图 1：成绩分布与行为特征分布
# ======================================================================
def fig_score_distribution(df: pd.DataFrame, thresholds: Dict[str, float],
                           out_name: str = "01_score_distribution.png") -> str:
    """
    三科成绩按"真实标签"分层叠加的直方图 + 行为特征分布。

    看图要点（答辩时可以讲的结论）：
      · 蓝（正常）与红（需关注）两条分布**明显重叠**，说明数据不是线性可分的；
      · 重叠区越宽，模型越难做，指标越真实（这也回应"防止指标虚高"的要求）。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    for j, col in enumerate(cfg.SCORE_COLUMNS):
        ax = axes[0, j]
        ax.hist(df.loc[df[cfg.LABEL_COLUMN] == 0, col], bins=25, alpha=0.65,
                label=_t("正常", "Normal"), color="#2E75B6", edgecolor="white")
        ax.hist(df.loc[df[cfg.LABEL_COLUMN] == 1, col], bins=25, alpha=0.65,
                label=_t("需要重点关注", "Needs attention"), color="#C00000", edgecolor="white")
        ax.set_title(f"{col.upper()} {_t('成绩分布（按真实标签分层）', 'score by true label')}", fontsize=11)
        ax.set_xlabel(_t("成绩（分）", "score"), fontsize=9)
        ax.set_ylabel(_t("人数", "count"), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, linestyle="--")

    # 行为特征：用箱线图对比两类学生的差异，比直方图更直观地展示重叠程度
    behaviors = [("attendance_rate", _t("出勤率", "attendance")),
                 ("homework_submit_rate", _t("作业提交率", "homework")),
                 ("self_study_hours", _t("每周自习时长", "study hours"))]
    for j, (col, label) in enumerate(behaviors):
        ax = axes[1, j]
        data = [df.loc[df[cfg.LABEL_COLUMN] == 0, col], df.loc[df[cfg.LABEL_COLUMN] == 1, col]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        labels=[_t("正常", "Normal"), _t("需关注", "Attention")])
        for patch, color in zip(bp["boxes"], ["#2E75B6", "#C00000"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(f"{label} {_t('分布对比', 'distribution')}", fontsize=11)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25, linestyle="--", axis="y")

    fig.suptitle(_t("图1 模拟数据的成绩与学习行为分布（注意两类分布存在明显重叠 → 任务有真实难度）",
                    "Fig.1 Score & behavior distributions (significant overlap → real task difficulty)"),
                 fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 三、图 2：风险等级分布
# ======================================================================
def fig_risk_distribution(df: pd.DataFrame, out_name: str = "02_risk_level_distribution.png") -> str:
    """风险等级分布 + 二分类标签分布 + 高风险比例校验 + 三科平均分随等级的走势。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) 三档等级人数
    ax = axes[0]
    counts = df[cfg.RISK_LEVEL_COLUMN].value_counts().reindex(cfg.RISK_LEVELS).fillna(0)
    colors = ["#70AD47", "#FFC000", "#C00000"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{int(v)}\n({v / len(df) * 100:.1f}%)",
                ha="center", fontsize=9)
    ax.set_title(_t("风险等级分布（三档）", "Risk level distribution"), fontsize=11)
    ax.set_ylabel(_t("人数", "count"), fontsize=9)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # (2) 二分类比例（与 20%~35% 的合规区间对照）
    ax = axes[1]
    n1 = int((df[cfg.LABEL_COLUMN] == 1).sum())
    n0 = len(df) - n1
    ax.pie([n0, n1], labels=[_t(f"正常 {n0} 人", f"Normal {n0}"),
                             _t(f"需关注 {n1} 人", f"Attention {n1}")],
           colors=["#2E75B6", "#C00000"], autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white"})
    ratio = n1 / len(df)
    ok = 0.20 <= ratio <= 0.35
    ax.set_title(_t(f"二分类标签比例（高风险率 {ratio:.1%}，"
                    f"{'符合' if ok else '不符合'} 20%~35% 要求）",
                    f"Binary label ratio ({ratio:.1%})"), fontsize=11)

    # (3) 各等级的三科平均分趋势（验证等级与成绩确实相关）
    ax = axes[2]
    means = [df.loc[df[cfg.RISK_LEVEL_COLUMN] == lv, "score_mean"].mean() for lv in cfg.RISK_LEVELS]
    stds = [df.loc[df[cfg.RISK_LEVEL_COLUMN] == lv, "score_mean"].std() for lv in cfg.RISK_LEVELS]
    ax.errorbar(cfg.RISK_LEVELS, means, yerr=stds, fmt="o-", color="#7030A0",
                capsize=5, markersize=8, linewidth=2)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.1f}", (i, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_title(_t("各风险等级的三科平均分（均值±标准差）", "Mean score by risk level"), fontsize=11)
    ax.set_ylabel(_t("三科平均分", "mean score"), fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle(_t("图2 风险等级与标签分布", "Fig.2 Risk level & label distribution"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 四、图 3：混淆矩阵
# ======================================================================
def _draw_cm(ax, cm: np.ndarray, title: str, metrics: Optional[Dict] = None) -> None:
    """在给定的 ax 上画一个带数值标注的混淆矩阵。"""
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    labels = [_t("正常(0)", "Normal(0)"), _t("需关注(1)", "Risk(1)")]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel(_t("预测类别", "Predicted"), fontsize=9)
    ax.set_ylabel(_t("真实类别", "Actual"), fontsize=9)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            pct = v / total * 100 if total else 0.0
            color = "white" if v > cm.max() * 0.6 else "black"
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    title_full = title
    if metrics:
        title_full += (f"\nacc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f}\n"
                       f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    ax.set_title(title_full, fontsize=10)
    # 标注四格的含义，方便不懂混淆矩阵的人也能看懂
    ax.text(0, -0.72, _t("TN 正确放过", "TN"), ha="center", fontsize=8, color="#555555")
    ax.text(1, -0.72, _t("FP 误报", "FP"), ha="center", fontsize=8, color="#555555")
    ax.text(1, 1.62, _t("TP 正确抓到", "TP"), ha="center", fontsize=8, color="#555555")
    ax.text(0, 1.62, _t("FN 漏报", "FN"), ha="center", fontsize=8, color="#555555")


def fig_confusion_matrices(summary: Dict, out_name: str = "03_confusion_matrix.png") -> str:
    """三联混淆矩阵：KNN 留一法、KNN 新样本、逻辑回归新样本。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    loo = summary.get("loo_knn", {})
    knn_new = summary.get("new_batch_knn", {})
    lr_new = summary.get("new_batch_logreg", {})

    _draw_cm(axes[0], loo.get("confusion_matrix", [[0, 0], [0, 0]]),
             _t(f"KNN(k={summary.get('main_k')}) 留一法\n（全部 {summary.get('n_samples')} 条模拟数据）",
                f"KNN(k={summary.get('main_k')}) LOO"), loo.get("metrics"))
    _draw_cm(axes[1], knn_new.get("confusion_matrix", [[0, 0], [0, 0]]),
             _t(f"KNN(k={summary.get('main_k')}) 新样本测试\n（{summary.get('new_batch_info', {}).get('n', '?')} 条从未参与训练）",
                "KNN new batch"), knn_new.get("metrics"))
    _draw_cm(axes[2], lr_new.get("confusion_matrix", [[0, 0], [0, 0]]),
             _t("逻辑回归（对照）新样本测试", "Logistic regression new batch"), lr_new.get("metrics"))

    fig.suptitle(_t("图3 混淆矩阵对比（行=真实，列=预测；FN 为漏报，是预警系统最需要关注的格子）",
                    "Fig.3 Confusion matrices"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 五、图 4：k 值对比
# ======================================================================
def fig_k_selection(summary: Dict, out_name: str = "04_k_selection.png") -> str:
    """
    k 值对比图：左图留一法指标曲线，右图 10 次随机划分的 F1 均值±标准差。

    结论要点：k 太小时对噪声敏感（指标波动大），k 太大时决策边界被过度平滑
    （欠拟合，recall 往往下降）。因此通常存在一个"中等 k"的综合最优区间。
    """
    ks = [r["k"] for r in summary.get("k_scan", [])]
    if not ks:
        raise RuntimeError("summary 中缺少 k_scan，无法绘制 k 值对比图")

    def col(name: str) -> List[float]:
        return [r[name] for r in summary["k_scan"]]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    ax.plot(ks, col("loo_f1"), "o-", color="#C00000", label="F1", linewidth=2)
    ax.plot(ks, col("loo_recall"), "s--", color="#ED7D31", label=_t("召回率 recall", "recall"))
    ax.plot(ks, col("loo_precision"), "^--", color="#2E75B6", label=_t("精确率 precision", "precision"))
    ax.plot(ks, col("loo_accuracy"), "d:", color="#70AD47", label=_t("准确率 accuracy", "accuracy"))
    ax.axvline(summary.get("main_k", cfg.MAIN_K), color="gray", linestyle="-.", alpha=0.7)
    ax.annotate(f"k={summary.get('main_k')}", (summary.get("main_k", cfg.MAIN_K), ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 6), fontsize=9, color="gray")
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
    ax.set_title(_t("留一法：不同 k 的指标曲线", "LOO metrics vs k"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    ax.errorbar(ks, col("split_f1_mean"), yerr=col("split_f1_std"), fmt="o-",
                color="#7030A0", capsize=4, linewidth=2)
    ax.axvline(summary.get("main_k", cfg.MAIN_K), color="gray", linestyle="-.", alpha=0.7)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_title(_t("10 次随机划分：F1 均值 ± 标准差", "10 random splits: F1 mean±std"), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    ax.plot(ks, col("split_f1_std"), "o-", color="#ED7D31", linewidth=2, label="F1")
    ax.plot(ks, col("split_acc_std"), "s--", color="#2E75B6", linewidth=2,
            label=_t("准确率 accuracy", "accuracy"))
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_t("标准差（越小越稳定）", "std (lower is steadier)"), fontsize=10)
    ax.set_title(_t("稳定性：指标波动随 k 的变化", "Stability vs k"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle(_t("图4 k 值选择依据（主 k 由留一法 F1 + 划分稳定性共同决定）",
                    "Fig.4 k selection"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 六、图 5：特征重要性
# ======================================================================
def fig_feature_importance(imp: pd.DataFrame, coef: Optional[pd.DataFrame] = None,
                           out_name: str = "05_feature_importance.png") -> str:
    """
    特征重要性：左图置换重要性（F1 下降量，含误差棒），右图逻辑回归系数（对照印证）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    ax = axes[0]
    order = imp.sort_values("importance_f1")
    colors = ["#C00000" if v > 0 else "#A6A6A6" for v in order["importance_f1"]]
    ax.barh(order["feature_cn"], order["importance_f1"], xerr=order["importance_f1_std"],
            color=colors, capsize=3, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(_t("打乱该特征后宏 F1 的下降量（越大越重要）", "ΔF1 after permutation"), fontsize=10)
    ax.set_title(_t("置换重要性（模型无关，重复 3 次取均值）", "Permutation importance"), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", axis="x")

    if coef is not None and len(coef):
        ax = axes[1]
        order2 = coef.sort_values("coef")
        colors2 = ["#C00000" if v > 0 else "#2E75B6" for v in order2["coef"]]
        ax.barh(order2["feature_cn"], order2["coef"], color=colors2, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(_t("逻辑回归系数（标准化特征下可比）", "logreg coefficient"), fontsize=10)
        ax.set_title(_t("对照：逻辑回归系数（红=推向高风险，蓝=拉向正常）",
                        "Logreg coefficients"), fontsize=11)
        ax.grid(alpha=0.25, linestyle="--", axis="x")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, _t("未找到逻辑回归系数文件\n请先运行 python main.py --step train",
                                  "logreg coef not available"), ha="center", va="center", fontsize=11)

    fig.suptitle(_t("图5 特征贡献说明：哪些特征真正在影响预测结果",
                    "Fig.5 Feature contribution"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 七、图 6：预测质量分析
# ======================================================================
def fig_prediction_analysis(summary: Dict, pred_df: Optional[pd.DataFrame] = None,
                            new_pred_df: Optional[pd.DataFrame] = None,
                            thresholds: Optional[Dict[str, float]] = None,
                            out_name: str = "06_prediction_analysis.png") -> str:
    """
    预测质量分析：
      (1) 高风险概率直方图（按真实标签着色）+ 低/中/高等级阈值线；
      (2) 错分样本的画像（FN 漏报 vs FP 误报在各特征上的均值对比）；
      (3) 阈值-指标权衡曲线（precision/recall/F1 随判定阈值变化）。
    """
    th = dict(cfg.RISK_LEVEL_THRESHOLDS if thresholds is None else thresholds)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # (1) 概率分布
    ax = axes[0]
    if pred_df is not None and len(pred_df):
        y_true = pred_df[cfg.LABEL_COLUMN].to_numpy()
        p = pred_df["loo_proba_risk"].to_numpy()
        ax.hist(p[y_true == 0], bins=20, alpha=0.65, color="#2E75B6",
                label=_t("真实正常", "true normal"), edgecolor="white")
        ax.hist(p[y_true == 1], bins=20, alpha=0.65, color="#C00000",
                label=_t("真实需关注", "true risk"), edgecolor="white")
        ax.axvline(th["low"], color="orange", linestyle="--", linewidth=1.5,
                   label=_t(f"低/中阈值 {th['low']}", f"th_low={th['low']}"))
        ax.axvline(th["high"], color="darkred", linestyle="--", linewidth=1.5,
                   label=_t(f"中/高阈值 {th['high']}", f"th_high={th['high']}"))
        ax.set_xlabel(_t("高风险概率（留一法）", "risk probability (LOO)"), fontsize=10)
        ax.set_ylabel(_t("人数", "count"), fontsize=10)
        ax.set_title(_t("概率分布：重叠区就是模型的困难区", "Probability distribution"), fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, _t("缺少全量预测文件", "no prediction file"), ha="center", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # (2) 错分样本画像
    ax = axes[1]
    if pred_df is not None and len(pred_df):
        feats = ["score_mean", "attendance_rate", "homework_submit_rate", "self_study_hours"]
        feats_cn = [cfg.FEATURE_CN[f] for f in feats]
        # 为了让四个量纲不同的特征同图比较，用"各特征的全样本均值"做归一化
        norm = pred_df[feats].mean()
        groups = {
            _t("FN 漏报（真需关注却判正常）", "FN"): pred_df[(pred_df[cfg.LABEL_COLUMN] == 1) &
                                                          (pred_df["loo_pred"] == 0)],
            _t("FP 误报（真正常却判需关注）", "FP"): pred_df[(pred_df[cfg.LABEL_COLUMN] == 0) &
                                                          (pred_df["loo_pred"] == 1)],
            _t("正确判定", "Correct"): pred_df[pred_df["pred_correct"] == 1],
        }
        width = 0.26
        xs = np.arange(len(feats))
        for i, (name, g) in enumerate(groups.items()):
            if len(g) == 0:
                continue
            vals = (g[feats].mean() / norm).values
            ax.bar(xs + (i - 1) * width, vals, width, label=f"{name} (n={len(g)})")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.9)
        ax.set_xticks(xs, feats_cn, fontsize=8, rotation=12)
        ax.set_ylabel(_t("相对全样本均值的倍数", "ratio to overall mean"), fontsize=10)
        ax.set_title(_t("错分样本画像（越接近 1 越像普通学生→越难判）",
                        "Misclassified profile"), fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # (3) 阈值权衡
    ax = axes[2]
    sweep = summary.get("threshold_sweep", [])
    if sweep:
        t = [r["threshold"] for r in sweep]
        ax.plot(t, [r["precision"] for r in sweep], "o-", color="#2E75B6",
                label=_t("精确率 precision", "precision"))
        ax.plot(t, [r["recall"] for r in sweep], "s-", color="#ED7D31",
                label=_t("召回率 recall", "recall"))
        ax.plot(t, [r["f1"] for r in sweep], "^-", color="#C00000",
                label="F1", linewidth=2)
        best = max(sweep, key=lambda r: r["f1"])
        ax.axvline(best["threshold"], color="gray", linestyle="-.", alpha=0.8)
        ax.annotate(_t(f"F1 最优阈值 {best['threshold']}", f"best th={best['threshold']}"),
                    (best["threshold"], 0.05), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color="gray")
        ax.set_xlabel(_t("判定阈值", "decision threshold"), fontsize=10)
        ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
        ax.set_title(_t("阈值权衡：人力充足可下调阈值提高召回", "Threshold trade-off"), fontsize=11)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, _t("缺少阈值扫描数据", "no threshold sweep"), ha="center", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle(_t("图6 预测质量分析", "Fig.6 Prediction quality analysis"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 八、总入口
# ======================================================================
def build_all(figures: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, str]:
    """
    生成全部图表。

    figures 可指定子集，例如 figures=["k"] 只画 k 值对比图；
    未指定时生成全部。返回 {图名: 路径}。
    """
    cfg.ensure_dirs()
    setup_chinese_font(verbose=verbose)

    if verbose:
        print("=" * 68)
        print("【步骤】生成可视化看板")
        print("=" * 68)

    df = load_simulated()
    outputs: Dict[str, str] = {}

    # 读取训练阶段的指标汇总（缺失时给出明确提示而不是崩溃）
    summary: Dict = {}
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    elif verbose:
        print("[提示] 未找到 metrics_summary.json，部分图表（混淆矩阵/k 对比/预测分析）将跳过。"
              "请先运行：python main.py --step train")

    pred_df = pd.read_csv(cfg.PREDICTIONS_PATH, encoding="utf-8-sig") \
        if os.path.exists(cfg.PREDICTIONS_PATH) else None
    imp = pd.read_csv(cfg.IMPORTANCE_PATH, encoding="utf-8-sig") \
        if os.path.exists(cfg.IMPORTANCE_PATH) else None

    want = lambda key: (figures is None or key in figures)  # noqa: E731

    # 图1、图2 只依赖数据文件，任何时候都能出
    if want("dist"):
        p = fig_score_distribution(df, cfg.RISK_LEVEL_THRESHOLDS)
        outputs["01_score_distribution"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("risk"):
        p = fig_risk_distribution(df)
        outputs["02_risk_level_distribution"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("cm") and summary:
        p = fig_confusion_matrices(summary)
        outputs["03_confusion_matrix"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("k") and summary.get("k_scan"):
        p = fig_k_selection(summary)
        outputs["04_k_selection"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("imp") and imp is not None:
        coef_df = None
        if summary.get("logreg_params", {}).get("coef"):
            coef_map = summary["logreg_params"]["coef"]
            coef_df = pd.DataFrame({
                "feature": list(coef_map.keys()),
                "feature_cn": [cfg.FEATURE_CN.get(k, k) for k in coef_map.keys()],
                "coef": list(coef_map.values()),
            })
        p = fig_feature_importance(imp, coef_df)
        outputs["05_feature_importance"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("pred") and summary:
        p = fig_prediction_analysis(summary, pred_df=pred_df)
        outputs["06_prediction_analysis"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")

    if verbose:
        print(f"      共生成 {len(outputs)} 张图 → {os.path.relpath(cfg.FIGURES_DIR, cfg.BASE_DIR)}")
        print("=" * 68)
    return outputs


def main() -> None:
    """命令行入口：python src/dashboard.py"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    build_all()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
