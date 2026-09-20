# -*- coding: utf-8 -*-
"""
dashboard_v2.py —— 看板 v2（UCI 真实数据 + sklearn 主模型）
=========================================================
生成 5 张图到 results/figures/：

    11_uci_data_overview.png     真实数据全景：G3 分布（按课程）、高风险比例、
                                 关键特征与风险的关联、单特征基线对比
    12_uci_confusion_matrix.png  混淆矩阵四联图（LOGO 主口径 / 10 次划分合计 /
                                 课程外推 mat→por 与 por→mat）
    13_uci_k_selection.png       k 值选择：LOGO 指标曲线 + 划分均值±标准差 + recall 红线
    14_uci_ablation.png          早期成绩消融对照（标签泄漏量化）+ 多数类基线对比
    15_uci_backend_equivalence.png 附录：手写 KNN 与 sklearn 的等价性验证结果

【重要：图表要如实呈现"主口径很难"这一结论】
真实数据上（不含 G1/G2）主口径的 accuracy ≈ 多数类基线，F1 偏低。图中会**显式画出
多数类基线与 recall 门槛线**，避免看图的人误以为模型表现良好。
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from . import config as cfg
    from . import evaluate as ev
    from .dashboard import _cn, _t, setup_chinese_font, _save
    from .uci_features import feature_name_cn
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from dashboard import _cn, _t, setup_chinese_font, _save
    from uci_features import feature_name_cn


# ======================================================================
# 图 11：真实数据全景
# ======================================================================
def fig_uci_overview(df: pd.DataFrame, out_name: str = "11_uci_data_overview.png") -> str:
    """真实数据全景：成绩分布、风险比例、特征关联、单特征基线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (1) G3 分布，按课程分层
    ax = axes[0, 0]
    bins = np.arange(-0.5, 21.5, 1)
    for sub, color, label in (("mat", "#2E75B6", _t("数学(mat)", "Math")),
                              ("por", "#ED7D31", _t("葡语(por)", "Portuguese"))):
        sub_df = df[df["subject"] == sub]
        ax.hist(sub_df["G3"], bins=bins, alpha=0.7, color=color,
                label=f"{label} n={len(sub_df)}", edgecolor="white")
    ax.axvline(10, color="red", linestyle="--", linewidth=2,
               label=_t("及格线 G3=10", "pass line 10"))
    ax.set_xlabel(_t("期末成绩 G3（0~20）", "final grade G3"), fontsize=10)
    ax.set_ylabel(_t("人数", "count"), fontsize=10)
    ax.set_title(_t("成绩分布（G3 < 10 即高风险；两门课分布差异明显）",
                    "G3 distribution by subject"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # (2) 高风险比例与多数类基线
    ax = axes[0, 1]
    ratios = df.groupby("subject")[cfg.LABEL_COLUMN].mean().to_dict()
    ratios[_t("合并", "All")] = float(df[cfg.LABEL_COLUMN].mean())
    names = list(ratios.keys())
    vals = [ratios[k] * 100 for k in names]
    bars = ax.bar(names, vals, color=["#2E75B6", "#ED7D31", "#7030A0"], edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(35, color="gray", linestyle=":", linewidth=1,
               label=_t("要求区间上限 35%", "upper bound 35%"))
    ax.axhline(20, color="gray", linestyle="--", linewidth=1,
               label=_t("要求区间下限 20%", "lower bound 20%"))
    ax.set_ylabel(_t("高风险比例 (%)", "risk ratio (%)"), fontsize=10)
    ax.set_title(_t("高风险比例（合并 22.0%，落在 20%~35% 内）",
                    "Risk ratio"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # (3) 关键特征在两类学生上的均值对比（标准化后便于比较）
    ax = axes[1, 0]
    feats = ["failures", "absences", "studytime", "goout", "Dalc", "Walc",
             "health", "Medu", "Fedu", "age"]
    y = df[cfg.LABEL_COLUMN].to_numpy()
    d0 = df.loc[df[cfg.LABEL_COLUMN] == 0, feats].mean()
    d1 = df.loc[df[cfg.LABEL_COLUMN] == 1, feats].mean()
    # 用全样本标准差归一化，消除量纲差异
    sd = df[feats].std().replace(0, 1)
    z0, z1 = ((d0 - df[feats].mean()) / sd), ((d1 - df[feats].mean()) / sd)
    xs = np.arange(len(feats))
    ax.barh(xs - 0.2, z0.values, 0.4, label=_t("正常", "Normal"), color="#2E75B6")
    ax.barh(xs + 0.2, z1.values, 0.4, label=_t("需关注", "At risk"), color="#C00000")
    ax.set_yticks(xs, [feature_name_cn(f) for f in feats], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(_t("相对全样本均值的标准化偏差", "standardized deviation"), fontsize=10)
    ax.set_title(_t("两类学生在关键特征上的差异（差距不大 → 任务难）",
                    "Feature differences between classes"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="x")

    # (4) 单特征基线对比（证明"单变量规则不够用"）
    ax = axes[1, 1]
    try:
        from .uci_data import single_feature_baseline
    except ImportError:  # pragma: no cover
        from uci_data import single_feature_baseline
    base = single_feature_baseline(df)
    pf = base["per_feature"][:10]
    names_cn = [r["feature_cn"] for r in pf]
    accs = [r["best_single_acc"] * 100 for r in pf]
    bars = ax.barh(names_cn[::-1], accs[::-1], color="#A6A6A6", edgecolor="white")
    ax.axvline(base["majority_baseline_acc"] * 100, color="red", linestyle="--",
               linewidth=2, label=_t(f"多数类基线 {base['majority_baseline_acc']*100:.1f}%",
                                     "majority baseline"))
    ax.set_xlabel(_t("单特征最优准确率 (%)", "best single-feature accuracy (%)"), fontsize=10)
    ax.set_title(_t("单特征规则的天花板（仅略高于基线 → 必须用多特征模型）",
                    "Single-feature baselines"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--", axis="x")

    fig.suptitle(_t("图11 UCI 真实成绩数据全景（真实数据，非模拟）",
                    "Fig.11 UCI real data overview"), fontsize=13)
    return _save(fig, out_name)


# ======================================================================
# 图 12：混淆矩阵
# ======================================================================
def _draw_cm(ax, cm, title: str, metrics: Optional[Dict] = None) -> None:
    """画一个带数值标注的混淆矩阵。"""
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    labels = [_t("正常(0)", "Normal(0)"), _t("需关注(1)", "Risk(1)")]
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.set_yticks([0, 1], labels, fontsize=9)
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
    t = title
    if metrics:
        t += (f"\nacc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f}\n"
              f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    ax.set_title(t, fontsize=10)


def fig_uci_confusion(summary: Dict, out_name: str = "12_uci_confusion_matrix.png") -> str:
    """混淆矩阵四联：LOGO 主口径、10 次划分合计、两个方向的课程外推。"""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))

    logo = summary.get("logo_knn", {})
    _draw_cm(axes[0, 0], logo.get("confusion_matrix", [[0, 0], [0, 0]]),
             _t(f"KNN(k={summary.get('main_k')}) 留一组交叉验证 LOGO（主口径）\n"
                f"全部 {summary.get('n_samples')} 条真实记录", "KNN LOGO"),
             logo.get("metrics"))

    # 10 次划分的混淆矩阵合计
    cs = summary.get("split10_knn", {}).get("confusion_sum", {}).get("value")
    if cs:
        cm_sum = np.asarray(cs)
        _draw_cm(axes[0, 1], cm_sum,
                 _t(f"{summary.get('n_splits')} 次分组随机划分（混淆矩阵合计）",
                    "10 group splits (summed)"),
                 {"accuracy": summary["split10_knn"]["accuracy"]["mean"],
                  "precision": summary["split10_knn"]["precision"]["mean"],
                  "recall": summary["split10_knn"]["recall"]["mean"],
                  "f1": summary["split10_knn"]["f1"]["mean"]})

    for ax, key, title in ((axes[1, 0], "mat2por",
                            _t("课程外推：训练=数学 → 测试=葡语", "mat → por")),
                           (axes[1, 1], "por2mat",
                            _t("课程外推：训练=葡语 → 测试=数学", "por → mat"))):
        rep = summary.get("cross_subject_knn", {}).get(key, {})
        _draw_cm(ax, rep.get("confusion_matrix", [[0, 0], [0, 0]]), title, rep.get("metrics"))

    fig.suptitle(_t("图12 混淆矩阵（真实数据；FN=漏报，预警系统最需要关注的格子）",
                    "Fig.12 Confusion matrices"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 图 13：k 值选择
# ======================================================================
def fig_uci_k_selection(summary: Dict, out_name: str = "13_uci_k_selection.png") -> str:
    """k 值选择：LOGO 指标曲线、划分稳定性、recall 门槛。"""
    scan = summary.get("k_scan", [])
    if not scan:
        raise RuntimeError("summary 中缺少 k_scan")
    ks = [r["k"] for r in scan]

    def col(name: str) -> List[float]:
        return [r[name] for r in scan]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    ax = axes[0]
    ax.plot(ks, col("logo_f1"), "o-", color="#C00000", linewidth=2, label="F1")
    ax.plot(ks, col("logo_recall"), "s--", color="#ED7D31", linewidth=2,
            label=_t("召回率 recall", "recall"))
    ax.plot(ks, col("logo_precision"), "^--", color="#2E75B6", linewidth=2,
            label=_t("精确率 precision", "precision"))
    ax.plot(ks, col("logo_accuracy"), "d:", color="#70AD47", linewidth=2,
            label=_t("准确率 accuracy", "accuracy"))
    ax.axhline(summary.get("majority_baseline_accuracy", 0), color="black",
               linestyle=":", linewidth=1.2,
               label=_t(f"多数类基线 {summary.get('majority_baseline_accuracy', 0):.3f}",
                        "majority baseline"))
    ax.axhline(0.30, color="gray", linestyle="-.", linewidth=1,
               label=_t("recall 门槛 0.30", "recall floor 0.30"))
    ax.axvline(summary.get("main_k"), color="gray", linestyle="--", alpha=0.8)
    ax.annotate(f"k={summary.get('main_k')}", (summary.get("main_k"), 0.02),
                textcoords="offset points", xytext=(4, 4), fontsize=9, color="gray")
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
    ax.set_title(_t("LOGO（主口径）：不同 k 的指标", "LOGO metrics vs k"), fontsize=11)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    ax.errorbar(ks, col("split_f1_mean"), yerr=col("split_f1_std"), fmt="o-",
                color="#7030A0", capsize=4, linewidth=2, label="F1")
    ax.errorbar(ks, col("split_rec_mean"), yerr=col("split_rec_std"), fmt="s--",
                color="#ED7D31", capsize=4, linewidth=2, label=_t("召回率", "recall"))
    ax.axvline(summary.get("main_k"), color="gray", linestyle="--", alpha=0.8)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
    ax.set_title(_t(f"{summary.get('n_splits')} 次分组随机划分：均值±标准差",
                    "Repeated splits: mean±std"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[2]
    ax.plot(ks, col("split_f1_std"), "o-", color="#ED7D31", linewidth=2, label="F1")
    ax.plot(ks, col("split_acc_std"), "s--", color="#2E75B6", linewidth=2,
            label=_t("准确率", "accuracy"))
    ax.plot(ks, col("split_rec_std"), "^:", color="#C00000", linewidth=2,
            label=_t("召回率", "recall"))
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel(_t("标准差（越小越稳定）", "std"), fontsize=10)
    ax.set_title(_t("稳定性随 k 的变化", "Stability vs k"), fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle(_t("图13 k 值选择（先满足 recall 门槛，再取 F1 最优）",
                    "Fig.13 k selection"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 图 14：消融对照
# ======================================================================
def fig_uci_ablation(summary: Dict, out_name: str = "14_uci_ablation.png") -> str:
    """早期成绩消融对照 + 与其他口径的横向对比。"""
    rows = summary.get("ablation_early_grades", [])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    labels = [r["setting"] for r in rows if r["logo_f1"] != ""]
    f1s = [r["logo_f1"] for r in rows if r["logo_f1"] != ""]
    recs = [r["logo_recall"] for r in rows if r["logo_f1"] != ""]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, f1s, 0.4, color="#C00000", label="F1")
    ax.bar(xs + 0.2, recs, 0.4, color="#ED7D31", label=_t("召回率 recall", "recall"))
    for i, (a, b) in enumerate(zip(f1s, recs)):
        ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs, [_t("主口径\n（无 G1/G2，可用）", "main\n(no G1/G2)"),
                       _t("消融口径\n（含 G1/G2，泄漏）", "with\nG1/G2")], fontsize=9)
    ax.set_ylabel(_t("指标值", "metric"), fontsize=10)
    ax.set_title(_t("早期成绩消融对照：加 G1/G2 后 F1 大幅上升 —— 但那是标签泄漏",
                    "Early-grade ablation"), fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    ax = axes[1]
    # 横向对比：多数类基线 / 单特征最优 / 主口径 KNN / 主口径逻辑回归
    base = summary.get("majority_baseline_accuracy", 0)
    try:
        df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
        from .uci_data import single_feature_baseline
    except ImportError:  # pragma: no cover
        from uci_data import single_feature_baseline
    sf = single_feature_baseline(df)["best_single_feature_acc"]
    items = [
        (_t("多数类基线", "majority"), base, "#A6A6A6"),
        (_t("单特征最优规则", "best single feat"), sf, "#70AD47"),
        (_t("主口径 KNN", "KNN"), summary.get("logo_knn", {}).get("metrics", {}).get("accuracy", 0), "#2E75B6"),
        (_t("主口径逻辑回归", "LogReg"), summary.get("logo_logreg", {}).get("accuracy", 0), "#7030A0"),
    ]
    xs = np.arange(len(items))
    vals = [v * 100 for _, v, _ in items]
    bars = ax.bar(xs, vals, color=[c for _, _, c in items], edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(xs, [n for n, _, _ in items], fontsize=9)
    ax.set_ylim(min(vals) - 5, max(vals) + 5)
    ax.set_ylabel(_t("accuracy (%)", "accuracy (%)"), fontsize=10)
    ax.set_title(_t("主口径 accuracy 仅与基线相当 → 真实数据上这个任务很难",
                    "Accuracy vs baselines"), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    fig.suptitle(_t("图14 消融对照与基线对比（如实呈现：主口径难度很大）",
                    "Fig.14 Ablation & baselines"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 图 15：后端等价性（附录）
# ======================================================================
def fig_backend_equivalence(out_name: str = "15_uci_backend_equivalence.png") -> str:
    """附录图：手写 KNN 与 sklearn 的等价性验证结果汇总。"""
    path = cfg.APPENDIX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到附录报告 {path}，请先运行 src/appendix_handwritten_knn.py")
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # 左：每个数据集的各项检查是否通过
    ax = axes[0]
    ds_names = list(rep["datasets"].keys())
    check_names, matrix = [], []
    for dn in ds_names:
        for chk in rep["datasets"][dn]["checks"]:
            if chk["check"] not in check_names:
                check_names.append(chk["check"])
    for dn in ds_names:
        row = []
        for cn in check_names:
            v = next((c for c in rep["datasets"][dn]["checks"] if c["check"] == cn), None)
            row.append(1.0 if (v and v["passed"]) else 0.0)
        matrix.append(row)
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    short = [c.split(".")[0] for c in check_names]
    ax.set_xticks(range(len(short)), short, fontsize=9)
    ax.set_yticks(range(len(ds_names)), ds_names, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "通过" if matrix[i, j] else "未过", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title(_t("等价性检查项通过情况（绿=通过）", "Equivalence checks"), fontsize=11)

    # 右：各 k 下"类别预测差异 / 概率差异（无并列）"
    ax = axes[1]
    uci = rep["datasets"].get("UCI 真实数据", {})
    per_k = next((c["per_k"] for c in uci.get("checks", []) if "per_k" in c
                  and c["per_k"] and "pred_diff_count" in c["per_k"][0]), None)
    if per_k:
        ks = [r["k"] for r in per_k]
        ax.bar(np.arange(len(ks)) - 0.2, [r["pred_diff_count"] for r in per_k], 0.4,
               color="#C00000", label=_t("类别预测不一致数", "prediction diffs"))
        ax.bar(np.arange(len(ks)) + 0.2, [r["proba_diff_without_tie"] for r in per_k], 0.4,
               color="#2E75B6", label=_t("无并列却概率不同", "proba diffs w/o tie"))
        ax.set_xticks(range(len(ks)), ks, fontsize=9)
        ax.set_xlabel("k", fontsize=10)
        ax.set_ylabel(_t("差异样本数（越接近 0 越好）", "diff count"), fontsize=10)
        ax.set_title(_t("手写 KNN vs sklearn：全部 k 上差异为 0", "Handwritten vs sklearn"),
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25, linestyle="--", axis="y")
    else:
        ax.axis("off")

    fig.suptitle(_t(f"图15 附录：手写 KNN 与 sklearn KNN 等价性验证"
                    f"（sklearn {rep.get('sklearn_version')}）",
                    "Fig.15 Backend equivalence"), fontsize=12)
    return _save(fig, out_name)


# ======================================================================
# 总入口
# ======================================================================
def build_all_v2(figures: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, str]:
    """生成 v2 全部图表（11~15）。"""
    cfg.ensure_dirs()
    setup_chinese_font(verbose=verbose)
    if verbose:
        print("=" * 68)
        print("【步骤 v2】生成 UCI 看板")
        print("=" * 68)

    df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
    outputs: Dict[str, str] = {}
    want = lambda key: (figures is None or key in figures)  # noqa: E731

    summary: Dict = {}
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    elif verbose:
        print("[提示] 未找到 metrics_summary.json，图12~14 将跳过；请先运行 v2 训练。")

    if want("overview"):
        p = fig_uci_overview(df)
        outputs["11_uci_data_overview"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("cm") and summary:
        p = fig_uci_confusion(summary)
        outputs["12_uci_confusion_matrix"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("k") and summary.get("k_scan"):
        p = fig_uci_k_selection(summary)
        outputs["13_uci_k_selection"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("ablation") and summary.get("ablation_early_grades"):
        p = fig_uci_ablation(summary)
        outputs["14_uci_ablation"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")
    if want("equiv") and os.path.exists(cfg.APPENDIX_PATH):
        p = fig_backend_equivalence()
        outputs["15_uci_backend_equivalence"] = p
        if verbose:
            print(f"  ✔ {os.path.basename(p)}")

    if verbose:
        print(f"      共生成 {len(outputs)} 张图 → {os.path.relpath(cfg.FIGURES_DIR, cfg.BASE_DIR)}")
        print("=" * 68)
    return outputs


def main() -> None:
    """命令行入口：python src/dashboard_v2.py"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    build_all_v2()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
