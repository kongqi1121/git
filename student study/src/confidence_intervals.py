# -*- coding: utf-8 -*-
"""
confidence_intervals.py —— 置信区间与显著性（建议 C3）
=========================================================
【为什么需要这个模块】
本项目所有指标都是**点估计**：留一组交叉验证 F1 = 0.3632、10 次分组划分 F1 = 0.3984 ± 0.0404。
但 0.0404 只是 10 次划分之间的离散程度，它**不是**指标的置信区间。
没有区间就无法判断"某次改进把 F1 从 0.3632 提到 0.3720"到底是真提升还是划分噪声。

【本模块给出三种不确定性，并严格说明各自含义（这是最容易混淆的地方）】

    ① **10 次划分的 t 区间**（推荐用于报告主结果）
       对 10 次划分的指标做单样本 t 区间：mean ± t(0.975, df=9) × s/√n。
       含义：如果反复重做"10 次随机划分"这一整套流程，均值会落在该区间的概率约 95%。
       为什么用 t 而不是正态：n 只有 10，自由度小，t 分位数（2.262）明显大于 1.96。

    ② **分层 bootstrap 区间**（对 F1 这类非线性指标更稳健）
       在**保持正负类比例**的前提下对样本做有放回重抽样，重算指标，取 2.5%/97.5% 分位。
       分层（stratified）是必要的：若整体重抽样，某些重抽样可能几乎没有正类，
       导致 precision/recall 剧烈跳动甚至无定义，区间被人为拉宽。
       适用对象：留一组交叉验证（LOGO）的点估计。

    ③ **按学生整组重抽的 bootstrap**（本数据集专有要求）
       UCI 合并数据里 **372 名学生同修两门课**，同一名学生的两条记录高度相关。
       若按"记录"重抽样，同一学生会以两条记录分别出现，等于**重复计数**，区间会偏窄。
       因此重抽单位必须是**学生**：抽到某学生就把他/她的**全部记录**一起放回。

【必须写进报告的三个诚实说明】
    · **LOGO 是确定性结果**（固定种子、无随机划分），严格说它没有"抽样置信区间"；
      它唯一的不确定性来源是"换一批学生"。因此我们**只对 LOGO 报 bootstrap 区间**，
      并明确标注其为"对样本重抽的不确定性"，而不是"对划分随机性的不确定性"。
    · **t 区间假设 10 次划分的指标近似服从正态**。样本量小，该假设无法严格检验，
      所以同时给出 bootstrap 分位区间作为交叉印证。
    · **区间不能用来做显著性检验**：两个模型区间重叠 ≠ 无差异，
      区间不重叠 ≠ 有显著差异（它们基于同一批数据，彼此相关）。
      因此本模块额外提供**配对比较**（在同一批划分上做配对 t 检验），这才是正确做法。

【产物】results/metrics/confidence_intervals.json
【命令】python src/confidence_intervals.py
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
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev


# ======================================================================
# 一、t 区间（10 次划分）
# ======================================================================
# t(0.975, df=9) = 2.262。这里不引入 scipy 依赖，直接内置所需自由度的临界值表，
# 覆盖本项目可能用到的划分次数（5~15），避免"改 splits 就报错"。
T_CRITICAL_975 = {
    4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    19: 2.093, 20: 2.086, 29: 2.045, 30: 2.042,
}


def t_critical_975(df: int) -> float:
    """
    取 95% 双侧 t 临界值（自由度 df = n-1）。

    表里没有精确自由度时，取**不大于该自由度的最近值**（更保守，区间更宽）。
    df < 4 时返回 12.706（df=1 的值），避免返回不可用结果。
    """
    if df <= 1:
        return 12.706
    keys = sorted(k for k in T_CRITICAL_975 if k <= df)
    return T_CRITICAL_975[keys[-1]] if keys else 12.706


def t_interval(values: Sequence[float], confidence: float = 0.95) -> Dict:
    """
    单样本 t 区间：mean ± t(0.975, n-1) × s/√n。

    参数
    ----
    values : 各次划分的指标值（本项目为 10 个）
    """
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "se": 0.0,
                "ci_low": mean, "ci_high": mean, "t_crit": None,
                "note": "样本量 < 2，无法计算区间"}
    s = float(a.std(ddof=1))                 # 样本标准差（无偏）
    se = s / np.sqrt(n)                      # 均值标准误
    tcrit = t_critical_975(n - 1)
    half = tcrit * se
    return {
        "n": int(n),
        "mean": round(mean, 6),
        "std": round(s, 6),
        "se": round(se, 6),
        "t_crit": tcrit,
        "half_width": round(half, 6),
        "ci_low": round(mean - half, 6),
        "ci_high": round(mean + half, 6),
        "method": f"单样本 t 区间（df={n - 1}, t={tcrit}）",
    }


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> Dict:
    """
    bootstrap 百分位区间（取 alpha/2 与 1-alpha/2 分位）。

    比"正态近似区间"更稳健：不假设指标分布对称，适合 F1 这类有偏、有界的指标。
    """
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0, "ci_low": None, "ci_high": None}
    lo_q = (1 - confidence) / 2 * 100
    hi_q = (1 + confidence) / 2 * 100
    lo = float(np.percentile(a, lo_q))
    hi = float(np.percentile(a, hi_q))
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "method": f"bootstrap 百分位区间（{int(confidence * 100)}%）",
    }


# ======================================================================
# 二、分层 / 按学生整组的 bootstrap
# ======================================================================
def stratified_bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 n_boot: int = 2000,
                                 random_state: int = cfg.RANDOM_SEED,
                                 confidence: float = 0.95) -> Dict:
    """
    分层 bootstrap：在**正类内部**与**负类内部**分别有放回重抽样。

    为什么必须分层：正类只有 230 条（占 22%）。若整体重抽，
    会有一定概率抽到极少正类的样本，使 precision/recall 剧烈跳动甚至无定义（分母为 0），
    区间被无意义地拉宽。分层后每次重抽都保持原始的正负比例，区间更可靠。

    注意：该口径假设**样本之间相互独立**，因此**只适用于单条记录即一个独立单位**的情形。
    本项目 UCI 数据存在"同一学生两条记录"，所以正式结果用
    `student_block_bootstrap_metrics()`，本函数保留用于对照说明。
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rng = np.random.RandomState(random_state)

    keys = ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                              rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        m = ev.metrics_from_counts(ev.confusion_counts(y_true[idx], y_pred[idx]))
        for k in keys:
            draws[k].append(m[k])

    out = {k: percentile_interval(draws[k], confidence) for k in keys}
    out["n_boot"] = int(n_boot)
    out["unit"] = "单条记录（行）"
    out["caveat"] = ("按记录重抽；在本数据集上同一学生的两条记录会被分别计数，"
                     "等于重复计数，区间偏窄。正式结果请用按学生整组重抽的口径。")
    return out


def student_block_bootstrap_metrics(df: pd.DataFrame, y_col: str,
                                    pred_col: str, group_col: str,
                                    n_boot: int = 2000,
                                    random_state: int = cfg.RANDOM_SEED,
                                    confidence: float = 0.95) -> Dict:
    """
    **按学生整组重抽**的 bootstrap —— 本项目正式采用的置信区间口径。

    做法：以 `student_group_id` 为单位有放回抽样；抽到某学生，
    就把他的**全部记录**（可能 1 条或 2 条）一起加入重抽样本，再重算指标。

    为什么必须这样做：合并数据中 **372 名学生同修数学与葡语**，
    同一个人的两条记录高度相关。若按记录重抽，同一学生会被分别计数、重复计数，
    使区间**偏窄**（人为显得更"确定"），这是本数据集最容易被忽略的统计陷阱。
    """
    d = df[[group_col, y_col, pred_col]].copy()
    y_true = d[y_col].to_numpy().astype(int)
    y_pred = d[pred_col].to_numpy().astype(int)
    groups = d[group_col].to_numpy()
    uniq = pd.unique(groups)
    # 预先按学生切好索引，避免重抽时反复做布尔索引（2000 次 × 668 组会很慢）
    idx_by_group: Dict[object, np.ndarray] = {g: np.where(groups == g)[0] for g in uniq}
    n_students = len(uniq)
    rng = np.random.RandomState(random_state)

    keys = ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    skipped = 0
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=n_students, replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        # 极端情况：重抽样本里可能没有正类或没有负类 → 指标无定义，跳过并计数
        c = ev.confusion_counts(y_true[idx], y_pred[idx])
        if c["TP"] + c["FN"] == 0 or c["TN"] + c["FP"] == 0:
            skipped += 1
            continue
        m = ev.metrics_from_counts(c)
        for k in keys:
            draws[k].append(m[k])

    out = {k: percentile_interval(draws[k], confidence) for k in keys}
    out["n_boot"] = int(n_boot)
    out["n_boot_effective"] = int(n_boot - skipped)
    out["n_boot_skipped"] = int(skipped)
    out["n_students"] = int(n_students)
    out["unit"] = "学生（同一学生全部记录一起重抽）"
    out["caveat"] = ("按学生整组重抽是本数据集唯一正确的 bootstrap 口径，"
                     "因为 372 名学生同修两门课、记录之间不独立。"
                     "该区间衡量的是『换一批学生』带来的不确定性，"
                     "而不是『换一次随机划分』带来的不确定性。")
    return out


# ======================================================================
# 三、配对比较（同一批划分上比较两个模型）
# ======================================================================
def paired_comparison(values_a: Sequence[float], values_b: Sequence[float],
                      name_a: str = "A", name_b: str = "B") -> Dict:
    """
    配对 t 检验（同一次划分下两个模型的结果相减）。

    为什么必须"配对"：10 次划分里，KNN 与逻辑回归跑的是**完全相同的 10 次划分**
    （同一批种子 42~51）。若按两组独立样本比较，会忽略"划分难度"这一共同因素，
    检验功效被白白浪费。配对后比较的是"同一次划分下谁更好"，显著性判断更可靠。

    这里只给 t 统计量与近似 p 值（不引入 scipy），p 值用正态近似给出并标注为近似值。
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"配对比较要求两组长度相同：{a.shape} vs {b.shape}")
    diff = a - b
    n = len(diff)
    mean_d = float(diff.mean())
    s_d = float(diff.std(ddof=1)) if n > 1 else 0.0
    if s_d == 0:
        # 差值的方差为 0（每次划分的差值完全一致）。此时 t 的分母为 0，无法按公式计算，
        # 但结论其实**比任何有限 t 统计量都强**：差异方向与幅度 100% 一致。
        # 早期版本返回 p_approx=None 且不带 significant 键，会被误读成"证据不足"，
        # 也会让调用方的 `res["significant_at_0.05"]` 直接 KeyError，故改为明确判定。
        return {
            "n": int(n), "pair_a": name_a, "pair_b": name_b,
            "mean_diff": round(mean_d, 6), "std_diff": 0.0,
            "t": None, "p_approx": 0.0 if mean_d != 0 else 1.0,
            "diff_ci95": [round(mean_d, 6), round(mean_d, 6)],
            "significant_at_0.05": bool(mean_d != 0),
            "note": "差值的标准差为 0（每次划分的差值完全一致）："
                    + ("差异方向与幅度 100% 一致，结论比任何有限 t 统计量都更强"
                       if mean_d != 0 else "两组结果完全相同，无差异"),
        }
    se_d = s_d / np.sqrt(n)
    t_stat = mean_d / se_d
    # 正态近似 p 值（双侧）。n=10 时真实 t 分布尾部更厚，故真实 p 值会比该近似更大，
    # 报告里必须标注"近似"，避免把边缘结果说成显著。
    from math import erfc, sqrt
    p_approx = float(erfc(abs(t_stat) / sqrt(2)))
    ci = t_interval(diff)
    return {
        "n": int(n),
        "pair_a": name_a, "pair_b": name_b,
        "mean_diff": round(mean_d, 6),
        "std_diff": round(s_d, 6),
        "t": round(float(t_stat), 4),
        "p_approx": round(p_approx, 4),
        "diff_ci95": [ci["ci_low"], ci["ci_high"]],
        "significant_at_0.05": bool(p_approx < 0.05),
        "note": "配对 t 检验（两组跑的是同一批划分，故必须配对）；"
                "p 值为正态近似，n=10 时真实 p 会略大，边缘结果需谨慎解读",
    }


# ======================================================================
# 四、总入口：读取已有产物，计算全部区间并落盘
# ======================================================================
def compute_all(n_boot: int = 2000, verbose: bool = True) -> Dict:
    """
    计算并保存全部置信区间。

    数据来源全部是**已有产物**（不需要重跑建模）：
      · results/metrics/split_details.csv  → 10 次划分逐次指标与混淆矩阵
      · results/metrics/full_predictions.csv → LOGO 逐条预测 + 学生编号
      · results/metrics/metrics_summary.json → 已归档的均值/标准差（用于交叉校验）
    """
    cfg.ensure_dirs()
    out: Dict = {"method_notes": {
        "t_interval": "10 次划分的均值用单样本 t 区间（df=9, t=2.262）",
        "bootstrap": f"分层/按学生整组 bootstrap，重抽 {n_boot} 次，取 2.5%/97.5% 分位",
        "logo": "LOGO 为确定性结果，其区间来自『按学生重抽样本』，"
                "衡量换一批学生的不确定性，而非划分随机性",
    }}

    # ---- ① 10 次划分：t 区间 + bootstrap 分位区间 ----
    split_path = cfg.SPLIT_DETAIL_PATH
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"缺少 {split_path}，请先运行：python main.py --step train")
    sd = pd.read_csv(split_path, encoding="utf-8-sig")
    keys = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"]
    out["splits"] = {}
    for model in [m for m in sd["model"].unique()]:
        sub = sd[sd["model"] == model]
        entry = {"n_splits": int(len(sub)), "t_interval": {}, "bootstrap_percentile": {}}
        for k in keys:
            vals = sub[k].to_numpy(dtype=float)
            entry["t_interval"][k] = t_interval(vals)
            # 对"每次划分的指标值"做重抽样，得到"均值"的 bootstrap 区间
            rng = np.random.RandomState(cfg.RANDOM_SEED)
            means = [float(rng.choice(vals, size=len(vals), replace=True).mean())
                     for _ in range(n_boot)]
            entry["bootstrap_percentile"][k] = percentile_interval(means)
        out["splits"][model] = entry

    # ---- ② LOGO：按学生整组重抽的 bootstrap ----
    pred_path = cfg.PREDICTIONS_PATH
    if os.path.exists(pred_path):
        pdf = pd.read_csv(pred_path, encoding="utf-8-sig")
        need = {cfg.LABEL_COLUMN, "logo_pred", "student_group_id"}
        if need.issubset(pdf.columns):
            out["logo"] = {
                "point_estimate": {
                    k: round(float(v), 6) for k, v in
                    ev.metrics_from_counts(ev.confusion_counts(
                        pdf[cfg.LABEL_COLUMN], pdf["logo_pred"])).items()
                },
                "counts": ev.confusion_counts(pdf[cfg.LABEL_COLUMN], pdf["logo_pred"]),
                "student_block_bootstrap": student_block_bootstrap_metrics(
                    pdf, cfg.LABEL_COLUMN, "logo_pred", "student_group_id",
                    n_boot=n_boot),
                "row_bootstrap_for_contrast": stratified_bootstrap_metrics(
                    pdf[cfg.LABEL_COLUMN].to_numpy(), pdf["logo_pred"].to_numpy(),
                    n_boot=n_boot),
            }

    # ---- ③ 配对比较：KNN vs 逻辑回归（同一批 10 次划分）----
    if "knn" in set(sd["model"]) and "logreg" in set(sd["model"]):
        knn = sd[sd["model"] == "knn"].sort_values("split")
        lr = sd[sd["model"] == "logreg"].sort_values("split")
        out["paired_knn_vs_logreg"] = {}
        for k in ["accuracy", "precision", "recall", "f1"]:
            out["paired_knn_vs_logreg"][k] = paired_comparison(
                knn[k].to_numpy(), lr[k].to_numpy(), "KNN", "逻辑回归")

    # ---- ④ 与已归档的 summary 交叉校验（防止统计口径漂移）----
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "split10_knn" in s:
            checks = {}
            for k in ["accuracy", "precision", "recall", "f1"]:
                archived = float(s["split10_knn"][k]["mean"])
                recomputed = out["splits"]["knn"]["t_interval"][k]["mean"]
                checks[k] = {
                    "archived_mean": round(archived, 6),
                    "recomputed_mean": round(recomputed, 6),
                    "consistent": bool(abs(archived - recomputed) < 5e-4),
                }
            out["cross_check_vs_summary"] = checks

    # ---- 保存 ----
    path = os.path.join(cfg.METRICS_DIR, "confidence_intervals.json")
    ev.save_json(out, path)

    if verbose:
        print("=" * 74)
        print("【置信区间】基于已有产物的离线计算（不重跑建模）")
        print("=" * 74)
        print(f"数据源：{os.path.basename(split_path)}、{os.path.basename(pred_path)}")
        print()
        print("① 10 次分组随机划分 —— t 区间（自由度 9，t=2.262）")
        print(f"  {'模型':<10}{'指标':<12}{'均值':>9}{'标准误':>9}{'95% CI':>22}")
        for model in out["splits"]:
            for k in ["f1", "recall", "accuracy", "precision"]:
                t = out["splits"][model]["t_interval"][k]
                print(f"  {model:<10}{k:<12}{t['mean']:>9.4f}{t['se']:>9.4f}"
                      f"{'[' + format(t['ci_low'], '.4f') + ', ' + format(t['ci_high'], '.4f') + ']':>22}")
            print()
        if "logo" in out:
            po = out["logo"]["point_estimate"]
            bs = out["logo"]["student_block_bootstrap"]
            rb = out["logo"]["row_bootstrap_for_contrast"]
            print("② LOGO 主口径 —— 点估计 + 按学生整组重抽的 95% bootstrap 区间")
            print(f"  点估计（确定性）：F1={po['f1']:.4f}  recall={po['recall']:.4f}  "
                  f"precision={po['precision']:.4f}  accuracy={po['accuracy']:.4f}")
            for k in ["f1", "recall", "precision", "accuracy"]:
                b, r = bs[k], rb[k]
                print(f"  {k:<10} 按学生整组 CI=[{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
                      f"   按记录（偏窄，仅对照）CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
            print(f"  有效重抽样 {bs['n_boot_effective']}/{bs['n_boot']}，"
                  f"跳过 {bs['n_boot_skipped']}（重抽样本缺某一类）")
            print()
        if "paired_knn_vs_logreg" in out:
            print("③ 配对比较：KNN vs 逻辑回归（同一次划分下相减，配对设计）")
            for k, v in out["paired_knn_vs_logreg"].items():
                sig = "显著" if v.get("significant_at_0.05") else "不显著"
                print(f"  {k:<10} 平均差值={v['mean_diff']:+.4f}  t={v['t']}  "
                      f"p≈{v['p_approx']}（{sig}）  差值 95%CI="
                      f"[{v['diff_ci95'][0]:+.4f}, {v['diff_ci95'][1]:+.4f}]")
            print()
        if "cross_check_vs_summary" in out:
            ok = all(v["consistent"] for v in out["cross_check_vs_summary"].values())
            print(f"④ 与已归档 summary 的交叉校验：{'全部一致 ✅' if ok else '存在不一致 ❌'}")
        print(f"\n已保存：{os.path.relpath(path, cfg.BASE_DIR)}")
        print("=" * 74)
    return out


def main() -> None:
    """命令行入口：python src/confidence_intervals.py [--boot 2000]"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    n_boot = 2000
    for i, a in enumerate(sys.argv):
        if a in ("--boot", "-b") and i + 1 < len(sys.argv):
            n_boot = int(sys.argv[i + 1])
    compute_all(n_boot=n_boot)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
