# -*- coding: utf-8 -*-
"""
calibration.py —— 概率校准（建议 A2）
=========================================================
【要解决的问题】
主模型 k = 1 时，`predict_proba` 只能输出 0 或 1（k+1 = 2 种取值），
于是三档风险等级（低 < 0.3、中 0.3~0.7、高 > 0.7）**实际退化为两档** ——
阈值 0.3 与 0.7 永远跨不到。这使"分档处置"这一使用方式失效。

【方法】Platt 校准（sigmoid），而不是 isotonic
    · k=1 的基础模型只有两个输出值，isotonic 在这种极低分辨率输入上容易过拟合出阶梯函数；
    · Platt 假设一个参数化的 S 形映射，参数少、更稳健。

【★ 最关键的实现约束：校准必须嵌进交叉验证的**训练折内层**】
`CalibratedClassifierCV` 自身会用内部 CV 拟合校准曲线。
如果直接在全量数据上包装再评估，校准器就"见过"测试折的数据 ——
概率会变得好看，但评测结果偏乐观（这是概率校准最容易被忽略的泄漏）。
因此本模块：
    ① 对每一折，只在**训练集**上创建并 fit `CalibratedClassifierCV`；
    ② 测试集只用于 transform/预测。

【必须讲清的一件事：校准不改善排序能力】
校准是**单调映射**（保序），因此：
    · ROC-AUC / F1 / recall（在固定阈值下）**不会有本质提升**；
    · 它改善的是概率的**数值意义**，从而让三档阈值真正可用。
所以本模块的验收指标是 **Brier 分数**与**可靠性曲线**，而不是 accuracy/F1。
混淆矩阵也一并报告，用于说明"校准对分类结果的影响通常很小"。

【产物】results/metrics/calibration.json（含校准前后概率向量，供画图）
【命令】python src/calibration.py
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .knn_numpy import assign_risk_levels
    from .preprocess import StandardScaler
    from .uci_data import group_train_test_split_indices, load_saved_dataset
    from .uci_features import prepare_uci_features
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from knn_numpy import assign_risk_levels
    from preprocess import StandardScaler
    from uci_data import group_train_test_split_indices, load_saved_dataset
    from uci_features import prepare_uci_features


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Brier 分数 = 均方误差 of (概率 vs 真实标签)，越小越好。

    它是**校准质量的正确指标**：只看概率是否接近真实频率，
    不像 accuracy/F1 那样只关心排序与阈值。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    proba = np.asarray(proba, dtype=float).ravel()
    return float(np.mean((proba - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    期望校准误差（ECE）：各分箱内 |平均预测概率 − 实际正类率| 的样本数加权平均。

    直观解释："模型说 60% 风险时，实际大约有多少比例真的是高风险" ——
    两者差距的加权平均就是 ECE。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    proba = np.asarray(proba, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(proba)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (proba >= lo) & (proba < hi if i < n_bins - 1 else proba <= hi)
        if m.sum() == 0:
            continue
        gap = abs(float(proba[m].mean()) - float(y_true[m].mean()))
        ece += gap * (m.sum() / n)
    return float(ece)


def _level_counts(proba: np.ndarray,
                  thresholds: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """按风险等级阈值统计各档人数（用于验证"三档是否恢复可用"）。"""
    levels = assign_risk_levels(np.asarray(proba, dtype=float), thresholds)
    out = {lv: 0 for lv in cfg.RISK_LEVELS}
    for lv in levels:
        out[str(lv)] = out.get(str(lv), 0) + 1
    return out


def pick_level_thresholds(proba: np.ndarray,
                          target: Tuple[float, float] = (0.25, 0.15)) -> Dict[str, float]:
    """
    在给定概率刻度上**重新标定**三档等级阈值。

    【为什么需要这一步】默认阈值（低 < 0.3，高 > 0.7）是针对**未校准**概率设定的。
    校准会把概率压向类别基准率（本项目基准率 0.22），校准后的概率上限可能远低于 0.7，
    于是"高"档会变成 0 人 —— 这并不是校准失败，而是**阈值没有跟着刻度一起改**。

    做法：按目标人群比例反推阈值。默认目标为
        低档约占 60%（0.25 + 0.15 = 0.40 以上为低），中档约 25%，高档约 15%。
    这与数据生成时三档等级的设定（低/中/高 ≈ 62%/26%/12%）保持一致，
    便于与未校准时的分档结果对照。
    """
    p = np.asarray(proba, dtype=float)
    q_low = 1.0 - target[0] - target[1]      # 低/中分位
    high = float(np.quantile(p, q_low, method="linear"))       # 高于此值为"低"
    low = float(np.quantile(p, 1.0 - target[1], method="linear"))  # 高于此值为"中"，否则"高"
    if low >= high:                          # 防御：概率几乎常量时避免阈值颠倒
        low, high = min(low, high) * 0.9, max(low, high) * 1.1
    return {"low": round(low, 4), "high": round(high, 4)}


def run_calibration(backend: str = "auto", method: str = "sigmoid",
                    base_model: str = "balanced", verbose: bool = True) -> Dict:
    """
    在校正的口径下评估概率校准：每一折的训练集内部 fit 校准器。

    参数
    ----
    base_model : 基础模型
        'balanced'  → 逻辑回归(class_weight='balanced')，**推荐**
                      原因：它有**分级分数**（连续概率），Platt 校准才有信息可用；
        'knn'       → KNN(k=主 k)。保留该选项用于对照与说明：
                      实测发现 k=1 时基础模型只有 0/1 两个取值，
                      Platt 校准在这种退化输入上**会破坏排序**（AUC 从 0.595 降到 0.463），
                      因此"给 k=1 的 KNN 做概率校准"是不可行的做法。

    在**留一组交叉验证**上做：对每一名学生（组）：
        ① 训练集 = 除该组外全部样本 → fit 标准化器 + 基础模型
        ② 在训练集内部取一个内层划分，用其 fit `CalibratedClassifierCV`（校准器只见训练数据）
        ③ 对该组预测原始概率与校准后概率
    最终把全体样本的原始/校准概率拼起来，计算 Brier、ECE、AUC、分档人数与混淆矩阵。
    """
    cfg.ensure_dirs()
    try:
        from .model_backends import make_knn, make_logreg
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from model_backends import make_knn, make_logreg
    from sklearn.calibration import CalibratedClassifierCV

    df = load_saved_dataset()
    X, y, _ = prepare_uci_features(df, include_early_grades=False, verbose=False)
    groups = df["student_group_id"].to_numpy()

    main_k = int(cfg.MAIN_K)
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if s.get("dataset", "").startswith("UCI"):
            main_k = int(s.get("main_k", main_k))

    def make_base():
        if base_model == "knn":
            return make_knn(k=main_k, backend=backend)
        return make_logreg(backend=backend, class_weight="balanced")

    base_desc = (f"KNN(k={main_k})" if base_model == "knn"
                 else "逻辑回归(class_weight='balanced')")

    uniq = pd.unique(groups)
    p_raw = np.zeros(len(y), dtype=float)
    p_cal = np.zeros(len(y), dtype=float)

    if verbose:
        print("=" * 78)
        print("【概率校准】Platt（sigmoid）校准，校准器只在训练折内层拟合")
        print("=" * 78)
        print(f"基础模型 = {base_desc}")
        print(f"学生数 = {len(uniq)}（留一组交叉验证，逐组预测）；请稍候……")

    for gi, g in enumerate(uniq):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        sc = StandardScaler().fit(X[tr])
        Xtr_s = sc.transform(X[tr])
        base = make_base().fit(Xtr_s, y[tr])

        p_raw[te] = base.predict_proba(sc.transform(X[te]))[:, 1]

        # 校准器：cv=3 的内部划分**只用传入的训练集**，不接触测试组
        cal = CalibratedClassifierCV(base, method=method, cv=3)
        cal.fit(Xtr_s, y[tr])
        p_cal[te] = cal.predict_proba(sc.transform(X[te]))[:, 1]

        if verbose and (gi + 1) % 150 == 0:
            print(f"    进度 {gi + 1}/{len(uniq)} 组")

    # 用 AUC 衡量排序能力（对单调变换不变）—— 这是判断"校准是否安全"的关键指标
    def _auc(yy: np.ndarray, ss: np.ndarray) -> float:
        pos, neg = ss[yy == 1], ss[yy == 0]
        cmp = pos[:, None] > neg[None, :]
        tie = pos[:, None] == neg[None, :]
        return float(cmp.mean() + 0.5 * tie.mean())

    base_pred = (p_raw > 0.5).astype(int)
    cal_pred = (p_cal > 0.5).astype(int)
    # 校准后 0.5 往往不再是合适工作点 → 同时给出"校准概率上的 F1 最优阈值"结果，
    # 避免把"阈值不匹配"误读成"校准把模型变差了"
    best_th, best_f1 = 0.5, -1.0
    for th in np.round(np.arange(0.05, 0.95, 0.01), 2):
        f1 = ev.metrics_from_counts(
            ev.confusion_counts(y, (p_cal >= th).astype(int)))["f1"]
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    cal_pred_best = (p_cal >= best_th).astype(int)

    # 校准后必须**重新标定**等级阈值（默认 0.3/0.7 是针对未校准概率设的）
    thresholds_cal = pick_level_thresholds(p_cal)
    result = {
        "method": f"CalibratedClassifierCV(method={method}, cv=3)，逐折训练集内拟合",
        "base_model": base_desc,
        "n_samples": int(len(y)),
        "n_students": int(len(uniq)),
        "main_k": main_k,
        "y_true": [int(v) for v in y],
        "proba_raw": [round(float(v), 6) for v in p_raw],
        "proba_calibrated": [round(float(v), 6) for v in p_cal],
        "brier": {"raw": round(brier_score(y, p_raw), 6),
                  "calibrated": round(brier_score(y, p_cal), 6)},
        "ece": {"raw": round(expected_calibration_error(y, p_raw), 6),
                "calibrated": round(expected_calibration_error(y, p_cal), 6)},
        "auc": {"raw": round(_auc(y, p_raw), 6), "calibrated": round(_auc(y, p_cal), 6)},
        "n_unique_proba": {
            "raw": int(len(np.unique(np.round(p_raw, 6)))),
            "calibrated": int(len(np.unique(np.round(p_cal, 6)))),
        },
        "level_counts_raw": _level_counts(p_raw),
        "level_counts_calibrated_default_thresholds": _level_counts(p_cal),
        "level_thresholds_calibrated": thresholds_cal,
        "level_counts_calibrated_retuned": _level_counts(p_cal, thresholds_cal),
        "metrics_raw": {k: round(v, 6) for k, v in ev.evaluate_binary(y, base_pred).items()},
        "metrics_calibrated_at_0.5": {k: round(v, 6) for k, v in
                                      ev.evaluate_binary(y, cal_pred).items()},
        "best_threshold_on_calibrated": best_th,
        "metrics_calibrated_at_best": {k: round(v, 6) for k, v in
                                       ev.evaluate_binary(y, cal_pred_best).items()},
        "note": "校准是单调映射，理想情况下不改变排序能力（AUC 应几乎不变），"
                "因此不会提升 F1/recall；它改善的是概率的数值意义（Brier / ECE）。"
                "**但若基础模型的输出本身已退化（例如 k=1 只有 0/1 两个取值），"
                "Platt 校准会破坏排序（实测 AUC 0.595→0.463），此时不应使用校准。**"
                "另外：使用校准概率时必须重新选择工作点，0.5 通常不再适用。",
        "disclaimer": cfg.DISCLAIMER,
    }
    out_name = "calibration.json" if base_model != "knn" else "calibration_knn.json"
    ev.save_json(result, os.path.join(cfg.METRICS_DIR, out_name))

    if verbose:
        b, e, a = result["brier"], result["ece"], result["auc"]
        print()
        print(f"  Brier 分数：{b['raw']:.4f} → {b['calibrated']:.4f}"
              f"  （越小越好，降幅 {(b['raw'] - b['calibrated']) / max(b['raw'], 1e-9) * 100:.1f}%）")
        print(f"  ECE（校准误差）：{e['raw']:.4f} → {e['calibrated']:.4f}")
        print(f"  AUC（排序能力）：{a['raw']:.4f} → {a['calibrated']:.4f}"
              f"  ← 应几乎不变；若明显下降说明基础模型输出已退化，不应校准")
        print(f"  概率取值个数：{result['n_unique_proba']['raw']} → "
              f"{result['n_unique_proba']['calibrated']}")
        print()
        print("  分档人数（低/中/高）：")
        print(f"    校准前（默认阈值）：{result['level_counts_raw']}")
        print(f"    校准后（默认阈值）：{result['level_counts_calibrated_default_thresholds']}"
              f"   ← 0.3/0.7 是针对未校准概率设的，刻度变了就不再适用")
        print(f"    校准后（重标定阈值 {thresholds_cal}）："
              f"{result['level_counts_calibrated_retuned']}   ← 三档才算真正可用")
        print()
        print(f"  校准后 F1 最优阈值 = {best_th:.2f}"
              f"（说明：校准后 0.5 通常不再是合适工作点，必须重新选阈值）")
        for tag, key in (("原始概率@0.5", "metrics_raw"),
                         ("校准概率@0.5", "metrics_calibrated_at_0.5"),
                         (f"校准概率@{best_th:.2f}", "metrics_calibrated_at_best")):
            m = result[key]
            print(f"    {tag:<18} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        print()
        print(f"  已保存：results/metrics/{out_name}")
        print("=" * 78)
    return result


def main() -> None:
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    method = "isotonic" if "--isotonic" in sys.argv else "sigmoid"
    base = "knn" if "--knn" in sys.argv else "balanced"
    run_calibration(method=method, base_model=base)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
