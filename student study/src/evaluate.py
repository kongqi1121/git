# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估指标模块
=========================================================
本模块只做一件事：把"真实标签 + 预测结果"翻译成可报告、可手工复算的指标。

为什么单独成模块？
    留一法、10 次随机划分、新样本测试三套口径都要算同一批指标，
    如果每处各写一遍，很容易出现 precision 分母用错（TP+FP 还是 n）
    之类的口径漂移。集中一处后，experiments.md 里的数字才有唯一来源。

【先复习混淆矩阵】（答辩必考）
                        预测：正常(0)   预测：需关注(1)
    实际：正常(0)            TN              FP
    实际：需关注(1)          FN              TP

    accuracy  = (TP + TN) / (TP + TN + FP + FN)   整体判对比例
    precision = TP / (TP + FP)                    判为需关注的人里，真的有多少确实需关注（查准）
    recall    = TP / (TP + FN)                    真的需关注的人里，抓出来多少（查全）
    F1        = 2·P·R / (P + R)                   查准与查全的调和平均
    specificity = TN / (TN + FP)                  正常学生被正确放过的比例

【本项目的业务取向】
    预警系统的核心风险是"漏报"（把真需要关注的学生判成正常），
    因此 **recall 比 precision 更重要**；但辅导员人力有限，precision 太低会导致
    "狼来了"，反而消耗信任。项目最终以 F1 为主指标，同时在阈值分析中给出
    recall/precision 的权衡曲线，让老师按人力情况自行选择工作点。
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ======================================================================
# 一、基础计数与指标
# ======================================================================
def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """统计混淆矩阵四个格子。正类固定为 1（需要重点关注）。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def confusion_matrix_array(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """
    返回 2×2 混淆矩阵，行=真实类别，列=预测类别，顺序为 [0, 1]。
    与 sklearn.metrics.confusion_matrix(labels=[0,1]) 的布局一致，便于对照。
    """
    c = confusion_counts(y_true, y_pred)
    return np.array([[c["TN"], c["FP"]],
                     [c["FN"], c["TP"]]], dtype=int)


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    """
    由混淆矩阵四个数推导全部指标（**这是人工复算要用的函数**）。

    边界情况处理（教学项目必须交代清楚，否则指标会出现 nan）：
      - TP+FP = 0（一个都没判为高风险）→ precision 无定义，约定取 0.0；
      - TP+FN = 0（测试集里没有高风险学生）→ recall 无定义，约定取 0.0；
      - P+R = 0 → F1 取 0.0。
    """
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # 负类查准率
    balanced_acc = 0.5 * (recall + specificity)             # 类别不平衡时比 accuracy 更公平

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": float(balanced_acc),
    }


def evaluate_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """一次性得到混淆矩阵计数 + 全部指标（主口径：正类=1）。"""
    c = confusion_counts(y_true, y_pred)
    out = dict(c)
    out.update(metrics_from_counts(c))
    return out


# ======================================================================
# 二、以"高风险"为关注类的完整报告（含 F1 的另一种算法用于交叉验证）
# ======================================================================
def full_report(y_true: Sequence[int], y_pred: Sequence[int],
                model_name: str = "model") -> Dict:
    """
    生成结构化评估报告。除指标外还附带：
      - f1_from_formula : 用 2TP/(2TP+FP+FN) 直接算的 F1（与 2PR/(P+R) 恒等）
      - support         : 各真实类别的样本数
    两者可互相校验，防止公式抄错。
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    m = evaluate_binary(y_true, y_pred)
    cm = confusion_matrix_array(y_true, y_pred)

    tp, fp, fn = m["TP"], m["FP"], m["FN"]
    # F1 的等价公式，用于交叉验证（两个公式结果一致说明实现正确）
    f1_from_formula = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "model": model_name,
        "n": int(len(y_true)),
        "confusion_matrix": cm.tolist(),      # [[TN, FP], [FN, TP]]
        "counts": {"TP": tp, "TN": m["TN"], "FP": fp, "FN": fn},
        "metrics": {k: round(float(v), 6) for k, v in m.items()},
        "f1_check": round(float(f1_from_formula), 6),
        "support": {"正常(0)": int((y_true == 0).sum()), "需关注(1)": int((y_true == 1).sum())},
        "pred_distribution": {"预测正常(0)": int((y_pred == 0).sum()),
                              "预测需关注(1)": int((y_pred == 1).sum())},
    }


# ======================================================================
# 三、多指标汇总（10 次划分的均值与标准差）
# ======================================================================
def aggregate_folds(reports: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    把多次重复实验的报告聚合为"均值 ± 标准差"。

    【为什么必须报方差】
    单次随机划分的指标受划分本身影响很大（同一模型不同 seed 可能差 5 个百分点以上）。
    只报一次最优结果属于选择性地报告，实验结论不可靠。
    因此本项目固定报告 10 次划分的均值与标准差，并把每次明细写入 CSV 以便复核。
    """
    keys = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in reports], dtype=float)
        out[k] = {
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6),
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
        }
    # 混淆矩阵也按元素累加，给出"10 次划分合计"的总体混淆矩阵
    cms = np.array([r["confusion_matrix"] for r in reports], dtype=float)
    out["confusion_sum"] = {"value": cms.sum(axis=0).astype(int).tolist()}
    return out


# ======================================================================
# 四、阈值与等级评估
# ======================================================================
def threshold_sweep(y_true: Sequence[int], proba_pos: Sequence[float]) -> List[Dict]:
    """
    预测阈值敏感性分析：把判定阈值从 0.1 扫到 0.9，观察 precision/recall/F1 的变化。

    用途：辅导员人力有限时可据此选择工作点——
      阈值调低 → recall 上升、precision 下降（多报，少漏）；
      阈值调高 → precision 上升、recall 下降（少报，可能漏人）。
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    proba_pos = np.asarray(proba_pos, dtype=float)
    rows = []
    for th in np.round(np.arange(0.10, 0.95, 0.05), 2):
        pred = (proba_pos >= th).astype(int)
        m = evaluate_binary(y_true, pred)
        rows.append({
            "threshold": float(th),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "n_predicted_risk": int(pred.sum()),
        })
    return rows


def evaluate_risk_levels(level_true: Sequence[str], level_pred: Sequence[str],
                         labels: Optional[List[str]] = None) -> Dict:
    """
    评估三档风险等级（低/中/高）的一致性。

    说明：risk_level 的"真实值"由数据生成时的分位数定档得到，与 risk_label 口径不同，
    因此等级准确率通常会明显低于二分类准确率 —— 这是正常的，不是 bug。
    该指标用于回答"三档标签本身是否可靠"，在 README 的风险对策中说明。
    """
    labels = list(cfg.RISK_LEVELS if labels is None else labels)
    level_true = list(map(str, level_true))
    level_pred = list(map(str, level_pred))
    n = len(level_true)

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {lab: i for i, lab in enumerate(labels)}
    for t, p in zip(level_true, level_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    acc = float(np.trace(cm) / n) if n else 0.0
    # 相邻档位算"部分正确"：把"高风险误判为中风险"视为可接受的降级（辅导员仍会关注）
    near = 0
    for t, p in zip(level_true, level_pred):
        if t == p:
            near += 1
        elif t in idx and p in idx and abs(idx[t] - idx[p]) == 1:
            near += 1

    per_class = {}
    for lab in labels:
        i = idx[lab]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4), "support": int(cm[i, :].sum())}

    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "accuracy": round(acc, 4),
        "accuracy_with_neighbor": round(near / n, 4) if n else 0.0,
        "per_class": per_class,
        "n": int(n),
        "note": "等级标签与二分类标签口径不同，等级准确率天然低于二分类准确率，属预期现象",
    }


# ======================================================================
# 五、人工复算（验收要求 5：至少 2 项指标手工核对）
# ======================================================================
def manual_recompute(c: Dict[str, int]) -> Dict[str, object]:
    """
    **人工复算函数**：只用混淆矩阵的四个整数，按定义一步步手算出 precision 与 recall，
    并把每一步的中间过程以字符串形式写出来，供 experiments.md 直接粘贴。

    这不是重复实现，而是"用另一种写法再算一遍"，
    用于验证 metrics_from_counts 的实现没有写错分母。
    """
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]

    # ---- 人工复算 1：precision = TP / (TP + FP) ----
    p_num, p_den = tp, tp + fp
    precision = (p_num / p_den) if p_den else 0.0
    p_steps = f"precision = TP/(TP+FP) = {p_num}/{p_den} = {precision:.6f}"

    # ---- 人工复算 2：recall = TP / (TP + FN) ----
    r_num, r_den = tp, tp + fn
    recall = (r_num / r_den) if r_den else 0.0
    r_steps = f"recall = TP/(TP+FN) = {r_num}/{r_den} = {recall:.6f}"

    # ---- 附带复算 3：accuracy 与 F1（用不同公式交叉验证） ----
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    a_steps = f"accuracy = (TP+TN)/N = ({tp}+{tn})/{total} = {accuracy:.6f}"

    f1_a = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    f1_b = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f_steps = (f"F1 = 2PR/(P+R) = {f1_a:.6f}；"
               f"交叉验证公式 2TP/(2TP+FP+FN) = {f1_b:.6f}"
               f"（两者一致：{abs(f1_a - f1_b) < 1e-9}）")

    return {
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "precision_manual": precision,
        "recall_manual": recall,
        "accuracy_manual": accuracy,
        "f1_manual_formula_a": f1_a,
        "f1_manual_formula_b": f1_b,
        "steps": {"precision": p_steps, "recall": r_steps,
                  "accuracy": a_steps, "f1": f_steps},
    }


def verify_against_implementation(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """
    把"人工复算值"与"程序实现值"逐项比对，输出是否一致的结论。
    这是验收要求"人工复算至少 2 项指标"的可执行证据。
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    c = confusion_counts(y_true, y_pred)
    manual = manual_recompute(c)
    impl = metrics_from_counts(c)

    checks = {
        "precision": abs(manual["precision_manual"] - impl["precision"]) < 1e-9,
        "recall": abs(manual["recall_manual"] - impl["recall"]) < 1e-9,
        "accuracy": abs(manual["accuracy_manual"] - impl["accuracy"]) < 1e-9,
        "f1_two_formulas": abs(manual["f1_manual_formula_a"] - manual["f1_manual_formula_b"]) < 1e-9,
    }
    return {
        "counts": c,
        "manual": manual,
        "implementation": {k: round(float(v), 6) for k, v in impl.items()},
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "steps": manual["steps"],
    }


# ======================================================================
# 六、输出辅助
# ======================================================================
def format_report(report: Dict, title: str = "") -> str:
    """把 full_report 的结果格式化成对齐的中文文本，方便终端阅读与粘贴进报告。"""
    m = report["metrics"]
    cm = report["confusion_matrix"]
    lines = []
    if title:
        lines.append(f"—— {title} ——")
    lines.append(f"样本数: {report['n']}    "
                 f"真实分布: {report['support']}    预测分布: {report['pred_distribution']}")
    lines.append("混淆矩阵（行=真实，列=预测，类序 [0正常, 1需关注]）:")
    lines.append("             预测正常(0)   预测需关注(1)")
    lines.append(f"  实际正常(0)   {cm[0][0]:>8d}      {cm[0][1]:>8d}")
    lines.append(f"  实际需关注(1) {cm[1][0]:>8d}      {cm[1][1]:>8d}")
    lines.append(f"accuracy ={m['accuracy']:.4f}   precision={m['precision']:.4f}   "
                 f"recall ={m['recall']:.4f}   F1 ={m['f1']:.4f}")
    lines.append(f"specificity={m['specificity']:.4f}   balanced_acc={m['balanced_accuracy']:.4f}")
    return "\n".join(lines)


def save_json(obj, path: str) -> str:
    """保存 JSON（统一 utf-8，ensure_ascii=False 便于中文直接阅读）。"""
    cfg.ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def _json_default(o):
    """json 序列化兜底：numpy 标量/数组转 Python 原生类型。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"无法序列化类型：{type(o)}")


def main() -> None:
    """自检：用一个人为构造的混淆矩阵，验证指标与人工复算完全一致。"""
    print("=" * 68)
    print("【自检】evaluate 指标与人工复算一致性")
    print("=" * 68)
    # 构造混淆矩阵：TN=88, FP=12, FN=7, TP=23（合计 130）
    y_true = np.array([0] * 100 + [1] * 30)
    y_pred = np.array([0] * 88 + [1] * 12 + [0] * 7 + [1] * 23)
    res = verify_against_implementation(y_true, y_pred)
    rep = full_report(y_true, y_pred, "自检用例")
    print(format_report(rep))
    print()
    for name, step in res["steps"].items():
        print("  " + step)
    print("逐项一致性检查：", res["checks"], "→", "全部通过 ✅" if res["all_passed"] else "存在不一致 ❌")
    assert res["all_passed"]
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
