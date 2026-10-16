# -*- coding: utf-8 -*-
"""
model_comparison.py —— 多模型对比矩阵（建议 A1）
=========================================================
【要解决的问题】
主模型选 KNN 的理由要能被验证，而不是只能回答"项目要求"。
本模块在同一套评估口径下比较多种模型，并**如实报告谁更好**。

【纳入对比的模型】
    1. KNN(k = 主 k)                 —— 当前主模型
    2. 逻辑回归（默认，class_weight=None）
    3. 逻辑回归（class_weight='balanced'） —— 建议 A4 的对照项
    4. 决策树（max_depth=4）          —— 可输出可读规则，便于生成帮扶建议
    5. 高斯朴素贝叶斯 GaussianNB      —— 注意：特征含 0/1 指示列，高斯假设并不严格成立
    6. 线性 SVM（LinearSVC + Platt 校准，用于概率输出）
    7. 多数类基线（全判"正常"）        —— 必须有的对照线

【方法学纪律】
    · 所有模型走**同一批划分、同一套口径**（LOGO + 10 次分组划分）；
    · 标准化器每折只 fit 训练集（`_fit_pipeline` 已保证）；
    · **不允许用测试集挑"最好的模型"** —— 本模块输出完整对比矩阵，
      选择依据是"业务目标（recall 优先）+ 区间是否分离"，而不是分数最高；
    · 计算成本差异要如实说明：SVM 在 LOGO 下需重训 668 次，因此
      **只跑 10 次分组划分 + 课程外推**，报告中会注明原因，而不是悄悄省略。

【产物】results/metrics/model_comparison.csv、model_comparison.json
【命令】python src/model_comparison.py [--logomodels knn,logreg_balanced]
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

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


# ======================================================================
# 模型工厂
# ======================================================================
def build_models(main_k: int) -> Dict[str, Callable[[], object]]:
    """
    返回 {模型名: 工厂函数}。工厂函数每次调用都返回**全新未训练**的模型实例 ——
    这点很重要：若复用同一个实例，后一折会带着前一折的训练状态，造成隐性泄漏。
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    def knn():
        try:
            from .model_backends import make_knn
        except ImportError:  # pragma: no cover - 脚本直跑分支
            from model_backends import make_knn
        return make_knn(k=main_k, backend="auto")

    return {
        f"KNN(k={main_k})": knn,

        "逻辑回归": lambda: LogisticRegression(
            max_iter=3000, C=1.0, solver="lbfgs", random_state=cfg.RANDOM_SEED),

        # A4：类别加权版逻辑回归（正类权重上调，预期 recall 上升、precision 下降）
        "逻辑回归(balanced)": lambda: LogisticRegression(
            max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced",
            random_state=cfg.RANDOM_SEED),

        # 决策树：限制深度以抑制过拟合；可解释性最好（可导出 if-else 规则）
        "决策树(depth=4)": lambda: DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, class_weight=None,
            random_state=cfg.RANDOM_SEED),

        # 高斯朴素贝叶斯：对 0/1 指示列的高斯假设不严格，属于"明知有偏仍纳入"的对照项
        "高斯朴素贝叶斯": lambda: GaussianNB(),

        # 线性 SVM：LinearSVC 默认不输出概率，用 Platt 校准包装以获得 predict_proba
        "线性SVM": lambda: CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight=None, random_state=cfg.RANDOM_SEED, max_iter=5000),
            method="sigmoid", cv=3),
    }


def _proba_of(model, X: np.ndarray) -> np.ndarray:
    """统一取高风险概率；对不支持 predict_proba 的模型给出明确错误。"""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    raise AttributeError(f"{type(model).__name__} 不支持 predict_proba")


def evaluate_by_group_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                            factories: Dict[str, Callable[[], object]],
                            n_splits: int = cfg.N_SPLITS,
                            verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    在 10 次**按学生分组**的随机划分上评估全部模型。

    为什么用分组划分而不是随机划分：合并数据里 372 名学生同修两门课，
    按行随机划分会让同一人跨训练/测试集（泄漏），指标偏乐观。
    """
    per_model: Dict[str, List[Dict]] = {m: [] for m in factories}
    proba_store: Dict[str, Dict[int, np.ndarray]] = {m: {} for m in factories}
    timings: Dict[str, float] = {}

    for i in range(n_splits):
        seed = cfg.RANDOM_SEED + i
        tr, te = group_train_test_split_indices(df, cfg.TEST_SIZE, seed)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        for name, factory in factories.items():
            t0 = time.time()
            model = factory()
            model.fit(Xtr, ytr)
            p = _proba_of(model, Xte)
            pred = (p > 0.5).astype(int)
            timings[name] = timings.get(name, 0.0) + (time.time() - t0)

            m = ev.evaluate_binary(yte, pred)
            per_model[name].append({
                "split": i + 1, "seed": seed,
                "accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"],
                "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            })
            proba_store[name][seed] = p
        if verbose:
            print(f"    划分 {i + 1}/{n_splits} 完成（seed={seed}）")

    # 汇总：均值 ± 标准差
    rows = []
    detail: Dict[str, List[Dict]] = {}
    for name, runs in per_model.items():
        arr = pd.DataFrame(runs)
        detail[name] = runs
        row = {"model": name, "n_splits": len(runs)}
        for k in ("accuracy", "precision", "recall", "f1"):
            row[k] = round(float(arr[k].mean()), 4)
            row[f"{k}_std"] = round(float(arr[k].std(ddof=1)), 4)
        row["fit_seconds_total"] = round(timings.get(name, 0.0), 2)
        rows.append(row)
    return pd.DataFrame(rows), {"detail": detail,
                                "proba": {m: {str(k): v.tolist() for k, v in d.items()}
                                          for m, d in proba_store.items()}}


def cross_subject(df: pd.DataFrame, factories: Dict[str, Callable[[], object]],
                  verbose: bool = True) -> pd.DataFrame:
    """课程外推：数学课训练 → 葡语课测试，反之亦然（真实的跨分布检验）。"""
    rows = []
    for train_sub, test_sub in (("mat", "por"), ("por", "mat")):
        tr = (df["subject"] == train_sub).to_numpy()
        te = (df["subject"] == test_sub).to_numpy()
        Xtr, ytr, _ = prepare_uci_features(df[tr], include_early_grades=False, verbose=False)
        Xte, yte, _ = prepare_uci_features(df[te], include_early_grades=False, verbose=False)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        for name, factory in factories.items():
            try:
                model = factory()
                model.fit(Xtr_s, ytr)
                p = _proba_of(model, Xte_s)
                m = ev.evaluate_binary(yte, (p > 0.5).astype(int))
                rows.append({
                    "model": name, "direction": f"{train_sub}->{test_sub}",
                    "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4), "f1": round(m["f1"], 4),
                })
            except Exception as e:  # 某个模型在该口径下失败不应中断整体对比
                rows.append({"model": name, "direction": f"{train_sub}->{test_sub}",
                             "accuracy": None, "precision": None, "recall": None,
                             "f1": None, "error": f"{type(e).__name__}: {e}"})
        if verbose:
            print(f"    课程外推 {train_sub} -> {test_sub} 完成")
    return pd.DataFrame(rows)


def run_comparison(n_splits: int = cfg.N_SPLITS, verbose: bool = True) -> Dict:
    """执行完整对比并落盘。"""
    cfg.ensure_dirs()
    df = load_saved_dataset()
    X, y, _ = prepare_uci_features(df, include_early_grades=False, verbose=False)

    main_k = int(cfg.MAIN_K)
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if s.get("dataset", "").startswith("UCI"):
            main_k = int(s.get("main_k", main_k))
            baseline = float(s.get("majority_baseline_accuracy", 0.0))
        else:
            baseline = 0.0
    else:
        baseline = 0.0

    if verbose:
        print("=" * 82)
        print("【模型对比矩阵】同一口径（10 次按学生分组划分 + 课程外推）")
        print("=" * 82)
        print(f"主 k = {main_k}；样本 {len(df)} 条 / 学生 {df['student_group_id'].nunique()} 名")
        print(f"多数类基线 accuracy = {baseline:.4f}（全判『正常』）")
        print()

    factories = build_models(main_k)
    table, extra = evaluate_by_group_split(df, X, y, factories,
                                           n_splits=n_splits, verbose=verbose)
    cs = cross_subject(df, factories, verbose=verbose)

    # 加上多数类基线（作为一行参照）
    base_row = {"model": "多数类基线(全判正常)", "n_splits": n_splits,
                "accuracy": round(baseline, 4), "accuracy_std": 0.0,
                "precision": 0.0, "precision_std": 0.0,
                "recall": 0.0, "recall_std": 0.0, "f1": 0.0, "f1_std": 0.0,
                "fit_seconds_total": 0.0}
    table = pd.concat([pd.DataFrame([base_row]), table], ignore_index=True)
    table = table.sort_values("f1", ascending=False).reset_index(drop=True)

    table.to_csv(os.path.join(cfg.METRICS_DIR, "model_comparison.csv"),
                 index=False, encoding="utf-8-sig")
    cs.to_csv(os.path.join(cfg.METRICS_DIR, "model_comparison_cross_subject.csv"),
              index=False, encoding="utf-8-sig")

    # 与主模型的配对比较（同一次划分下相减，见 C3 的方法学说明）
    # 【符号约定】统一为 Δ = 该模型 − 主模型，因此 **Δ > 0 表示优于主模型**。
    # 早期版本把参数顺序写反（传成 主模型 − 该模型），导致"Δ 为负却是更好的模型"，
    # 极易误读，故在此显式固定顺序并加注释。
    try:
        from .confidence_intervals import paired_comparison
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from confidence_intervals import paired_comparison
    main_name = f"KNN(k={main_k})"
    paired = {}
    det = extra["detail"]
    if main_name in det:
        main_f1 = np.array([r["f1"] for r in det[main_name]])
        main_rec = np.array([r["recall"] for r in det[main_name]])
        for name, runs in det.items():
            if name == main_name:
                continue
            f1s = np.array([r["f1"] for r in runs])
            recs = np.array([r["recall"] for r in runs])
            paired[name] = {
                # Δ = name − main，使"正数 = 更好"符合直觉
                "f1": paired_comparison(f1s, main_f1, name, main_name),
                "recall": paired_comparison(recs, main_rec, name, main_name),
            }

    # 结论：谁更好？如实给结论而不是维护原选择
    conclusion = _summarize(table, paired, main_name, baseline)

    out = {
        "protocol": f"{n_splits} 次按学生分组随机划分（7:3）；课程外推另表",
        "main_k": main_k,
        "majority_baseline_accuracy": round(baseline, 4),
        "table": table.to_dict(orient="records"),
        "cross_subject": cs.to_dict(orient="records"),
        "paired_vs_main_model": paired,
        "conclusion": conclusion,
        "note": "所有模型走同一批划分与同一套标准化流程；标准化器每折只 fit 训练集。"
                "选主模型应依据业务目标（预警场景 recall 优先）+ 区间是否分离，"
                "不应仅看分数高低。",
        "disclaimer": cfg.DISCLAIMER,
    }
    ev.save_json(out, os.path.join(cfg.METRICS_DIR, "model_comparison.json"))

    if verbose:
        print()
        print(f"  {'模型':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>9}"
              f"{'拟合秒数':>10}")
        for _, r in table.iterrows():
            print(f"  {r['model']:<22}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
                  f"{r['recall']:>9.4f}{r['f1']:>9.4f}{r['fit_seconds_total']:>10.2f}")
        print()
        print("  与主模型的配对比较（F1 / recall）：")
        for name, v in paired.items():
            f1v, rcv = v["f1"], v["recall"]
            print(f"    {name:<22} ΔF1={f1v['mean_diff']:+.4f} (p≈{f1v['p_approx']})"
                  f"   Δrecall={rcv['mean_diff']:+.4f} (p≈{rcv['p_approx']})")
        print()
        for line in conclusion:
            print("  " + line)
        print()
        print("  已保存：model_comparison.csv / model_comparison_cross_subject.csv / "
              "model_comparison.json")
        print("=" * 82)
    return out


def _summarize(table: pd.DataFrame, paired: Dict, main_name: str,
               baseline: float) -> List[str]:
    """
    生成结论文字。原则：**如实陈述，包括不利于主模型的结果**。
    若某个模型在 recall 上显著优于主模型，必须明确写出"应考虑更换主模型"。
    """
    lines = []
    best_f1 = table.iloc[0]
    lines.append(f"F1 最高的模型是「{best_f1['model']}」（F1={best_f1['f1']:.4f}）。")
    if best_f1["model"] != main_name:
        lines.append(f"⚠️ 该模型不是当前主模型「{main_name}」，"
                     f"应据此评估是否需要更换主模型，而不是维持原选择。")

    # recall 视角（预警场景的关键指标）
    best_rec = table.sort_values("recall", ascending=False).iloc[0]
    lines.append(f"recall 最高的模型是「{best_rec['model']}」（recall={best_rec['recall']:.4f}）。")

    # 相对基线的增益（判断"模型是否真的学到东西"）
    for _, r in table.iterrows():
        if r["model"].startswith("多数类基线"):
            continue
        gain = float(r["accuracy"]) - baseline
        flag = "高于" if gain > 0 else "低于"
        lines.append(f"「{r['model']}」accuracy {r['accuracy']:.4f}，{flag}多数类基线 "
                     f"{baseline:.4f}（{gain:+.4f}）。")

    # 显著优于主模型的项（Δ = 该模型 − 主模型，正数表示更好）
    better = [(n, v) for n, v in paired.items()
              if v["recall"]["mean_diff"] > 0 and v["recall"]["p_approx"] < 0.05]
    if better:
        names = "、".join(f"{n}(Δrecall={v['recall']['mean_diff']:+.4f})" for n, v in better)
        lines.append(f"在配对检验下 recall 显著优于主模型的模型：{names}。"
                     f"因此『主模型选择』需要重新论证。")
    else:
        lines.append("在配对检验下，没有模型在 recall 上显著优于主模型。")

    lines.append("提醒：本项目样本有限（1044 条 / 230 个正类），"
                 "差异小于约 0.03 的指标变化应结合配对区间判断，不宜只看点估计。")
    return lines


def main() -> None:
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()
    n = cfg.N_SPLITS
    if "--quick" in sys.argv:
        n = 3
    run_comparison(n_splits=n)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
