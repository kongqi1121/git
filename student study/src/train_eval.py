# -*- coding: utf-8 -*-
"""
train_eval.py —— 训练与评估主流程
=========================================================
这是项目的"实验中枢"，被 main.py 的菜单项 2 / --auto 调用。完整流程：

    ① 读取模拟数据 + 新样本测试集，做数据体检
    ② 在训练集上 fit 缺失值填补器与标准化器（防止数据泄漏）
    ③ k 值扫描：留一法（LOO）+ 10 次随机划分，双口径对比 9 个候选 k
    ④ 依据规则选出主 k，并给出可答辩的选择理由
    ⑤ 用主 k 训练最终 KNN；同时训练对照模型（逻辑回归，sklearn 优先 / 手写兜底）
    ⑥ 在**从未参与训练的 100 条新样本**上做最终测试（两个模型同口径对比）
    ⑦ 人工复算：由混淆矩阵手算 precision / recall / accuracy / F1 并与程序实现比对
    ⑧ 保存模型、预处理统计量、全部指标（JSON + CSV）与留一法逐样本预测

输出文件（全部落在 results/ 下）：
    models/models.npz                 训练好的 KNN 与逻辑回归参数 + 标准化参数
    models/preprocess_stats.json      填补中位数、均值/标准差、风险等级阈值
    metrics/metrics_summary.json      汇总（含 k 扫描、双口径、新样本、人工复算）
    metrics/metrics_table.csv         指标总表（便于贴进报告）
    metrics/k_scan.csv                k 值对比明细
    metrics/split_details.csv         10 次划分逐次明细（含随机种子，可复现）
    metrics/new_batch_metrics.csv     新样本测试指标表
    metrics/full_predictions.csv      全体样本留一法预测（学生风险名单）
    metrics/manual_verification.json  人工复算过程与一致性检查
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .generate_data import simulate_students, save_simulation
    from .knn_numpy import (KNNClassifierNumpy, assign_risk_levels, loo_predict_proba,
                            proba_to_risk_level)
    from .logreg_numpy import LogisticRegressionNumpy
    from .preprocess import (Dataset, MedianImputer, StandardScaler, build_dataset,
                             load_simulated, random_split, save_preprocess_stats,
                             validate_dataframe)
    from .cross_validate import k_scan, select_best_k, repeated_random_split
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from generate_data import simulate_students, save_simulation
    from knn_numpy import (KNNClassifierNumpy, assign_risk_levels, loo_predict_proba,
                           proba_to_risk_level)
    from logreg_numpy import LogisticRegressionNumpy
    from preprocess import (Dataset, MedianImputer, StandardScaler, build_dataset,
                           load_simulated, random_split, save_preprocess_stats,
                           validate_dataframe)
    from cross_validate import k_scan, select_best_k, repeated_random_split


# ======================================================================
# 一、对照模型后端选择（sklearn 优先，缺失则用手写实现）
# ======================================================================
def choose_logreg_backend() -> Dict:
    """
    探测环境里是否有 sklearn：
      - 有 → 使用 sklearn.linear_model.LogisticRegression（实训约束的首选）；
      - 没有 → 使用本项目的 numpy 手写逻辑回归。
    本机实测无 sklearn，因此实际走手写分支，实验记录中会如实写明。
    """
    try:
        import sklearn  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        return {"backend": "sklearn", "version": getattr(sklearn, "__version__", "unknown"),
                "detail": "检测到 scikit-learn，对照模型使用 sklearn.linear_model.LogisticRegression"}
    except Exception as e:  # ImportError 或其他导入异常
        return {"backend": "numpy_manual", "version": None,
                "detail": f"未检测到可用的 scikit-learn（{type(e).__name__}: {e}），"
                          f"按实训约束改用手写 numpy 逻辑回归作为对照模型"}


def make_logreg(backend: str, random_state: int = cfg.RANDOM_SEED):
    """按后端创建对照模型实例（保持统一的 fit/predict/predict_proba 接口）。"""
    if backend == "sklearn":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                  random_state=random_state)
    return LogisticRegressionNumpy(lr=cfg.LOGREG_LR, epochs=cfg.LOGREG_EPOCHS,
                                   l2=cfg.LOGREG_L2, random_state=random_state)


def logreg_params(clf, backend: str) -> Dict:
    """提取对照模型的可读参数（写进实验记录用于答辩）。"""
    if backend == "sklearn":
        return {"C": float(getattr(clf, "C", 1.0)), "solver": getattr(clf, "solver", "lbfgs"),
                "coef": {k: round(float(v), 4) for k, v in zip(cfg.FEATURE_COLUMNS, clf.coef_.ravel())},
                "intercept": round(float(clf.intercept_[0]), 4)}
    return clf.params_dict()


# ======================================================================
# 二、新样本（从未参与训练）测试集准备
# ======================================================================
def load_or_make_new_batch(n_new: int = cfg.N_NEW_SAMPLES,
                           seed: int = cfg.NEW_BATCH_SEED) -> pd.DataFrame:
    """
    读取"从未参与训练"的新样本测试集；不存在则现场生成。

    为什么必须有这一步？
      训练集上的留一法/交叉验证都属于"在同一批数据内部反复评估"，
      模型多少会间接适配这批数据的分布。用一批**完全独立生成**的新样本做最终测试，
      才能回答"换一批新学生，系统还准吗"这个真正重要的问题。
    """
    if os.path.exists(cfg.NEW_BATCH_PATH):
        df = pd.read_csv(cfg.NEW_BATCH_PATH, encoding="utf-8-sig")
        # 样本数与要求不符时重新生成，避免读到旧的、规模不对的文件
        if len(df) == n_new:
            return df
    df, _ = simulate_students(n_samples=n_new, random_state=seed,
                              risk_ratio_target=cfg.RISK_RATIO_TARGET,
                              enforce_size=False)  # 新样本批次允许小于 300 条
    df["student_id"] = [f"NEW{i:04d}" for i in range(1, len(df) + 1)]
    save_simulation(df, cfg.NEW_BATCH_PATH)
    return df


# ======================================================================
# 三、模型保存与加载
# ======================================================================
def save_models(knn: KNNClassifierNumpy, logreg, backend: str,
                imputer: MedianImputer, scaler: StandardScaler,
                path: str = cfg.MODEL_PATH) -> str:
    """
    保存模型到单个 .npz 文件（numpy 原生格式，无需 pickle，无 sklearn 依赖也能读）。

    保存内容：
      KNN        : 训练特征矩阵、标签、k（KNN 是懒惰学习，"模型"就是训练数据本身）
      逻辑回归    : 权重 w 与偏置 b（手写实现）；sklearn 后端则单独存 coef/intercept
      预处理      : 填补中位数、标准化均值/标准差、特征列顺序、风险等级阈值
    特征列顺序必须一起保存：否则查询时列顺序错位会导致"静默的错误预测"。
    """
    cfg.ensure_dirs()
    payload = {
        "feature_columns": np.array(cfg.FEATURE_COLUMNS, dtype=object),
        "knn_k": np.array([knn.k]),
        "knn_X_train": knn.X_train_,
        "knn_y_train": knn.y_train_,
        "imputer_fill": imputer.fill_values_,
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "logreg_backend": np.array([backend], dtype=object),
        "risk_th_low": np.array([cfg.RISK_LEVEL_THRESHOLDS["low"]]),
        "risk_th_high": np.array([cfg.RISK_LEVEL_THRESHOLDS["high"]]),
    }
    if backend == "sklearn":
        payload["logreg_w"] = np.asarray(logreg.coef_, dtype=float).ravel()
        payload["logreg_b"] = np.asarray(logreg.intercept_, dtype=float).ravel()
    else:
        payload["logreg_w"] = np.asarray(logreg.w_, dtype=float)
        payload["logreg_b"] = np.asarray([logreg.b_], dtype=float)
    np.savez_compressed(path, **payload)
    return path


def load_saved_models(path: str = cfg.MODEL_PATH) -> Dict:
    """
    加载模型与预处理参数，供 query_student.py / explain.py 复用。

    返回 dict：{knn, logreg, imputer, scaler, feature_columns, thresholds, backend}
    其中 logreg 统一用 LogisticRegressionNumpy 重建（参数已在训练时保存），
    因此即使答辩环境没有 sklearn 也能加载并预测。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件 {path}，请先运行训练评估：python main.py --step train")
    data = np.load(path, allow_pickle=True)

    feature_columns = [str(c) for c in data["feature_columns"]]
    if feature_columns != cfg.FEATURE_COLUMNS:
        raise ValueError(f"模型的特征列顺序与当前配置不一致：{feature_columns} vs {cfg.FEATURE_COLUMNS}")

    knn = KNNClassifierNumpy(k=int(data["knn_k"][0])).fit(data["knn_X_train"], data["knn_y_train"])

    logreg = LogisticRegressionNumpy()
    logreg.w_ = np.asarray(data["logreg_w"], dtype=float)
    logreg.b_ = float(np.asarray(data["logreg_b"], dtype=float).ravel()[0])

    imputer = MedianImputer()
    imputer.fill_values_ = np.asarray(data["imputer_fill"], dtype=float)

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(data["scaler_mean"], dtype=float)
    scaler.std_ = np.asarray(data["scaler_std"], dtype=float)

    return {
        "knn": knn, "logreg": logreg, "imputer": imputer, "scaler": scaler,
        "feature_columns": feature_columns,
        "thresholds": {"low": float(data["risk_th_low"][0]), "high": float(data["risk_th_high"][0])},
        "backend": str(data["logreg_backend"][0]),
    }


# ======================================================================
# 四、主流程
# ======================================================================
def run_training_eval(main_k: Optional[int] = None,
                      n_splits: int = cfg.N_SPLITS,
                      seed: int = cfg.RANDOM_SEED,
                      k_list: Optional[List[int]] = None,
                      verbose: bool = True) -> Dict:
    """执行完整训练与评估流程，返回汇总字典（同时写入 results/）。"""
    cfg.ensure_dirs()
    t0 = time.time()
    summary: Dict = {"seed": seed, "n_splits": n_splits,
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if verbose:
        print("=" * 68)
        print("【步骤】训练与评估（主模型：手写 KNN；对照模型：逻辑回归）")
        print("=" * 68)

    # ---------------- ① 读数据 ----------------
    df = load_simulated()
    if len(df) < 300:
        # 数据规模不足时自动重生成，保证满足 300~500 的硬性要求
        if verbose:
            print(f"[警告] 现有数据仅 {len(df)} 条，少于要求的 300 条，自动重新生成……")
        df, gen_report = simulate_students(n_samples=cfg.N_SAMPLES, random_state=seed)
        save_simulation(df)
        summary["regenerated"] = True
    health = validate_dataframe(df)
    summary["data_health"] = health
    if verbose:
        print(f"[1/8] 数据体检：样本 {health['n_samples']} 条，特征 {health['n_features']} 个，"
              f"缺失值 {health['missing_total']} 个")
        for w in health["warnings"]:
            print(f"      [警告] {w}")
        print(f"      高风险比例 = {df[cfg.LABEL_COLUMN].mean():.4f}"
              f"（要求 0.20~0.35）；等级分布 = "
              f"{df[cfg.RISK_LEVEL_COLUMN].value_counts().to_dict()}")

    # ---------------- ② 构建数据集（标准化只在训练集上 fit） ----------------
    ds, imputer, scaler = build_dataset(df)
    summary["n_samples"] = len(ds)
    summary["feature_columns"] = list(cfg.FEATURE_COLUMNS)
    if verbose:
        print(f"[2/8] 构建数据集：中位数填补 + z-score 标准化（统计量仅来自训练集）")

    # ---------------- ③ k 值扫描（双口径） ----------------
    df_scan, loo_results, split_results = k_scan(
        ds, k_list=k_list, n_splits=n_splits, seed=seed, verbose=verbose)
    df_scan.to_csv(cfg.K_SCAN_PATH, index=False, encoding="utf-8-sig")
    summary["k_scan"] = df_scan.round(6).to_dict(orient="records")

    # 10 次划分逐次明细（保留 seed，任何一条都能被复现）
    split_rows: List[Dict] = []
    for k, reps in split_results.items():
        for i, rep in enumerate(reps):
            split_rows.append({
                "k": k, "split": i + 1, "seed": rep["seed"],
                "n_train": rep["n_train"], "n_test": rep["n_test"],
                "test_risk_ratio": round(float(np.mean([rep["counts"]["TP"] + rep["counts"]["FN"]]) /
                                               rep["n_test"]), 4),
                **{key: round(float(rep["metrics"][key]), 6) for key in
                   ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")},
                **rep["counts"],
            })
    pd.DataFrame(split_rows).to_csv(cfg.SPLIT_DETAIL_PATH, index=False, encoding="utf-8-sig")

    # ---------------- ④ 选择主 k ----------------
    sel = select_best_k(df_scan)
    main_k = int(sel["chosen_k"] if main_k is None else main_k)
    summary["k_selection"] = sel
    summary["main_k"] = main_k
    if verbose:
        print(f"[3/8] 主 k 选择：k = {main_k}")
        print(f"      理由：{sel['reason']}")

    # ---------------- ⑤ 主模型：留一法 + 10 次划分（主 k） ----------------
    loo_main = loo_results.get(main_k)
    if loo_main is None:  # 用户自定义 k 不在候选列表里时现算
        from .knn_numpy import loo_predict  # noqa: F401  局部导入，避免顶层循环依赖
        loo_main = dict(ev.full_report(ds.y, loo_predict(ds.X, ds.y, main_k),
                                       model_name=f"KNN_LOO_k{main_k}"))
        loo_main["k"] = main_k
        loo_main["proba_pos"] = loo_predict_proba(ds.X, ds.y, main_k).tolist()
    loo_proba = np.asarray(loo_main["proba_pos"], dtype=float)
    loo_level_pred = assign_risk_levels(loo_proba)

    summary["loo_knn"] = {k: v for k, v in loo_main.items() if k != "proba_pos"}
    # 等级评估（真实等级 vs 由概率划出的等级）
    level_eval = ev.evaluate_risk_levels(ds.risk_level if ds.risk_level is not None else ["未知"] * len(ds),
                                         loo_level_pred)
    summary["risk_level_eval"] = level_eval

    reps_main, details_main = repeated_random_split(ds, main_k, n_splits=n_splits, base_seed=seed)
    agg_knn = ev.aggregate_folds(reps_main)
    summary["split10_knn"] = agg_knn
    if verbose:
        print(f"[4/8] 主模型 KNN(k={main_k}) 双口径结果：")
        print("      留一法     ：" + _fmt_metrics(loo_main["metrics"]))
        print(f"      {n_splits} 次划分 ：" + _fmt_agg(agg_knn))

    # 阈值敏感性（基于留一法概率，可据此调整工作点）
    sweep = ev.threshold_sweep(ds.y, loo_proba)
    summary["threshold_sweep"] = sweep
    pd.DataFrame(sweep).to_csv(os.path.join(cfg.METRICS_DIR, "threshold_sweep.csv"),
                               index=False, encoding="utf-8-sig")

    # ---------------- ⑥ 对照模型：逻辑回归 ----------------
    backend_info = choose_logreg_backend()
    summary["logreg_backend"] = backend_info
    if verbose:
        print(f"[5/8] 对照模型：{backend_info['detail']}")

    # 逻辑回归的 10 次划分评估（复用同一套划分函数，保证与 KNN 完全同口径）
    lr_factory = lambda k: make_logreg(backend_info["backend"], random_state=seed)  # noqa: E731
    lr_reps, lr_details = repeated_random_split(ds, main_k, n_splits=n_splits,
                                                base_seed=seed, model_factory=lr_factory)
    agg_lr = ev.aggregate_folds(lr_reps)
    summary["split10_logreg"] = agg_lr

    # 逻辑回归的留一法：它是参数模型，无法用"距离矩阵掩对角线"的技巧，
    # 因此老老实实训练 n 次（400 条数据 × 8 维，耗时可接受）。
    loo_lr_pred, loo_lr_proba = _logreg_loo(ds, backend_info["backend"], seed)
    loo_lr = ev.full_report(ds.y, loo_lr_pred, model_name="LogReg_LOO")
    summary["loo_logreg"] = loo_lr["metrics"]
    if verbose:
        print("      逻辑回归 留一法：" + _fmt_metrics(loo_lr["metrics"]))
        print(f"      逻辑回归 {n_splits} 次划分：" + _fmt_agg(agg_lr))

    # ---------------- ⑦ 最终模型：全量训练 + 新样本测试 ----------------
    knn_final = KNNClassifierNumpy(k=main_k).fit(ds.X, ds.y)
    lr_final = make_logreg(backend_info["backend"], random_state=seed).fit(ds.X, ds.y)

    new_df = load_or_make_new_batch(n_new=cfg.N_NEW_SAMPLES, seed=cfg.NEW_BATCH_SEED)
    # 新样本必须复用训练集的填补中位数与标准化参数，绝不能用新样本自身的统计量
    X_new = scaler.transform(imputer.transform(
        new_df[cfg.FEATURE_COLUMNS].to_numpy(dtype=float)))
    y_new = new_df[cfg.LABEL_COLUMN].to_numpy(dtype=int)

    knn_new_pred = knn_final.predict(X_new)
    knn_new_proba = knn_final.predict_proba(X_new)[:, 1]
    lr_new_proba = lr_final.predict_proba(X_new)[:, 1]
    lr_new_pred = (lr_new_proba >= 0.5).astype(int)

    new_knn = ev.full_report(y_new, knn_new_pred, model_name=f"KNN_new_batch_k{main_k}")
    new_lr = ev.full_report(y_new, lr_new_pred, model_name="LogReg_new_batch")
    summary["new_batch_knn"] = new_knn
    summary["new_batch_logreg"] = new_lr
    summary["new_batch_info"] = {
        "n": int(len(new_df)),
        "path": cfg.NEW_BATCH_PATH,
        "risk_ratio": round(float(y_new.mean()), 4),
        "note": "该批次由独立随机种子生成，从未参与任何训练与调参",
    }

    # 新样本的等级评估与明细
    new_level_true = new_df[cfg.RISK_LEVEL_COLUMN].to_numpy()
    new_level_pred = assign_risk_levels(knn_new_proba)
    summary["new_batch_risk_level_eval"] = ev.evaluate_risk_levels(new_level_true, new_level_pred)

    new_batch_table = pd.DataFrame([
        {"model": f"KNN(k={main_k})", **_round_metrics(new_knn["metrics"]), **new_knn["counts"]},
        {"model": "逻辑回归", **_round_metrics(new_lr["metrics"]), **new_lr["counts"]},
    ])
    new_batch_table.to_csv(cfg.NEW_BATCH_TABLE_PATH, index=False, encoding="utf-8-sig")

    new_detail = new_df[["student_id"] + cfg.SCORE_COLUMNS + [
        "attendance_rate", "homework_submit_rate", "self_study_hours",
        cfg.LABEL_COLUMN, cfg.RISK_LEVEL_COLUMN]].copy()
    new_detail["knn_proba_risk"] = np.round(knn_new_proba, 4)
    new_detail["knn_pred"] = knn_new_pred
    new_detail["knn_pred_level"] = new_level_pred
    new_detail["logreg_proba_risk"] = np.round(lr_new_proba, 4)
    new_detail["logreg_pred"] = lr_new_pred
    new_detail["knn_correct"] = (knn_new_pred == y_new).astype(int)
    new_detail.to_csv(os.path.join(cfg.METRICS_DIR, "new_batch_predictions.csv"),
                      index=False, encoding="utf-8-sig")

    if verbose:
        print(f"[6/8] 新样本最终测试（{len(new_df)} 条，从未参与训练）：")
        print("      KNN      ：" + _fmt_metrics(new_knn["metrics"]))
        print("      逻辑回归  ：" + _fmt_metrics(new_lr["metrics"]))

    # ---------------- ⑧ 人工复算 ----------------
    verify = ev.verify_against_implementation(y_new, knn_new_pred)
    verify["scope"] = "KNN 在新样本测试集上的混淆矩阵"
    # 追加一项：用留一法混淆矩阵再做一次复算，避免"只在新样本上对过一次"的偶然性
    verify_loo = ev.verify_against_implementation(ds.y, _pred_from_proba(loo_proba))
    verify_loo["scope"] = "KNN 在全体训练数据留一法下的混淆矩阵"
    verify["loo_check"] = verify_loo
    verify["all_passed"] = bool(verify["all_passed"] and verify_loo["all_passed"])
    ev.save_json(verify, cfg.VERIFY_PATH)
    if verbose:
        st = verify["steps"]
        print("[7/8] 人工复算（用混淆矩阵四格手算，并与程序实现比对）：")
        print("      " + st["precision"])
        print("      " + st["recall"])
        print("      " + st["accuracy"])
        print("      " + st["f1"])
        print(f"      一致性检查：{verify['checks']} → "
              f"{'全部通过 ✅' if verify['all_passed'] else '存在不一致 ❌'}")

    # ---------------- ⑨ 保存模型与指标 ----------------
    # 把人工复算结果也写进汇总（与 v2 的 summary 结构保持一致，
    # 便于统一读取；早期只在独立文件里保存，汇总中查不到）。
    summary["manual_verification"] = {
        "counts": verify["counts"], "steps": verify["steps"],
        "all_passed": verify["all_passed"],
    }
    summary["dataset"] = "自建模拟/脱敏数据（v1 口径：手写 KNN 为主模型）"
    save_models(knn_final, lr_final, backend_info["backend"], imputer, scaler)
    save_preprocess_stats(imputer, scaler, extra={
        "main_k": main_k,
        "logreg_backend": backend_info["backend"],
    })

    # 全体样本的留一法预测 → 直接就是"需要关注的学生名单"
    pred_df = df[["student_id"] + cfg.SCORE_COLUMNS + [
        "score_mean", "score_std",
        "attendance_rate", "homework_submit_rate", "self_study_hours",
        cfg.LABEL_COLUMN, cfg.RISK_LEVEL_COLUMN]].copy()
    pred_df["loo_proba_risk"] = np.round(loo_proba, 4)
    pred_df["loo_pred"] = _pred_from_proba(loo_proba)
    pred_df["loo_pred_level"] = loo_level_pred
    pred_df["pred_correct"] = (pred_df["loo_pred"] == pred_df[cfg.LABEL_COLUMN]).astype(int)
    pred_df.to_csv(cfg.PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    # 指标总表（双口径 + 两个模型）
    rows = [
        {"model": f"KNN(k={main_k})", "protocol": "留一法(LOO)", **_round_metrics(loo_main["metrics"]),
         **loo_main["counts"]},
        {"model": f"KNN(k={main_k})", "protocol": f"{n_splits}次随机划分(均值)",
         **_round_metrics({k: agg_knn[k]["mean"] for k in agg_knn if k != "confusion_sum"}), **{"TP": "", "TN": "", "FP": "", "FN": ""}},
        {"model": "逻辑回归", "protocol": "留一法(LOO)", **_round_metrics(loo_lr["metrics"]),
         **loo_lr["counts"]},
        {"model": "逻辑回归", "protocol": f"{n_splits}次随机划分(均值)",
         **_round_metrics({k: agg_lr[k]["mean"] for k in agg_lr if k != "confusion_sum"}), **{"TP": "", "TN": "", "FP": "", "FN": ""}},
        {"model": f"KNN(k={main_k})", "protocol": "新样本测试", **_round_metrics(new_knn["metrics"]),
         **new_knn["counts"]},
        {"model": "逻辑回归", "protocol": "新样本测试", **_round_metrics(new_lr["metrics"]),
         **new_lr["counts"]},
    ]
    table = pd.DataFrame(rows)
    for c in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"):
        if c in table.columns:
            table[c] = table[c].apply(lambda v: "" if v == "" else round(float(v), 4))
    table.to_csv(cfg.METRICS_TABLE_PATH, index=False, encoding="utf-8-sig")

    summary["metrics_table"] = table.to_dict(orient="records")
    summary["logreg_params"] = logreg_params(lr_final, backend_info["backend"])
    summary["knn_params"] = {"k": main_k, "n_train": int(len(ds)),
                             "n_features": len(cfg.FEATURE_COLUMNS)}
    summary["risk_level_thresholds"] = cfg.RISK_LEVEL_THRESHOLDS
    summary["elapsed_sec"] = round(time.time() - t0, 2)

    # v1 的汇总写到**独立文件**，不再覆盖 v2 的 metrics_summary.json ——
    # 两者口径完全不同（v1：模拟数据 + 手写 KNN；v2：UCI 真实数据 + sklearn），
    # 混写会让下游按错口径读指标（实测遇到过读取时 KeyError: logo_knn）。
    ev.save_json(summary, cfg.LEGACY_METRICS_SUMMARY_PATH)

    if verbose:
        print(f"[8/8] 已保存：模型 → {os.path.relpath(cfg.MODEL_PATH, cfg.BASE_DIR)}；"
              f"指标 → {os.path.relpath(cfg.METRICS_DIR, cfg.BASE_DIR)}")
        print(f"      总耗时 {summary['elapsed_sec']} 秒")
        print("=" * 68)
    return summary


# ======================================================================
# 五、内部工具
# ======================================================================
def _pred_from_proba(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """概率 → 类别，严格大于阈值判为 1（与 KNN predict 的平票口径一致）。"""
    return (np.asarray(p, dtype=float) > threshold).astype(int)


def _logreg_loo(ds: Dataset, backend: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    逻辑回归的留一法：训练 n 次（每次少一个样本）。

    为什么不像 KNN 那样优化？因为逻辑回归是参数模型，去掉任意一个样本都会
    改变梯度下降的解，必须重训。400 条数据 × 1200 轮全批量梯度下降 ≈ 数秒，
    教学规模下可接受；数据量再大就应改用解析解或增量更新。
    """
    n = len(ds)
    proba = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = make_logreg(backend, random_state=seed).fit(ds.X[mask], ds.y[mask])
        proba[i] = float(clf.predict_proba(ds.X[i:i + 1])[:, 1][0])
    return _pred_from_proba(proba), proba


def _fmt_metrics(m: Dict) -> str:
    """格式化一行指标，终端输出用。"""
    return (f"acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  spec={m['specificity']:.4f}")


def _fmt_agg(agg: Dict) -> str:
    """格式化"均值±标准差"，终端输出用。"""
    return (f"acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}  "
            f"P={agg['precision']['mean']:.4f}±{agg['precision']['std']:.4f}  "
            f"R={agg['recall']['mean']:.4f}±{agg['recall']['std']:.4f}  "
            f"F1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")


def _round_metrics(m: Dict, nd: int = 4) -> Dict:
    """指标保留 4 位小数（报告展示精度足够，避免长小数干扰阅读）。"""
    return {k: round(float(v), nd) for k, v in m.items()
            if k in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")}


def main(argv: Optional[List[str]] = None) -> None:
    """命令行入口：python src/train_eval.py --k 7 --splits 10"""
    parser = argparse.ArgumentParser(description="训练与评估（手写 KNN 主模型 + 逻辑回归对照）")
    parser.add_argument("--k", type=int, default=None, help="指定主 k；不指定则由扫描结果自动选择")
    parser.add_argument("--splits", type=int, default=cfg.N_SPLITS, help="随机划分次数，默认 10")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="随机种子")
    args = parser.parse_args(argv)
    run_training_eval(main_k=args.k, n_splits=args.splits, seed=args.seed)


if __name__ == "__main__":
    main()
