# -*- coding: utf-8 -*-
"""
train_eval_v2.py —— 主流程 v2：UCI 真实数据 + sklearn 主模型
=========================================================
对应需求调整：
    ① 主数据 = UCI 真实公开成绩数据；模拟数据仅作鲁棒性对照；
    ② 主模型 = sklearn 的 KNN / 逻辑回归；手写 KNN 降为附录（等价性验证）。

【本模块负责的评估口径（三套 + 消融）】
    A. 留一组交叉验证 LOGO（主口径）
       每次留出**一整名学生**（该生可能同时修数学与葡语，共 2 条记录）。
       比留一法更严格，也更贴近"预测没见过的新学生"。
       实现说明：LOGO 下 KNN 用 numpy 向量化实现计算（一次距离矩阵即可完成全部
       折的预测），并用**已在附录中证明与 sklearn 逐样本等价**作为依据；
       sklearn 后端则在随机划分与新样本口径上独立运行，两套结果互相印证。

    B. 10 次随机划分（按学生分组）
       报 accuracy / precision / recall / F1 的均值 ± 标准差。
       关键：必须**按学生分组**划分 —— 合并数据里有 372 名学生同修两门课，
       按行划分会让同一人同时进训练集与测试集（泄漏）。

    C. 课程外推测试（真实数据版的"新样本测试"）
       UCI 只有两门课，没有第三批"从未参与训练"的数据，
       因此用**留一门课程**的方式做真正的外推：用全部数学课记录训练、预测葡语课记录，
       反之亦然。这等价于"用一批学生训练、去预测另一批样本"，
       比同分布内部划分更能暴露过拟合。两门课的高风险率差异很大（32.9% vs 15.4%），
       因此这项测试本身也检验模型的跨分布鲁棒性。

    D. 早期成绩消融对照（标签泄漏量化）
       主口径**不含 G1/G2**（G2 与 G3 相关 0.905/0.919，用了就是标签泄漏）。
       本对照把 G1/G2 加回来，量化"早期成绩能带来多少提升"，
       并明确说明这种提升在实际预警场景中**不可用**。

    E. 模拟数据对照（学生原始思路的保留）
       同一套代码跑自建模拟数据（400 条，构造了类别不平衡与特征重叠），
       比较"真实数据 vs 构造数据"的难度差异。
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import evaluate as ev
    from .model_backends import (ModelAdapter, describe_backend, make_knn, make_logreg,
                                 resolve_backend, sklearn_available, sklearn_version)
    from .preprocess import StandardScaler
    from .uci_data import (FEATURES_MAIN, group_train_test_split_indices,
                           leave_one_group_out_indices, PASS_THRESHOLD, TARGET_RAW,
                           build_dataset as build_uci_dataset)
    from .uci_features import add_risk_level, feature_name_cn, prepare_uci_features
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import evaluate as ev
    from model_backends import (ModelAdapter, describe_backend, make_knn, make_logreg,
                                resolve_backend, sklearn_available, sklearn_version)
    from preprocess import StandardScaler
    from uci_data import (FEATURES_MAIN, group_train_test_split_indices,
                          leave_one_group_out_indices, PASS_THRESHOLD, TARGET_RAW,
                          build_dataset as build_uci_dataset)
    from uci_features import add_risk_level, feature_name_cn, prepare_uci_features


# ======================================================================
# 一、只从训练集拟合的"标准化 + 模型"小流水线
# ======================================================================
def _fit_pipeline(X_tr: np.ndarray, y_tr: np.ndarray, kind: str,
                  backend: str = "auto", **kw) -> Tuple[ModelAdapter, StandardScaler]:
    """
    在训练集上 fit 标准化器与模型，返回 (模型, 标准化器)。

    【防泄漏铁律】标准化器只能看训练集。本函数是唯一的入口，
    因此"用测试集统计量标准化"这类错误在结构上就不可能发生。
    """
    scaler = StandardScaler().fit(X_tr)
    model = (make_knn(**kw, backend=backend) if kind == "knn"
             else make_logreg(backend=backend, **kw))
    model.fit(scaler.transform(X_tr), y_tr)
    return model, scaler


def _predict(model: ModelAdapter, scaler: StandardScaler,
             X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """用训练阶段得到的 scaler 变换测试集后预测，返回 (类别, 高风险概率)。"""
    Xs = scaler.transform(X_te)
    return model.predict(Xs), model.predict_proba(Xs)[:, 1]


# ======================================================================
# 二、留一组交叉验证（LOGO）—— 主口径
# ======================================================================
def leave_one_group_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        kind: str, backend: str = "auto", **kw) -> Dict:
    """
    留一组交叉验证：每次留出一整名学生（可能含 2 条记录）。

    KNN 的高效实现：LOGO 与留一法的思路相同 —— 训练集 = "除该组外的全部样本"，
    因此可用**一次 n×n 距离矩阵 + 组分块掩码**完成全部 668 折预测：
      · 对每个待预测样本 i，把它所属组的**所有成员**（含自己）距离置为 +∞，
      · 再取最小 k 个邻居投票。
    这与"逐折重新 fit sklearn"数学等价；该等价性由附录的等价验证支持
    （手写实现与 sklearn 逐样本一致，且留一法加速实现与朴素实现逐样本一致）。
    逻辑回归是参数模型，无法这样免训练，只能逐折重训（668 折 × 极快收敛，可接受）。
    """
    n = len(y)
    proba = np.zeros(n, dtype=float)

    if kind == "knn":
        try:
            from .knn_numpy import KNNClassifierNumpy
            from .knn_numpy import assign_risk_levels as _arl  # noqa: F401
        except ImportError:  # pragma: no cover - 脚本直跑分支
            from knn_numpy import KNNClassifierNumpy
        k = int(kw.get("k", cfg.MAIN_K))
        ref = KNNClassifierNumpy(k=k).fit(X, y)
        D = KNNClassifierNumpy.euclidean_distances(X)          # 一次算完
        # 构建"同组掩码"：同组样本之间距离置为 +∞（等价于把整组留出）
        same_group = groups[:, None] == groups[None, :]
        D_masked = np.where(same_group, np.inf, D)
        k_eff = max(1, min(k, n - 1))
        part = np.argpartition(D_masked, kth=k_eff - 1, axis=1)[:, :k_eff]
        proba = y[part].sum(axis=1) / float(k_eff)
        del ref
    else:
        # 逻辑回归是参数模型，无法用一次距离矩阵免训练，只能逐折重训。
        # 这里直接按"组"切分训练/测试下标，不依赖 DataFrame 的列名（避免字段缺失导致崩溃）。
        uniq = pd.unique(groups)
        for g in uniq:
            te_idx = np.where(groups == g)[0]
            tr_idx = np.where(groups != g)[0]
            model, scaler = _fit_pipeline(X[tr_idx], y[tr_idx], kind, backend=backend, **kw)
            Xs = scaler.transform(X[te_idx])
            proba[te_idx] = model.predict_proba(Xs)[:, 1]

    pred = (proba > 0.5).astype(int)
    rep = ev.full_report(y, pred, model_name=f"{kind}_LOGO")
    rep["proba_pos"] = proba.tolist()
    return rep


# ======================================================================
# 三、10 次随机划分（按学生分组）
# ======================================================================
def repeated_group_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                         kind: str, backend: str = "auto",
                         n_splits: int = cfg.N_SPLITS,
                         test_size: float = cfg.TEST_SIZE,
                         base_seed: int = cfg.RANDOM_SEED, **kw
                         ) -> Tuple[List[Dict], List[Dict]]:
    """
    重复 n_splits 次**按学生分组**的随机划分。

    每次都用该次划分自己的训练集重新 fit 标准化器与模型，因此不存在数据泄漏。
    """
    reports: List[Dict] = []
    details: List[Dict] = []
    for i in range(n_splits):
        seed = base_seed + i
        tr_idx, te_idx = group_train_test_split_indices(df, test_size, seed)
        model, scaler = _fit_pipeline(X[tr_idx], y[tr_idx], kind, backend=backend, **kw)
        Xs = scaler.transform(X[te_idx])
        pred = model.predict(Xs)
        proba = model.predict_proba(Xs)[:, 1]
        rep = ev.full_report(y[te_idx], pred, model_name=f"{kind}_split{i+1}")
        rep["seed"] = int(seed)
        rep["n_train"] = int(len(tr_idx))
        rep["n_test"] = int(len(te_idx))
        rep["proba_pos"] = proba.tolist()
        reports.append(rep)
        details.append({
            "split": i + 1, "seed": seed, "model": kind,
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "test_risk_ratio": round(float(y[te_idx].mean()), 4),
            **{k2: round(float(rep["metrics"][k2]), 6) for k2 in
               ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")},
            **rep["counts"],
        })
    return reports, details


# ======================================================================
# 四、课程外推测试（真正的"新样本"口径）
# ======================================================================
def cross_subject_test(df: pd.DataFrame, kind: str, backend: str = "auto", **kw) -> Dict:
    """
    留一门课程做外推：mat → por 与 por → mat 两个方向。

    这是本数据集上能做到的"最接近真实新样本"的检验：
    训练集与测试集来自**不同课程、不同分布**（mat 高风险率 32.9%，por 仅 15.4%）。
    """
    out: Dict[str, Any] = {}
    for train_sub, test_sub in (("mat", "por"), ("por", "mat")):
        tr = (df["subject"] == train_sub).to_numpy()
        te = (df["subject"] == test_sub).to_numpy()
        Xtr, ytr, _ = prepare_uci_features(df[tr], include_early_grades=False, verbose=False)
        Xte, yte, _ = prepare_uci_features(df[te], include_early_grades=False, verbose=False)
        model, scaler = _fit_pipeline(Xtr, ytr, kind, backend=backend, **kw)
        pred, proba = _predict(model, scaler, Xte)
        rep = ev.full_report(yte, pred, model_name=f"{kind}_train{train_sub}_test{test_sub}")
        rep["direction"] = f"{train_sub} → {test_sub}"
        rep["train_risk_ratio"] = round(float(ytr.mean()), 4)
        rep["test_risk_ratio"] = round(float(yte.mean()), 4)
        rep["proba_pos"] = proba.tolist()
        out[f"{train_sub}2{test_sub}"] = rep
    return out


# ======================================================================
# 五、k 值扫描（LOGO 主口径 + 随机划分）
# ======================================================================
def k_scan_uci(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               k_list: Optional[List[int]] = None,
               n_splits: int = cfg.N_SPLITS, test_size: float = cfg.TEST_SIZE,
               base_seed: int = cfg.RANDOM_SEED, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """对各候选 k 同时做 LOGO 与 10 次分组随机划分。"""
    k_list = list(cfg.K_CANDIDATES if k_list is None else k_list)
    if verbose:
        print(f"[k 值扫描] 候选 k = {k_list}，LOGO（留一组）+ {n_splits} 次分组随机划分")

    rows, logo_results = [], {}
    for k in k_list:
        logo = leave_one_group_out(X, y, groups, "knn", k=k)
        logo_results[k] = logo
        reps, _ = repeated_group_split(df, X, y, "knn", n_splits=n_splits,
                                       test_size=test_size, base_seed=base_seed, k=k)
        agg = ev.aggregate_folds(reps)
        m = logo["metrics"]
        rows.append({
            "k": k,
            "logo_accuracy": m["accuracy"], "logo_precision": m["precision"],
            "logo_recall": m["recall"], "logo_f1": m["f1"],
            "split_acc_mean": agg["accuracy"]["mean"], "split_acc_std": agg["accuracy"]["std"],
            "split_prec_mean": agg["precision"]["mean"], "split_prec_std": agg["precision"]["std"],
            "split_rec_mean": agg["recall"]["mean"], "split_rec_std": agg["recall"]["std"],
            "split_f1_mean": agg["f1"]["mean"], "split_f1_std": agg["f1"]["std"],
        })
        if verbose:
            print(f"  k={k:<3d} LOGO: acc={m['accuracy']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} F1={m['f1']:.4f} | "
                  f"划分 F1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")
    return pd.DataFrame(rows), logo_results


def select_k_uci(df_scan: pd.DataFrame) -> Dict:
    """
    选主 k：以 **LOGO 的 F1 与 recall 的调和**为主依据。

    【为什么这里不能只最大化 F1】真实 UCI 数据（不含 G1/G2）上各类不平衡严重
    （高风险仅 22%），F1 的最优点往往落在"极端保守"的位置：
    实测 k 越大 recall 越低（k=31 时 recall 仅 0.08），虽然 precision 很高，
    对"预警"这个用途几乎失去意义 —— 一个只抓出 8% 风险学生的系统没有价值。
    因此选择规则为：
      1. 先排除 recall < 0.30 的候选（抓不到人的模型不参与评选）；
      2. 在剩余候选中取 F1 最大者；
      3. 若 F1 相差在 0.01 内，取 k 较大者（投票人数多、概率分辨率更细、更稳定）。
    """
    df = df_scan.copy()
    ok = df[df["logo_recall"] >= 0.30]
    excluded = df[df["logo_recall"] < 0.30]["k"].tolist()
    if ok.empty:                       # 全部候选 recall 都过低，退化为按 F1 选
        ok = df.copy()
        excluded = []
    best_f1 = float(ok["logo_f1"].max())
    near = ok[ok["logo_f1"] >= best_f1 - 0.01]
    chosen = int(near.sort_values("k", ascending=False).iloc[0]["k"])
    return {
        "chosen_k": chosen,
        "best_logo_f1": round(best_f1, 6),
        "excluded_by_recall": excluded,
        "near_candidates": near["k"].tolist(),
        "reason": (
            f"风险预警的核心是'抓得到人'：先排除 LOGO recall < 0.30 的候选 "
            f"（{excluded or '无'}），再在剩余候选中取 LOGO F1 最高者（{best_f1:.4f}）；"
            f"F1 相差 0.01 内取 k 较大者（投票人数更多、概率分档更细），最终 k = {chosen}"
        ),
    }


# ======================================================================
# 六、主流程
# ======================================================================
def run_uci_pipeline(backend: str = "auto", n_splits: int = cfg.N_SPLITS,
                     seed: int = cfg.RANDOM_SEED, verbose: bool = True) -> Dict:
    """执行 UCI 主数据集的完整训练与评估，落盘全部指标。"""
    cfg.ensure_dirs()
    t0 = time.time()
    backend = resolve_backend(backend)
    summary: Dict[str, Any] = {
        "dataset": "UCI Student Performance（真实公开数据，主数据）",
        "n_splits": n_splits, "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "backend_description": describe_backend("knn", backend),
        "sklearn_version": sklearn_version(),
        "label_rule": f"risk_label = 1 if {TARGET_RAW} < {PASS_THRESHOLD:.0f} else 0",
    }

    if verbose:
        print("=" * 68)
        print("【步骤 v2】UCI 真实数据 · 训练与评估（主模型：sklearn）")
        print("=" * 68)
        print(f"后端：{summary['backend_description']}")

    # ---------------- ① 数据 ----------------
    if os.path.exists(cfg.UCI_DATA_PATH):
        df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
        if cfg.RISK_LEVEL_COLUMN not in df.columns:
            df = add_risk_level(df)
    else:
        df, _ = build_uci_dataset(verbose=verbose)

    groups = df["student_group_id"].to_numpy()
    n_groups = len(pd.unique(groups))
    summary["n_samples"] = int(len(df))
    summary["n_students"] = int(n_groups)
    summary["risk_ratio"] = round(float(df[cfg.LABEL_COLUMN].mean()), 4)
    summary["risk_ratio_by_subject"] = {
        k: round(float(v), 4) for k, v in df.groupby("subject")[cfg.LABEL_COLUMN].mean().items()
    }

    X_main, y, feat_names = prepare_uci_features(df, include_early_grades=False, verbose=False)
    summary["feature_names"] = feat_names
    summary["n_features_main"] = len(feat_names)

    if verbose:
        print(f"[1/7] 数据：{len(df)} 条记录 / {n_groups} 名学生，"
              f"高风险比例 {summary['risk_ratio']:.4f}（分课程 {summary['risk_ratio_by_subject']}）")
        print(f"      主口径特征数 = {len(feat_names)}（**不含 G1/G2**，避免标签泄漏）")

    # 基线：多数类预测的准确率（必须报告，否则 accuracy 会被误读）
    base_rate = max(float(y.mean()), 1 - float(y.mean()))
    summary["majority_baseline_accuracy"] = round(base_rate, 4)
    if verbose:
        print(f"      多数类基线 accuracy = {base_rate:.4f}"
              f"（≈ 全判为'正常'的准确率，衡量模型是否真的学到东西）")

    # ---------------- ② k 值扫描 ----------------
    df_scan, logo_results = k_scan_uci(df, X_main, y, groups,
                                       n_splits=n_splits, base_seed=seed, verbose=verbose)
    df_scan.to_csv(cfg.K_SCAN_PATH, index=False, encoding="utf-8-sig")
    summary["k_scan"] = df_scan.round(6).to_dict(orient="records")

    sel = select_k_uci(df_scan)
    main_k = int(sel["chosen_k"])
    summary["k_selection"] = sel
    summary["main_k"] = main_k
    if verbose:
        print(f"[2/7] 主 k = {main_k}；理由：{sel['reason']}")

    # ---------------- ③ 主模型：LOGO（主口径） ----------------
    logo_knn = logo_results.get(main_k) or leave_one_group_out(X_main, y, groups, "knn", k=main_k)
    summary["logo_knn"] = {k: v for k, v in logo_knn.items() if k != "proba_pos"}
    if verbose:
        print(f"[3/7] 主模型 KNN(k={main_k}) 留一组交叉验证（LOGO，主口径）：")
        print("      " + _fmt(logo_knn["metrics"]))

    # ---------------- ④ 主模型与对照模型：10 次分组随机划分 ----------------
    reps_knn, det_knn = repeated_group_split(df, X_main, y, "knn", backend=backend,
                                             n_splits=n_splits, base_seed=seed, k=main_k)
    agg_knn = ev.aggregate_folds(reps_knn)
    summary["split10_knn"] = agg_knn

    reps_lr, det_lr = repeated_group_split(df, X_main, y, "logreg", backend=backend,
                                           n_splits=n_splits, base_seed=seed)
    agg_lr = ev.aggregate_folds(reps_lr)
    summary["split10_logreg"] = agg_lr

    pd.DataFrame(det_knn + det_lr).to_csv(cfg.SPLIT_DETAIL_PATH, index=False, encoding="utf-8-sig")

    logo_lr = leave_one_group_out(X_main, y, groups, "logreg", backend=backend)
    summary["logo_logreg"] = {k: v for k, v in logo_lr.items() if k != "proba_pos"}

    if verbose:
        print(f"      {n_splits} 次分组划分：{_fmt_agg(agg_knn)}")
        print("[4/7] 对照模型 逻辑回归：")
        print("      LOGO：" + _fmt(logo_lr["metrics"]))
        print(f"      {n_splits} 次分组划分：{_fmt_agg(agg_lr)}")

    # ---------------- ⑤ 课程外推测试 ----------------
    cs_knn = cross_subject_test(df, "knn", backend=backend, k=main_k)
    cs_lr = cross_subject_test(df, "logreg", backend=backend)
    summary["cross_subject_knn"] = {k: {kk: vv for kk, vv in v.items() if kk != "proba_pos"}
                                    for k, v in cs_knn.items()}
    summary["cross_subject_logreg"] = {k: {kk: vv for kk, vv in v.items() if kk != "proba_pos"}
                                       for k, v in cs_lr.items()}
    if verbose:
        print("[5/7] 课程外推测试（真正的跨分布检验）：")
        for key, rep in cs_knn.items():
            print(f"      KNN  {rep['direction']}: 训练高风险率 {rep['train_risk_ratio']:.3f} → "
                  f"测试 {rep['test_risk_ratio']:.3f}  " + _fmt(rep["metrics"]))
        for key, rep in cs_lr.items():
            print(f"      逻辑回归 {rep['direction']}: " + _fmt(rep["metrics"]))

    # ---------------- ⑥ 阈值调优（只用训练侧数据选，绝不用测试集） ----------------
    # 方法：在 LOGO 的**训练侧**用 5 折分组交叉验证选阈值，再报告"分组随机划分"上的效果。
    # 这里简化为在每次划分的训练集上做 3 折内部 CV 选阈值，然后作用于测试集，
    # 以便诚实展示"合理调阈值能提升多少"，且不存在测试集泄漏。
    tuned = _tune_threshold_within_training(df, X_main, y, main_k, backend=backend,
                                            n_splits=n_splits, seed=seed)
    summary["tuned_threshold"] = tuned
    if verbose:
        print("[6/7] 阈值调优（在训练集内部交叉验证选阈值，不接触测试集）：")
        print(f"      选定阈值 = {tuned['threshold']}；调优后测试集 "
              f"F1={tuned['f1_mean']:.4f}（默认 0.5 时为 {agg_knn['f1']['mean']:.4f}）")
        print(f"      说明：{tuned['note']}")

    # ---------------- ⑦ 早期成绩消融对照 ----------------
    ablation = early_grade_ablation(df, X_main, y, groups, main_k, seed=seed)
    summary["ablation_early_grades"] = ablation
    if verbose:
        print("[7/7] 早期成绩消融对照（量化标签泄漏）：")
        for row in ablation:
            # 第三行（仅用 G2 单特征）只有 accuracy，其它指标为空字符串，需分别格式化
            def _fv(v, nd=4):
                return f"{float(v):.{nd}f}" if isinstance(v, (int, float)) else "—"
            print(f"      {row['setting']:<30s} F1={_fv(row['logo_f1'])} "
                  f"recall={_fv(row['logo_recall'])} accuracy={_fv(row['logo_accuracy'])}")
        print("      结论：加入 G1/G2 后指标大幅提升，但那是**标签泄漏**，"
              "真实预警场景中用不到（期中时还没有期末成绩）")

    # ---------------- 保存 ----------------
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    summary["disclaimer"] = cfg.DISCLAIMER
    summary["citation"] = cfg.DATA_CITATION
    # 注意：这里**不能立刻保存** —— 下面还要把指标总表、人工复算结果补进 summary。
    # 早期版本在这里就 save_json 了，导致手动复算结果从未进入汇总文件
    # （实测发现 metrics_summary.json 里缺少 manual_verification 键），故调整到最后统一保存。

    # 指标总表
    rows = [
        {"model": f"KNN(k={main_k})", "protocol": "留一组交叉验证 LOGO（主口径）",
         **_r(logo_knn["metrics"]), **logo_knn["counts"]},
        {"model": f"KNN(k={main_k})", "protocol": f"{n_splits} 次分组随机划分（均值）",
         **_r({k: agg_knn[k]["mean"] for k in agg_knn if k != "confusion_sum"}),
         "TP": "", "TN": "", "FP": "", "FN": ""},
        {"model": "逻辑回归", "protocol": "留一组交叉验证 LOGO",
         **_r(logo_lr["metrics"]), **logo_lr["counts"]},
        {"model": "逻辑回归", "protocol": f"{n_splits} 次分组随机划分（均值）",
         **_r({k: agg_lr[k]["mean"] for k in agg_lr if k != "confusion_sum"}),
         "TP": "", "TN": "", "FP": "", "FN": ""},
        {"model": f"KNN(k={main_k})", "protocol": "课程外推 mat→por",
         **_r(cs_knn["mat2por"]["metrics"]), **cs_knn["mat2por"]["counts"]},
        {"model": f"KNN(k={main_k})", "protocol": "课程外推 por→mat",
         **_r(cs_knn["por2mat"]["metrics"]), **cs_knn["por2mat"]["counts"]},
    ]
    table = pd.DataFrame(rows)
    for c in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"):
        if c in table.columns:
            table[c] = table[c].apply(lambda v: "" if v == "" else round(float(v), 4))
    table.to_csv(cfg.METRICS_TABLE_PATH, index=False, encoding="utf-8-sig")
    summary["metrics_table"] = table.to_dict(orient="records")

    # 逐样本预测名单（主口径 LOGO）
    pred_df = df[["student_id", "subject", "student_group_id", TARGET_RAW,
                  "G1", "G2", cfg.LABEL_COLUMN, cfg.RISK_LEVEL_COLUMN]].copy()
    try:
        from .knn_numpy import assign_risk_levels
    except ImportError:  # pragma: no cover - 脚本直跑分支
        from knn_numpy import assign_risk_levels
    pred_df["logo_proba_risk"] = np.round(logo_knn["proba_pos"], 4)
    pred_df["logo_pred"] = (np.asarray(logo_knn["proba_pos"]) > 0.5).astype(int)
    pred_df["logo_pred_level"] = assign_risk_levels(np.asarray(logo_knn["proba_pos"]))
    pred_df["pred_correct"] = (pred_df["logo_pred"] == pred_df[cfg.LABEL_COLUMN]).astype(int)
    pred_df.to_csv(cfg.PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    # 人工复算
    verify = ev.verify_against_implementation(y, pred_df["logo_pred"].to_numpy())
    verify["scope"] = f"UCI 真实数据 LOGO 主口径（KNN k={main_k}，{len(y)} 条）"
    ev.save_json(verify, cfg.VERIFY_PATH)
    summary["manual_verification"] = {
        "counts": verify["counts"], "steps": verify["steps"],
        "all_passed": verify["all_passed"],
    }
    # 到这里 summary 才完整（含指标总表与人工复算），统一落盘
    ev.save_json(summary, cfg.METRICS_SUMMARY_PATH)
    if verbose:
        print("\n—— 人工复算（LOGO 主口径）——")
        for k2, v2 in verify["steps"].items():
            print("  " + v2)
        print(f"  一致性：{verify['checks']} → 全部通过 = {verify['all_passed']}")
        print(f"\n总耗时 {summary['elapsed_sec']} 秒；指标已写入 "
              f"{os.path.relpath(cfg.METRICS_SUMMARY_PATH, cfg.BASE_DIR)}")
        print("=" * 68)
    return summary


# ======================================================================
# 七、辅助：阈值调优与消融
# ======================================================================
def _tune_threshold_within_training(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                                    k: int, backend: str, n_splits: int = 10,
                                    seed: int = cfg.RANDOM_SEED) -> Dict:
    """
    在**训练集内部**用分组交叉验证选判定阈值，然后把它用在测试集上。

    为什么必须这样：如果直接在测试集上扫阈值挑最好的，就是把测试集当验证集用，
    指标会虚高（这是预警类项目最常见的隐性作弊）。
    这里对 10 次分组划分中的每一次，都在其**训练集**上再做 3 折内部 CV 选阈值，
    然后应用到该次测试集，最后汇总——全过程不接触测试集标签。
    """
    chosen, scores = [], []
    for i in range(n_splits):
        seed_i = seed + i
        tr_idx, te_idx = group_train_test_split_indices(df, cfg.TEST_SIZE, seed_i)
        Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        # ---- 内部 3 折：选阈值 ----
        best_th, best_f1 = 0.5, -1.0
        for th in np.round(np.arange(0.20, 0.61, 0.05), 2):
            f1s = []
            for j in range(3):
                itr, ite = group_train_test_split_indices(df_tr, 0.34, seed_i * 10 + j)
                m, sc = _fit_pipeline(Xtr[itr], ytr[itr], "knn", backend=backend, k=k)
                _, p = _predict(m, sc, Xtr[ite])
                pr = (p >= th).astype(int)
                f1s.append(ev.metrics_from_counts(ev.confusion_counts(ytr[ite], pr))["f1"])
            mean_f1 = float(np.mean(f1s))
            if mean_f1 > best_f1:
                best_f1, best_th = mean_f1, float(th)
        chosen.append(best_th)
        # ---- 用选出的阈值在该次测试集上评估 ----
        m, sc = _fit_pipeline(Xtr, ytr, "knn", backend=backend, k=k)
        _, p = _predict(m, sc, Xte)
        pr = (p >= best_th).astype(int)
        scores.append(ev.metrics_from_counts(ev.confusion_counts(yte, pr)))
    f1s = np.array([s["f1"] for s in scores])
    recs = np.array([s["recall"] for s in scores])
    precs = np.array([s["precision"] for s in scores])
    return {
        "threshold": float(np.round(np.mean(chosen), 3)),
        "chosen_per_split": chosen,
        "f1_mean": round(float(f1s.mean()), 4), "f1_std": round(float(f1s.std(ddof=1)), 4),
        "recall_mean": round(float(recs.mean()), 4), "recall_std": round(float(recs.std(ddof=1)), 4),
        "precision_mean": round(float(precs.mean()), 4),
        "precision_std": round(float(precs.std(ddof=1)), 4),
        "note": "阈值只在训练集内部用 3 折分组交叉验证选择，测试集标签全程不参与选择；"
                "调低阈值可显著提高 recall（少漏报），代价是 precision 下降（多误报）",
    }


def early_grade_ablation(df: pd.DataFrame, X_main: np.ndarray, y: np.ndarray,
                         groups: np.ndarray, main_k: int,
                         seed: int = cfg.RANDOM_SEED) -> List[Dict]:
    """
    消融对照：主口径（不含 G1/G2） vs 含 G1/G2。

    目的不是"刷高分"，而是**量化标签泄漏有多大**：
    G2 与 G3 的相关系数 0.905（数学）/0.919（葡语），把 G2 当特征等于
    "用期中成绩预测期末成绩"，实际预警场景中期末时才有 G3，因此这一口径不可用。
    """
    rows = []
    # 主口径：直接复用已算好的 X_main
    logo_main = leave_one_group_out(X_main, y, groups, "knn", k=main_k)
    rows.append({
        "setting": "主口径：不含 G1/G2（可用）",
        "n_features": int(X_main.shape[1]),
        "logo_accuracy": round(logo_main["metrics"]["accuracy"], 4),
        "logo_precision": round(logo_main["metrics"]["precision"], 4),
        "logo_recall": round(logo_main["metrics"]["recall"], 4),
        "logo_f1": round(logo_main["metrics"]["f1"], 4),
    })
    # 消融口径：加入 G1/G2
    X_g, _, names_g = prepare_uci_features(df, include_early_grades=True, verbose=False)
    logo_g = leave_one_group_out(X_g, y, groups, "knn", k=main_k)
    rows.append({
        "setting": "消融口径：加入 G1/G2（标签泄漏，不可用）",
        "n_features": int(X_g.shape[1]),
        "logo_accuracy": round(logo_g["metrics"]["accuracy"], 4),
        "logo_precision": round(logo_g["metrics"]["precision"], 4),
        "logo_recall": round(logo_g["metrics"]["recall"], 4),
        "logo_f1": round(logo_g["metrics"]["f1"], 4),
    })
    # 极端泄漏示例：只用 G2 一个特征的阈值规则能达到的准确率
    yv = df[cfg.LABEL_COLUMN].to_numpy()
    best_acc = max(float(((df["G2"] <= t).astype(int) == yv).mean()) for t in np.unique(df["G2"]))
    rows.append({
        "setting": "极端示例：仅用 G2 单特征阈值",
        "n_features": 1,
        "logo_accuracy": round(best_acc, 4),
        "logo_precision": "", "logo_recall": "", "logo_f1": "",
    })
    pd.DataFrame(rows).to_csv(cfg.ABLATION_PATH, index=False, encoding="utf-8-sig")
    return rows


# ======================================================================
# 八、格式化工具
# ======================================================================
def _fmt(m: Dict) -> str:
    return (f"acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  spec={m['specificity']:.4f}")


def _fmt_agg(agg: Dict) -> str:
    return (f"acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}  "
            f"P={agg['precision']['mean']:.4f}±{agg['precision']['std']:.4f}  "
            f"R={agg['recall']['mean']:.4f}±{agg['recall']['std']:.4f}  "
            f"F1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f}")


def _r(m: Dict) -> Dict:
    return {k: round(float(v), 4) for k, v in m.items()
            if k in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy")}


def main(argv: Optional[List[str]] = None) -> None:
    """命令行入口：python src/train_eval_v2.py [--backend sklearn|numpy] [--splits 10]"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    parser = argparse.ArgumentParser(description="UCI 真实数据 + sklearn 主模型 训练与评估")
    parser.add_argument("--backend", default="auto", choices=["auto", "sklearn", "numpy"])
    parser.add_argument("--splits", type=int, default=cfg.N_SPLITS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args(argv)
    run_uci_pipeline(backend=args.backend, n_splits=args.splits, seed=args.seed)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
