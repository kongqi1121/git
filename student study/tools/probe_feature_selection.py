# -*- coding: utf-8 -*-
"""
probe_feature_selection.py —— A3 特征选择的快速探底（诊断脚本）

【为什么先探底再做完整实现】
评审阶段即判定特征选择"收益不确定、风险较高"，理由是：
  · 最优单特征准确率仅 0.8017（基线 0.7797），信号本就稀疏；
  · 置换重要性最大值与最小值仅差约 2 倍，没有单特征独揽贡献。
因此先用**不含泄漏风险的单特征筛选**快速判断"删特征到底有没有用"，
再决定是否值得投入完整实现（RFE 必须嵌进 CV 内层，成本高）。

【本探底的做法】
在每一折训练集内部：按"与标签的互信息"排序特征 → 保留前 K 个 → 训练模型 → 在测试折评估。
互信息只用训练集计算，因此**不构成标签泄漏**。
对比 K ∈ {5, 8, 12, 20, 39}（39 = 不筛选）在 10 次分组划分上的表现。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

import config as cfg          # noqa: E402
import evaluate as ev         # noqa: E402
from model_backends import make_knn, make_logreg  # noqa: E402
from preprocess import StandardScaler  # noqa: E402
from uci_data import group_train_test_split_indices, load_saved_dataset  # noqa: E402
from uci_features import feature_name_cn, prepare_uci_features  # noqa: E402


def run(model_kind: str = "logreg_balanced") -> int:
    df = load_saved_dataset()
    X, y, names = prepare_uci_features(df, include_early_grades=False, verbose=False)
    n_feat = X.shape[1]

    def build():
        if model_kind == "knn":
            return make_knn(k=int(cfg.MAIN_K), backend="auto")
        return make_logreg(backend="auto", class_weight="balanced")

    print("=" * 84)
    print("【A3 探底】单特征筛选（按互信息，只在训练折内计算）对性能的影响")
    print("=" * 84)
    print(f"模型 = {model_kind}；特征总数 = {n_feat}；划分 = {cfg.N_SPLITS} 次按学生分组")
    print()

    ks = [5, 8, 12, 20, n_feat]
    results = {k: [] for k in ks}
    # 记录每次划分里被选中的特征（用于回答"删掉了什么"）
    picked_log = []

    for i in range(cfg.N_SPLITS):
        seed = cfg.RANDOM_SEED + i
        tr, te = group_train_split = None, None
        tr, te = group_train_test_split_indices(df, cfg.TEST_SIZE, seed)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        # 互信息只用训练折计算 —— 这是不泄漏的关键
        mi = mutual_info_classif(Xtr, ytr, random_state=seed)
        order = np.argsort(-mi)
        if i == 0:
            picked_log = [(names[j], round(float(mi[j]), 4)) for j in order]

        for k in ks:
            idx = order[:k]
            m = build().fit(Xtr[:, idx], ytr)
            p = m.predict_proba(Xte[:, idx])[:, 1]
            r = ev.evaluate_binary(yte, (p > 0.5).astype(int))
            results[k].append({"precision": r["precision"], "recall": r["recall"],
                               "f1": r["f1"], "accuracy": r["accuracy"]})

    print(f"  {'特征数':>6}{'accuracy':>11}{'precision':>11}{'recall':>10}{'F1':>10}")
    base_f1 = None
    for k in ks:
        arr = pd.DataFrame(results[k])
        f1 = float(arr["f1"].mean())
        if k == n_feat:
            base_f1 = f1
        tag = "（不筛选）" if k == n_feat else ""
        print(f"  {k:>6}{float(arr['accuracy'].mean()):>11.4f}"
              f"{float(arr['precision'].mean()):>11.4f}{float(arr['recall'].mean()):>10.4f}"
              f"{f1:>10.4f} {tag}")

    print()
    print("  被选中的前 12 个特征（按互信息，第 1 折）：")
    for rank, (nm, v) in enumerate(picked_log[:12], 1):
        print(f"    {rank:>2}. {feature_name_cn(nm):<28} MI={v}")
    print()
    print("  结论判据：")
    print("    · 若某个 K 的 F1 明显高于不筛选（且超过 0.03），说明特征选择有价值；")
    print("    · 若各 K 差异都在噪声范围内（<0.03），则应如实写成『未带来提升』。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    kind = "knn" if "--knn" in sys.argv else "logreg_balanced"
    raise SystemExit(run(kind))
