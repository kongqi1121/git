# -*- coding: utf-8 -*-
"""
tests/test_evaluation_protocols.py —— 评估协议单元测试（建议 D2 补充）
=========================================================
覆盖"评估口径"本身是否正确，而不只是指标公式：

    confidence_intervals  t 区间、分层 bootstrap、按学生整组重抽、配对比较
    model_backends        后端可用性、两后端一致性、n_jobs 约束
    uci_data              按学生分组的划分（防同一学生跨集合泄漏）

这些是本项目最容易"静默算错"的部分：
区间公式用错不会报错，分组划分写错只会让指标悄悄偏乐观。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as cfg          # noqa: E402
import uci_data as ud         # noqa: E402
from confidence_intervals import (paired_comparison, percentile_interval,  # noqa: E402
                                  student_block_bootstrap_metrics, t_critical_975,
                                  t_interval)
from model_backends import (make_knn, make_logreg, resolve_backend,  # noqa: E402
                            sklearn_available)


# ======================================================================
# 1. t 区间
# ======================================================================
def test_t_interval_matches_hand_calculation():
    """
    t 区间必须与手算一致：mean ± t(0.975,9)·s/√n。

    容差说明：`t_interval` 对外输出**保留 6 位小数**（便于报告展示），
    因此比对容差取 1e-5 量级；早期测试用 1e-9 属于容差写错，
    会把正常的四舍五入误判为错误。
    """
    vals = [0.40, 0.44, 0.38, 0.42, 0.39, 0.41, 0.43, 0.37, 0.45, 0.40]
    res = t_interval(vals)
    a = np.asarray(vals)
    s = a.std(ddof=1)
    se = s / np.sqrt(len(a))
    assert res["n"] == 10
    assert abs(res["mean"] - a.mean()) < 1e-5
    assert abs(res["se"] - se) < 1e-5
    assert abs(res["half_width"] - 2.262 * se) < 1e-3
    assert res["ci_low"] < res["mean"] < res["ci_high"]


def test_t_critical_table_sane():
    """t 临界值应随自由度增加而下降，且 df=9 时为 2.262。"""
    assert t_critical_975(9) == 2.262
    assert t_critical_975(4) == 2.776
    assert t_critical_975(30) < t_critical_975(9)
    # 表里没有的中间自由度取"不大于它的最近值"（更保守）
    assert t_critical_975(13) == 2.160


def test_t_interval_single_value():
    """只有一个样本时不应抛错，而是退化为点区间。"""
    res = t_interval([0.5])
    assert res["ci_low"] == res["ci_high"] == 0.5


# ======================================================================
# 2. bootstrap 区间
# ======================================================================
def test_percentile_interval_brackets_median():
    rng = np.random.RandomState(0)
    vals = rng.normal(1.0, 0.1, size=500)
    res = percentile_interval(vals)
    assert res["ci_low"] < res["median"] < res["ci_high"]
    assert res["ci_low"] < res["ci_high"]


def test_student_block_bootstrap_uses_full_records():
    """
    按学生整组重抽：抽到某学生必须带上他的**全部记录**。
    用一个刻意构造的数据集验证 —— 若实现退化为"按记录重抽"，指标会明显不同。
    """
    # 200 名学生，其中 100 名各 2 条记录（同标签），100 名各 1 条记录
    rows = []
    for i in range(100):
        rows.append({"g": "A%03d" % i, "y": 1, "p": 1})
        rows.append({"g": "A%03d" % i, "y": 1, "p": 0})   # 同一学生一错一对
    for i in range(100):
        rows.append({"g": "B%03d" % i, "y": 0, "p": 0})
    df = pd.DataFrame(rows)
    res = student_block_bootstrap_metrics(df, "y", "p", "g", n_boot=200, random_state=0)
    assert res["unit"].startswith("学生")
    assert res["n_students"] == 200
    # 该数据集上 recall 恒为 0.5（每名学生恰好一半记录命中），区间应集中在 0.5 附近
    assert abs(res["recall"]["mean"] - 0.5) < 0.06, res["recall"]
    assert res["n_boot_effective"] > 0


def test_student_block_bootstrap_handles_missing_class():
    """退化情形：数据只有一类时不应崩溃，且统计跳过次数。"""
    df = pd.DataFrame({"g": ["g1", "g2", "g3"], "y": [0, 0, 0], "p": [0, 0, 0]})
    res = student_block_bootstrap_metrics(df, "y", "p", "g", n_boot=20, random_state=0)
    assert res["n_boot_skipped"] == 20          # 全部缺失正类 → 全部跳过
    assert res["n_boot_effective"] == 0


# ======================================================================
# 3. 配对比较
# ======================================================================
def test_paired_comparison_detects_consistent_difference():
    """A 每次都比 B 高固定量 → 配对差值显著。"""
    a = np.array([0.40, 0.42, 0.41, 0.39, 0.43, 0.40, 0.44, 0.38, 0.42, 0.41])
    b = a - 0.05
    res = paired_comparison(a, b, "A", "B")
    assert abs(res["mean_diff"] - 0.05) < 1e-9
    assert res["significant_at_0.05"], res


def test_paired_comparison_no_difference_is_not_significant():
    """两组完全相同时差值恒为 0，不应报显著。"""
    a = np.array([0.4, 0.42, 0.41, 0.39, 0.43, 0.40, 0.44, 0.38, 0.42, 0.41])
    res = paired_comparison(a, a.copy(), "A", "B")
    assert res["mean_diff"] == 0.0
    assert not res.get("significant_at_0.05", False)


def test_paired_comparison_requires_same_length():
    with pytest.raises(ValueError):
        paired_comparison([1, 2, 3], [1, 2], "A", "B")


# ======================================================================
# 4. 按学生分组划分（防泄漏）
# ======================================================================
def _fake_uci(n_students=20):
    """构造含"同一学生两条记录"的假数据。"""
    rows = []
    for i in range(n_students):
        g = "STU%04d" % i
        rows.append({"student_group_id": g, "subject": "mat", "risk_label": i % 2})
        rows.append({"student_group_id": g, "subject": "por", "risk_label": i % 2})
    return pd.DataFrame(rows)


def test_group_split_keeps_students_together():
    """
    按学生分组划分：同一学生的所有记录必须在同一侧。
    这是本项目最关键的防泄漏措施（372 名同修两门课的学生）。
    """
    df = _fake_uci(20)
    tr, te = ud.group_train_test_split_indices(df, test_size=0.3, random_state=42)
    assert len(set(tr) & set(te)) == 0, "训练集与测试集索引重叠"
    tr_g = set(df.iloc[tr]["student_group_id"])
    te_g = set(df.iloc[te]["student_group_id"])
    assert tr_g & te_g == set(), "同一学生同时出现在训练集与测试集"
    assert len(tr) + len(te) == len(df), "划分未覆盖全部记录"


def test_group_split_is_reproducible():
    df = _fake_uci(20)
    tr1, te1 = ud.group_train_test_split_indices(df, 0.3, 42)
    tr2, te2 = ud.group_train_test_split_indices(df, 0.3, 42)
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    tr3, _ = ud.group_train_test_split_indices(df, 0.3, 43)
    assert not np.array_equal(tr1, tr3), "不同种子应产生不同划分"


def test_leave_one_group_out_never_leaks():
    """留一组交叉验证：每一折的测试组都不能出现在训练组里。"""
    df = _fake_uci(15)
    for g, tr_idx, te_idx in ud.leave_one_group_out_indices(df):
        tr_g = set(df.iloc[tr_idx]["student_group_id"])
        te_g = set(df.iloc[te_idx]["student_group_id"])
        assert te_g == {g}
        assert tr_g & te_g == set(), "折 %s 训练集混入了测试学生" % g


# ======================================================================
# 5. 模型后端
# ======================================================================
def test_backend_resolution_defaults_to_sklearn_when_available():
    b = resolve_backend("auto")
    if sklearn_available():
        assert b == "sklearn"
    else:
        assert b == "numpy"


def test_both_backends_agree_on_separable_data():
    """两套后端在易分数据上应给出一致的预测（附录等价性结论的守门测试）。"""
    rng = np.random.RandomState(0)
    X = np.r_[rng.normal(-1.5, 0.6, size=(60, 3)), rng.normal(1.5, 0.6, size=(60, 3))]
    y = np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)]
    if not sklearn_available():
        pytest.skip("环境无 sklearn，跳过后端一致性检查")
    a = make_knn(k=5, backend="sklearn").fit(X, y)
    b = make_knn(k=5, backend="numpy").fit(X, y)
    assert (a.predict(X) == b.predict(X)).all(), "两后端预测不一致"
    pa = a.predict_proba(X)[:, 1]
    pb = b.predict_proba(X)[:, 1]
    assert np.max(np.abs(pa - pb)) < 1e-9, "两后端概率不一致（最大差异 %g）" % np.max(np.abs(pa - pb))


def test_knn_n_jobs_is_one():
    """
    本项目固定 n_jobs=1：n_jobs=-1 会在受限环境触发 joblib 线程池的命名管道错误
    （实测 PermissionError: [WinError 5]），且 1044 条数据下多线程无收益。
    """
    assert cfg.SKLEARN_KNN_PARAMS["n_jobs"] == 1


def test_logreg_class_weight_configurable():
    """逻辑回归的 class_weight 必须可在不改动全局配置的前提下被覆盖（A4 实验依赖）。"""
    if not sklearn_available():
        pytest.skip("环境无 sklearn")
    X = np.random.RandomState(0).normal(size=(80, 4))
    y = (X[:, 0] > 0).astype(int)
    m = make_logreg(backend="sklearn", class_weight="balanced").fit(X, y)
    assert m.model.class_weight == "balanced"
    # 默认配置不应被修改
    assert cfg.SKLEARN_LOGREG_PARAMS.get("class_weight") is None
