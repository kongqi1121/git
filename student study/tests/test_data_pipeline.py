# -*- coding: utf-8 -*-
"""
tests/test_data_pipeline.py —— 数据管线单元测试（建议 D2）
=========================================================
【为什么优先测这几个点】
本项目的失败大多发生在"不会主动报错"的地方：数据被污染、等级档位退化、
自匹配判错、指标口径漂移 —— 这些都不会抛异常，只会静默给出错误结论。
因此测试重点覆盖**曾经真实出过问题的环节**，把踩过的坑变成回归防线。

覆盖模块：
    uci_data      清洗脏行、内容体检（防污染）、学生分组
    uci_features  三档等级定档（防"中档消失"）
    evaluate      指标与人工复算一致、边界不产生 nan
    knn_numpy     手工可复算用例、自匹配排除、邻居数恒为 k
    llm_helper    结论防篡改校验

【重要】全部测试**不依赖网络**：数据用内置小样本在 tmp_path 里现场构造。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# 让 tests/ 能导入 src/ 下的模块（与 main.py 的做法一致）
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as cfg            # noqa: E402
import evaluate as ev            # noqa: E402
import uci_data as ud            # noqa: E402
import uci_features as uf        # noqa: E402
from knn_numpy import KNNClassifierNumpy  # noqa: E402
from llm_helper import build_facts, validate_no_conclusion_change  # noqa: E402


# ======================================================================
# 1. 数据清洗：脏行必须被剔除
# ======================================================================
def _raw_row(school="GP", sex="F", age=17, mjob="other", fjob="other",
             reason="course", guardian="mother", g3=10, g1=10, g2=10):
    """构造一行 UCI 原始格式（分号分隔）数据，便于测试。"""
    return ("{school};{sex};{age};U;GT3;T;2;2;{mjob};{fjob};{reason};{guardian};"
            "1;2;0;no;yes;no;no;yes;yes;yes;no;4;3;3;1;1;3;4;{g1};{g2};{g3}").format(
        school=school, sex=sex, age=age, mjob=mjob, fjob=fjob,
        reason=reason, guardian=guardian, g1=g1, g2=g2, g3=g3)


_HEADER = ("school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;"
           "guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;"
           "nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;"
           "absences;G1;G2;G3")


def test_clean_raw_drops_dirty_rows(tmp_path):
    """
    混入表头行与字段不足的残行，必须被识别为脏行并剔除。

    容差说明：清洗报告里的数值经过四舍五入，因此只做整数级断言
    （n_after == len(cleaned)），不比对浮点小数。
    """
    lines = [_HEADER, _raw_row(g3=10), _raw_row(g3=5), "", _raw_row(g3=8)]
    p = tmp_path / "student-mat.csv"
    p.write_text("\n".join(lines), encoding="utf-8")

    df = ud.load_raw("mat", str(p))
    cleaned, report = ud.clean_raw(df)
    # 空行会被 pandas 跳过；有效数据行应保留 3 条
    assert len(cleaned) == 3, "有效数据行数不符：%d" % len(cleaned)
    assert report["n_missing_after"] == 0
    assert cleaned["G3"].notna().all()


def test_clean_raw_keeps_row_count_consistent(tmp_path):
    """
    清洗报告的 n_before / n_after 必须自洽。

    注意：**必须写入表头**，否则 pandas 会把第一行数据当成列名，
    后续按列名取值会直接 KeyError —— 这是测试数据构造的常见错误。
    """
    lines = [_HEADER, _raw_row(g3=10), _raw_row(g3=12)]
    p = tmp_path / "student-por.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    df = ud.load_raw("por", str(p))
    cleaned, rep = ud.clean_raw(df)
    assert rep["n_before"] == 2, "n_before 应为 2（表头不算数据行）"
    assert rep["n_after"] == len(cleaned) == 2
    assert rep["n_missing_after"] == 0


# ======================================================================
# 2. 内容体检：必须能识破"被模拟数据伪装"的文件（针对真实事故）
# ======================================================================
def _write_fake_uci(path, n=400):
    """写一个**模拟数据格式**的假 UCI 文件（模拟曾经发生的污染事故）。"""
    df = pd.DataFrame({
        "student_id": ["SIM%04d" % i for i in range(1, n + 1)],
        "python": np.random.RandomState(0).uniform(40, 100, n),
        "ml": np.random.RandomState(1).uniform(40, 100, n),
        "cv": np.random.RandomState(2).uniform(40, 100, n),
        "attendance_rate": 0.9,
        "homework_submit_rate": 0.9,
        "self_study_hours": 4.0,
        "risk_label": 0,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_verify_saved_dataset_detects_simulated_pollution(tmp_path):
    """
    核心回归测试：把模拟数据写到 UCI 数据集路径上，体检必须判定为**不合格**。
    这正是本项目真实发生过的数据污染事故。
    """
    fake = tmp_path / "uci_student_performance.csv"
    _write_fake_uci(fake, n=400)
    rep = ud.verify_saved_dataset(str(fake))
    assert rep["ok"] is False, "体检未能识破被模拟数据污染的文件"
    assert "缺少 UCI 必需字段" in rep["reason"] or "行数异常" in rep["reason"]


def test_verify_saved_dataset_rejects_wrong_row_count(tmp_path):
    """行数不对（例如只有 400 行）也必须判为不合格。"""
    fake = tmp_path / "uci.csv"
    df = pd.DataFrame({
        "student_id": ["x"] * 400,
        "subject": ["mat"] * 400,
        "G3": [10] * 400,
        "risk_label": [0] * 400,
        "student_group_id": ["STU0001"] * 400,
        "risk_level": ["低"] * 400,
    })
    df.to_csv(fake, index=False, encoding="utf-8-sig")
    rep = ud.verify_saved_dataset(str(fake))
    assert rep["ok"] is False
    assert "行数异常" in rep["reason"]


def test_verify_saved_dataset_missing_file(tmp_path):
    rep = ud.verify_saved_dataset(str(tmp_path / "not_exist.csv"))
    assert rep["ok"] is False
    assert "不存在" in rep["reason"]


# ======================================================================
# 3. 三档等级定档：必须三档都有人（针对"中档消失"的历史 bug）
# ======================================================================
def test_add_risk_level_gives_three_nonempty_levels():
    """G3 是 0~20 的整数，定档后三档都必须有样本（此前"中"档曾被完全跳过）。"""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"G3": np.clip(rng.normal(11, 3.5, 600).round(), 0, 20).astype(int)})
    out = uf.add_risk_level(df)
    counts = out[cfg.RISK_LEVEL_COLUMN].value_counts()
    assert set(counts.index) == set(cfg.RISK_LEVELS), "档位不全：%s" % list(counts.index)
    assert counts.min() > 0, "存在空档：%s" % counts.to_dict()


def test_add_risk_level_is_monotone_in_grade():
    """等级必须随成绩单调：成绩越高等级越不危险（低 > 中 > 高 的分数区间连续）。"""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"G3": np.arange(0, 21)})
    out = uf.add_risk_level(df)
    order = {lv: i for i, lv in enumerate(cfg.RISK_LEVELS)}   # 低=0, 中=1, 高=2
    codes = [order[v] for v in out[cfg.RISK_LEVEL_COLUMN]]
    # 成绩升序时，危险度编码应单调不增
    assert all(codes[i] >= codes[i + 1] for i in range(len(codes) - 1)), \
        "等级与成绩不单调：%s" % list(zip(df["G3"], out[cfg.RISK_LEVEL_COLUMN]))


def test_feature_names_are_stable():
    """特征列名与顺序必须稳定（不同划分下维度漂移会导致指标不可比）。"""
    a = uf.feature_names(include_early_grades=False)
    b = uf.feature_names(include_early_grades=False)
    assert a == b, "两次取特征名结果不一致"
    assert not any(n in ("G1", "G2") for n in a), "主口径不应包含 G1/G2（标签泄漏）"
    c = uf.feature_names(include_early_grades=True)
    assert set(c) - set(a) == {"G1", "G2", "prior_avg"}, "消融口径新增列不符"


# ======================================================================
# 4. 指标与混淆矩阵
# ======================================================================
def test_metrics_match_manual_recompute():
    """指标实现必须与"只看混淆矩阵四格手算"的结果一致。"""
    y_true = np.array([0] * 100 + [1] * 30)
    y_pred = np.array([0] * 88 + [1] * 12 + [0] * 7 + [1] * 23)
    res = ev.verify_against_implementation(y_true, y_pred)
    assert res["all_passed"], res["checks"]
    assert abs(res["manual"]["precision_manual"] - 23 / 35) < 1e-12
    assert abs(res["manual"]["recall_manual"] - 23 / 30) < 1e-12


def test_metrics_edge_cases_no_nan():
    """全判一类时 precision/recall 的分母为 0，必须返回 0 而不是 nan。"""
    y_true = np.array([0, 0, 1, 1])
    all_neg = np.zeros(4, dtype=int)
    m = ev.evaluate_binary(y_true, all_neg)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert not any(np.isnan([m["precision"], m["recall"], m["f1"], m["accuracy"]]))

    all_pos = np.ones(4, dtype=int)
    m2 = ev.evaluate_binary(y_true, all_pos)
    assert not any(np.isnan([m2["precision"], m2["recall"], m2["f1"]]))


def test_confusion_matrix_layout():
    """混淆矩阵布局必须是 [[TN, FP], [FN, TP]]（与文档一致）。"""
    cm = ev.confusion_matrix_array([0, 0, 1, 1], [0, 1, 0, 1])
    assert cm.tolist() == [[1, 1], [1, 1]]
    cm2 = ev.confusion_matrix_array([0, 0, 0, 1], [0, 0, 0, 1])
    assert cm2.tolist() == [[3, 0], [0, 1]]


# ======================================================================
# 5. 手写 KNN：可复算、自匹配、邻居数
# ======================================================================
def test_knn_manual_case():
    """手工可复算用例：5 个点、查询 (0.9,0.9)、k=3 → 高风险比例 2/3。"""
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 2.]])
    y = np.array([0, 0, 1, 1, 1])
    clf = KNNClassifierNumpy(k=3).fit(X, y)
    p = clf.predict_proba(np.array([[0.9, 0.9]]))[0]
    assert abs(p[1] - 2 / 3) < 1e-12
    assert clf.predict(np.array([[0.9, 0.9]]))[0] == 1


def test_knn_returned_neighbor_count_is_always_k():
    """无论是否排除自匹配，返回的近邻数都必须等于 k（曾因剔除后不补足而出错）。"""
    rng = np.random.RandomState(3)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    for k in (1, 3, 7, 15):
        clf = KNNClassifierNumpy(k=k).fit(X, y)
        idx, _ = clf._knn_search(X[:10], k=k, self_indices=np.arange(10))
        assert idx.shape == (10, k), "k=%d 邻居数错误：%s" % (k, idx.shape)
        # 自匹配必须被剔除
        for r in range(10):
            assert r not in idx[r], "k=%d 时第 %d 行仍包含自身" % (k, r)


def test_knn_duplicate_samples_are_not_mistaken_for_self():
    """
    针对"误删重复样本"的历史 bug：若数据中存在**特征完全相同的不同样本**，
    按索引排除自身时，它们必须被保留为合法邻居。
    """
    X = np.array([[0., 0.], [0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    y = np.array([1, 0, 0, 0, 0])       # 第 0、1 行特征完全相同但标签不同
    clf = KNNClassifierNumpy(k=2).fit(X, y)
    nbs = clf.kneighbors(X[0], k=2, self_index=0)
    idxs = [n["train_index"] for n in nbs]
    assert 0 not in idxs, "自身未被排除"
    assert 1 in idxs, "特征相同的另一个样本被误删"
    assert len(nbs) == 2


def test_knn_loo_matches_naive():
    """留一法加速实现必须与朴素实现逐样本一致。"""
    from knn_numpy import loo_predict
    rng = np.random.RandomState(5)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=50) > 0).astype(int)
    for k in (1, 3, 7):
        fast = loo_predict(X, y, k)
        slow = []
        for i in range(len(X)):
            mask = np.ones(len(X), dtype=bool)
            mask[i] = False
            clf = KNNClassifierNumpy(k=k).fit(X[mask], y[mask])
            slow.append(int(clf.predict(X[i:i + 1])[0]))
        assert (fast == np.array(slow)).all(), "k=%d 留一法实现不一致" % k


def test_distance_matrix_symmetric_and_small_diagonal():
    """
    距离矩阵必须精确对称；对角线**不要求**为 0（灾难性抵消），
    但必须远小于真实样本间距离 —— 这是本项目不敢用 `D == 0` 判自匹配的原因。
    """
    rng = np.random.RandomState(7)
    X = rng.normal(size=(60, 6))
    D = KNNClassifierNumpy.euclidean_distances(X)
    assert np.max(np.abs(D - D.T)) == 0.0, "距离矩阵不对称"
    diag = np.max(np.abs(np.diag(D)))
    assert diag < 1e-6, "对角线误差超出预期量级：%g" % diag


# ======================================================================
# 6. LLM 结论防篡改
# ======================================================================
def _facts(p=0.7143, level="高"):
    return build_facts(student_id="t1", proba_risk=p, risk_level=level,
                       top_contributions=[{"feature": "absences", "feature_cn": "缺课次数",
                                           "value_raw": 12, "risk_gap": 0.5}],
                       neighbors=[{"student_id": "n1", "label": 1}], k=7)


def test_llm_guard_allows_consistent_text():
    v = validate_no_conclusion_change("该生高风险概率为 0.7143，等级为高风险。", _facts())
    assert v["passed"], v["issues"]


def test_llm_guard_blocks_probability_rewrite():
    v = validate_no_conclusion_change("该生高风险概率为 0.2143，等级为高风险。", _facts())
    assert not v["passed"], "概率被改写却未被拦截"


def test_llm_guard_blocks_level_rewrite():
    v = validate_no_conclusion_change("该生高风险概率为 0.7143，判定为低风险，无需关注。", _facts())
    assert not v["passed"], "等级被改写却未被拦截"


def test_llm_guard_allows_threshold_numbers():
    """文中出现等级阈值（0.3 / 0.7）属正常口径说明，不应被误判为篡改。"""
    text = ("该生高风险概率为 0.7143。等级口径：低 < 0.3，中 0.3 ~ 0.7，高 > 0.7。")
    v = validate_no_conclusion_change(text, _facts())
    assert v["passed"], v["issues"]


# ======================================================================
# 7. 配置一致性（防止路径/口径再次被改乱）
# ======================================================================
def test_config_paths_are_distinct():
    """UCI 主数据集与模拟数据集必须是两个不同文件（曾因共用路径造成污染）。"""
    assert os.path.abspath(cfg.UCI_DATA_PATH) != os.path.abspath(cfg.SIM_DATA_PATH)
    assert not hasattr(cfg, "DATA_PATH"), \
        "config 中不应再存在 DATA_PATH 别名（它曾导致模拟数据写入主数据集）"


def test_label_rule_consistent_with_threshold():
    """标签定义必须与阈值常量一致。"""
    assert ud.PASS_THRESHOLD == 10.0
    assert cfg.LABEL_COLUMN == "risk_label"
    assert set(cfg.RISK_LEVELS) == {"低", "中", "高"}
