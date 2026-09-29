# -*- coding: utf-8 -*-
"""
generate_data.py —— 模拟数据生成模块
=========================================================
职责：生成 300~500 条"模拟 / 脱敏"学生学业数据，供 KNN 风险预警模型训练。

【为什么不用 sklearn 的 make_classification】
sklearn 生成的数据往往"太干净"（近线性可分），会导致指标虚高到 1.00，
无法体现模型优劣。本项目刻意采用"潜在能力因子 + 行为因子 + 噪声 +
概率化打标 + 分位数定档"的方式，主动制造：
    1) 类别不平衡：高风险比例控制在 20%~35%；
    2) 特征重叠：能力强但态度差 / 能力弱但很努力 的交叉样本；
    3) 标签歧义：边界学生以概率形式打标，不是硬阈值，天然存在难分样本；
    4) 等级歧义：risk_label（二分类）与 risk_level（三档）口径不同，
       允许出现"中等档但判为需要关注"这类真实场景中的边界情况。
这样数据才"有真实区分难度"，指标才有说服力。

【可复现性】所有随机过程都由 random_state 控制，同一 seed 必然得到同一份数据。

【合规声明】全部数据由随机数模拟生成，不含任何真实学生信息，
            仅用于教学实验，不得用于真实学籍、评奖、处分等决策。
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# 兼容两种运行方式：python main.py（包导入）与 python src/generate_data.py（脚本直跑）
try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ======================================================================
# 一、通用小工具
# ======================================================================
def _clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """把数组裁剪到 [low, high] 区间（成绩、比例类字段的物理边界）。"""
    return np.clip(arr, low, high)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，用于把"风险倾向"映射到 0~1 概率。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _quantile_thresholds(x: np.ndarray, q1: float, q2: float) -> Tuple[float, float]:
    """
    计算分位数阈值，使用 numpy 默认的'线性插值'算法，且显式写入 ndarray.quantile
    以保证任何 numpy 版本下结果一致（避免不同版本插值方式差异导致不可复现）。
    """
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, q1, method="linear")), float(np.quantile(x, q2, method="linear"))


# ======================================================================
# 二、核心数据生成
# ======================================================================
def simulate_students(
    n_samples: int = cfg.N_SAMPLES,
    random_state: int = cfg.RANDOM_SEED,
    risk_ratio_target: float = cfg.RISK_RATIO_TARGET,
    enforce_size: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    生成模拟学生学业数据。

    参数
    ----
    n_samples       : 样本条数。训练/评估主数据集要求在 300~500 之间；
    random_state    : 随机种子，决定数据完全可复现
    risk_ratio_target : 目标高风险比例（仅用于记录与校准断言）
    enforce_size    : 是否强制 300~500 的规模约束。
                      主数据集必须为 True；"最终测试用的新样本批次"允许更小规模
                      （本项目取 100 条），因此调用时可置 False。

    返回
    ----
    df     : 含全部字段的 DataFrame
    report : 生成过程统计信息（比例、重叠度、可分性等），用于防止"数据过干净"
    """
    if enforce_size and not 300 <= n_samples <= 500:
        # 硬性约束校验：提前报错优于生成不合规数据后再排查
        raise ValueError(f"主数据集 n_samples 需在 300~500 之间，当前为 {n_samples}；"
                         f"若确需生成其他规模（如新样本测试集），请传 enforce_size=False")
    if n_samples < 30:
        raise ValueError(f"n_samples 过小（{n_samples}），无法进行有效的划分评估")

    rng = np.random.RandomState(random_state)
    n = n_samples

    # ---------------- 1) 潜在能力因子与学习行为因子 ----------------
    # 能力分层：为了让分布呈现真实的"双峰偏态"，用 3 档混合分布生成基础能力。
    # 48% 基础较好、34% 中等、18% 基础较弱——与高风险比例同一量级，
    # 但**不是**标签本身（标签后面还要叠加行为与噪声），因此不会线性可分。
    mix = rng.choice([0, 1, 2], size=n, p=[0.48, 0.34, 0.18])
    ability_center = np.array([78.0, 66.0, 52.0])[mix]
    ability = ability_center + rng.normal(0.0, 6.5, size=n)  # 个体能力波动

    # 投入因子：出勤 / 作业 / 自习 三者正相关（同一"态度"潜变量），
    # 但与能力仅是弱相关，故意让"高能力低投入""低能力高投入"大量存在。
    effort = 0.45 * (ability - 66.0) / 12.0 + rng.normal(0.0, 0.85, size=n)

    attendance = _clip(0.86 + 0.075 * effort + rng.normal(0.0, 0.070, size=n), 0.35, 1.00)
    homework = _clip(0.82 + 0.095 * effort + rng.normal(0.0, 0.100, size=n), 0.05, 1.00)
    study_hours = _clip(4.6 + 1.35 * effort + rng.normal(0.0, 1.350, size=n), 0.0, 10.0)

    # ---------------- 2) 三科成绩（相关信息 + 科目噪声 + 个体噪声） ----------------
    # 三科成绩共享同一个"能力"信号，因此彼此高度相关（符合真实成绩单），
    # 但各自带科目难度偏移与随机噪声，使三科不完全一致（存在偏科）。
    subject_bias = np.array([2.0, -6.5, -1.0])       # python 较易、ml 较难
    subject_sigma = np.array([7.0, 9.0, 8.0])        # 各科噪声强度
    student_noise = rng.normal(0.0, 3.0, size=n)     # 个体一次性波动（影响三科）

    # 出勤/作业差的同学会额外丢分（但不必然，系数较小以保留重叠）
    behavior_penalty = 12.0 * (1.0 - attendance) + 8.0 * (1.0 - homework)

    base = ability[:, None] + student_noise[:, None] - behavior_penalty[:, None]
    scores = base + subject_bias[None, :] + rng.normal(0.0, 1.0, size=(n, 3)) * subject_sigma[None, :]
    scores = _clip(scores, 25.0, 100.0)

    df = pd.DataFrame({
        "student_id": [f"SIM{i:04d}" for i in range(1, n + 1)],
        "python": np.round(scores[:, 0], 1),
        "ml": np.round(scores[:, 1], 1),
        "cv": np.round(scores[:, 2], 1),
        "attendance_rate": np.round(attendance, 3),
        "homework_submit_rate": np.round(homework, 3),
        "self_study_hours": np.round(study_hours, 2),
    })

    # 派生特征：平均分与偏科程度（后续也作为模型输入特征）
    df["score_mean"] = np.round(df[cfg.SCORE_COLUMNS].mean(axis=1), 2)
    df["score_std"] = np.round(df[cfg.SCORE_COLUMNS].std(axis=1, ddof=0), 2)

    # ---------------- 3) 风险评分（加权规则 + 显式噪声） ----------------
    # 规则权重是"业务先验"，不是用标签拟合出来的，因此不存在标签泄漏。
    risk_raw = (
        1.00 * (70.0 - df["score_mean"]) / 10.0                 # 平均分越低头部风险越大
        + 0.85 * (0.90 - df["attendance_rate"]) / 0.10          # 出勤不足
        + 0.75 * (0.85 - df["homework_submit_rate"]) / 0.10     # 作业不交
        + 0.45 * (4.5 - df["self_study_hours"]) / 1.5           # 自习时长不足
        + 0.20 * df["score_std"] / 10.0                         # 偏科加成
        + rng.normal(0.0, 0.90, size=n)                         # 显式噪声：制造真实歧义
    )

    # ---------------- 4) 二分类标签：概率化打标（关键防"满分"设计） ----------------
    # 先按分位数确定"倾向红线"risk_tau，再把风险评分经 sigmoid 变成概率后**随机抽样**，
    # 而不是硬阈值切割。这样边界学生天然存在难分样本，指标不会虚高到 1.0。
    risk_tau = float(np.quantile(risk_raw, 1.0 - risk_ratio_target, method="linear"))
    p_risk = _sigmoid(1.55 * (risk_raw - risk_tau))  # 1.55 为陡峭度，太大则退化为硬阈值
    risk_label = (rng.rand(n) < p_risk).astype(int)

    # ---------------- 5) 校准：让高风险比例落进 20%~35% 的合规区间 ----------------
    lo, hi = 0.20, 0.35
    ratio = risk_label.mean()
    if ratio < lo or ratio > hi:
        # 用"评分的分位数"反推需要的红线位置：评分越高越该被判为风险
        target_ratio = float(np.clip(risk_ratio_target, lo + 0.01, hi - 0.01))
        new_tau = float(np.quantile(risk_raw, 1.0 - target_ratio, method="linear"))
        p_risk = _sigmoid(1.55 * (risk_raw - new_tau))
        risk_label = (rng.rand(n) < p_risk).astype(int)
        ratio = risk_label.mean()
        if ratio < lo or ratio > hi:
            # 极端情况兜底：直接按评分的秩切出目标比例（仍然是"有难度"的：见可分性检查）
            k = int(round(n * target_ratio))
            order = np.argsort(-risk_raw)
            risk_label = np.zeros(n, dtype=int)
            risk_label[order[:k]] = 1
            ratio = risk_label.mean()

    df[cfg.LABEL_COLUMN] = risk_label
    df["risk_score"] = np.round(risk_raw, 4)     # 中间量，仅用于实验分析，不参与建模
    df["p_risk_true"] = np.round(p_risk, 4)      # 理论风险概率，仅用于分析

    # ---------------- 6) 风险等级：另一套口径（故意与二分类不完全一致） ----------------
    # risk_level 是"辅导员排查优先级"的三档，用独立分位数定档：
    #   低 = 前 62%，中 = 62%~88%，高 = 后 12%
    # 与 risk_label 口径不同，会出现"中等档但被判为需要关注"等边界情况，
    # 这正是真实系统中"概率输出 + 人工判断"并存的原因。
    t_mid, t_high = _quantile_thresholds(risk_raw, 0.62, 0.88)
    risk_level = np.where(risk_raw >= t_high, "高", np.where(risk_raw >= t_mid, "中", "低"))
    df[cfg.RISK_LEVEL_COLUMN] = risk_level

    # ---------------- 7) 统计报告（用于证明"数据不过于干净"） ----------------
    report = _build_report(df, risk_raw, risk_tau, t_mid, t_high, random_state)
    return df, report


def _build_report(df: pd.DataFrame, risk_raw: np.ndarray, risk_tau: float,
                  t_mid: float, t_high: float, random_state: int) -> Dict:
    """
    汇总数据质量报告。重点回答一个问题：数据是不是"太干净"了？

    关键指标 best_single_feature_acc —— 用**单个特征 + 最优阈值**能达到的最高准确率。
    如果它接近 1.0，说明一个简单的 if-else 就能做完任务，KNN 没有价值；
    本项目要求它明显低于 1.0，以证明特征之间有重叠、任务有真实难度。
    """
    X = df[cfg.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[cfg.LABEL_COLUMN].to_numpy(dtype=int)

    best_acc, best_feat = 0.0, None
    for j, name in enumerate(cfg.FEATURE_COLUMNS):
        col = X[:, j]
        # 该特征在正负类上的均值大小关系决定判别方向（分数越低越危险）
        direction = 1.0 if col[y == 1].mean() < col[y == 0].mean() else -1.0
        signed = direction * col
        # 在分位数候选点中搜索最佳阈值（等价于全阈值扫描的近似，速度快且足够说明问题）
        for t in np.quantile(signed, np.linspace(0.05, 0.95, 37)):
            # direction=1 表示值越小越危险 → 小于阈值判为风险
            pred = (signed <= t).astype(int) if direction == 1 else (signed >= t).astype(int)
            acc = float((pred == y).mean())
            if acc > best_acc:
                best_acc, best_feat = acc, name

    report = {
        "random_state": int(random_state),
        "n_samples": int(len(df)),
        "risk_label_counts": {str(k): int(v) for k, v in df[cfg.LABEL_COLUMN].value_counts().sort_index().items()},
        "risk_label_ratio": round(float(y.mean()), 4),
        "risk_level_counts": {k: int(v) for k, v in df[cfg.RISK_LEVEL_COLUMN].value_counts().items()},
        "risk_score_tau": round(float(risk_tau), 4),
        "risk_level_cuts": {"mid": round(float(t_mid), 4), "high": round(float(t_high), 4)},
        "best_single_feature": best_feat,
        "best_single_feature_acc": round(best_acc, 4),
        "data_is_too_clean": bool(best_acc >= 0.90),
        # 边界歧义样本量：二分类标签与三档等级"不一致"的样本（如 中风险 却 判为需要关注）
        "label_level_conflict": int(((df[cfg.RISK_LEVEL_COLUMN].isin(["中", "高"])) &
                                     (df[cfg.LABEL_COLUMN] == 0)).sum()),
        "feature_mean_by_label": {
            name: {
                "正常": round(float(df.loc[df[cfg.LABEL_COLUMN] == 0, name].mean()), 3),
                "需关注": round(float(df.loc[df[cfg.LABEL_COLUMN] == 1, name].mean()), 3),
            }
            for name in cfg.FEATURE_COLUMNS
        },
    }
    return report


def save_simulation(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """
    保存**模拟数据**到 CSV（utf-8-sig 便于 Excel 直接打开不乱码）。

    默认路径必须是 cfg.SIM_DATA_PATH，**不能**用 cfg.DATA_PATH：
    v2 起 config.DATA_PATH 已指向 UCI 真实数据集，早期版本因此把模拟数据
    写进了 data/uci_student_performance.csv，造成主数据集被静默污染
    （实测后果：v2 流程读到 400×13 的模拟数据后报 KeyError: student_group_id）。
    """
    cfg.ensure_dirs()
    path = path or cfg.SIM_DATA_PATH
    out = df.copy()
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ======================================================================
# 三、原始 5 人种子数据（兼容项目早期存在的 scores.csv）
# ======================================================================
def build_seed_scores(path: str = cfg.SEED_CSV_PATH) -> str:
    """
    生成最初版 5 人 × (python/ml/cv) 的 scores.csv 种子数据。

    这 5 条数据刻意覆盖典型画像（学霸、偏科、低投入、中等、临界），
    在 README 中作为"数据扩写来源"说明；扩写后的主数据集由 simulate_students 生成。
    """
    rows = [
        # student_id, python, ml, cv, attendance, homework, study, note
        ("S001", 92.0, 88.0, 85.0, 0.98, 1.00, 8.0, "学霸，全科均衡"),
        ("S002", 78.0, 55.0, 62.0, 0.90, 0.85, 5.0, "机器学习偏弱，偏科"),
        ("S003", 58.0, 52.0, 60.0, 0.62, 0.45, 2.0, "低投入，出勤与作业均不足"),
        ("S004", 70.0, 68.0, 72.0, 0.85, 0.80, 4.5, "中等水平，态度尚可"),
        ("S005", 64.0, 61.0, 55.0, 0.75, 0.70, 3.5, "临界学生，需要重点观察"),
    ]
    df = pd.DataFrame(rows, columns=[
        "student_id", "python", "ml", "cv",
        "attendance_rate", "homework_submit_rate", "self_study_hours", "note",
    ])
    df["score_mean"] = np.round(df[cfg.SCORE_COLUMNS].mean(axis=1), 2)
    df["score_std"] = np.round(df[cfg.SCORE_COLUMNS].std(axis=1, ddof=0), 2)
    cfg.ensure_dirs()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ======================================================================
# 四、命令行入口
# ======================================================================
def main(argv: List[str] | None = None) -> Dict:
    """命令行入口：生成种子数据 + 主模拟数据集 + 新样本测试集。"""
    parser = argparse.ArgumentParser(description="生成校园学生成绩模拟数据（教学用，非真实数据）")
    parser.add_argument("--n", type=int, default=cfg.N_SAMPLES, help=f"样本条数（300~500），默认 {cfg.N_SAMPLES}")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help=f"随机种子，默认 {cfg.RANDOM_SEED}")
    parser.add_argument("--n-new", type=int, default=cfg.N_NEW_SAMPLES, help=f"新样本条数，默认 {cfg.N_NEW_SAMPLES}")
    parser.add_argument("--new-seed", type=int, default=cfg.NEW_BATCH_SEED, help="新样本随机种子")
    parser.add_argument("--ratio", type=float, default=cfg.RISK_RATIO_TARGET, help="目标高风险比例")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("【步骤】生成模拟数据（全部为随机模拟，不含真实学生信息）")
    print("=" * 68)

    # 1) 种子数据
    seed_path = build_seed_scores()
    print(f"[1/3] 原始 5 人种子数据已保存：{seed_path}")

    # 2) 主数据集
    df, report = simulate_students(n_samples=args.n, random_state=args.seed,
                                   risk_ratio_target=args.ratio)
    path = save_simulation(df)
    print(f"[2/3] 模拟数据集已保存：{path}")
    print(f"      样本数={report['n_samples']}，"
          f"高风险比例={report['risk_label_ratio']:.4f}（要求 0.20~0.35）")
    print(f"      标签分布={report['risk_label_counts']}")
    print(f"      等级分布={report['risk_level_counts']}")
    print(f"      单特征最优准确率={report['best_single_feature_acc']:.4f}"
          f"（特征={report['best_single_feature']}）"
          f"→ 数据是否过于干净：{'是（需要重新调参）' if report['data_is_too_clean'] else '否（有真实区分难度）'}")
    print(f"      标签与等级口径不一致的边界样本数={report['label_level_conflict']}")

    # 3) 新样本（从未参与训练，用于最终测试）
    #    注意 enforce_size=False：新样本批次不属于"主数据集"，规模可小于 300
    new_df, new_report = simulate_students(n_samples=args.n_new, random_state=args.new_seed,
                                           risk_ratio_target=args.ratio, enforce_size=False)
    # 新样本学生编号独立，避免与训练集 ID 混同
    new_df["student_id"] = [f"NEW{i:04d}" for i in range(1, len(new_df) + 1)]
    new_path = os.path.join(cfg.DATA_DIR, "scores_new_batch.csv")
    save_simulation(new_df, new_path)
    print(f"[3/3] 新样本测试集已保存：{new_path}")
    print(f"      样本数={new_report['n_samples']}，高风险比例={new_report['risk_label_ratio']:.4f}")
    print("-" * 68)
    print(cfg.DISCLAIMER)

    return {"data_path": path, "new_path": new_path, "report": report, "new_report": new_report}


if __name__ == "__main__":
    main()
