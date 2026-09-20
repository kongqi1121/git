# -*- coding: utf-8 -*-
"""
uci_features.py —— UCI 特征编码与标签派生
=========================================================
把 UCI 原始字段转成模型可用的**数值矩阵**，并把"口径"固定下来。

【三个必须做对的地方（答辩重点）】

1. **分类字段编码**：UCI 里有 9 个分类型字段（school/sex/address/…/guardian）。
   本项目采用**确定性的二元指示编码**（每个字段取一个参考水平，其余水平各占一列），
   而不是在每次划分时临时 `get_dummies`。原因：
     · 随机划分后训练集里可能缺少某个类别（例如 10 次划分中某次没有 'teacher'），
       临时编码会让不同划分的**特征维度都不一样**，指标根本不可比；
     · 固定 schema 后，任何划分、任何新样本都用同一套列顺序，
       与 sklearn 的 `OneHotEncoder(handle_unknown='ignore')` 思路一致但更可控。

2. **标签定义**：葡萄牙成绩 0~20 分，10 分及格 → `risk_label = (G3 < 10)`。
   两门课程统一口径；合并后高风险比例 22.03%，落在 20%~35% 区间内。
   三档风险等级 `risk_level` 由 **G3 的分位数**派生（与模拟数据的做法一致，
   保证下游看板/评估模块无需改动）：
       低 = G3 前 62% 区间，中 = 62%~88%，高 = 后 12%
   注意这与二分类标签**口径不同**，会出现"中档但判为正常"这类边界样本 ——
   这是刻意保留的：真实系统中"等级"与"是否干预"本来就由不同标准决定。

3. **主口径禁用 G1/G2**：这两个早期成绩与 G3 的相关系数达 0.905/0.919，
   用它预测期末成绩属于**标签泄漏**。因此：
     · `include_early_grades=False`（默认，主口径）→ 16 个原始字段 + 1 个派生字段；
     · `include_early_grades=True`（消融对照）→ 额外加入 G1、G2。
   两组口径在 `train_eval` 里做消融对照，量化"早期成绩带来多少提升"。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ----------------------------------------------------------------------
# 一、固定编码 schema
# ----------------------------------------------------------------------
# 每个分类型字段的"参考水平"（取该字段的一个合法取值作为基准，不单独占列）。
# 其余水平各生成一列，命名为 f"{字段}_{水平}"，取值 0/1。
# 注意：基准水平不代表"更好/更差"，只是编码上的参照，系数解释时相对基准而言。
CATEGORICAL_REFERENCE = {
    "school": "GP",
    "sex": "F",
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
}

# 已知取值（用于固定列顺序，并支持"新样本出现未知类别时填 0"）
CATEGORICAL_LEVELS = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["other", "teacher", "health", "services", "at_home"],
    "Fjob": ["other", "teacher", "health", "services", "at_home"],
    "reason": ["course", "home", "reputation", "other"],
    "guardian": ["mother", "father", "other"],
}

# 二元 yes/no 字段（直接映射 1/0）
YN_FEATURES = list(cfg.UCI_BINARY_FEATURES)

# 数值字段（保持原始量纲，由标准化器统一处理）
NUM_FEATURES = list(cfg.UCI_NUMERIC_FEATURES)

# 派生字段
DERIVED_FEATURES = ["prior_avg"]   # 历史成绩均值（仅 when include_early_grades=True）

# 派生特征中文名
DERIVED_FEATURE_CN = {"prior_avg": "历史平均成绩(G1/G2均值)"}

# 分类指示列取值的中文说明（供图表与解释使用）
LEVEL_CN = {
    "MS": "MS 校", "GP": "GP 校",
    "M": "男", "F": "女",
    "R": "农村", "U": "城市",
    "LE3": "≤3 人", "GT3": ">3 人",
    "T": "父母同住", "A": "父母分居",
    "teacher": "教师", "health": "医疗", "services": "服务业",
    "at_home": "居家", "other": "其他",
    "home": "离家近", "reputation": "学校声誉", "course": "课程偏好",
    "mother": "母亲", "father": "父亲",
}


# ======================================================================
# 二、特征名生成
# ======================================================================
def feature_names(include_early_grades: bool = False) -> List[str]:
    """
    返回**固定顺序**的特征列名。

    顺序 = 数值特征 → 分类指示列 → 二元特征 → [早期成绩/派生]
    任何调用方都用这个函数取列名，绝不在别处硬编码顺序。
    """
    names: List[str] = []
    names.extend(NUM_FEATURES)
    for col, levels in CATEGORICAL_LEVELS.items():
        ref = CATEGORICAL_REFERENCE[col]
        for lv in levels:
            if lv == ref:
                continue
            names.append(f"{col}_{lv}")
    names.extend(YN_FEATURES)
    if include_early_grades:
        names.extend(cfg.UCI_EARLY_GRADE_FEATURES)   # G1, G2
        names.extend(DERIVED_FEATURES)               # prior_avg
    return names


def feature_name_cn(name: str) -> str:
    """把（可能带后缀的）特征名翻译成中文，供图表与解释使用。"""
    if name in cfg.FEATURE_CN_UCI:
        return cfg.FEATURE_CN_UCI[name]
    if name in DERIVED_FEATURE_CN:
        return DERIVED_FEATURE_CN[name]
    if "_" in name:
        col, lv = name.split("_", 1)
        base = cfg.FEATURE_CN_UCI.get(col, col)
        return f"{base}={LEVEL_CN.get(lv, lv)}"
    return name


# ======================================================================
# 三、核心：把 DataFrame 转成矩阵
# ======================================================================
def prepare_uci_features(df: pd.DataFrame,
                         include_early_grades: bool = False,
                         standardize: bool = True,
                         scaler=None,
                         verbose: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    编码 UCI 特征。

    参数
    ----
    df                   : 含 UCI 原始字段与 risk_label 的 DataFrame
    include_early_grades : 是否把 G1/G2（及派生 prior_avg）纳入特征。
                           False = 主口径（避免标签泄漏，模拟"期中预警"）
                           True  = 消融对照口径
    standardize          : 是否做 z-score 标准化（KNN 必需；逻辑回归也受益）
    scaler               : 已 fit 的 StandardScaler（传入则复用，避免泄漏）

    返回 (X, y, names)
    """
    names = feature_names(include_early_grades)
    n = len(df)
    X = np.zeros((n, len(names)), dtype=float)

    def col_index(name: str) -> int:
        return names.index(name)

    # ---- ① 数值特征 ----
    for c in NUM_FEATURES:
        if c not in df.columns:
            raise KeyError(f"缺少数值特征列 {c}")
        X[:, col_index(c)] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

    # ---- ② 分类指示列 ----
    for col, levels in CATEGORICAL_LEVELS.items():
        ref = CATEGORICAL_REFERENCE[col]
        vals = df[col].astype(str).to_numpy()
        for lv in levels:
            if lv == ref:
                continue
            X[:, col_index(f"{col}_{lv}")] = (vals == lv).astype(float)
        if verbose:
            unknown = sorted(set(vals) - set(levels))
            if unknown:
                print(f"    [提示] {col} 出现未知取值 {unknown}，已全部按 0 处理")

    # ---- ③ 二元 yes/no 特征 ----
    for c in YN_FEATURES:
        vals = df[c].astype(str).str.lower().to_numpy()
        X[:, col_index(c)] = np.where(vals == "yes", 1.0, np.where(vals == "no", 0.0, np.nan))

    # ---- ④ 早期成绩（仅消融口径）----
    if include_early_grades:
        for c in cfg.UCI_EARLY_GRADE_FEATURES:
            X[:, col_index(c)] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        g = df[cfg.UCI_EARLY_GRADE_FEATURES].to_numpy(dtype=float)
        X[:, col_index("prior_avg")] = np.nanmean(g, axis=1)

    # ---- 缺失值：先用列中位数填补（真实数据里可能有少量缺失）----
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        inds = np.where(np.isnan(X))
        X[inds] = med[inds[1]]

    # ---- 标准化 ----
    if standardize:
        if scaler is None:
            # 不传 scaler 时必须在本批数据上 fit（只能用于训练集！）
            try:
                from .preprocess import StandardScaler
            except ImportError:  # pragma: no cover - 脚本直跑分支
                from preprocess import StandardScaler
            scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    y = df[cfg.LABEL_COLUMN].to_numpy(dtype=int)
    return X, y, names


# ======================================================================
# 四、标签与派生列
# ======================================================================
def add_risk_level(df: pd.DataFrame,
                   q_low: float = 0.62,
                   q_high: float = 0.88) -> pd.DataFrame:
    """
    由 G3 派生三档风险等级（低/中/高）。

    【为什么不能用"G3 的分位数"直接当阈值】G3 是 **0~20 的整数**，高度离散
    （1044 条里只有 19 个不同取值）。若直接取 62% 分位数的浮点值作为切点，
    该点恰好落在某个整分数上，`>= 切点 → 低` 会把整个分数段一次性归入"低"，
    结果"中"档被完全跳过（实测只剩"高 637 / 低 407"两档，三档失效）。

    因此这里改为：**在每个目标分位数附近，选择使各档人数最接近目标比例、
    且落在整数分值上的切点**，并按阈值从低到高依次判定（保证等级连续、无跳跃）。
    这样三档都有样本，且切点是有意义的整数分数。
    """
    out = df.copy()
    g3 = out["G3"].to_numpy(dtype=float)
    levels = np.unique(g3)                      # 只有 19 个取值，枚举所有切点组合成本极低
    n = len(g3)
    # 目标比例：低 / 中 / 高
    w_low, w_high = q_low, 1.0 - q_high
    w_mid = 1.0 - w_low - w_high

    best = None
    # 直接枚举所有合法的整数切点对（t_high < t_mid），选使三档人数比例最接近目标的一组。
    # 这比"取分位数再四舍五入"更稳健：后者在离散分数上会出现两档合并、中档被跳过的问题。
    for i, t_high in enumerate(levels):
        for t_mid in levels[i + 1:]:
            n_high = int((g3 < t_high).sum())
            n_mid = int(((g3 >= t_high) & (g3 < t_mid)).sum())
            n_low = n - n_high - n_mid
            if min(n_high, n_mid, n_low) == 0:      # 必须三档都有人
                continue
            # 用三个比例的分箱损失衡量"接近目标"的程度
            loss = (abs(n_low / n - w_low) + abs(n_mid / n - w_mid)
                    + abs(n_high / n - w_high))
            if best is None or loss < best[0]:
                best = (loss, float(t_high), float(t_mid), n_low, n_mid, n_high)

    if best is None:
        # 极端情况兜底：按唯一分值三等分（几乎不会发生）
        t_high, t_mid = float(levels[len(levels) // 3]), float(levels[2 * len(levels) // 3])
        counts = None
    else:
        _, t_high, t_mid, n_low, n_mid, n_high = best
        counts = {"低": n_low, "中": n_mid, "高": n_high}

    # 按阈值从高到低判定，保证"高 ⊂ 中 ⊂ 低"的包含关系，等级自然连续
    level = np.where(g3 < t_high, "高", np.where(g3 < t_mid, "中", "低"))
    out[cfg.RISK_LEVEL_COLUMN] = level
    out.attrs["risk_level_cuts"] = {
        "mid": t_mid, "high": t_high, "counts": counts,
        "rule": f"G3 < {t_high:g} → 高；{t_high:g} ≤ G3 < {t_mid:g} → 中；G3 ≥ {t_mid:g} → 低",
        "method": "枚举整数切点，选取使三档人数比例最接近 0.62/0.26/0.12 的组合",
    }
    return out


# ======================================================================
# 五、自检
# ======================================================================
def main() -> None:
    """自检：编码后矩阵形状、取值分布、两套口径的特征数、标签比例。"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
    df = add_risk_level(df)

    print("=" * 68)
    print("【自检】UCI 特征编码")
    print("=" * 68)
    for flag in (False, True):
        X, y, names = prepare_uci_features(df, include_early_grades=flag)
        tag = "消融口径（含 G1/G2）" if flag else "主口径（不含 G1/G2）"
        print(f"\n{tag}: X.shape={X.shape}, 特征数={len(names)}")
        print(f"   高风险比例={y.mean():.4f}  标准化后均值范围=[{X.mean(axis=0).min():.3f},"
              f"{X.mean(axis=0).max():.3f}]  标准差范围=[{X.std(axis=0).min():.3f},"
              f"{X.std(axis=0).max():.3f}]")
        print(f"   特征名（前 8）: {names[:8]}")
    print()
    print("三档等级分布:", df[cfg.RISK_LEVEL_COLUMN].value_counts().to_dict())
    print("特征中文名示例:", [(n, feature_name_cn(n)) for n in
                             ["failures", "studytime", "school_MS", "Mjob_teacher"]])
    print("=" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
