# -*- coding: utf-8 -*-
"""
preprocess.py —— 数据预处理模块
=========================================================
职责：
    1. 读取模拟数据，校验字段与取值范围；
    2. 缺失值处理（中位数填补，且中位数**只从训练集统计**）；
    3. 特征归一化（z-score 标准化，均值和标准差**只从训练集统计**）；
    4. 训练集 / 测试集随机划分、留一法索引生成。

【为什么强调"只从训练集统计"】
如果把全量数据的均值/标准差/中位数拿去标准化和填补，测试集的信息就"泄漏"进了
训练过程，指标会偏乐观。教学项目里这是最常见的隐性作弊，因此本项目把 scaler
和 imputer 都做成独立对象：`fit` 只允许在训练集上调用，`transform` 才用于测试集。

【为什么选 z-score 而不是 min-max】
KNN 用欧氏距离度量相似度，字段量纲必须统一：
    - 成绩 0~100、出勤率 0~1、自习 0~10，直接算距离会被成绩主导；
    - z-score 让每个特征变成"偏离均值几个标准差"，距离才有可比性；
    - z-score 对异常值比 min-max 稳健（min-max 会被单个极端值压缩整体尺度）。
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ======================================================================
# 一、数据结构
# ======================================================================
@dataclass
class Dataset:
    """封装一次建模所需的全部数组，避免到处传 DataFrame 造成口径不一致。"""
    X: np.ndarray            # 标准化后的特征矩阵 (n, d)
    y: np.ndarray            # 标签 (n,)
    ids: np.ndarray          # 学生编号 (n,)
    feature_names: List[str]  # 特征列名，顺序与 X 的列一一对应
    raw: Optional[pd.DataFrame] = None  # 原始（未标准化）数据，供解释模块做中文描述
    risk_level: Optional[np.ndarray] = None  # 三档风险等级（真实值，用于看板）

    def __len__(self) -> int:  # 方便直接 len(dataset)
        return int(self.X.shape[0])

    def subset(self, idx: np.ndarray) -> "Dataset":
        """按索引取子集（用于训练/测试划分，保证 raw 与数组同步切分）。"""
        idx = np.asarray(idx)
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            ids=self.ids[idx],
            feature_names=list(self.feature_names),
            raw=None if self.raw is None else self.raw.iloc[idx].reset_index(drop=True),
            risk_level=None if self.risk_level is None else self.risk_level[idx],
        )


# ======================================================================
# 二、缺失值处理
# ======================================================================
class MedianImputer:
    """
    中位数填补器。

    为什么用中位数而不是均值？
      - 学业数据里"成绩特别低"的样本会把均值拉偏，中位数更稳健；
      - 缺失通常不是完全随机（例如缺考者往往成绩差），中位数受极端值影响小。

    注意：fill_values_ 只能由训练集 fit 得到，之后用于测试集/新样本。
    """

    def __init__(self) -> None:
        self.fill_values_: Optional[np.ndarray] = None
        self.n_missing_per_feature_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MedianImputer":
        """统计每一列的中位数（忽略 NaN）。全列为 NaN 时退化为 0。"""
        X = np.asarray(X, dtype=float)
        self.n_missing_per_feature_ = np.isnan(X).sum(axis=0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        self.fill_values_ = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """按训练集得到的中位数填补缺失值。"""
        if self.fill_values_ is None:
            raise RuntimeError("MedianImputer 尚未 fit，请先调用 fit(train_X)")
        X = np.array(X, dtype=float, copy=True)
        inds = np.where(np.isnan(X))
        if len(inds[0]) > 0:
            X[inds] = np.take(self.fill_values_, inds[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def total_missing(self) -> int:
        """本次 fit 时遇到的缺失值总数（用于报告）。"""
        return 0 if self.n_missing_per_feature_ is None else int(self.n_missing_per_feature_.sum())


# ======================================================================
# 三、特征标准化
# ======================================================================
class StandardScaler:
    """
    z-score 标准化：x' = (x - mean) / std。

    KNN 依赖距离，量纲不统一会让"成绩(0~100)"完全压过"出勤率(0~1)"，
    因此标准化是 KNN 的必需步骤，而不是可选优化。
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """在训练集上统计均值与标准差（ddof=0）。std 为 0 的常量列置 1，避免除零。"""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler 尚未 fit，请先调用 fit(train_X)")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """反标准化：把标准化空间还原成原始量纲，用于把解释文字写成"多少分"。"""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler 尚未 fit")
        return np.asarray(X, dtype=float) * self.std_ + self.mean_

    def to_dict(self) -> Dict[str, List[float]]:
        """导出为可 JSON 序列化的字典，随模型一起保存供查询功能复用。"""
        return {
            "mean": [float(v) for v in (self.mean_ if self.mean_ is not None else [])],
            "std": [float(v) for v in (self.std_ if self.std_ is not None else [])],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "StandardScaler":
        obj = cls()
        obj.mean_ = np.asarray(d["mean"], dtype=float)
        obj.std_ = np.asarray(d["std"], dtype=float)
        return obj


# ======================================================================
# 四、读取与校验
# ======================================================================
def load_simulated(path: Optional[str] = None) -> pd.DataFrame:
    """
    读取**模拟对照数据集**（scores_simulated.csv）。

    注意：v2 起 config.DATA_PATH 指向 UCI 主数据集，因此这里默认显式使用
    cfg.SIM_DATA_PATH，避免"以为在读模拟数据、其实读到了 UCI 数据"的口径错误。
    """
    path = path or cfg.SIM_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到模拟数据文件 {path}。请先运行：python src/generate_data.py"
        )
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:  # pragma: no cover
        return pd.read_csv(path, encoding="utf-8")


def validate_dataframe(df: pd.DataFrame, require_label: bool = True) -> Dict:
    """
    数据体检：字段是否齐全、取值范围是否合理、是否有重复学生编号。

    返回体检报告字典；发现问题时给出 `warnings` 列表（不直接抛错，
    以免影响"能运行"这一验收要求，但会明确打印出来）。
    """
    warnings: List[str] = []
    required = ["student_id"] + cfg.SCORE_COLUMNS + [
        "attendance_rate", "homework_submit_rate", "self_study_hours",
    ]
    if require_label:
        required += [cfg.LABEL_COLUMN, cfg.RISK_LEVEL_COLUMN]

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必需字段：{missing_cols}")

    # 派生特征缺失时自动补算（兼容早期只有 5 列的 scores.csv）
    if "score_mean" not in df.columns:
        df["score_mean"] = df[cfg.SCORE_COLUMNS].mean(axis=1)
        warnings.append("缺少 score_mean，已自动补算")
    if "score_std" not in df.columns:
        df["score_std"] = df[cfg.SCORE_COLUMNS].std(axis=1, ddof=0)
        warnings.append("缺少 score_std，已自动补算")

    # 取值合法性检查
    for c in cfg.SCORE_COLUMNS + ["score_mean"]:
        bad = int(((df[c] < 0) | (df[c] > 100)).sum())
        if bad:
            warnings.append(f"{c} 有 {bad} 条超出 0~100")
    for c in ["attendance_rate", "homework_submit_rate"]:
        bad = int(((df[c] < 0) | (df[c] > 1)).sum())
        if bad:
            warnings.append(f"{c} 有 {bad} 条超出 0~1")
    bad = int(((df["self_study_hours"] < 0) | (df["self_study_hours"] > 10)).sum())
    if bad:
        warnings.append(f"self_study_hours 有 {bad} 条超出 0~10")

    if df["student_id"].duplicated().any():
        warnings.append("存在重复的 student_id")

    if require_label and not set(pd.unique(df[cfg.LABEL_COLUMN])).issubset({0, 1}):
        warnings.append(f"{cfg.LABEL_COLUMN} 取值不是 0/1")

    n_missing = int(df[cfg.FEATURE_COLUMNS].isna().sum().sum())

    return {
        "n_samples": int(len(df)),
        "n_features": len(cfg.FEATURE_COLUMNS),
        "missing_total": n_missing,
        "missing_per_feature": {c: int(df[c].isna().sum()) for c in cfg.FEATURE_COLUMNS},
        "warnings": warnings,
    }


# ======================================================================
# 五、构建 Dataset（含缺失值填补与标准化）
# ======================================================================
def build_dataset(
    df: pd.DataFrame,
    imputer: Optional[MedianImputer] = None,
    scaler: Optional[StandardScaler] = None,
    require_label: bool = True,
) -> Tuple[Dataset, MedianImputer, StandardScaler]:
    """
    把 DataFrame 转成 Dataset。

    - 若传入已 fit 的 imputer / scaler，则沿用其统计量（用于测试集、新样本、单学生查询）；
    - 若不传，则在本次数据上 fit（只能对训练集这样用）。

    返回 (Dataset, imputer, scaler)。
    """
    validate_dataframe(df, require_label=require_label)

    X_raw = df[cfg.FEATURE_COLUMNS].to_numpy(dtype=float)
    if imputer is None:
        imputer = MedianImputer().fit(X_raw)
    X_filled = imputer.transform(X_raw)

    if scaler is None:
        scaler = StandardScaler().fit(X_filled)
    X_std = scaler.transform(X_filled)

    y = df[cfg.LABEL_COLUMN].to_numpy(dtype=int) if require_label else np.full(len(df), -1, dtype=int)
    risk_level = df[cfg.RISK_LEVEL_COLUMN].to_numpy() if cfg.RISK_LEVEL_COLUMN in df.columns else None

    ds = Dataset(
        X=np.asarray(X_std, dtype=float),
        y=y,
        ids=df["student_id"].to_numpy(),
        feature_names=list(cfg.FEATURE_COLUMNS),
        raw=df.reset_index(drop=True),
        risk_level=risk_level,
    )
    return ds, imputer, scaler


def random_split(
    ds: Dataset,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_SEED,
) -> Tuple[Dataset, Dataset]:
    """
    按比例随机划分训练集/测试集。

    使用 RandomState(random_state) 而非全局 np.random，保证"10 次随机划分"
    每一条划分都能被单独复现（实验记录里要能写出第 i 次划分的确切种子）。
    """
    n = len(ds)
    n_test = int(round(n * test_size))
    n_test = max(1, min(n - 1, n_test))  # 防止极端比例导致训练集/测试集为空
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return ds.subset(train_idx), ds.subset(test_idx)


def make_loocv_indices(n: int):
    """
    生成留一法索引序列：第 i 折用"除第 i 个样本外的全部数据"训练。

    这里返回的是索引而不是真的复制数据——留一法在 KNN 中的高效实现见
    cross_validate.py：只需一次距离矩阵，掩掉对角线即可，无需训练 n 次。
    """
    for i in range(n):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        yield i, train_idx, np.array([i])


def save_preprocess_stats(imputer: MedianImputer, scaler: StandardScaler,
                          path: str = cfg.PREPROCESS_STATS_PATH,
                          extra: Optional[Dict] = None) -> str:
    """
    保存预处理统计量到 JSON。

    查询单个学生时必须复用训练时的中位数与均值/标准差，
    否则同一个学生在"训练时"和"查询时"会被标准化成不同向量，预测结果不一致。
    """
    cfg.ensure_dirs()
    payload = {
        "feature_columns": cfg.FEATURE_COLUMNS,
        "imputer_fill_values": [float(v) for v in imputer.fill_values_],
        "scaler": scaler.to_dict(),
        "risk_level_thresholds": cfg.RISK_LEVEL_THRESHOLDS,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_preprocess_stats(path: str = cfg.PREPROCESS_STATS_PATH) -> Dict:
    """读取预处理统计量。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """命令行自检：读取数据 → 体检 → 构建 Dataset → 做一次 7:3 划分并打印统计。"""
    print("=" * 68)
    print("【步骤】预处理自检")
    print("=" * 68)

    df = load_simulated()
    report = validate_dataframe(df)
    print(f"样本数={report['n_samples']}，特征数={report['n_features']}，缺失值总数={report['missing_total']}")
    for w in report["warnings"]:
        print(f"  [警告] {w}")

    ds, imputer, scaler = build_dataset(df)
    print("标准化后各特征均值（应接近 0）：",
          np.round(ds.X.mean(axis=0), 4))
    print("标准化后各特征标准差（应接近 1）：",
          np.round(ds.X.std(axis=0), 4))

    train, test = random_split(ds, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED)
    print(f"随机划分（seed={cfg.RANDOM_SEED}，test_size={cfg.TEST_SIZE}）："
          f"训练集 {len(train)} 条 / 测试集 {len(test)} 条")
    print(f"训练集高风险比例={train.y.mean():.4f}，测试集高风险比例={test.y.mean():.4f}")
    print("示例（前 3 个测试样本的原始特征）：")
    print(test.raw[["student_id"] + cfg.FEATURE_COLUMNS].head(3).to_string(index=False))
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
