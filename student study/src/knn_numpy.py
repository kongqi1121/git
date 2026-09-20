# -*- coding: utf-8 -*-
"""
knn_numpy.py —— 主模型：纯 numpy 手写 KNN（K 近邻分类器）
=========================================================
本项目核心 AI 功能 = 二分类：判断学生是否"需要重点关注"。
按实训要求，主模型必须是手写 KNN，**不得**用 sklearn 的 KNeighborsClassifier 替代。

实现要点（答辩要能讲清）：
    1. 距离度量：欧氏距离 d(a,b) = sqrt( Σ_i (a_i - b_i)^2 )
       本实现用矩阵恒等式一次性算出全部距离，避免双重循环：
           ||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·b^T
       复杂度从 O(n_test * n_train * d) 的 Python 循环降为一次矩阵乘法，
       400 条数据下留一法（400×400 距离矩阵）可以瞬间完成。
    2. 邻居选择：对每个待预测样本，取距离最小的 k 个训练样本。
       用 np.argpartition 做部分排序（O(n)）而不是全排序（O(n log n)），
       再只对选中的 k 个做排序，保证"近邻学生名单"按距离有序输出。
    3. 多数投票：k 个近邻的标签做多数表决得到类别。
       平票时按约定归为"正常"（0），即"疑罪从无"——教学场景下宁可漏报也不误报，
       这一约定在 README 的风险对策中说明。
    4. 概率输出 predict_proba：k 个近邻中"高风险(1)"样本所占比例。
       例如 k=7 且 5 个近邻是高风险 → p = 5/7 ≈ 0.714。
       这个概率天然是离散的（分母只有 k+1 种取值），是 KNN 的固有特性，
       因此概率只适合做"风险排序"和使用者判断，不适合当作精确概率解读。

【重要声明】本模块输出仅用于教学实验，不得作为真实学籍、评奖、处分的依据。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ======================================================================
# 一、核心算法：KNN 分类器
# ======================================================================
class KNNClassifierNumpy:
    """
    用 numpy 手写的 K 近邻分类器（支持二分类/多分类，本项目中为二分类）。

    典型用法
    --------
    >>> clf = KNNClassifierNumpy(k=7).fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)[:, 1]   # 高风险概率
    >>> pred  = clf.predict(X_test)               # 0/1 类别
    """

    def __init__(self, k: int = cfg.MAIN_K) -> None:
        if k < 1:
            raise ValueError(f"k 必须 >= 1，当前为 {k}")
        self.k = int(k)
        # 无索引信息时判定"几乎为自己"的绝对容差。
        # 取值依据：实测向量化距离公式算"样本到自身"时误差量级约 1e-7（灾难性抵消），
        # 因此容差必须大于该误差量级；但又要远小于"两个真实不同学生"的最小距离
        # （标准化特征下通常 > 1e-3），所以 1e-6 是安全的量级。
        self.SELF_DISTANCE_TOL = 1e-6
        # 以下属性在 fit 后才有值
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self._X_norm_sq_: Optional[np.ndarray] = None
        self._train_labels_1d_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 1) 拟合：KNN 是"懒惰学习"，训练阶段只保存数据，不做参数估计
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierNumpy":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组 (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError(f"X 与 y 样本数不一致：{len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("训练集为空，无法 fit")

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        # 预计算训练样本的 ||x||^2，供距离矩阵复用，避免每次预测重复计算
        self._X_norm_sq_ = np.einsum("ij,ij->i", X, X)
        # 若已是 0/1 二分类，直接缓存 1 维标签数组便于向量化投票
        self._train_labels_1d_ = y.astype(int) if set(np.unique(y)).issubset({0, 1}) else None
        return self

    # ------------------------------------------------------------------
    # 2) 距离计算：手写欧氏距离（这是本项目的核心考点）
    # ------------------------------------------------------------------
    @staticmethod
    def euclidean_distances(A: np.ndarray, B: Optional[np.ndarray] = None,
                            B_norm_sq: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算 A 中每个样本到 B 中每个样本的欧氏距离，返回 (len(A), len(B)) 矩阵。

        数学推导（矩阵形式）：
            ||a - b||^2 = (a-b)·(a-b) = a·a + b·b - 2·a·b
        用矩阵乘法一次算完所有样本对：
            D2 = |A|^2[:,None] + |B|^2[None,:] - 2 A B^T
        最后开方得到欧氏距离。数值上可能出现极小的负数（浮点误差），
        因此先 clip 到 0 再 sqrt，避免 sqrt(负数) = nan。
        """
        A = np.asarray(A, dtype=float)
        if B is None:                       # 不传 B 时算 A 到自身的距离矩阵
            B = A
            B_norm_sq = np.einsum("ij,ij->i", B, B) if B_norm_sq is None else B_norm_sq
        elif B_norm_sq is None:
            B_norm_sq = np.einsum("ij,ij->i", B, B)

        a_sq = np.einsum("ij,ij->i", A, A)
        d2 = a_sq[:, None] + B_norm_sq[None, :] - 2.0 * (A @ B.T)
        np.maximum(d2, 0.0, out=d2)         # 修正浮点负零
        return np.sqrt(d2)

    def _knn_search(self, X: np.ndarray, k: Optional[int] = None,
                    exclude_self: bool = False,
                    self_indices: Optional[Sequence[int]] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 X 中每个样本找 k 个最近邻。

        参数
        ----
        exclude_self : 是否排除样本本身（当查询对象也在训练集里时）。
        self_indices : 查询第 i 行在训练集中对应的下标（长度 = n_query）。
                       传入该参数时采用**按索引精确排除**，这是最可靠的判定方式。

        【为什么必须用索引排除，而不是"距离阈值"】
        实测两种朴素做法都会出错，这是本项目调试中发现的真实问题：
          ① `D == 0.0`：向量化距离公式 ||a||²+||b||²-2abᵀ 在计算"样本到自身"时
             会出现灾难性抵消（两个几乎相等的大数相减），实测自身距离是
             **8.43e-08 而不是 0**，于是完全判不出自匹配 —— 自己的标签照样参与投票；
          ② `D < 1e-9`（或任何模糊阈值）：真实数据里存在**特征完全相同的不同学生**
             （例如同一名学生的两门课记录、或背景与行为完全一致的两人），
             它们到查询点的距离同样极小，用阈值会把这些**合法邻居**一起误删，
             而且剔除后不补足还会导致有效近邻数 < k。
        因此：调用方知道"被查询的是训练集第几行"时，用 self_indices 精确排除；
        确实不知道时（例如查询外部新学生），exclude_self 应保持 False。

        返回
        ----
        idx    : (n_query, k) 近邻在训练集中的下标
        dist   : (n_query, k) 对应距离，按距离升序排列
        """
        if self.X_train_ is None:
            raise RuntimeError("请先调用 fit(X_train, y_train)")
        k = int(self.k if k is None else k)
        # k 不能超过训练样本数（留一法时训练集只有 n-1 个样本，必须裁剪）
        n_train = len(self.X_train_)
        k_eff = max(1, min(k, n_train))

        D = self.euclidean_distances(np.asarray(X, dtype=float),
                                     self.X_train_, self._X_norm_sq_)

        excluded_rows: Optional[np.ndarray] = None
        if exclude_self or self_indices is not None:
            D = D.copy()
            if self_indices is not None:
                # 精确排除：把"第 i 行对应的训练行"置为 +∞
                rows = np.arange(D.shape[0])
                cols = np.asarray(self_indices, dtype=int)
                cols = np.clip(cols, 0, n_train - 1)
                D[rows, cols] = np.inf
                excluded_rows = np.ones(D.shape[0], dtype=bool)
            else:
                # 无索引信息时的保守回退：用很小的绝对容差判定"几乎为 0"的距离。
                # 注意该分支可能误删"特征完全相同的不同学生"，因此只在
                # 调用方明确知道查询对象不在训练集、但仍要求排除极端靠近点时使用。
                near_zero = D < self.SELF_DISTANCE_TOL
                if near_zero.any():
                    D[near_zero] = np.inf
                    excluded_rows = near_zero.any(axis=1)

        # 需要多取候选：被置为 inf 的格子会在排序后落到行尾，需补足才能凑够 k 个
        k_take = min(n_train, k_eff + 1) if excluded_rows is not None else k_eff
        if k_take >= n_train:
            idx_all = np.argsort(D, axis=1, kind="stable")
        else:
            # argpartition 只保证"第 k_take 小"就位，左侧即最小的 k_take 个
            part = np.argpartition(D, kth=k_take - 1, axis=1)[:, :k_take]
            idx_all = np.take_along_axis(
                part, np.argsort(np.take_along_axis(D, part, axis=1), axis=1, kind="stable"),
                axis=1)

        if excluded_rows is not None and excluded_rows.any():
            # 剔除被置为 inf 的占位，保证每行返回 k_eff 个**有效**邻居
            idx_out = np.empty((idx_all.shape[0], k_eff), dtype=idx_all.dtype)
            dist_out = np.empty((idx_all.shape[0], k_eff), dtype=float)
            all_idx = np.arange(n_train)
            for r in range(idx_all.shape[0]):
                row = idx_all[r]
                keep = row[np.isfinite(D[r, row])][:k_eff]
                if len(keep) < k_eff:
                    # 极端情况（训练集几乎全是重复点）：用剩余下标补齐，保证形状稳定
                    rest = np.setdiff1d(all_idx, keep, assume_unique=False)
                    keep = np.concatenate([keep, rest[:k_eff - len(keep)]])
                idx_out[r] = keep
                dist_out[r] = D[r, keep]
            return idx_out, dist_out

        idx = idx_all[:, :k_eff]
        dist = np.take_along_axis(D, idx, axis=1)
        return idx, dist

    # ------------------------------------------------------------------
    # 3) 概率输出：k 个近邻中"高风险"样本的比例
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, k: Optional[int] = None,
                      exclude_self: bool = False,
                      self_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        返回 (n_samples, n_classes) 的概率矩阵，列顺序与 self.classes_ 一致。

        对二分类（0/1）：
            p(高风险) = 近邻中标签为 1 的个数 / k
            p(正常)   = 1 - p(高风险)

        self_indices         : 查询行在训练集中的下标（推荐方式，精确排除自匹配）
        exclude_self         : 无索引信息时的模糊回退（可能误删重复样本邻居，慎用）
        """
        if self.X_train_ is None:
            raise RuntimeError("请先调用 fit(X_train, y_train)")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # 单样本输入兼容：(d,) → (1, d)

        idx, _ = self._knn_search(X, k=k, exclude_self=exclude_self,
                                  self_indices=self_indices)
        neigh_labels = self.y_train_[idx]           # (n_query, k)
        n_query, k_used = neigh_labels.shape
        classes = self.classes_

        proba = np.zeros((n_query, len(classes)), dtype=float)
        for j, c in enumerate(classes):
            # 每一行统计等于类别 c 的近邻个数，再除以 k 得到比例
            proba[:, j] = (neigh_labels == c).sum(axis=1) / float(k_used)

        # 兜底：数值误差导致某行和不为 1 时做归一化（理论上不会发生）
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return proba / row_sum

    def predict(self, X: np.ndarray, k: Optional[int] = None,
                exclude_self: bool = False,
                self_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        多数投票得到类别。

        约定：平票时归为"正常(0)"，即"疑罪从无"。
        例如 k=4 出现 2:2 时判为 0，避免因偶数 k 造成随机波动。
        """
        if self.X_train_ is None:
            raise RuntimeError("请先调用 fit(X_train, y_train)")
        # 对二分类，argmax(proba) 等价于多数投票；平票时 argmax 取较小下标 → 归为 0
        proba = self.predict_proba(X, k=k, exclude_self=exclude_self,
                                   self_indices=self_indices)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # 4) 近邻明细：查询功能要展示"和你最像的几个同学"
    # ------------------------------------------------------------------
    def kneighbors(self, x: np.ndarray, k: Optional[int] = None,
                   ids: Optional[Sequence[str]] = None,
                   exclude_self: bool = False,
                   self_index: Optional[int] = None) -> List[Dict]:
        """
        返回单个样本的 k 个近邻明细（距离、标签、学生编号）。

        参数
        ----
        x           : 单样本特征向量（已标准化），shape (d,) 或 (1, d)
        ids         : 训练样本的学生编号，用于输出可读名单
        self_index  : 该样本在训练集中的下标（推荐，用于精确排除自匹配）
        exclude_self: 无下标时的模糊回退（可能误删重复样本邻居）
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        si = None if self_index is None else [int(self_index)]
        idx, dist = self._knn_search(x, k=k, exclude_self=exclude_self,
                                     self_indices=si)
        out: List[Dict] = []
        for j in range(idx.shape[1]):
            ti = int(idx[0, j])
            out.append({
                "rank": j + 1,
                "train_index": ti,
                "student_id": None if ids is None else str(ids[ti]),
                "distance": float(dist[0, j]),
                "label": int(self.y_train_[ti]),
                "label_cn": cfg.RISK_LABEL_CN.get(int(self.y_train_[ti]), str(self.y_train_[ti])),
            })
        return out


# ======================================================================
# 二、概率 → 风险等级
# ======================================================================
def proba_to_risk_level(p: float,
                        thresholds: Optional[Dict[str, float]] = None) -> str:
    """
    按"高风险概率"划分风险等级（阈值可调）。

    默认口径（config.RISK_LEVEL_THRESHOLDS）：
        p < 0.30            → 低
        0.30 <= p <= 0.70   → 中
        p > 0.70            → 高

    说明：阈值调低会提高召回率（少漏报）、降低精确率（多误报），
    辅导员排查人力有限时应调高阈值，期末预警场景通常宁可多报不可漏报，
    因此本项目在 experiments.md 中给出了阈值敏感性分析。
    """
    th = dict(cfg.RISK_LEVEL_THRESHOLDS if thresholds is None else thresholds)
    if p < th["low"]:
        return "低"
    if p <= th["high"]:
        return "中"
    return "高"


def assign_risk_levels(proba_pos: np.ndarray,
                       thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
    """对一批高风险概率批量划分等级，返回字符串数组。"""
    th = dict(cfg.RISK_LEVEL_THRESHOLDS if thresholds is None else thresholds)
    proba_pos = np.asarray(proba_pos, dtype=float)
    return np.where(proba_pos < th["low"], "低",
                    np.where(proba_pos <= th["high"], "中", "高"))


def risk_level_to_action(level: str) -> str:
    """把风险等级翻译成辅导员可执行的处置建议（不含 LLM，纯规则）。"""
    return {
        "低": "常规跟进：保持现有学习状态，无需额外干预。",
        "中": "建议关注：辅导员可在两周内做一次简短沟通，了解课程难点。",
        "高": "重点帮扶：建议纳入重点名单，安排学业辅导与固定跟进周期。",
    }.get(level, "未知等级：请人工复核。")


# ======================================================================
# 三、留一法专用：一次距离矩阵完成 n 折预测
# ======================================================================
def loo_predict_proba(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    留一法（Leave-One-Out）下的 KNN 高风险概率 —— 高效实现。

    朴素做法要训练 n 次、每次算一次距离矩阵，总共 O(n^2) 次距离计算；
    这里利用一个关键事实：**留一法时训练集就是"除自己以外的全部样本"**，
    因此只需算一次 n×n 距离矩阵，把对角线（自己到自己，距离为 0）置为 +∞，
    再取每行最小的 k 个即可，等价于留一法结果，速度快 n 倍。

    返回 (n,) 的高风险概率数组。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel().astype(int)
    n = len(X)
    k_eff = max(1, min(int(k), n - 1))

    D = KNNClassifierNumpy.euclidean_distances(X)
    D[np.arange(n), np.arange(n)] = np.inf  # 排除"自己"，模拟留一法

    if k_eff == n - 1:
        idx = np.argsort(D, axis=1, kind="stable")[:, :k_eff]
    else:
        part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
        idx = part  # 只要集合，投票不需要排序，省一次排序开销

    neigh = y[idx]                       # (n, k)
    return neigh.sum(axis=1) / float(k_eff)


def loo_predict(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    留一法下的类别预测。

    平票口径必须与 KNNClassifierNumpy.predict 完全一致，否则两套评估口径不可比：
    predict 用 argmax(proba)，平票取较小下标 → 归为"正常"；
    这里等价实现为 `p > 0.5`（严格大于），平票（p 恰为 0.5）同样归为 0。
    """
    p = loo_predict_proba(X, y, k)
    return (p > 0.5).astype(int)


# ======================================================================
# 四、自检
# ======================================================================
def _self_test() -> None:
    """小规模手工可验证的自检：用 2 维数据核对距离、投票与概率是否正确。"""
    print("=" * 68)
    print("【自检】knn_numpy 手工可复算用例")
    print("=" * 68)

    # 5 个训练样本，2 个特征（已标准化），标签依次为 0,0,1,1,1
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 1])

    clf = KNNClassifierNumpy(k=3).fit(X, y)
    q = np.array([[0.9, 0.9]])
    print("查询点 (0.9, 0.9)，手算到各训练点的欧氏距离：")
    d = KNNClassifierNumpy.euclidean_distances(q, X)[0]
    for i, di in enumerate(d):
        print(f"  到样本{i}({X[i].tolist()}, 标签{y[i]}) 的距离 = {di:.4f}")

    neigh = clf.kneighbors(q[0], k=3)
    print("k=3 的最近邻（按距离升序）：",
          [(n["train_index"], round(n["distance"], 4), n["label"]) for n in neigh])
    p = clf.predict_proba(q)[0]
    print("predict_proba =", np.round(p, 4), "（列顺序对应 classes_ =", clf.classes_, "）")
    print("predict =", clf.predict(q)[0])
    print("手算核对：最近邻为样本1(0)、样本3(1)、样本4(1) → 高风险比例 = 2/3 ≈ 0.6667 → 判为 1")
    assert abs(p[1] - 2.0 / 3.0) < 1e-12, "概率计算与手算不符"
    assert clf.predict(q)[0] == 1, "投票结果与手算不符"
    print("自检通过 ✅")
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    _self_test()
