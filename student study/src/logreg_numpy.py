# -*- coding: utf-8 -*-
"""
logreg_numpy.py —— 对照模型：纯 numpy 手写逻辑回归
=========================================================
为什么手写而不用 sklearn？
    本机环境实测 **未安装 scikit-learn**（import sklearn 报 ModuleNotFoundError），
    按实训约束"优先使用 sklearn；如果环境没有 sklearn，则用 numpy 手写逻辑回归"，
    因此对照模型用手写实现，保证项目在裸 numpy 环境下也能完整跑通。

如果答辩机器上装了 sklearn，程序会自动优先调用 sklearn 的 LogisticRegression
（见 train_eval.py 的 choose_logreg_backend），并把所用后端写进实验记录，
两种后端下指标口径完全一致，可直接对比。

模型形式（答辩要点）
    z = w·x + b
    p(高风险) = sigmoid(z) = 1 / (1 + e^(-z))
    损失函数 = 对数损失（交叉熵）+ L2 正则：
        L = -(1/n) Σ [ y·log(p) + (1-y)·log(1-p) ] + (λ/2)·||w||^2
    梯度：
        ∂L/∂w = (1/n)·X^T·(p - y) + λ·w
        ∂L/∂b = (1/n)·Σ(p - y)
    更新：
        w ← w - η·∂L/∂w
    与 KNN 的本质差别：逻辑回归是**参数模型**，训练阶段求出全局线性边界，
    预测阶段只做一次向量乘法，速度快但只能拟合线性可分结构；
    KNN 是**非参数模型**，训练不学习参数，预测时依赖局部近邻，能拟合非线性边界，
    但预测代价随样本量线性增长。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid：对 |z| 很大的值做分段处理，避免 exp 溢出。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionNumpy:
    """
    numpy 手写逻辑回归（二分类）。

    参数
    ----
    lr          : 学习率 η
    epochs      : 全量梯度下降迭代轮数
    l2          : L2 正则强度 λ
    batch_size  : 小批量大小；None 表示全批量（本项目数据量小，默认全批量）
    random_state: 便于可复现的实验（仅在启用小批量时用于打乱顺序）

    实现细节：使用"按参数自适应步长"的更新（RMSProp 风格），
    因为在特征已完成 z-score 标准化的情况下，纯固定步长对 λ 较敏感；
    自适应步长能让损失稳定下降，避免因学习率不当导致 exp 溢出或发散。
    """

    def __init__(self, lr: float = cfg.LOGREG_LR, epochs: int = cfg.LOGREG_EPOCHS,
                 l2: float = cfg.LOGREG_L2, batch_size: Optional[int] = None,
                 random_state: int = cfg.RANDOM_SEED) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.batch_size = batch_size
        self.random_state = int(random_state)

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_history_: List[float] = []
        self.classes_: np.ndarray = np.array([0, 1])

    # ------------------------------------------------------------------
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """当前参数下的对数损失（含 L2 项），用于观察是否收敛。"""
        p = sigmoid(X @ self.w_ + self.b_)
        eps = 1e-12
        data_loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * self.l2 * float(self.w_ @ self.w_)
        return float(data_loss + reg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNumpy":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(float)
        n, d = X.shape

        self.w_ = np.zeros(d, dtype=float)
        self.b_ = 0.0
        self.loss_history_ = []

        # 自适应步长的累积梯度平方（初始为 1，等价于初始步长 = lr）
        cache_w = np.ones(d, dtype=float)
        cache_b = 1.0
        eps = 1e-8
        rng = np.random.RandomState(self.random_state)

        bs = n if not self.batch_size else int(min(self.batch_size, n))
        for epoch in range(self.epochs):
            # 小批量时每轮打乱样本顺序；全批量时顺序无关
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                Xb, yb = X[idx], y[idx]
                m = len(idx)

                p = sigmoid(Xb @ self.w_ + self.b_)
                err = p - yb                       # 交叉熵对 z 的梯度
                gw = (Xb.T @ err) / m + self.l2 * self.w_
                gb = float(err.mean())

                # 自适应步长更新（除以历史梯度均方根）
                cache_w += gw * gw
                cache_b += gb * gb
                self.w_ -= self.lr * gw / np.sqrt(cache_w + eps)
                self.b_ -= self.lr * gb / np.sqrt(cache_b + eps)

            # 每轮记录一次全量损失，用于收敛曲线与"是否真的训练成功"的验证
            if epoch % max(1, self.epochs // 60) == 0 or epoch == self.epochs - 1:
                self.loss_history_.append(self._loss(X, y))
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """返回线性得分 z = w·x + b（未过 sigmoid 的原始输出）。"""
        if self.w_ is None:
            raise RuntimeError("请先调用 fit(X, y)")
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (n, 2) 概率矩阵，列为 [p(正常), p(高风险)]。"""
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """按阈值输出类别。默认 0.5；调低阈值可提高召回率（少漏报）。"""
        return (sigmoid(self.decision_function(X)) >= threshold).astype(int)

    # ------------------------------------------------------------------
    def feature_contributions(self, x: np.ndarray) -> List[Tuple[str, float]]:
        """
        线性模型的天然可解释性：每个特征的贡献 = w_i · x_i。
        正贡献表示"把该学生推向高风险"，负贡献表示"拉向正常"。
        返回按贡献绝对值降序的 (特征名, 贡献值) 列表。
        """
        if self.w_ is None:
            raise RuntimeError("请先调用 fit(X, y)")
        x = np.asarray(x, dtype=float).ravel()
        contrib = self.w_ * x
        pairs = list(zip(cfg.FEATURE_COLUMNS, contrib.tolist()))
        return sorted(pairs, key=lambda t: abs(t[1]), reverse=True)

    def params_dict(self) -> Dict:
        """导出可读参数，写进实验记录用于答辩说明。"""
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "l2": self.l2,
            "batch_size": self.batch_size,
            "n_loss_records": len(self.loss_history_),
            "final_loss": None if not self.loss_history_ else round(self.loss_history_[-1], 6),
            "intercept_b": round(float(self.b_), 4),
            "coef": None if self.w_ is None else {k: round(float(v), 4)
                                                  for k, v in zip(cfg.FEATURE_COLUMNS, self.w_)},
        }


def _self_test() -> None:
    """自检：在一个线性可分的小数据上，逻辑回归应能 100% 分开，损失应单调下降到很小。"""
    print("=" * 68)
    print("【自检】logreg_numpy 小数据收敛性检查")
    print("=" * 68)
    rng = np.random.RandomState(0)
    X = np.r_[rng.normal(-1.5, 0.5, size=(30, 2)), rng.normal(1.5, 0.5, size=(30, 2))]
    y = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]

    clf = LogisticRegressionNumpy(lr=0.2, epochs=600, l2=1e-3).fit(X, y)
    acc = float((clf.predict(X) == y).mean())
    print(f"训练集准确率 = {acc:.4f}（线性可分数据应接近 1.0）")
    print(f"损失：首次 {clf.loss_history_[0]:.6f} → 末次 {clf.loss_history_[-1]:.6f}")
    print("参数：", clf.params_dict()["coef"], "b =", clf.params_dict()["intercept_b"])
    assert clf.loss_history_[-1] < clf.loss_history_[0], "损失未下降，训练存在问题"
    assert acc > 0.95, "线性可分数据上准确率异常偏低"
    print("自检通过 ✅")
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    _self_test()
