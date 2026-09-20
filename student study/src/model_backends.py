# -*- coding: utf-8 -*-
"""
model_backends.py —— 模型后端抽象层（sklearn 主力 / 手写附录）
=========================================================
【为什么需要这一层】
本项目 v2 要求：
    · **主模型由 sklearn 承担**（KNN + 逻辑回归），这是核心功能；
    · **手写 KNN 保留为附录亮点**（证明自己实现了算法，并做等价性验证），
      但不再承担核心功能。
如果全项目直接用 `sklearn.neighbors.KNeighborsClassifier`，手写实现就无法
在同一套评估代码里做对照；如果到处都是 `if sklearn else 手写` 分支，
评估逻辑会立刻分叉、指标口径不再可比。

因此这里做一个极薄的适配层：**统一接口 + 可切换后端**。
    统一接口：fit / predict / predict_proba / decision_function / get_params / set_params
    可切换后端：`type='knn', backend='sklearn' | 'numpy'`，
                以及 `type='logreg', backend='sklearn' | 'numpy'`
这样"10 次随机划分""留一组交叉验证""置换重要性"等评估函数只写一份，
两套实现（sklearn 与手写）自动获得**同口径**指标，可比性由构造保证。

【诚实性保障】`describe_backend()` 会报告当前**实际使用**的后端；
若环境缺 sklearn 自动回退手写实现，程序会明确打印出来，
不会出现"声称用了 sklearn 其实没用"的情况。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from . import config as cfg
    from .knn_numpy import KNNClassifierNumpy
    from .logreg_numpy import LogisticRegressionNumpy
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    from knn_numpy import KNNClassifierNumpy
    from logreg_numpy import LogisticRegressionNumpy


# ======================================================================
# 一、sklearn 可用性探测
# ======================================================================
def sklearn_available() -> bool:
    """探测 scikit-learn 是否可用（不抛异常）。"""
    try:
        import sklearn  # noqa: F401
        from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        return True
    except Exception:
        return False


def sklearn_version() -> Optional[str]:
    """返回 sklearn 版本号，未安装则返回 None。"""
    try:
        import sklearn
        return getattr(sklearn, "__version__", "unknown")
    except Exception:
        return None


def resolve_backend(requested: str = "auto") -> str:
    """
    决定实际使用的后端。

    requested: 'auto'（默认，有 sklearn 就用）/ 'sklearn' / 'numpy'
    · 'auto'    → sklearn 可用则用 sklearn，否则回退 numpy 手写（并打印提示）
    · 'sklearn' → 强制 sklearn；不可用时报错（避免静默降级）
    · 'numpy'   → 强制手写实现（用于附录对照实验）
    """
    ok = sklearn_available()
    if requested == "auto":
        return "sklearn" if ok else "numpy"
    if requested == "sklearn" and not ok:
        raise RuntimeError(
            "指定 backend='sklearn' 但当前环境不可用。请执行：python -m pip install scikit-learn"
        )
    if requested == "numpy" and ok:
        return "numpy"
    return requested


def describe_backend(kind: str, backend: str) -> str:
    """生成一行可读的后端说明（写进实验记录）。"""
    if backend == "sklearn":
        return f"{kind}: sklearn 实现（scikit-learn {sklearn_version()}）—— 主模型承担核心功能"
    return f"{kind}: numpy 手写实现 —— 因环境缺少 scikit-learn 而自动回退（或按参数强制指定）"


# ======================================================================
# 二、统一接口的模型包装器
# ======================================================================
class ModelAdapter:
    """
    统一 sklearn 与手写实现的方法签名。

    接口：
        fit(X, y) -> self
        predict(X) -> ndarray[int]
        predict_proba(X) -> ndarray (n, 2)，第 1 列为高风险概率
        decision_function(X) -> ndarray（仅逻辑回归有）
        get_feature_importance() -> (names, values)（KNN 返回 None）
    """

    def __init__(self, kind: str, backend: str = "auto", **kwargs: Any) -> None:
        assert kind in ("knn", "logreg"), f"未知模型类型 {kind}"
        self.kind = kind
        self.requested_backend = backend
        self.backend = resolve_backend(backend)
        self.kwargs = dict(kwargs)
        self.model = self._build()
        # 记录预测阶段是否需要 exclude_self（仅手写实现支持该参数）
        self._supports_exclude_self = isinstance(self.model, KNNClassifierNumpy)

    # ------------------------------------------------------------------
    def _build(self):
        """按后端实例化具体模型对象。"""
        if self.kind == "knn":
            if self.backend == "sklearn":
                from sklearn.neighbors import KNeighborsClassifier
                params = dict(cfg.SKLEARN_KNN_PARAMS)
                params.update(self.kwargs)
                # 参数名统一：本项目内部用 k，sklearn 用 n_neighbors
                if "k" in params:
                    params["n_neighbors"] = params.pop("k")
                self.k = int(params["n_neighbors"])
                return KNeighborsClassifier(**params)
            else:
                k = int(self.kwargs.get("k", self.kwargs.get("n_neighbors", cfg.MAIN_K)))
                self.k = k
                return KNNClassifierNumpy(k=k)

        # ---- 逻辑回归 ----
        if self.backend == "sklearn":
            from sklearn.linear_model import LogisticRegression
            params = dict(cfg.SKLEARN_LOGREG_PARAMS)
            params.update(self.kwargs)
            return LogisticRegression(**params)
        return LogisticRegressionNumpy(
            lr=float(self.kwargs.get("lr", cfg.LOGREG_LR)),
            epochs=int(self.kwargs.get("epochs", cfg.LOGREG_EPOCHS)),
            l2=float(self.kwargs.get("l2", cfg.LOGREG_L2)),
        )

    # ------------------------------------------------------------------
    @property
    def n_neighbors(self) -> Optional[int]:
        """KNN 的 k 值（非 KNN 时返回 None）。"""
        if self.kind != "knn":
            return None
        if self.backend == "sklearn":
            return int(self.model.n_neighbors)
        return int(self.model.k)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelAdapter":
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y).ravel())
        return self

    def predict(self, X: np.ndarray, exclude_self: bool = False,
                self_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        预测类别。

        self_indices / exclude_self 仅对手写 KNN 生效：查询数据集中已有学生时，
        需要排除"自己"这个自身匹配（否则等于用答案喂自己）。
        推荐用 self_indices（按训练集下标精确排除，不会误删"特征完全相同的其他学生"）；
        exclude_self 是无下标时的模糊回退。
        sklearn 的 KNN 不提供该开关，因此 sklearn 后端要求查询集与训练集无重叠
        （留一组交叉验证、随机划分天然满足）。
        """
        X = np.asarray(X, dtype=float)
        if self._supports_exclude_self and (self_indices is not None or exclude_self):
            return self.model.predict(X, exclude_self=exclude_self,
                                      self_indices=self_indices)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, exclude_self: bool = False,
                      self_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """返回 (n, 2) 概率矩阵；第 1 列为高风险概率。"""
        X = np.asarray(X, dtype=float)
        if self._supports_exclude_self and (self_indices is not None or exclude_self):
            return np.asarray(self.model.predict_proba(X, exclude_self=exclude_self,
                                                       self_indices=self_indices),
                              dtype=float)
        return np.asarray(self.model.predict_proba(X), dtype=float)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """线性得分（仅逻辑回归）。KNN 不支持时抛出明确异常。"""
        if self.kind != "logreg":
            raise AttributeError("KNN 不支持 decision_function")
        return np.asarray(self.model.decision_function(np.asarray(X, dtype=float)))

    # ------------------------------------------------------------------
    def coefficients(self) -> Optional[np.ndarray]:
        """逻辑回归系数（按训练特征顺序）；KNN 返回 None。"""
        if self.kind != "logreg":
            return None
        if self.backend == "sklearn":
            return np.asarray(self.model.coef_, dtype=float).ravel()
        return np.asarray(self.model.w_, dtype=float)

    def intercept(self) -> Optional[float]:
        """逻辑回归截距；KNN 返回 None。"""
        if self.kind != "logreg":
            return None
        if self.backend == "sklearn":
            return float(np.asarray(self.model.intercept_).ravel()[0])
        return float(self.model.b_)

    def feature_importance(self) -> Optional[np.ndarray]:
        """仅逻辑回归可用：|系数| 作为重要度。KNN 返回 None（需用置换重要性）。"""
        coef = self.coefficients()
        return None if coef is None else np.abs(coef)

    # ------------------------------------------------------------------
    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None,
                   exclude_self: bool = False,
                   self_indices: Optional[Sequence[int]] = None):
        """
        返回近邻明细 (distances, indices)。

        手写实现返回的是字典列表，这里统一成与 sklearn 相同的 (dist, idx) 二元组，
        便于查询模块无差别调用。
        """
        X2 = np.asarray(X, dtype=float)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        if self._supports_exclude_self:
            nq = len(X2)
            si = None if self_indices is None else list(map(int, self_indices))
            dist_rows, idx_rows = [], []
            for i in range(nq):
                one = None if si is None else [si[i]]
                nbs = self.model.kneighbors(X2[i], k=n_neighbors,
                                            exclude_self=exclude_self, self_index=None
                                            if one is None else one[0])
                dist_rows.append([n["distance"] for n in nbs])
                idx_rows.append([n["train_index"] for n in nbs])
            return np.array(dist_rows, dtype=float), np.array(idx_rows, dtype=int)
        return self.model.kneighbors(X2, n_neighbors=n_neighbors)

    # ------------------------------------------------------------------
    def backend_info(self) -> Dict[str, Any]:
        """返回后端与超参数信息，写进实验记录用于答辩说明。"""
        info: Dict[str, Any] = {
            "kind": self.kind,
            "backend_requested": self.requested_backend,
            "backend_used": self.backend,
            "description": describe_backend(self.kind, self.backend),
        }
        if self.kind == "knn":
            info["k"] = self.n_neighbors
            info["weights"] = ("uniform（与手写多数投票口径一致）"
                               if self.backend == "sklearn" else "uniform（多数投票）")
            info["metric"] = "euclidean"
        else:
            coef = self.coefficients()
            info["coef"] = None if coef is None else [round(float(v), 4) for v in coef]
            info["intercept"] = None if self.intercept() is None else round(self.intercept(), 4)
            if self.backend == "sklearn":
                info["C"] = float(getattr(self.model, "C", 1.0))
                info["solver"] = getattr(self.model, "solver", None)
                info["class_weight"] = getattr(self.model, "class_weight", None)
                info["final_loss"] = None  # sklearn 不暴露损失曲线
            else:
                info["lr"] = self.model.lr
                info["epochs"] = self.model.epochs
                info["l2"] = self.model.l2
                info["final_loss"] = (round(float(self.model.loss_history_[-1]), 6)
                                      if getattr(self.model, "loss_history_", None) else None)
        return info


# ======================================================================
# 三、便捷工厂
# ======================================================================
def make_knn(k: int, backend: str = "auto", **kwargs) -> ModelAdapter:
    """构造 KNN（主模型）。"""
    return ModelAdapter("knn", backend=backend, k=int(k), **kwargs)


def make_logreg(backend: str = "auto", **kwargs) -> ModelAdapter:
    """构造逻辑回归（对照模型）。"""
    return ModelAdapter("logreg", backend=backend, **kwargs)


def make_model(kind: str, backend: str = "auto", **kwargs) -> ModelAdapter:
    """按类型构造模型，供评估流水线统一调用。"""
    return ModelAdapter(kind, backend=backend, **kwargs)


def main() -> None:
    """自检：报告后端可用性，并在小数据上验证两套后端都能正常 fit/predict。"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    print("=" * 68)
    print("【自检】模型后端抽象层")
    print("=" * 68)
    print(f"sklearn 可用 : {sklearn_available()}（版本 {sklearn_version()}）")
    print(f"resolve_backend('auto') = {resolve_backend('auto')}")
    print()

    rng = np.random.RandomState(0)
    X = np.r_[rng.normal(-1.5, 0.6, size=(50, 3)), rng.normal(1.5, 0.6, size=(50, 3))]
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]

    for backend in ["sklearn", "numpy"]:
        if backend == "sklearn" and not sklearn_available():
            print("[跳过] sklearn 不可用，无法测试该后端")
            continue
        knn = make_knn(k=5, backend=backend).fit(X, y)
        lr = make_logreg(backend=backend).fit(X, y)
        acc_k = float((knn.predict(X) == y).mean())
        acc_l = float((lr.predict(X) == y).mean())
        print(f"backend={backend:8s} KNN acc={acc_k:.4f}  LogReg acc={acc_l:.4f}  "
              f"KNN k={knn.n_neighbors}")
        print(f"          {knn.backend_info()['description']}")
    print("=" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
