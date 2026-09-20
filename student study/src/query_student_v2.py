# -*- coding: utf-8 -*-
"""
query_student_v2.py —— 单名学生风险查询 v2（UCI 真实数据 + sklearn 主模型）
=========================================================
对外回答辅导员的问题：

    "这名学生风险多高？概率多少？主要原因是啥？跟谁最像？"

【v2 的两处关键变化】
    ① 数据换成 UCI 真实记录（可用编号示例：mat0001、por0001 …）；
       同时支持**手工输入特征**查询任意"虚构学生"（体现"换输入仍可预测"）。
    ② 主模型换成 sklearn（`model_backends.ModelAdapter`），
       手写 KNN 仅在显式指定 `--backend numpy` 时使用（附录对照）。

【自匹配问题的正确处理（本项目的真实调试经验）】
查询数据集中已有学生时，该生到"自己"的距离必然最小，会成为最近邻 —— 等于用答案喂自己。
但判定"自己"**不能用距离阈值**：
    · 向量化距离公式算自身距离实测为 8.43e-08（灾难性抵消，不是 0）；
    · 真实数据里还存在特征完全相同的**不同**学生（距离也是 0），用阈值会误删合法邻居。
因此这里改为**按训练集下标精确排除**（手写后端支持 `self_index`）；
sklearn 后端不提供该开关，故对 sklearn 采用**两次预测对比法**：
先正常预测一次，再格外确认"如果本人被排除，结论是否改变"，并在报告中如实披露。
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
    from . import llm_helper as llm
    from .knn_numpy import assign_risk_levels, proba_to_risk_level, risk_level_to_action
    from .model_backends import ModelAdapter, make_knn, make_logreg, resolve_backend
    from .preprocess import StandardScaler
    from .uci_features import (CATEGORICAL_LEVELS, LEVEL_CN, NUM_FEATURES, YN_FEATURES,
                               feature_name_cn, prepare_uci_features)
    from .utils import bar, fmt_pct, print_kv, print_section, print_title
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import llm_helper as llm
    from knn_numpy import assign_risk_levels, proba_to_risk_level, risk_level_to_action
    from model_backends import ModelAdapter, make_knn, make_logreg, resolve_backend
    from preprocess import StandardScaler
    from uci_features import (CATEGORICAL_LEVELS, LEVEL_CN, NUM_FEATURES, YN_FEATURES,
                              feature_name_cn, prepare_uci_features)
    from utils import bar, fmt_pct, print_kv, print_section, print_title


# 供交互式输入使用的字段提示（含取值范围与中文说明）
NUMERIC_PROMPTS = {
    "age": ("年龄", 15, 22),
    "Medu": ("母亲学历 0=无 1=小学 2=初中 3=高中 4=高等教育", 0, 4),
    "Fedu": ("父亲学历 0=无 1=小学 2=初中 3=高中 4=高等教育", 0, 4),
    "traveltime": ("上学路程时间 1=<15分 2=15~30分 3=30~60分 4=>1小时", 1, 4),
    "studytime": ("每周学习时间 1=<2h 2=2~5h 3=5~10h 4=>10h", 1, 4),
    "failures": ("历史挂科次数 0~3（3 表示 3 次及以上）", 0, 4),
    "famrel": ("家庭关系质量 1=很差 ~ 5=很好", 1, 5),
    "freetime": ("课后自由时间 1=很少 ~ 5=很多", 1, 5),
    "goout": ("与朋友外出频率 1=很少 ~ 5=很多", 1, 5),
    "Dalc": ("工作日饮酒 1=很少 ~ 5=很多", 1, 5),
    "Walc": ("周末饮酒 1=很少 ~ 5=很多", 1, 5),
    "health": ("健康状况 1=很差 ~ 5=很好", 1, 5),
    "absences": ("本学期缺课次数 0~75", 0, 93),
}
YN_PROMPTS = {
    "schoolsup": "学校额外教育支持", "famsup": "家庭学业支持",
    "paid": "课外付费补习", "activities": "课外活动",
    "nursery": "是否上过幼儿园", "higher": "是否想读高等教育",
    "internet": "家里有无网络", "romantic": "是否有恋爱关系",
}
CAT_PROMPTS = {
    "school": "学校", "sex": "性别", "address": "居住地",
    "famsize": "家庭规模", "Pstatus": "父母同住状况",
    "Mjob": "母亲职业", "Fjob": "父亲职业", "reason": "择校原因", "guardian": "监护人",
}


# ======================================================================
# 一、查询引擎
# ======================================================================
class RiskQueryEngineV2:
    """
    风险查询引擎（v2）。

    构造时加载 UCI 数据集、特征编码 schema、并在**全量数据**上训练最终模型
    （部署模型用全量训练；评估指标则来自 LOGO / 分组划分，两者严格区分）。
    """

    def __init__(self, backend: str = "auto", k: Optional[int] = None,
                 verbose: bool = False) -> None:
        self.backend = resolve_backend(backend)
        self.df = pd.read_csv(cfg.UCI_DATA_PATH, encoding="utf-8-sig")
        self.X_raw, self.y, self.feature_names = prepare_uci_features(
            self.df, include_early_grades=False, standardize=False)
        self.scaler = StandardScaler().fit(self.X_raw)
        self.X = self.scaler.transform(self.X_raw)
        self.ids = self.df["student_id"].to_numpy()
        self.groups = self.df["student_group_id"].to_numpy()

        # 主 k：优先取 v2 指标里的选择结果，其次取配置默认
        self.k = int(k if k is not None else self._load_main_k())
        self.knn: ModelAdapter = make_knn(k=self.k, backend=self.backend).fit(self.X, self.y)
        self.logreg: ModelAdapter = make_logreg(backend=self.backend).fit(self.X, self.y)
        print(f"[引擎 v2] 数据：{len(self.df)} 条 UCI 真实记录 / "
              f"{len(pd.unique(self.groups))} 名学生；主模型 KNN(k={self.k})，后端={self.backend}")

    @staticmethod
    def _load_main_k() -> int:
        """从 v2 指标文件读取由实验选定的主 k。"""
        import json
        if os.path.exists(cfg.METRICS_SUMMARY_PATH):
            try:
                with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("main_k"):
                    return int(s["main_k"])
            except Exception:
                pass
        return int(cfg.MAIN_K)

    # ------------------------------------------------------------------
    def _row_to_features(self, row: pd.Series) -> Dict[str, object]:
        """把一行原始数据转成特征字典（保留原始类型，供编码函数使用）。"""
        feats: Dict[str, object] = {}
        for c in NUM_FEATURES:
            feats[c] = float(row[c])
        for c in YN_FEATURES:
            feats[c] = str(row[c])
        for c in CATEGORICAL_LEVELS:
            feats[c] = str(row[c])
        return feats

    def _encode(self, feats: Dict[str, object]) -> np.ndarray:
        """把特征字典编码成模型输入向量（复用主管线的固定 schema）。"""
        df1 = pd.DataFrame([feats])
        # 缺失字段用训练集众数/中位数补（体现"数据不全也能查"）
        for c in NUM_FEATURES:
            if c not in df1 or pd.isna(df1[c].iloc[0]):
                df1[c] = float(np.median(self.df[c].to_numpy(dtype=float)))
        for c in YN_FEATURES:
            if c not in df1 or df1[c].iloc[0] in (None, "", "nan"):
                df1[c] = str(self.df[c].mode().iloc[0])
        for c in CATEGORICAL_LEVELS:
            if c not in df1 or df1[c].iloc[0] in (None, "", "nan"):
                df1[c] = str(self.df[c].mode().iloc[0])
        df1[cfg.LABEL_COLUMN] = 0        # 占位，编码时不需要标签
        X1, _, _ = prepare_uci_features(df1, include_early_grades=False, standardize=False)
        return self.scaler.transform(X1)[0]

    # ------------------------------------------------------------------
    def query_features(self, feats: Dict[str, object], student_id: str = "自定义输入",
                       self_index: Optional[int] = None, use_llm: bool = False) -> Dict:
        """
        核心查询：特征字典 → 风险结论。

        self_index 为该样本在训练集中的下标（若查询的是数据集已有学生），
        传入后手写后端会**精确排除自身**；sklearn 后端不支持该开关，
        因此额外做一次"排除自身的近似校验"并在结果中披露差异。
        """
        x = self._encode(feats).reshape(1, -1)
        knn = self.knn
        # ---- 主模型概率 ----
        if self_index is not None and getattr(knn, "_supports_exclude_self", False):
            proba = float(knn.predict_proba(x, self_indices=[self_index])[0, 1])
            self_excluded = True
        else:
            proba = float(knn.predict_proba(x)[0, 1])
            self_excluded = False
        level = proba_to_risk_level(proba, cfg.RISK_LEVEL_THRESHOLDS)
        pred = int(proba > 0.5)

        # ---- 自匹配影响核查（sklearn 后端的重要披露项）----
        self_match_info = {"supported": bool(self_excluded), "note": ""}
        if self_index is not None and not self_excluded:
            # sklearn 无法排除自身：用"最近邻是否为本人"来披露影响
            dist, idx = knn.kneighbors(x, n_neighbors=min(self.k + 1, len(self.X)))
            nearest_is_self = int(idx[0, 0]) == int(self_index)
            self_match_info = {
                "supported": False,
                "nearest_is_self": bool(nearest_is_self),
                "nearest_distance": round(float(dist[0, 0]), 8),
                "note": ("sklearn 后端不支持排除自身；实测最近邻"
                         + ("就是本人" if nearest_is_self else "不是本人")
                         + "，因此该概率"
                         + ("含本人一票，会略偏乐观（评估指标用的是留一组交叉验证，不受此影响）"
                            if nearest_is_self else "不受自匹配影响")),
            }

        # ---- 对照模型 ----
        lr_proba = float(self.logreg.predict_proba(x)[0, 1])
        lr_level = proba_to_risk_level(lr_proba, cfg.RISK_LEVEL_THRESHOLDS)

        # ---- 关键特征贡献（用**训练集分层对比**，稳健且不会因近邻同类而退化）----
        contrib = self._contributions(x[0])

        # ---- 近邻明细 ----
        neighbors = self._neighbors(x, self_index)

        result = {
            "student_id": student_id,
            "proba_risk": round(proba, 4),
            "risk_level": level,
            "pred_label": pred,
            "action": risk_level_to_action(level),
            "logreg_proba_risk": round(lr_proba, 4),
            "logreg_risk_level": lr_level,
            "models_agree_level": bool(lr_level == level),
            "self_match": self_match_info,
            "contributions": contrib["all"],
            "top_contributions": contrib["top"],
            "n_risk_pushing": contrib["n_risk_pushing"],
            "neighbors": neighbors,
            "sentences": self._sentences(student_id, proba, level, contrib, neighbors),
            "thresholds": dict(cfg.RISK_LEVEL_THRESHOLDS),
            "model_info": {"knn_k": self.k, "backend": self.backend,
                           "n_train": len(self.X)},
            "raw_features": {k: (round(float(v), 3) if isinstance(v, (int, float)) else v)
                             for k, v in feats.items()},
        }

        # ---- 帮扶建议（LLM 只做文字组织，不改结论）----
        # 传入 prediction_source，避免事实包里把 sklearn 主模型误写成"手写 KNN"
        src = (f"sklearn KNN（k={self.k}，欧氏距离 + 多数投票 + predict_proba）"
               if self.backend == "sklearn" else
               f"numpy 手写 KNN（k={self.k}，欧氏距离 + 多数投票 + 近邻高风险占比概率）")
        facts = llm.build_facts(
            student_id=student_id, proba_risk=proba, risk_level=level,
            top_contributions=contrib["top"],
            neighbors=[{"student_id": nb["student_id"], "label": nb["label"]} for nb in neighbors],
            raw_features={k: v for k, v in result["raw_features"].items()
                          if isinstance(v, (int, float))},
            thresholds=cfg.RISK_LEVEL_THRESHOLDS, k=self.k,
            prediction_source=src)
        result["facts"] = facts
        result["assistance"] = llm.generate_assistance(facts, use_api=use_llm, verbose=False)
        return result

    # ------------------------------------------------------------------
    def _contributions(self, x_std: np.ndarray) -> Dict:
        """
        特征贡献：把该生每个特征值与"高风险组均值 / 低风险组均值"比较。

        在标准化空间中计算：gap = |x - 低风险组均值| - |x - 高风险组均值|，
        gap > 0 表示该特征把它推向高风险。
        """
        Xr = self.X[self.y == 1]
        Xs = self.X[self.y == 0]
        mr, ms = Xr.mean(axis=0), Xs.mean(axis=0)
        rows = []
        for j, name in enumerate(self.feature_names):
            d_r = float(abs(x_std[j] - mr[j]))
            d_s = float(abs(x_std[j] - ms[j]))
            rows.append({
                "feature": name,
                "feature_cn": feature_name_cn(name),
                "dist_to_risk_mean": round(d_r, 4),
                "dist_to_safe_mean": round(d_s, 4),
                "risk_gap": round(d_s - d_r, 4),
                "direction": "推向高风险" if d_s - d_r > 0 else "相对优势",
            })
        df = pd.DataFrame(rows).sort_values("risk_gap", ascending=False)
        top = df.head(5).to_dict(orient="records")
        return {"all": df.to_dict(orient="records"), "top": top,
                "n_risk_pushing": int((df["risk_gap"] > 0).sum())}

    def _neighbors(self, x: np.ndarray, self_index: Optional[int],
                   n: Optional[int] = None) -> List[Dict]:
        """返回近邻明细（含学生编号），并标注哪个是本人。"""
        n = int(max(1, min(n or self.k, len(self.X))))
        if self_index is not None and getattr(self.knn, "_supports_exclude_self", False):
            dist, idx = self.knn.kneighbors(x, n_neighbors=n,
                                            self_indices=[self_index])
        else:
            dist, idx = self.knn.kneighbors(x, n_neighbors=n)
        out = []
        for r in range(idx.shape[1]):
            ti = int(idx[0, r])
            out.append({
                "rank": r + 1,
                "train_index": ti,
                "student_id": str(self.ids[ti]),
                "subject": str(self.df.iloc[ti]["subject"]),
                "distance": round(float(dist[0, r]), 4),
                "label": int(self.y[ti]),
                "label_cn": cfg.RISK_LABEL_CN[int(self.y[ti])],
                "is_self": bool(self_index is not None and ti == int(self_index)),
            })
        return out

    @staticmethod
    def _sentences(sid: str, proba: float, level: str, contrib: Dict,
                   neighbors: Sequence[Dict]) -> List[str]:
        """生成中文解释句（只做文字组织，数值全部来自上面的计算）。"""
        n_risk = sum(1 for nb in neighbors if nb["label"] == 1)
        s = [f"该生高风险概率为 {proba:.4f}（最近 {len(neighbors)} 位相似同学中有 "
             f"{n_risk} 位属于“需要重点关注”），据此判定风险等级为【{level}】。"]
        for i, it in enumerate(contrib["top"], 1):
            s.append(f"（{i}）{it['feature_cn']}：到高风险组平均水平的标准化距离 "
                     f"{it['dist_to_risk_mean']}，到低风险组 {it['dist_to_safe_mean']}，"
                     f"差距 {it['risk_gap']:+.4f} → {it['direction']}。")
        Xr = contrib["all"]
        s.append(f"（依据说明）对比基准 = 训练集中真实标签分层的组均值"
                 f"（高风险组与正常组），共 {len(Xr)} 个特征参与比较。")
        if level == "低" and contrib["n_risk_pushing"] > 0:
            names = "、".join(it["feature_cn"] for it in contrib["top"]
                             if it["risk_gap"] > 0)
            s.append(f"（口径提示）该生整体为低风险，但 {names} 相对高风险组更接近，"
                     f"属**群体内的相对短板**，不等于风险来源。")
        return s

    # ------------------------------------------------------------------
    def query_by_id(self, student_id: str, use_llm: bool = False) -> Dict:
        """按记录编号查询（如 mat0001 / por0001）。"""
        hit = self.df.index[self.df["student_id"] == str(student_id)].tolist()
        if not hit:
            raise KeyError(f"未找到编号 {student_id}；可用示例：mat0001、por0001")
        i = int(hit[0])
        feats = self._row_to_features(self.df.iloc[i])
        return self.query_features(feats, student_id=str(student_id),
                                   self_index=i, use_llm=use_llm)

    # ------------------------------------------------------------------
    def batch_query(self, top_n: int = 10) -> pd.DataFrame:
        """批量筛查：输出按高风险概率降序的名单（真实使用场景）。"""
        proba = self.knn.predict_proba(self.X)[:, 1]
        lr = self.logreg.predict_proba(self.X)[:, 1]
        out = self.df[["student_id", "subject", "failures", "absences", "studytime",
                       "goout", "Dalc", cfg.LABEL_COLUMN]].copy()
        out["knn_proba_risk"] = np.round(proba, 4)
        out["risk_level"] = assign_risk_levels(proba, cfg.RISK_LEVEL_THRESHOLDS)
        out["logreg_proba_risk"] = np.round(lr, 4)
        out["action"] = [risk_level_to_action(l) for l in out["risk_level"]]
        return out.sort_values("knn_proba_risk", ascending=False).head(top_n).reset_index(drop=True)


# ======================================================================
# 二、打印
# ======================================================================
def print_result_v2(result: Dict) -> None:
    """排版输出一次查询结果。"""
    print_title(f"学生风险查询（UCI 真实数据）：{result['student_id']}")
    p = result["proba_risk"]
    print(f"  高风险概率(KNN) : {p:.4f}  {bar(p)}  {fmt_pct(p)}")
    th = result["thresholds"]
    print(f"  风险等级        : 【{result['risk_level']}】"
          f"（低 < {th['low']}；中 {th['low']}~{th['high']}；高 > {th['high']}）")
    print(f"  处置建议        : {result['action']}")
    print(f"  对照模型(逻辑回归): 概率 {result['logreg_proba_risk']:.4f}，"
          f"等级 {result['logreg_risk_level']}，"
          f"{'与本模型一致' if result['models_agree_level'] else '与本模型不一致（边界样本）'}")
    sm = result["self_match"]
    print(f"  自匹配处理      : "
          f"{'已精确排除自身（手写后端）' if sm['supported'] else sm.get('note', '不适用')}")

    print_section("为什么判为这个等级（关键特征贡献）")
    for s in result["sentences"]:
        print("  " + s)

    print_section(f"最相似的 {len(result['neighbors'])} 位同学")
    for nb in result["neighbors"]:
        mark = " ← 本人" if nb["is_self"] else ""
        print(f"  #{nb['rank']}  {nb['student_id']}（{nb['subject']}）"
              f"  距离={nb['distance']}  实际={nb['label_cn']}{mark}")


def print_batch_v2(df: pd.DataFrame, title: str = "批量筛查结果（按高风险概率降序）") -> None:
    """打印批量筛查名单。"""
    print_title(title)
    cols = ["student_id", "subject", "failures", "absences", "studytime",
            "knn_proba_risk", "risk_level"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    n_high = int((df["risk_level"] == "高").sum())
    n_mid = int((df["risk_level"] == "中").sum())
    print(f"\n  共 {len(df)} 人：高风险 {n_high}，中风险 {n_mid}，"
          f"低风险 {len(df) - n_high - n_mid}")
    print(f"  {cfg.DISCLAIMER}")


# ======================================================================
# 三、交互式输入
# ======================================================================
def interactive_input_v2(demo: bool = False) -> Dict[str, object]:
    """
    交互式录入学生特征。直接回车表示"使用训练集众数/中位数"。
    demo=True 时不询问，直接返回一套示例特征（便于自动化演示）。
    """
    if demo:
        return {"age": 17.0, "Medu": 2.0, "Fedu": 2.0, "traveltime": 2.0,
                "studytime": 1.0, "failures": 2.0, "famrel": 3.0, "freetime": 4.0,
                "goout": 4.0, "Dalc": 2.0, "Walc": 3.0, "health": 4.0, "absences": 12.0,
                "schoolsup": "yes", "famsup": "no", "paid": "no", "activities": "no",
                "nursery": "yes", "higher": "no", "internet": "no", "romantic": "yes",
                "school": "MS", "sex": "M", "address": "R", "famsize": "GT3",
                "Pstatus": "T", "Mjob": "other", "Fjob": "other",
                "reason": "course", "guardian": "mother"}

    print_section("请输入学生特征（直接回车 = 使用训练集统计值；输入 q 放弃）")
    feats: Dict[str, object] = {}
    for key, (desc, lo, hi) in NUMERIC_PROMPTS.items():
        while True:
            s = input(f"  {desc}: ").strip()
            if s.lower() == "q":
                return {}
            if s == "":
                break
            try:
                v = float(s)
            except ValueError:
                print("    [错误] 请输入数字或直接回车")
                continue
            if not (lo <= v <= hi):
                print(f"    [警告] {v} 超出建议范围 [{lo}, {hi}]，仍按原值计算")
            feats[key] = v
            break
    for key, desc in YN_PROMPTS.items():
        s = input(f"  {desc} (yes/no，回车跳过): ").strip().lower()
        if s in ("yes", "no"):
            feats[key] = s
    for key, desc in CAT_PROMPTS.items():
        levels = "/".join(CATEGORICAL_LEVELS[key])
        s = input(f"  {desc} [{levels}]（回车跳过）: ").strip()
        if s in CATEGORICAL_LEVELS[key]:
            feats[key] = s
    return feats


# ======================================================================
# 四、命令行入口
# ======================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """python src/query_student_v2.py --id mat0001 / --top 10 / --interactive / --demo"""
    parser = argparse.ArgumentParser(description="UCI 真实数据 · 单学生风险查询（sklearn 主模型）")
    parser.add_argument("--id", type=str, default=None, help="记录编号，如 mat0001")
    parser.add_argument("--top", type=int, default=None, help="批量筛查前 N 名")
    parser.add_argument("--interactive", action="store_true", help="交互式录入特征")
    parser.add_argument("--demo", action="store_true", help="用内置示例特征直接演示（无需输入）")
    parser.add_argument("--backend", default="auto", choices=["auto", "sklearn", "numpy"])
    parser.add_argument("--k", type=int, default=None, help="指定 k（默认取实验选定的主 k）")
    parser.add_argument("--llm", action="store_true", help="尝试联网调用 LLM 组织建议文字")
    args = parser.parse_args(argv)

    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    engine = RiskQueryEngineV2(backend=args.backend, k=args.k)

    if args.top:
        print_batch_v2(engine.batch_query(top_n=args.top))
    elif args.id:
        r = engine.query_by_id(args.id, use_llm=args.llm)
        print_result_v2(r)
        print()
        print(llm.format_assistance(r["assistance"]))
    elif args.interactive or args.demo:
        feats = interactive_input_v2(demo=args.demo)
        if not feats:
            print("已取消。")
            return
        r = engine.query_features(feats, student_id="自定义输入(虚构)", use_llm=args.llm)
        print_result_v2(r)
        print()
        print(llm.format_assistance(r["assistance"]))
    else:
        parser.print_help()
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
