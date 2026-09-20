# -*- coding: utf-8 -*-
"""
query_student.py —— 单名学生风险查询（辅助功能 2）
=========================================================
对外回答辅导员最关心的问题：

    "这个学生风险多高？概率多少？主要原因是啥？跟哪几位同学最像？"

实现要点（答辩要能讲清）：
    1. 输入特征是**原始量纲**（成绩 0~100、出勤率 0~1、自习 0~10），
       程序内部自动套用"训练时保存的"填补中位数与标准化参数。
       —— 这一点极其重要：如果查询时用查询样本自己的均值做标准化，
       同一个学生每次查都得到不同的概率。因此必须复用 preprocess_stats.json。
    2. 缺失值：用训练集中位数填补，并在输出中明确标注哪些字段缺失，
       避免使用者误以为那是真实观测值。
    3. 输出包含：
       · 风险等级 + 高风险概率（来自 KNN，主模型结论，**不可被任何文字修改**）
       · 对照模型（逻辑回归）的概率，用于交叉参考
       · 关键原因（近邻对比解释，中文句子）
       · 最相似的 k 位同学名单（辅导员可据此找参照对象）
       · 复算校验：把近邻标签手工数一遍，验证 predict_proba 的分数是否正确
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
    from . import explain as ex
    from . import llm_helper as llm
    from .knn_numpy import assign_risk_levels, proba_to_risk_level, risk_level_to_action
    from .preprocess import load_simulated
    from .train_eval import load_saved_models
    from .utils import bar, fmt_pct, print_kv, print_section, print_title
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    import explain as ex
    import llm_helper as llm
    from knn_numpy import assign_risk_levels, proba_to_risk_level, risk_level_to_action
    from preprocess import load_simulated
    from train_eval import load_saved_models
    from utils import bar, fmt_pct, print_kv, print_section, print_title


# ======================================================================
# 一、模型与训练集上下文
# ======================================================================
class RiskQueryEngine:
    """
    风险查询引擎：加载模型一次，之后可以反复查询任意学生（这才是"可封装成简单应用"）。

    构造时做三件事：
      1. 加载 models.npz（KNN 训练数据 + 逻辑回归参数 + 预处理统计量）；
      2. 加载训练集原始数据（供按学号查询、输出近邻姓名）；
      3. 把训练集特征标准化，得到 KNN 的"标准空间参照系"。
    """

    def __init__(self, model_path: str = cfg.MODEL_PATH, verbose: bool = False) -> None:
        saved = load_saved_models(model_path)
        self.knn = saved["knn"]
        self.logreg = saved["logreg"]
        self.imputer = saved["imputer"]
        self.scaler = saved["scaler"]
        self.feature_columns: List[str] = saved["feature_columns"]
        self.thresholds: Dict[str, float] = saved["thresholds"]
        self.backend: str = saved["backend"]

        # 训练集原始数据（用于按学号查询与展示近邻信息）
        self.train_df = load_simulated()
        self.train_ids = self.train_df["student_id"].to_numpy()
        # KNN 内部已保存标准化后的训练特征，这里直接复用，避免重复计算
        self.X_train_std = self.knn.X_train_
        self.y_train = self.knn.y_train_
        if verbose:
            print(f"[引擎] 已加载模型：KNN(k={self.knn.k})，训练样本 {len(self.train_ids)} 条，"
                  f"对照模型后端={self.backend}")

    # ------------------------------------------------------------------
    def _prepare(self, features: Dict[str, float]
                 ) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, float]]:
        """
        把原始特征字典整理成"模型可用的标准化向量"。

        返回 (标准化向量, 缺失字段名列表, 填补后的原始向量, 完整原始特征字典)
        """
        missing: List[str] = []
        raw_values: List[float] = []
        full: Dict[str, float] = {}

        for i, col in enumerate(self.feature_columns):
            v = features.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                # 记录缺失，用训练集中位数填补（而不是 0，也不是查询样本自身统计量）
                missing.append(col)
                filled = float(self.imputer.fill_values_[i])
                raw_values.append(filled)
                full[col] = filled
            else:
                raw_values.append(float(v))
                full[col] = float(v)

        x_raw = np.asarray(raw_values, dtype=float).reshape(1, -1)
        x_std = self.scaler.transform(x_raw)[0]
        return x_std, missing, x_raw[0], full

    # ------------------------------------------------------------------
    def query_features(self, features: Dict[str, float],
                       student_id: str = "自定义输入",
                       use_llm: bool = False) -> Dict:
        """
        核心查询：输入原始特征 → 输出完整风险结论（结论全部来自模型，不由文字改写）。
        """
        x_std, missing, x_raw, full = self._prepare(features)

        # ---- 判断被查询者是否为训练集内的学生（原始特征完全一致）----
        # 若是，则必须排除"自己"这个 0 距离近邻，否则等于用答案喂自己：
        # 概率会被自己的标签拉偏（甚至出现 0/1 的极端值），参考同学名单里还会出现本人。
        raw_dist = np.abs(self.train_df[self.feature_columns].to_numpy(dtype=float) - x_raw).max(axis=1)
        in_training_set = bool(raw_dist.min() < 1e-9)

        # ---- ① 主模型：KNN 概率 = k 个近邻中高风险样本占比 ----
        proba = float(self.knn.predict_proba(x_std.reshape(1, -1),
                                             exclude_self=in_training_set)[0, 1])
        level = proba_to_risk_level(proba, self.thresholds)
        # 平票（概率恰为 0.5）归为"正常"，与 KNNClassifierNumpy.predict 的口径一致
        pred = int(proba > 0.5)

        # ---- ② 对照模型：逻辑回归概率（仅作交叉参考，不参与最终结论） ----
        lr_proba = float(self.logreg.predict_proba(x_std.reshape(1, -1))[0, 1])
        lr_level = proba_to_risk_level(lr_proba, self.thresholds)

        # ---- ③ 局部解释：训练集分层对比 + 近邻构成 ----
        raw_series = pd.Series(full)
        local = ex.explain_local(
            knn=self.knn, x_std=x_std, y_train=self.y_train,
            X_train_std=self.X_train_std, feature_names=self.feature_columns,
            raw_row=raw_series, scaler=self.scaler, top_n=4,
            neighbor_ids=self.train_ids, exclude_self=in_training_set,
        )

        # ---- ④ 复算校验：手工数一遍近邻标签，验证概率 ----
        nb_labels = np.array([n["label"] for n in local["neighbors"]], dtype=int)
        manual_proba = float(nb_labels.sum()) / len(nb_labels)
        proba_check = abs(manual_proba - proba) < 1e-9

        result = {
            "student_id": student_id,
            "proba_risk": round(proba, 4),
            "risk_level": level,
            "pred_label": pred,
            "logreg_proba_risk": round(lr_proba, 4),
            "logreg_risk_level": lr_level,
            "models_agree_level": bool(lr_level == level),
            "missing_fields": missing,
            "in_training_set": in_training_set,
            "raw_features": {k: round(float(v), 4) for k, v in full.items()},
            "action": risk_level_to_action(level),
            "explain": local,
            "top_contributions": local["top_contributions"],
            "sentences": local["sentences"],
            "neighbors": local["neighbors"],
            "proba_manual_check": {
                "manual_from_neighbors": round(manual_proba, 4),
                "model_output": round(proba, 4),
                "consistent": bool(proba_check),
            },
            "thresholds": self.thresholds,
            "model_info": {
                "knn_k": int(self.knn.k),
                "n_train": int(len(self.train_ids)),
                "logreg_backend": self.backend,
            },
        }

        # ---- ⑤ 帮扶建议（LLM 只做文字组织，不改结论） ----
        facts = llm.build_facts(
            student_id=student_id, proba_risk=proba, risk_level=level,
            top_contributions=local["top_contributions"], neighbors=local["neighbors"],
            raw_features=result["raw_features"], thresholds=self.thresholds, k=self.knn.k,
        )
        result["facts"] = facts
        result["assistance"] = llm.generate_assistance(facts, use_api=use_llm, verbose=False)
        return result

    # ------------------------------------------------------------------
    def query_by_id(self, student_id: str, use_llm: bool = False) -> Dict:
        """按学号查询（数据集中已存在的学生）。"""
        row = self.train_df[self.train_df["student_id"] == str(student_id)]
        if row.empty:
            raise KeyError(f"数据集中没有学号 {student_id}。"
                           f"可用学号范围示例：{self.train_ids[0]} ~ {self.train_ids[-1]}")
        feats = {c: float(row.iloc[0][c]) for c in self.feature_columns}
        return self.query_features(feats, student_id=str(student_id), use_llm=use_llm)

    # ------------------------------------------------------------------
    def batch_query(self, df: Optional[pd.DataFrame] = None, top_n: int = 10) -> pd.DataFrame:
        """
        批量查询（对应真实使用场景：期中/期末批量筛查）。
        返回按高风险概率降序排列的名单，便于辅导员从最需要关注的人开始跟进。
        """
        df = self.train_df if df is None else df
        X_raw = df[self.feature_columns].to_numpy(dtype=float)
        X_std = self.scaler.transform(self.imputer.transform(X_raw))
        # 若查询对象就是训练集本身，必须排除"自己"这个 0 距离近邻，
        # 这样得到的概率才与留一法口径一致（否则每人都自带一票，概率整体偏高）。
        exclude_self = df is self.train_df
        proba = self.knn.predict_proba(X_std, exclude_self=exclude_self)[:, 1]
        lr_proba = self.logreg.predict_proba(X_std)[:, 1]

        out = df[["student_id"] + cfg.SCORE_COLUMNS + [
            "attendance_rate", "homework_submit_rate", "self_study_hours"]].copy()
        out["knn_proba_risk"] = np.round(proba, 4)
        out["risk_level"] = assign_risk_levels(proba, self.thresholds)
        out["logreg_proba_risk"] = np.round(lr_proba, 4)
        out["action"] = [risk_level_to_action(l) for l in out["risk_level"]]
        if cfg.LABEL_COLUMN in df.columns:
            out["true_label"] = df[cfg.LABEL_COLUMN].to_numpy()
        return out.sort_values("knn_proba_risk", ascending=False).head(top_n).reset_index(drop=True)


# ======================================================================
# 二、结果打印
# ======================================================================
def print_result(result: Dict) -> None:
    """把一次查询结果排版输出（终端友好，中文可读）。"""
    print_title(f"学生风险查询结果：{result['student_id']}")
    p = result["proba_risk"]
    print(f"  KNN 高风险概率 : {p:.4f}  {bar(p)}  {fmt_pct(p)}")
    print(f"  风险等级       : 【{result['risk_level']}】"
          f"（低 < {result['thresholds']['low']}；中 {result['thresholds']['low']}~{result['thresholds']['high']}；"
          f"高 > {result['thresholds']['high']}）")
    print(f"  二分类判定     : {cfg.RISK_LABEL_CN[result['pred_label']]}"
          f"（概率 > 0.5 判为需要重点关注）")
    print(f"  处置建议       : {result['action']}")
    print(f"  对照模型(逻辑回归) : 概率 {result['logreg_proba_risk']:.4f}，"
          f"等级 {result['logreg_risk_level']}，"
          f"{'与本模型等级一致' if result['models_agree_level'] else '与本模型等级不一致（边界样本，建议人工判定）'}")

    chk = result["proba_manual_check"]
    print_section("概率复算校验（手工数近邻标签）")
    print(f"  近邻中高风险人数/近邻总数 = {chk['manual_from_neighbors']:.4f}；"
          f"模型输出 = {chk['model_output']:.4f}；一致 = {chk['consistent']}")
    if result.get("in_training_set"):
        print("  说明：该生已在训练集中，已自动排除“自己”这个 0 距离近邻，"
              "避免用自己的标签抬高概率。")
    if result["explain"].get("notes"):
        for n in result["explain"]["notes"]:
            print(f"  [提示] {n}")

    print_section("输入特征（原始量纲）")
    print_kv({cfg.FEATURE_CN.get(k, k): v for k, v in result["raw_features"].items()})
    if result["missing_fields"]:
        print(f"  [注意] 以下字段缺失，已用训练集中位数填补："
              f"{[cfg.FEATURE_CN.get(f, f) for f in result['missing_fields']]}")

    print_section("为什么判为这个等级（模型给出的关键原因）")
    for s in result["sentences"]:
        print("  " + s)

    print_section(f"最相似的 {len(result['neighbors'])} 位同学（k 近邻，按距离升序）")
    for nb in result["neighbors"]:
        print(f"  #{nb['rank']}  {nb['student_id']}  距离={nb['distance']:.4f}  "
              f"实际标签={nb['label_cn']}")


def print_neighbor_table(engine: "RiskQueryEngine", result: Dict) -> None:
    """额外打印近邻同学的原始特征对比表（辅导员想"跟谁比"时很有用）。"""
    ids = [nb["student_id"] for nb in result["neighbors"] if nb["student_id"]]
    if not ids:
        return
    sub = engine.train_df[engine.train_df["student_id"].isin(ids)].copy()
    cols = ["student_id"] + cfg.SCORE_COLUMNS + ["attendance_rate",
                                                 "homework_submit_rate", "self_study_hours"]
    print()
    print("近邻同学原始特征对比：")
    print(sub[cols].to_string(index=False))


def print_batch(df: pd.DataFrame, title: str = "批量筛查结果（按高风险概率降序）") -> None:
    """打印批量筛查名单。"""
    print_title(title)
    cols = ["student_id", "score_mean", "attendance_rate", "homework_submit_rate",
            "self_study_hours", "knn_proba_risk", "risk_level"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    n_high = int((df["risk_level"] == "高").sum())
    n_mid = int((df["risk_level"] == "中").sum())
    print(f"\n  共 {len(df)} 人：高风险 {n_high} 人，中风险 {n_mid} 人，"
          f"低风险 {len(df) - n_high - n_mid} 人")
    print(f"  {cfg.DISCLAIMER}")


# ======================================================================
# 三、交互式输入
# ======================================================================
PROMPTS = {
    "python": ("Python 程序设计成绩 (0~100)", 0.0, 100.0),
    "ml": ("机器学习成绩 (0~100)", 0.0, 100.0),
    "cv": ("计算机视觉成绩 (0~100)", 0.0, 100.0),
    "attendance_rate": ("出勤率 (0~1，如 0.85)", 0.0, 1.0),
    "homework_submit_rate": ("作业提交率 (0~1，如 0.80)", 0.0, 1.0),
    "self_study_hours": ("每周自习时长 (0~10 小时)", 0.0, 10.0),
}


def interactive_input() -> Dict[str, float]:
    """
    交互式录入学生特征。直接回车表示"该字段缺失"，由程序用训练集中位数填补。
    输入非法时提示重填，不抛异常，保证答辩现场不会因为手误而中断。
    """
    print_section("请输入学生特征（直接回车表示缺失，将由训练集中位数填补）")
    feats: Dict[str, float] = {}
    for key, (label, lo, hi) in PROMPTS.items():
        while True:
            s = input(f"  {label}: ").strip()
            if s == "":
                print("    → 记为缺失")
                break
            try:
                v = float(s)
            except ValueError:
                print("    [错误] 请输入数字，或直接回车表示缺失")
                continue
            if not (lo <= v <= hi):
                print(f"    [警告] 数值 {v} 超出建议范围 [{lo}, {hi}]，仍将按原值计算")
            feats[key] = v
            break
    # 派生特征由程序自动计算，避免使用者手工算错
    scores = [feats[k] for k in cfg.SCORE_COLUMNS if k in feats]
    if scores:
        feats["score_mean"] = float(np.mean(scores))
        feats["score_std"] = float(np.std(scores))
        print(f"  （已自动计算派生特征：三科平均分={feats['score_mean']:.2f}，"
              f"三科波动={feats['score_std']:.2f}）")
    return feats


# ======================================================================
# 四、命令行入口
# ======================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """python src/query_student.py --id SIM0007 / --top 15 / --interactive"""
    parser = argparse.ArgumentParser(description="单名学生风险查询（KNN 主模型，含概率/原因/近邻）")
    parser.add_argument("--id", type=str, default=None, help="按学号查询，如 SIM0007")
    parser.add_argument("--top", type=int, default=None, help="批量筛查前 N 名高风险学生")
    parser.add_argument("--interactive", action="store_true", help="交互式录入特征查询")
    parser.add_argument("--batch-all", action="store_true", help="对全体学生做批量筛查并保存 CSV")
    parser.add_argument("--llm", action="store_true", help="尝试联网调用 LLM 组织建议文字（需 API Key）")
    args = parser.parse_args(argv)

    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    engine = RiskQueryEngine(verbose=True)

    # 默认行为：交互式查询（最贴近"可封装成简单应用"的验收要求）
    do_default_interactive = not (args.id or args.top or args.batch_all or args.interactive)

    if args.id:
        result = engine.query_by_id(args.id, use_llm=args.llm)
        print_result(result)
        print_neighbor_table(engine, result)
        print()
        print(llm.format_assistance(result["assistance"]))
    elif args.batch_all or args.top:
        top_n = args.top if args.top else len(engine.train_df)
        table = engine.batch_query(top_n=top_n)
        print_batch(table, title=f"批量筛查结果（前 {len(table)} 名高风险学生）")
        if args.batch_all:
            out = os.path.join(cfg.RESULTS_DIR, "risk_list_all.csv")
            engine.batch_query(top_n=len(engine.train_df)).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n  已保存全量风险名单：{os.path.relpath(out, cfg.BASE_DIR)}")
    elif args.interactive or do_default_interactive:
        feats = interactive_input()
        result = engine.query_features(feats, student_id="自定义输入")
        print_result(result)
        print()
        print(llm.format_assistance(result["assistance"]))

    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
