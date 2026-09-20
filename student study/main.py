# -*- coding: utf-8 -*-
"""
main.py —— 项目菜单式主程序
=========================================================
运行方式（三种，任选其一）：

    python main.py                     # 菜单模式（答辩现场最直观）
    python main.py --auto              # 自动跑完整流程：数据→训练→解释→看板→示例查询
    python main.py --step train        # 只跑某一步（data/train/explain/dashboard/query/verify/report）

菜单功能：
    1. 生成/重建模拟数据（支持随机种子）
    2. 训练与评估（留一法 + 10 次随机划分 + k 值对比 + 新样本测试 + 人工复算）
    3. 生成可视化看板（6 张图 → results/figures/）
    4. 特征贡献解释（全局置换重要性）
    5. 单名学生查询（按学号 / 交互式输入 / 批量筛查）
    6. 生成学业帮扶建议（LLM 只做文字组织）
    7. 查看当前实验指标汇总（从 results/metrics/metrics_summary.json 读取）
    8. 一键跑完整流程

【重要声明】本项目仅作教学辅助，全部数据为模拟/脱敏数据，
            不得用于任何真实学籍、评奖、处分或决策。
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import platform
import time
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# 路径处理：把 src 加入 sys.path，使 `python main.py` 能直接 import 各模块。
# 这样无论从项目根目录还是 src 目录运行都能正常工作。
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config as cfg            # noqa: E402
from utils import (bar, enable_utf8_console, fmt_pct, print_kv,  # noqa: E402
                   print_list, print_section, print_title)


# ======================================================================
# 0. 环境自检
# ======================================================================
def check_environment(verbose: bool = True) -> Dict:
    """
    环境自检：Python 版本、依赖版本、sklearn 是否存在、数据/模型文件是否就绪。

    这一步是答辩时的"开场保障"：先证明环境合规（Python 3.10 + numpy/pandas/matplotlib，
    不需要 GPU、不需要联网），再演示功能。
    """
    info: Dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "缺失（必需！）"
    try:
        import sklearn
        info["sklearn"] = getattr(sklearn, "__version__", "unknown")
    except ImportError:
        info["sklearn"] = "未安装 → 对照模型使用 numpy 手写逻辑回归（符合实训约束）"

    info["data_file"] = "存在" if os.path.exists(cfg.DATA_PATH) else "缺失（请先执行步骤 1）"
    info["new_batch_file"] = "存在" if os.path.exists(cfg.NEW_BATCH_PATH) else "缺失（请先执行步骤 1）"
    info["model_file"] = "存在" if os.path.exists(cfg.MODEL_PATH) else "缺失（请先执行步骤 2）"
    info["metrics_file"] = "存在" if os.path.exists(cfg.METRICS_SUMMARY_PATH) else "缺失（请先执行步骤 2）"

    if verbose:
        print_title("环境与文件自检")
        print_kv(info)
        ok_py = info["python"].startswith("3.10")
        print(f"  Python 3.10 要求：{'满足 ✅' if ok_py else '不满足（当前 ' + info['python'] + '，可能仍可运行）'}")
        need = [m for m in ("numpy", "pandas", "matplotlib") if info[m].startswith("缺失")]
        print(f"  必需依赖：{'全部就绪 ✅' if not need else '缺少 ' + str(need) + ' ❌'}")
    return info


# ======================================================================
# 1. 生成模拟数据
# ======================================================================
def step_generate_data(n: int = cfg.N_SAMPLES, seed: int = cfg.RANDOM_SEED,
                       n_new: int = cfg.N_NEW_SAMPLES,
                       new_seed: int = cfg.NEW_BATCH_SEED) -> Dict:
    """步骤 1：生成模拟数据（主数据集 + 新样本测试集 + 5 人种子数据）。"""
    from generate_data import main as gen_main
    return gen_main(["--n", str(n), "--seed", str(seed),
                     "--n-new", str(n_new), "--new-seed", str(new_seed)])


# ======================================================================
# 2. 训练与评估
# ======================================================================
def step_train_eval(k: Optional[int] = None, splits: int = cfg.N_SPLITS,
                    seed: int = cfg.RANDOM_SEED) -> Dict:
    """步骤 2：训练与评估（手写 KNN 主模型 + 逻辑回归对照，三套评估口径）。"""
    from train_eval import run_training_eval
    if not os.path.exists(cfg.DATA_PATH):
        print("[提示] 未找到数据文件，先自动执行步骤 1（生成模拟数据）……")
        step_generate_data(seed=seed)
    return run_training_eval(main_k=k, n_splits=splits, seed=seed)


# ======================================================================
# 3. 特征解释
# ======================================================================
def step_explain(k: Optional[int] = None) -> Dict:
    """步骤 3：全局置换重要性（解释"哪些特征真正在影响预测"）。"""
    from explain import run_explain
    if not os.path.exists(cfg.MODEL_PATH):
        print("[提示] 未找到模型文件，先自动执行步骤 2（训练）……")
        step_train_eval()
    return run_explain(k=k)


# ======================================================================
# 4. 可视化看板
# ======================================================================
def step_dashboard(figures: Optional[List[str]] = None) -> Dict:
    """步骤 4：生成 matplotlib 看板（6 张图）。"""
    from dashboard import build_all
    return build_all(figures=figures)


# ======================================================================
# 5. 单学生查询
# ======================================================================
def step_query(student_id: Optional[str] = None, interactive: bool = False,
               top: Optional[int] = None, use_llm: bool = False) -> Dict:
    """
    步骤 5：查询。三种模式：
      · 指定学号 → 输出该生完整报告；
      · interactive=True → 交互式录入任意特征（体现"随机换输入仍可预测"）；
      · top=N → 批量筛查前 N 名高风险学生。
    """
    from query_student import (RiskQueryEngine, interactive_input, print_batch,
                               print_neighbor_table, print_result)
    from llm_helper import format_assistance

    if not os.path.exists(cfg.MODEL_PATH):
        print("[提示] 未找到模型文件，先自动执行步骤 2（训练）……")
        step_train_eval()

    engine = RiskQueryEngine(verbose=True)

    if top:
        table = engine.batch_query(top_n=top)
        print_batch(table, title=f"批量筛查结果（前 {len(table)} 名高风险学生）")
        return {"mode": "batch", "table": table}

    if interactive:
        feats = interactive_input()
        result = engine.query_features(feats, student_id="自定义输入", use_llm=use_llm)
    else:
        sid = student_id or "SIM0007"
        result = engine.query_by_id(sid, use_llm=use_llm)

    print_result(result)
    print_neighbor_table(engine, result)
    print()
    print(format_assistance(result["assistance"]))
    return {"mode": "single", "result": result}


# ======================================================================
# 6. 帮扶建议（单独入口，对应辅助功能 4）
# ======================================================================
def step_assistance(student_id: str = "SIM0007", use_llm: bool = False) -> Dict:
    """步骤 6：为指定学生生成帮扶建议（文字可由 LLM 组织，结论不可修改）。"""
    from llm_helper import format_assistance
    from query_student import RiskQueryEngine
    if not os.path.exists(cfg.MODEL_PATH):
        print("[提示] 未找到模型文件，先自动执行步骤 2（训练）……")
        step_train_eval()
    engine = RiskQueryEngine(verbose=False)
    result = engine.query_by_id(student_id, use_llm=use_llm)
    print(format_assistance(result["assistance"]))
    return result


# ======================================================================
# 7. 查看指标汇总
# ======================================================================
def step_show_metrics() -> None:
    """步骤 7：从 results 读取并展示真实实验指标（不重新计算，保证"看到的即保存的"）。"""
    if not os.path.exists(cfg.METRICS_SUMMARY_PATH):
        print("[提示] 尚未生成指标文件，请先执行步骤 2（训练与评估）。")
        return
    with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)

    print_title("实验指标汇总（来自 results/metrics/metrics_summary.json）")
    print_kv({
        "样本数": s.get("n_samples"),
        "特征数": len(s.get("feature_columns", [])),
        "主 k": s.get("main_k"),
        "随机种子": s.get("seed"),
        "划分次数": s.get("n_splits"),
        "耗时(秒)": s.get("elapsed_sec"),
    })

    print_section("① 主模型 KNN —— 留一法（主报告口径）")
    m = s.get("loo_knn", {}).get("metrics", {})
    print_kv({k: m.get(k) for k in ("accuracy", "precision", "recall", "f1",
                                    "specificity", "balanced_accuracy")})
    print("  混淆矩阵 [[TN, FP], [FN, TP]] =", s.get("loo_knn", {}).get("confusion_matrix"))

    print_section(f"② 主模型 KNN —— {s.get('n_splits')} 次随机划分（均值 ± 标准差）")
    for name, label in (("accuracy", "准确率"), ("precision", "精确率"),
                        ("recall", "召回率"), ("f1", "F1")):
        a = s.get("split10_knn", {}).get(name, {})
        print(f"  {label:<4}: {a.get('mean')} ± {a.get('std')}  "
              f"(min {a.get('min')} / max {a.get('max')})")

    print_section("③ 对照模型 逻辑回归")
    lm = s.get("loo_logreg", {})
    print("  留一法：" + "  ".join(f"{k}={lm.get(k)}" for k in
                                  ("accuracy", "precision", "recall", "f1")))
    a = s.get("split10_logreg", {})
    if a:
        print(f"  {s.get('n_splits')} 次划分 F1: {a.get('f1', {}).get('mean')} "
              f"± {a.get('f1', {}).get('std')}")
    print(f"  后端：{s.get('logreg_backend', {}).get('detail')}")

    print_section("④ 新样本最终测试（从未参与训练）")
    for key, label in (("new_batch_knn", "KNN"), ("new_batch_logreg", "逻辑回归")):
        r = s.get(key, {})
        mm = r.get("metrics", {})
        print(f"  {label:<6}: " + "  ".join(f"{k}={mm.get(k)}" for k in
                                            ("accuracy", "precision", "recall", "f1")))
        print(f"          混淆矩阵 {r.get('confusion_matrix')}")
    print(f"  批次信息：{s.get('new_batch_info')}")

    print_section("⑤ k 值选择理由")
    print(f"  主 k = {s.get('k_selection', {}).get('chosen_k')}")
    print(f"  {s.get('k_selection', {}).get('reason')}")

    print_section("⑥ 人工复算（验收要求：至少 2 项指标手工核对）")
    if os.path.exists(cfg.VERIFY_PATH):
        with open(cfg.VERIFY_PATH, "r", encoding="utf-8") as f:
            v = json.load(f)
        for name, step in v.get("steps", {}).items():
            print("  " + step)
        print(f"  一致性检查：{v.get('checks')} → 全部通过 = {v.get('all_passed')}")
    else:
        print("  [提示] 未找到 manual_verification.json")

    print("-" * 68)
    print(cfg.DISCLAIMER)


def step_verify() -> None:
    """
    步骤 8：人工复算专项（把"手工算指标"的过程单独跑一遍并打印/落盘）。

    这是验收要求 5 的独立证据：不依赖训练流程，任何时候都能复现。
    """
    import numpy as np
    import pandas as pd
    import evaluate as ev
    from preprocess import load_simulated
    from knn_numpy import loo_predict
    from train_eval import load_saved_models

    print_title("人工复算专项：由混淆矩阵手算 precision / recall")
    if not os.path.exists(cfg.MODEL_PATH):
        print("[提示] 未找到模型文件，请先执行步骤 2（训练与评估）。")
        return
    saved = load_saved_models()
    knn = saved["knn"]
    df = load_simulated()
    from preprocess import build_dataset
    ds, _, _ = build_dataset(df)
    y_pred = loo_predict(ds.X, ds.y, knn.k)
    res = ev.verify_against_implementation(ds.y, y_pred)
    print(f"口径：全部 {len(df)} 条模拟数据的留一法预测（k={knn.k}）")
    print(f"混淆矩阵四格（TN, FP, FN, TP）= {res['counts']}")
    print()
    print_list([f"{k}：{v}" for k, v in res["steps"].items()], bullet="→")
    print()
    print(f"程序实现值：{res['implementation']}")
    print(f"逐项一致性：{res['checks']}")
    print(f"结论：{'全部一致，指标可信 ✅' if res['all_passed'] else '存在不一致，需检查代码 ❌'}")
    ev.save_json(res, cfg.VERIFY_PATH)
    print(f"已保存：{os.path.relpath(cfg.VERIFY_PATH, cfg.BASE_DIR)}")
    print("-" * 68)
    print(cfg.DISCLAIMER)


# ======================================================================
# 9. 一键完整流程
# ======================================================================
def run_full_pipeline(seed: int = cfg.RANDOM_SEED, k: Optional[int] = None,
                      splits: int = cfg.N_SPLITS, with_llm: bool = False) -> Dict:
    """
    一键跑完整流程（验收要求："能运行：python main.py 可完成主要流程"）。

    顺序：数据 → 训练评估 → 解释 → 看板 → 示例查询（高风险/低风险各一例）→ 指标汇总
    每步都有明确的失败提示，任何一步出错都会停在那一步并打印原因，
    而不是静默地给出不完整结果。
    """
    t0 = time.time()
    enable_utf8_console()
    print_title("《基于机器学习的校园学生成绩风险预警与学业帮扶系统》完整流程")
    print(f"  {cfg.DISCLAIMER}\n")

    check_environment(verbose=True)
    print()

    step_generate_data(seed=seed)
    print()
    summary = step_train_eval(k=k, splits=splits, seed=seed)
    print()
    step_explain()
    print()
    step_dashboard()
    print()

    # 示例查询：从留一法结果里挑一个"高风险被正确识别"和一个"低风险"的学生，更有说服力
    print_title("示例查询：随机换输入仍可预测")
    example_ids = _pick_example_students()
    for sid in example_ids:
        try:
            step_query(student_id=sid, use_llm=with_llm)
        except Exception as e:  # 单条查询失败不应中断整个流程
            print(f"  [警告] 查询 {sid} 失败：{type(e).__name__}: {e}")
        print()
    # 批量筛查（真实使用场景：期中/期末批量拉名单）
    step_query(top=10)
    print()

    step_verify()
    print()
    step_show_metrics()

    print_title("完整流程执行结束")
    print(f"  总耗时：{time.time() - t0:.1f} 秒")
    print("  产物目录：")
    print(f"    data/                模拟数据（含新样本批次）")
    print(f"    results/figures/     可视化图（6 张）")
    print(f"    results/metrics/     指标、混淆矩阵、人工复算、风险名单")
    print(f"    results/models/      训练好的模型与预处理参数")
    print(f"  {cfg.DISCLAIMER}")
    return summary


def _pick_example_students() -> List[str]:
    """从留一法预测结果中挑出有代表性的示例学生（1 个高风险、1 个低风险）。"""
    if not os.path.exists(cfg.PREDICTIONS_PATH):
        return ["SIM0007"]
    try:
        import pandas as pd
        df = pd.read_csv(cfg.PREDICTIONS_PATH, encoding="utf-8-sig")
        ids: List[str] = []
        high = df[(df["loo_pred_level"] == "高")].sort_values("loo_proba_risk", ascending=False)
        low = df[(df["loo_pred_level"] == "低")].sort_values("loo_proba_risk")
        if len(high):
            ids.append(str(high.iloc[0]["student_id"]))
        if len(low):
            ids.append(str(low.iloc[0]["student_id"]))
        return ids or ["SIM0007"]
    except Exception:
        return ["SIM0007"]


# ======================================================================
# 10. 菜单
# ======================================================================
MENU = """
=================== 校园学生成绩风险预警系统（教学实验版） ===================
  主模型：numpy 手写 KNN（欧氏距离 + k 近邻多数投票 + predict_proba）
  对照模型：逻辑回归（sklearn 优先，未安装则用 numpy 手写）
  数据：300~500 条自建模拟/脱敏数据，高风险比例 20%~35%，含特征重叠与标签噪声

  1. 生成/重建模拟数据（可设随机种子）
  2. 训练与评估（留一法 + 10 次随机划分 + k 值对比 + 新样本测试 + 人工复算）
  3. 特征贡献解释（全局置换重要性）
  4. 生成可视化看板（6 张图）
  5. 单名学生查询（按学号 / 交互式输入 / 批量筛查）
  6. 生成学业帮扶建议（LLM 仅做文字组织，不改预测结论）
  7. 查看当前实验指标汇总
  8. 人工复算专项（手算 precision / recall 并核对）
  9. 一键跑完整流程
  0. 退出
==============================================================================
  {disclaimer}
"""


def interactive_menu() -> None:
    """菜单循环。每一项都做了异常兜底，避免一个错误就退出整个程序。"""
    enable_utf8_console()
    print_title("环境自检（启动时自动执行）")
    check_environment(verbose=True)

    while True:
        print(MENU.format(disclaimer=cfg.DISCLAIMER))
        choice = input("请选择功能编号 [0-9]: ").strip()
        try:
            if choice == "0":
                print("已退出。提醒：" + cfg.DISCLAIMER)
                break
            elif choice == "1":
                s = input(f"  样本条数（回车默认 {cfg.N_SAMPLES}）: ").strip()
                sd = input(f"  随机种子（回车默认 {cfg.RANDOM_SEED}）: ").strip()
                n = int(s) if s else cfg.N_SAMPLES
                seed = int(sd) if sd else cfg.RANDOM_SEED
                step_generate_data(n=n, seed=seed)
            elif choice == "2":
                s = input(f"  是否指定主 k（回车=由实验自动选择，默认 {cfg.MAIN_K}）: ").strip()
                k = int(s) if s else None
                step_train_eval(k=k)
            elif choice == "3":
                step_explain()
            elif choice == "4":
                step_dashboard()
            elif choice == "5":
                print("  查询方式：1=按学号  2=交互式输入  3=批量筛查前 N 名")
                mode = input("  请选择 [1/2/3]（回车默认 2）: ").strip() or "2"
                if mode == "1":
                    sid = input(f"  学号（如 SIM0007）: ").strip() or "SIM0007"
                    step_query(student_id=sid)
                elif mode == "3":
                    s = input("  显示前几名（回车默认 10）: ").strip()
                    step_query(top=int(s) if s else 10)
                else:
                    step_query(interactive=True)
            elif choice == "6":
                sid = input(f"  学号（如 SIM0007）: ").strip() or "SIM0007"
                step_assistance(student_id=sid)
            elif choice == "7":
                step_show_metrics()
            elif choice == "8":
                step_verify()
            elif choice == "9":
                run_full_pipeline()
            else:
                print("  [提示] 请输入 0~9 之间的编号。")
        except KeyboardInterrupt:
            print("\n  [已取消当前操作，返回菜单]")
        except Exception as e:
            # 教学项目的原则：出错要说清是什么错、下一步怎么办，而不是抛一堆堆栈
            print(f"\n  [错误] {type(e).__name__}: {e}")
            print("  [建议] 若提示缺少数据/模型文件，请先执行步骤 1 → 步骤 2。")


# ======================================================================
# v2 流程：UCI 真实数据 + sklearn 主模型（默认主流程）
# ======================================================================
def step_uci_data(seed: int = cfg.RANDOM_SEED) -> Dict:
    """步骤 v2-1：下载并构建 UCI 真实数据集（主数据）。"""
    from uci_data import build_dataset as build_uci
    df, report = build_uci(verbose=True)
    return report


def step_train_eval_v2(backend: str = "auto", splits: int = cfg.N_SPLITS,
                       seed: int = cfg.RANDOM_SEED) -> Dict:
    """步骤 v2-2：UCI 数据 + sklearn 主模型的训练与评估（LOGO/分组划分/课程外推/消融）。"""
    from train_eval_v2 import run_uci_pipeline
    if not os.path.exists(cfg.UCI_DATA_PATH):
        print("[提示] 未找到 UCI 数据，先自动构建……")
        step_uci_data(seed=seed)
    return run_uci_pipeline(backend=backend, n_splits=splits, seed=seed)


def step_appendix(verbose: bool = True) -> Dict:
    """步骤 v2-3：附录——手写 KNN 与 sklearn 的等价性验证。"""
    from appendix_handwritten_knn import run_appendix
    try:
        return run_appendix(verbose=verbose)
    except RuntimeError as e:
        print(f"[跳过] {e}")
        return {}


def step_nested_cv() -> Dict:
    """步骤 v2-4：嵌套交叉验证（检验 k 选择是否存在选择偏差）。"""
    from nested_cv_v2 import run_nested_cv
    return run_nested_cv(verbose=True)


def step_dashboard_v2() -> Dict:
    """步骤 v2-5：生成 UCI 看板（图 11~15）。"""
    from dashboard_v2 import build_all_v2
    return build_all_v2()


def step_query_v2(student_id: Optional[str] = None, top: Optional[int] = None,
                  interactive: bool = False, demo: bool = False,
                  backend: str = "auto", use_llm: bool = False) -> Dict:
    """步骤 v2-6：单学生查询 / 批量筛查（sklearn 主模型）。"""
    from llm_helper import format_assistance
    from query_student_v2 import (RiskQueryEngineV2, interactive_input_v2,
                                  print_batch_v2, print_result_v2)
    if not os.path.exists(cfg.UCI_DATA_PATH):
        step_uci_data()
    engine = RiskQueryEngineV2(backend=backend)
    if top:
        print_batch_v2(engine.batch_query(top_n=top))
        return {"mode": "batch"}
    if interactive or demo:
        feats = interactive_input_v2(demo=demo)
        if not feats:
            print("已取消。")
            return {"mode": "cancelled"}
        r = engine.query_features(feats, student_id="自定义输入(虚构)", use_llm=use_llm)
    else:
        r = engine.query_by_id(student_id or "mat0001", use_llm=use_llm)
    print_result_v2(r)
    print()
    print(format_assistance(r["assistance"]))
    return {"mode": "single", "result": r}


def step_sim_compare() -> None:
    """
    步骤 v2-7：模拟数据 vs UCI 真实数据 的鲁棒性对照。

    保留最初的"主动构造特征重叠样本"思路作为**对照组**：同一套评估代码，
    只换数据源，比较"可控构造数据"与"真实数据"的难度差异。
    """
    print_title("模拟数据 vs UCI 真实数据 对照")
    rows = []
    if os.path.exists(cfg.METRICS_SUMMARY_PATH):
        with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if str(s.get("dataset", "")).startswith("UCI"):
            m = s.get("logo_knn", {}).get("metrics", {})
            rows.append({"数据": "UCI 真实数据", "样本数": s.get("n_samples"),
                         "高风险比例": s.get("risk_ratio"),
                         "多数类基线acc": s.get("majority_baseline_accuracy"),
                         "F1(LOGO)": m.get("f1"), "recall(LOGO)": m.get("recall"),
                         "acc(LOGO)": m.get("accuracy")})
    sim_summary = os.path.join(cfg.METRICS_DIR, "sim_metrics_summary.json")
    if os.path.exists(sim_summary):
        with open(sim_summary, "r", encoding="utf-8") as f:
            s2 = json.load(f)
        m2 = s2.get("loo_knn", {}).get("metrics", {})
        r2 = s2.get("risk_ratio", 0)
        rows.append({"数据": "自建模拟数据", "样本数": s2.get("n_samples"),
                     "高风险比例": r2,
                     "多数类基线acc": round(max(r2, 1 - r2), 4),
                     "F1(LOGO)": m2.get("f1"), "recall(LOGO)": m2.get("recall"),
                     "acc(LOGO)": m2.get("accuracy")})
    if not rows:
        print("  [提示] 尚未生成对照指标；请先运行 `python src/run_sim_compare.py`。")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n  说明：模拟数据由「加权规则 + 随机噪声 + 分位数定档」构造，难度可控；"
          "UCI 真实数据在剔除早期成绩（避免泄漏）后可用信息更稀疏，难度明显更高。")
    df.to_csv(cfg.SIM_COMPARE_PATH, index=False, encoding="utf-8-sig")
    print(f"  已保存：{os.path.relpath(cfg.SIM_COMPARE_PATH, cfg.BASE_DIR)}")
    print("-" * 68)
    print(cfg.DISCLAIMER)


def run_full_pipeline_v2(seed: int = cfg.RANDOM_SEED, splits: int = cfg.N_SPLITS,
                         backend: str = "auto") -> Dict:
    """v2 一键完整流程：UCI 数据 → sklearn 训练评估 → 附录等价性 → 看板 → 示例查询。"""
    t0 = time.time()
    enable_utf8_console()
    print_title("《基于机器学习的校园学生成绩风险预警与学业帮扶系统》v2 完整流程")
    print("  主数据：UCI Student Performance 真实公开数据集（CC BY 4.0，已匿名化）")
    print("  主模型：sklearn KNN / 逻辑回归（手写 KNN 作为附录做等价性验证）")
    print(f"  {cfg.DISCLAIMER}\n")

    check_environment(verbose=True)
    print()
    step_uci_data(seed=seed)
    print()
    step_train_eval_v2(backend=backend, splits=splits, seed=seed)
    print()
    step_appendix()
    print()
    step_dashboard_v2()
    print()

    print_title("示例查询：随机换输入仍可预测")
    for sid in ["mat0001", "por0001"]:
        try:
            step_query_v2(student_id=sid)
        except Exception as e:
            print(f"  [警告] 查询 {sid} 失败：{type(e).__name__}: {e}")
        print()
    step_query_v2(top=10)

    print_title("v2 完整流程执行结束")
    print(f"  总耗时：{time.time() - t0:.1f} 秒")
    print("  主要产物：")
    print("    data/uci_raw/                     UCI 原始 CSV（含 MD5 校验）")
    print("    data/uci_student_performance.csv  清洗与编码后的主数据集")
    print("    results/figures/11~15*.png        UCI 看板（5 张）")
    print("    results/metrics/                  指标、k 扫描、消融、嵌套 CV、人工复算")
    print(f"  {cfg.DISCLAIMER}")
    return {"elapsed": round(time.time() - t0, 1)}


def step_show_metrics_v2() -> None:
    """步骤 v2-8：展示 v2（UCI 真实数据）的真实实验指标。"""
    if not os.path.exists(cfg.METRICS_SUMMARY_PATH):
        print("[提示] 尚未生成 v2 指标，请先运行：python main.py --step train")
        return
    with open(cfg.METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)

    print_title("v2 实验指标汇总（UCI 真实数据 + sklearn 主模型）")
    print_kv({
        "数据集": s.get("dataset"),
        "样本数": s.get("n_samples"),
        "独立学生数": s.get("n_students"),
        "高风险比例": s.get("risk_ratio"),
        "主口径特征数": s.get("n_features_main"),
        "主 k": s.get("main_k"),
        "模型后端": s.get("backend"),
        "多数类基线 accuracy": s.get("majority_baseline_accuracy"),
        "标签规则": s.get("label_rule"),
    })

    print_section("① 留一组交叉验证 LOGO（主口径）")
    for name, key in (("KNN", "logo_knn"), ("逻辑回归", "logo_logreg")):
        m = s.get(key, {}).get("metrics", {})
        print(f"  {name:<6}: " + "  ".join(
            f"{k}={m.get(k)}" for k in ("accuracy", "precision", "recall", "f1")))
        print(f"          混淆矩阵 {s.get(key, {}).get('confusion_matrix')}")

    print_section(f"② {s.get('n_splits')} 次分组随机划分（均值 ± 标准差）")
    for name, key in (("KNN", "split10_knn"), ("逻辑回归", "split10_logreg")):
        a = s.get(key, {})
        print(f"  {name:<6}: " + "  ".join(
            f"{k}={a.get(k, {}).get('mean')}±{a.get(k, {}).get('std')}"
            for k in ("accuracy", "precision", "recall", "f1")))

    print_section("③ 课程外推测试（跨分布）")
    for key, rep in s.get("cross_subject_knn", {}).items():
        m = rep.get("metrics", {})
        print(f"  KNN {rep.get('direction')}: " + "  ".join(
            f"{k}={m.get(k)}" for k in ("accuracy", "precision", "recall", "f1")))
        print(f"      训练高风险率 {rep.get('train_risk_ratio')} → 测试 {rep.get('test_risk_ratio')}")

    print_section("④ 早期成绩消融（标签泄漏量化）")
    for row in s.get("ablation_early_grades", []):
        print(f"  {row['setting']:<32s} 特征数={row['n_features']:<3} "
              f"F1={row['logo_f1']} recall={row['logo_recall']} acc={row['logo_accuracy']}")

    print_section("⑤ k 值选择理由")
    print(f"  主 k = {s.get('k_selection', {}).get('chosen_k')}")
    print(f"  {s.get('k_selection', {}).get('reason')}")

    print_section("⑥ 人工复算")
    for name, step in s.get("manual_verification", {}).get("steps", {}).items():
        print("  " + step)
    print(f"  一致性全部通过 = {s.get('manual_verification', {}).get('all_passed')}")

    nc_path = os.path.join(cfg.METRICS_DIR, "nested_cv.json")
    if os.path.exists(nc_path):
        with open(nc_path, "r", encoding="utf-8") as f:
            nc = json.load(f)
        print_section("⑦ 嵌套交叉验证（无偏估计）")
        m = nc["nested_cv"]["metrics"]
        print("  " + "  ".join(f"{k}={m.get(k)}" for k in
                               ("accuracy", "precision", "recall", "f1")))
        print(f"  各折选出的 k 分布：{nc['chosen_k_distribution']}；"
              f"选择偏差 = {nc['selection_bias_f1']:+.4f}")

    print("-" * 68)
    print(cfg.DISCLAIMER)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="基于机器学习的校园学生成绩风险预警与学业帮扶系统（教学实验版）")
    parser.add_argument("--auto", action="store_true", help="自动跑完整流程（无需交互）")
    parser.add_argument("--step", type=str, default=None,
                        choices=["data", "train", "appendix", "nested", "dashboard",
                                 "query", "compare", "metrics",
                                 # v1 步骤（配合 --legacy 使用）
                                 "explain", "assist", "verify", "all"],
                        help="只执行某一步（默认 v2 口径；v1 步骤需加 --legacy）")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="随机种子")
    parser.add_argument("--k", type=int, default=None, help="指定主 k（默认由实验选择）")
    parser.add_argument("--splits", type=int, default=cfg.N_SPLITS, help="随机划分次数")
    parser.add_argument("--n", type=int, default=cfg.N_SAMPLES, help="模拟数据条数（300~500）")
    parser.add_argument("--id", type=str, default=None, help="查询记录编号（如 mat0001 / por0001）")
    parser.add_argument("--top", type=int, default=None, help="批量筛查前 N 名")
    parser.add_argument("--interactive-query", action="store_true", help="交互式录入特征查询")
    parser.add_argument("--demo-query", action="store_true", help="用内置示例特征直接演示查询")
    parser.add_argument("--llm", action="store_true", help="尝试联网调用 LLM 组织建议文字")
    parser.add_argument("--backend", default="auto", choices=["auto", "sklearn", "numpy"],
                        help="模型后端：auto=有 sklearn 就用 sklearn（默认）")
    parser.add_argument("--legacy", action="store_true",
                        help="使用旧流程（模拟数据 + 手写 KNN 为主模型）")
    args = parser.parse_args(argv)

    enable_utf8_console()

    # ---- 默认走 v2 流程（UCI 真实数据 + sklearn 主模型）----
    if not args.legacy:
        if args.auto or args.step == "all":
            run_full_pipeline_v2(seed=args.seed, splits=args.splits, backend=args.backend)
            return
        if args.step:
            if args.step == "data":
                step_uci_data(seed=args.seed)
            elif args.step == "train":
                step_train_eval_v2(backend=args.backend, splits=args.splits, seed=args.seed)
            elif args.step == "appendix":
                step_appendix()
            elif args.step == "nested":
                step_nested_cv()
            elif args.step == "dashboard":
                step_dashboard_v2()
            elif args.step == "query":
                step_query_v2(student_id=args.id, top=args.top,
                              interactive=args.interactive_query, demo=args.demo_query,
                              backend=args.backend, use_llm=args.llm)
            elif args.step == "metrics":
                step_show_metrics_v2()
            elif args.step == "compare":
                step_sim_compare()
            return
        interactive_menu()          # 无参数 → 菜单模式（菜单默认也走 v2）
        return

    # ---- 旧流程（legacy：模拟数据 + 手写 KNN 主模型）----
    if args.auto or args.step == "all":
        run_full_pipeline(seed=args.seed, k=args.k, splits=args.splits, with_llm=args.llm)
        return

    if args.step:
        if args.step == "data":
            step_generate_data(n=args.n, seed=args.seed)
        elif args.step == "train":
            step_train_eval(k=args.k, splits=args.splits, seed=args.seed)
        elif args.step == "explain":
            step_explain(k=args.k)
        elif args.step == "dashboard":
            step_dashboard()
        elif args.step == "query":
            step_query(student_id=args.id, interactive=args.interactive_query,
                       top=args.top, use_llm=args.llm)
        elif args.step == "assist":
            step_assistance(student_id=args.id or "SIM0007", use_llm=args.llm)
        elif args.step == "metrics":
            step_show_metrics()
        elif args.step == "verify":
            step_verify()
        return

    # 无参数 → 菜单模式
    interactive_menu()


if __name__ == "__main__":
    main()
