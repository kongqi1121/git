# -*- coding: utf-8 -*-
"""
config.py —— 全局配置模块
=========================================================
本文件把"项目口径"集中到一处，避免各模块各写一套参数导致的
指标不可比、复现困难等问题。所有其他模块都从这里导入常量。

【数据源口径（v2，重要）】
    · 主数据  ：**UCI Student Performance 真实公开数据集**（Cortez & Silva, 2008，
                dataset id 320，CC BY 4.0），1044 条真实学生记录（含两门课程）；
    · 对照数据：本项目自建的**模拟/脱敏数据**（400 条），仅用于鲁棒性对照实验，
                验证"类别不平衡 + 特征重叠 + 标签噪声"对模型的影响。
    两套数据、同一套评估代码，是本项目防"指标虚高"的核心设计。

【合规声明】UCI 数据为已匿名化的公开学术数据集，本项目按其许可（CC BY 4.0）署名使用；
    所有输出仅用于教学实验，**不得用于任何真实学籍、评奖、评优或处分决策**。

设计原则：
1. 所有随机过程都必须显式传入 random_state，保证结果可复现；
2. 特征列、标签列、k 值候选、风险等级阈值都只在这里定义一次；
3. 不引入任何外部依赖（除标准库），方便被任意模块安全导入。
"""

from __future__ import annotations

import os

# ----------------------------------------------------------------------
# 一、路径配置
# ----------------------------------------------------------------------
# SRC_DIR  = .../project/src
# BASE_DIR = .../project      （即 src 的上一级）
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")            # 数据目录
RESULTS_DIR = os.path.join(BASE_DIR, "results")      # 结果目录
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")   # 图表目录
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")   # 指标目录
MODELS_DIR = os.path.join(RESULTS_DIR, "models")     # 模型目录

# ---- 主数据集：UCI 真实数据 ----
UCI_DATA_FILENAME = "uci_student_performance.csv"
UCI_DATA_PATH = os.path.join(DATA_DIR, UCI_DATA_FILENAME)
UCI_RAW_DIR = os.path.join(DATA_DIR, "uci_raw")       # UCI 原始 CSV 存放目录

# ---- 对照数据集：模拟/脱敏数据 ----
SIM_DATA_FILENAME = "scores_simulated.csv"
SIM_DATA_PATH = os.path.join(DATA_DIR, SIM_DATA_FILENAME)
SIM_NEW_BATCH_PATH = os.path.join(DATA_DIR, "scores_new_batch.csv")
SEED_CSV_PATH = os.path.join(DATA_DIR, "scores.csv")  # 早期 5 人种子数据

# 数据集注册表：主/对照两套数据用同一套代码路径处理
DATASETS = {
    "uci": {
        "name": "UCI 真实公开数据集",
        "path": UCI_DATA_PATH,
        "role": "主数据",
        "description": "UCI Student Performance（Cortez & Silva, 2008，id=320，CC BY 4.0）",
    },
    "sim": {
        "name": "自建模拟/脱敏数据集",
        "path": SIM_DATA_PATH,
        "role": "鲁棒性对照",
        "description": "本项目随机生成的模拟数据（400 条，构造类别不平衡与特征重叠）",
    },
}
DEFAULT_DATASET = "uci"

# ---- 向后兼容别名（旧代码用 DATA_PATH 指向主数据集）----
DATA_PATH = UCI_DATA_PATH
NEW_BATCH_PATH = SIM_NEW_BATCH_PATH

# 模型与统计参数文件
MODEL_PATH = os.path.join(MODELS_DIR, "models.npz")          # 训练好的模型参数
PREPROCESS_STATS_PATH = os.path.join(MODELS_DIR, "preprocess_stats.json")
PREDICTIONS_PATH = os.path.join(METRICS_DIR, "full_predictions.csv")

# 指标输出文件
METRICS_SUMMARY_PATH = os.path.join(METRICS_DIR, "metrics_summary.json")
METRICS_TABLE_PATH = os.path.join(METRICS_DIR, "metrics_table.csv")
K_SCAN_PATH = os.path.join(METRICS_DIR, "k_scan.csv")
SPLIT_DETAIL_PATH = os.path.join(METRICS_DIR, "split_details.csv")
NEW_BATCH_TABLE_PATH = os.path.join(METRICS_DIR, "new_batch_metrics.csv")
IMPORTANCE_PATH = os.path.join(METRICS_DIR, "feature_importance.csv")
VERIFY_PATH = os.path.join(METRICS_DIR, "manual_verification.json")

# v2 新增产物
APPENDIX_PATH = os.path.join(METRICS_DIR, "appendix_handwritten_knn.json")
ABLATION_PATH = os.path.join(METRICS_DIR, "ablation_early_grades.csv")
SIM_COMPARE_PATH = os.path.join(METRICS_DIR, "sim_vs_uci_comparison.csv")
UCI_REPORT_PATH = os.path.join(METRICS_DIR, "uci_dataset_report.json")
GROUP_CV_PATH = os.path.join(METRICS_DIR, "group_cv_details.csv")


def ensure_dirs() -> None:
    """确保所有输出目录存在（幂等，可重复调用）。"""
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, UCI_RAW_DIR):
        os.makedirs(d, exist_ok=True)


def dataset_path(key: str = DEFAULT_DATASET) -> str:
    """取某个数据集的路径。"""
    if key not in DATASETS:
        raise KeyError(f"未知数据集 '{key}'，可选：{list(DATASETS)}")
    return str(DATASETS[key]["path"])


# ----------------------------------------------------------------------
# 二、字段配置
# ----------------------------------------------------------------------
# ---- 模拟数据的建模特征（保留，供对照实验复用）----
FEATURE_COLUMNS = [
    "python",                # Python 程序设计成绩
    "ml",                    # 机器学习成绩
    "cv",                    # 计算机视觉成绩
    "attendance_rate",       # 出勤率 0~1
    "homework_submit_rate",  # 作业提交率 0~1
    "self_study_hours",      # 每周自习时长 0~10 小时
    "score_mean",            # 三科平均分（派生特征）
    "score_std",             # 三科标准差（派生特征，衡量偏科程度）
]
SCORE_COLUMNS = ["python", "ml", "cv"]

# ---- UCI 真实数据的建模特征（主口径）----
# 说明：**不含 G1/G2**。G2 与 G3 的相关系数高达 0.905（数学）/0.919（葡语），
# 用它们预测期末成绩属于标签泄漏，指标会虚高到 0.9+。主口径只用
# 人口学 + 学习行为 + 家庭与学校支持特征，即"期中预警真正拿得到的信息"，
# 因此主口径的列名需要在 uci_data 中做 one-hot 编码后才能确定。
UCI_NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
UCI_CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
]
UCI_BINARY_FEATURES = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
UCI_EARLY_GRADE_FEATURES = ["G1", "G2"]   # 仅用于消融对照，主口径禁用

# UCI 特征中文名（图表与解释用）
FEATURE_CN_UCI = {
    "age": "年龄",
    "Medu": "母亲学历",
    "Fedu": "父亲学历",
    "traveltime": "上学路程时间",
    "studytime": "每周学习时间",
    "failures": "历史挂科次数",
    "famrel": "家庭关系质量",
    "freetime": "课后自由时间",
    "goout": "与朋友外出频率",
    "Dalc": "工作日饮酒",
    "Walc": "周末饮酒",
    "health": "健康状况",
    "absences": "缺课次数",
    "G1": "第一学期成绩",
    "G2": "第二学期成绩",
    "G3": "期末成绩",
    "school": "学校",
    "sex": "性别",
    "address": "居住地类型",
    "famsize": "家庭规模",
    "Pstatus": "父母同住状况",
    "Mjob": "母亲职业",
    "Fjob": "父亲职业",
    "reason": "择校原因",
    "guardian": "监护人",
    "schoolsup": "学校额外支持",
    "famsup": "家庭学业支持",
    "paid": "付费补习",
    "activities": "课外活动",
    "nursery": "是否上过幼儿园",
    "higher": "是否想读高等教育",
    "internet": "家里有无网络",
    "romantic": "是否有恋爱关系",
    "prior_avg": "历史平均成绩(G1/G2均值)",
    "subject": "课程",
}

# 特征中文名（模拟数据）
FEATURE_CN = {
    "python": "Python程序设计成绩",
    "ml": "机器学习成绩",
    "cv": "计算机视觉成绩",
    "attendance_rate": "出勤率",
    "homework_submit_rate": "作业提交率",
    "self_study_hours": "每周自习时长",
    "score_mean": "三科平均分",
    "score_std": "三科成绩波动(偏科程度)",
}

# 特征单位说明，用于生成可读的解释文本
FEATURE_UNIT = {
    "python": "分",
    "ml": "分",
    "cv": "分",
    "attendance_rate": "",
    "homework_submit_rate": "",
    "self_study_hours": "小时/周",
    "score_mean": "分",
    "score_std": "分",
}

LABEL_COLUMN = "risk_label"   # 二分类标签：0 正常 / 1 需要重点关注
RISK_LEVEL_COLUMN = "risk_level"  # 三档风险等级：低 / 中 / 高

RISK_LABEL_CN = {0: "正常", 1: "需要重点关注"}
RISK_LEVELS = ["低", "中", "高"]

# ----------------------------------------------------------------------
# 三、数据生成配置（模拟对照数据集）
# ----------------------------------------------------------------------
RANDOM_SEED = 42          # 全局默认随机种子
N_SAMPLES = 400           # 模拟数据集样本量（要求 300~500）
N_NEW_SAMPLES = 100       # 模拟数据的"新样本"测试集规模
NEW_BATCH_SEED = 2025     # 新样本使用不同种子，避免与训练集重复
RISK_RATIO_TARGET = 0.28  # 模拟数据的目标高风险比例

# ----------------------------------------------------------------------
# 四、模型与评估配置
# ----------------------------------------------------------------------
# ★ 主模型后端：'auto' = 有 sklearn 用 sklearn，否则回退手写实现
#   本项目要求主模型由 sklearn 承担，手写 KNN 作为附录做等价性验证。
MODEL_BACKEND = os.environ.get("DSH_MODEL_BACKEND", "auto")

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21, 31]  # k 值对比候选
MAIN_K = 31                                      # 主 k（由实验扫描确定，此处为默认值）

# 风险等级划分阈值（基于 predict_proba 输出的高风险概率）
RISK_LEVEL_THRESHOLDS = {"low": 0.30, "high": 0.70}

N_SPLITS = 10             # 10 次随机划分的次数
TEST_SIZE = 0.3           # 每次随机划分的测试集比例
N_IMPORTANCE_REPEATS = 3  # 置换重要性重复次数（取均值，降低随机波动）

# sklearn 模型超参数
SKLEARN_KNN_PARAMS = {
    "n_neighbors": MAIN_K,
    "weights": "uniform",    # 与手写"多数投票"口径一致，便于等价性对照
    "metric": "euclidean",   # 欧氏距离
    # 【为什么 n_jobs=1 而不是 -1】
    # ① 本项目数据量只有 1044 条，sklearn 在 n_jobs=-1 时会用 joblib 起线程池，
    #    线程池需要创建命名管道；在受限沙箱（如本项目的执行环境）里会被拒绝
    #    （实测 PermissionError: [WinError 5] 拒绝访问），导致完全无法运行；
    # ② 数据量这么小，多线程的调度开销本身就大于并行收益（实测单线程更快）。
    # 因此固定 n_jobs=1：既安全又可移植，指标完全不受影响（KNN 结果与线程数无关）。
    "n_jobs": 1,
}
SKLEARN_LOGREG_PARAMS = {
    "max_iter": 3000,
    "C": 1.0,                # 与手写版 L2 强度对应（C = 1/lambda）
    "solver": "lbfgs",
    "class_weight": None,    # 主口径不做过采样/加权，保持与手写版一致
}

# 手写逻辑回归（对照组）超参数
LOGREG_LR = 0.15          # 学习率
LOGREG_EPOCHS = 1200      # 迭代轮数
LOGREG_L2 = 1e-3          # L2 正则强度

# ---- UCI 数据集专有评估口径 ----
# 留一组交叉验证（Leave-One-Group-Out）：每次留出一整名学生（可能含两门课记录）
USE_GROUP_CV = True

# ----------------------------------------------------------------------
# 五、中文绘图配置
# ----------------------------------------------------------------------
# Windows 常见中文字体，按优先级尝试；缺失时自动回退，不影响运行。
CN_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "DejaVu Sans"]

# 免责声明（v2：主数据为真实公开数据，措辞必须准确）
DISCLAIMER = (
    "本系统为教学实验原型。主数据为 UCI 公开学术数据集"
    "（Student Performance, Cortez & Silva 2008, CC BY 4.0，已匿名化），"
    "对照数据为本项目生成的模拟数据。系统输出仅供教学演示与人工复核参考，"
    "不得用于任何真实学籍、评奖、评优或处分决策。"
)

# 数据来源引用（写进报告与界面，满足 CC BY 4.0 署名要求）
DATA_CITATION = (
    "Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School "
    "Student Performance. In Proceedings of 5th FUTure BUsiness TEChnology Conference "
    "(FUBUTEC 2008), pp. 5-12, Porto, Portugal. UCI ML Repository, id 320, "
    "DOI 10.24432/C5TG7T. Licensed under CC BY 4.0."
)


def project_info() -> str:
    """返回项目信息摘要字符串，供 main.py 头部展示。"""
    return (
        "项目：基于机器学习的校园学生成绩风险预警与学业帮扶系统\n"
        f"主数据：{DATASETS['uci']['description']}（1044 条真实学生记录）\n"
        f"对照数据：{DATASETS['sim']['description']}\n"
        f"主模型：sklearn KNN（k={MAIN_K}，欧氏距离 + 多数投票 + predict_proba）\n"
        "对照模型：sklearn 逻辑回归\n"
        "附录亮点：numpy 手写 KNN（与 sklearn 逐样本等价性验证通过）\n"
        f"声明：{DISCLAIMER}"
    )


if __name__ == "__main__":
    # 直接运行本文件时，打印当前配置，便于检查路径是否正确
    print(project_info())
    print("-" * 68)
    print("数据集注册表：", {k: v["role"] for k, v in DATASETS.items()})
    print("模拟数据特征列：", FEATURE_COLUMNS)
    print("UCI 主口径特征：数值", len(UCI_NUMERIC_FEATURES), "+ 分类",
          len(UCI_CATEGORICAL_FEATURES), "+ 二元", len(UCI_BINARY_FEATURES),
          "（不含 G1/G2，避免标签泄漏）")
    print("k 候选：", K_CANDIDATES)
    print("风险等级阈值：", RISK_LEVEL_THRESHOLDS)
    print("模型后端设置：", MODEL_BACKEND)
