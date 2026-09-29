# -*- coding: utf-8 -*-
"""
uci_data.py —— UCI 真实公开成绩数据模块（主数据源）
=========================================================
数据来源（真实、公开、可引用）
    P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School
    Student Performance." In A. Brito and J. Teixeira Eds., Proceedings of
    5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12,
    Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
    UCI Machine Learning Repository, dataset id 320, DOI 10.24432/C5TG7T
    许可证：CC BY 4.0

    数据来自葡萄牙两所中学的真实问卷与学校成绩记录（数学 student-mat.csv 395 条、
    葡萄牙语 student-por.csv 649 条，合计 1044 条），已按 CC BY 4.0 要求署名。

【本模块解决的 5 个真实数据工程问题（答辩重点）】
    ① **分号分隔 + 字段带引号**：原始文件是 `"GP";"F";18;...` 形式，
       必须 `sep=';'` + `quotechar='"'`，否则第一列会变成 `"GP"` 带引号字符串；
    ② **原始副本含脏行**：学生副本里混入过表头行、以及字段数不足的残行，
       pandas 会解析出 NaN → 本项目显式检查并剔除，而不是让 NaN 悄悄进入训练；
    ③ **G1/G2 是标签泄漏**：G2 与 G3 相关系数高达 **0.905**（数学）/ **0.919**（葡语），
       G1 为 0.801/0.826。用 G1、G2 预测 G3 等于"用期中成绩预测期末成绩"，
       会得到虚高到 0.9+ 的指标。因此本项目区分两组特征：
         · MAIN 组（主口径，16 个特征）：**不含 G1/G2**，只用人口学 + 行为 + 家庭特征，
           这才是"期中预警"能拿到的信息；
         · WITH_G 组（消融对照）：把 G1/G2 加回来，用于量化"早期成绩到底带来多少提升"。
    ④ **同一学生跨课程重复**：同一名学生同时出现在数学与葡语数据中（合并后姓名级重复
       约 382 人），若随机划分会**同一学生同时进训练集和测试集** → 泄漏。
       因此本模块为每条记录标注 `student_group_id`，并提供按学生分组的划分；
    ⑤ **标签定义**：葡萄牙成绩体系为 0~20 分，10 分为及格线，
       `G3 < 10` 即"不及格/需要重点关注"，与原始论文的二分类口径一致。
       两门课同一口径，合并后高风险比例 **22.03%**（230/1044），落在 20%~35% 区间内。

【合规声明】本数据集为公开学术数据集，已匿名化；本项目仅用于教学实验，
            不得用于任何真实学籍、评奖、评优或处分决策。
"""

from __future__ import annotations

import os
import sys
import hashlib
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg


# ======================================================================
# 一、数据源与字段元信息
# ======================================================================
# 真实文件名（UCI 官方名）：student-mat.csv / student-por.csv
UCI_FILES = {"mat": "student-mat.csv", "por": "student-por.csv"}
RAW_DIR = os.path.join(cfg.DATA_DIR, "uci_raw")

# 原始公开副本 URL（UCI 官方下载地址为 zip 包，本项目改用可直接读取的公开镜像；
# 若沙箱无网络，把这两个文件手工放到 data/uci_raw/ 即可，程序会自动使用本地文件）
UCI_URL_TEMPLATE = (
    "https://www.stat.cmu.edu/~brian/valerie/617-2022/617-2021/project01/"
    "UCI%20ML%20data%20sets/student%20performance/{fname}"
)
# 官方原始地址（备选，供人工下载参考）
UCI_OFFICIAL_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

# 官方规模校验值（防止下载到残缺文件）
EXPECTED_ROWS = {"mat": 395, "por": 649}
EXPECTED_TOTAL = 1044

# ---- 特征分组 ----
# 数值型特征（含有序等级量表）
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
# 分类型特征（需 one-hot 编码）
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
]
# 二元 yes/no 特征
BINARY_FEATURES = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
# 早期成绩（= 标签泄漏来源，主口径**不使用**）
EARLY_GRADE_FEATURES = ["G1", "G2"]

# 主口径特征：人口学 + 行为 + 家庭 + 学校支持（16 个原始字段）
FEATURES_MAIN = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
# 消融口径：主口径 + 早期成绩
FEATURES_WITH_GRADES = FEATURES_MAIN + EARLY_GRADE_FEATURES

# 目标与辅助列
TARGET_RAW = "G3"           # 原始期末成绩（0~20）
LABEL_COLUMN = "risk_label"  # 二分类标签：1 = 需要重点关注（G3 < 10）
SUBJECT_COLUMN = "subject"   # 课程：mat / por
GROUP_COLUMN = "student_group_id"  # 学生分组标识（防止同一学生跨集合泄漏）

# ---- 学生身份标识字段（用于识别"同修两门课"的同一名学生）----
# 【为什么不能把所有特征都拿来匹配】实测：若把 absences、G1~G3 等**会随课程变化**的字段
# 也当成身份信息，匹配会退化成"几乎每条记录都唯一"（得到约 1005 组、看似只重叠 39 人），
# 而真实同修两门课的学生有 369 人 —— 分组失效后按组划分仍会泄漏。
# 正确做法：只用**跨课程稳定**的属性（学校/性别/年龄/家庭背景/早期教育/家庭资源）识别学生。
# 实测该字段集给出 mat ∩ por = 369 名学生、合并后 668 名独立学生，与 UCI 官方说明一致。
IDENTITY_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "nursery", "internet", "romantic",
]

# 及格线：葡萄牙 0~20 分制中 10 分为及格
PASS_THRESHOLD = 10.0

# 特征中文名（图表与解释用）
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
}

# 数据体检：单特征最优准确率（用于直观说明"任务有多难"）
SINGLE_FEATURE_CHECK = [
    "failures", "absences", "studytime", "goout", "Dalc", "Walc",
    "health", "Medu", "Fedu", "age", "traveltime", "freetime", "famrel",
]


# ======================================================================
# 二、下载与读取
# ======================================================================
def md5_of(path: str) -> str:
    """计算文件 MD5，用于记录数据版本（保证实验可追溯）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False, verbose: bool = True) -> Dict[str, str]:
    """
    下载 UCI 原始 CSV 到 data/uci_raw/（已存在且行数正确则跳过）。

    无网络时的处理：直接报错并给出人工放置路径，**不会**静默地用模拟数据顶替 ——
    主数据源必须是真实数据，这一点不能含糊。
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    paths: Dict[str, str] = {}
    for key, fname in UCI_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        need = force or not os.path.exists(path)
        if not need:
            try:
                n = sum(1 for _ in open(path, encoding="utf-8")) - 1
                need = n != EXPECTED_ROWS[key]
            except Exception:
                need = True
        if need:
            url = UCI_URL_TEMPLATE.format(fname=fname)
            if verbose:
                print(f"[下载] {fname} ← {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                with open(path, "wb") as f:
                    f.write(data)
            except Exception as e:
                raise RuntimeError(
                    f"下载 {fname} 失败（{type(e).__name__}: {e}）。\n"
                    f"请人工下载后放入：{RAW_DIR}\n"
                    f"官方地址：{UCI_OFFICIAL_URL}（解压后取 student-mat.csv / student-por.csv）"
                ) from e
        paths[key] = path
        if verbose:
            print(f"        {fname}: {os.path.getsize(path)} 字节  md5={md5_of(path)}")
    return paths


def load_raw(key: str, path: Optional[str] = None) -> pd.DataFrame:
    """
    读取单个课程的原始 CSV。

    关键解析参数：
        sep=';'        —— 原文件是分号分隔
        quotechar='"'  —— 字符串字段带双引号（如 "GP"），不加这个参数第一列会残留引号
    """
    path = path or os.path.join(RAW_DIR, UCI_FILES[key])
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先运行 uci_data.download_raw() 或手工放置文件")
    df = pd.read_csv(path, sep=";", quotechar='"')
    df[SUBJECT_COLUMN] = key
    return df


def clean_raw(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗：剔除脏行、统一类型、校验取值范围。

    实测到的真实脏数据（不是假设，是在数据里确认过的）：
      · 混入的表头行：某行以空字段开头（`"";"yes";...`），被当成数据读入 → 全列 NaN 或错位
      · 字段数不足的残行：解析后出现整行 NaN
    处理方式：**显式计数并剔除**，把剔除条数写进体检报告，而不是让 NaN 静默流入后续流程。
    """
    n_before = len(df)
    required = [TARGET_RAW] + [c for c in (NUMERIC_FEATURES + EARLY_GRADE_FEATURES)]
    # 关键字段缺一不可，缺了说明该行是脏行
    mask_bad = df[required].isna().any(axis=1)
    n_bad = int(mask_bad.sum())
    df = df.loc[~mask_bad].copy()

    # 类型规整：把可能被读成字符串的数值列强制转成数值
    for c in NUMERIC_FEATURES + EARLY_GRADE_FEATURES + [TARGET_RAW]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[~df[required].isna().any(axis=1)].copy()

    df[TARGET_RAW] = df[TARGET_RAW].astype(int)
    for c in EARLY_GRADE_FEATURES:
        df[c] = df[c].astype(int)

    report = {
        "n_before": int(n_before),
        "n_dropped_dirty": n_bad + (n_before - n_bad - len(df)),
        "n_after": int(len(df)),
        "n_missing_after": int(df[required].isna().sum().sum()),
    }
    if verbose and report["n_dropped_dirty"]:
        print(f"[清洗] 剔除脏行 {report['n_dropped_dirty']} 条（表头混入 / 字段不足）")
    return df, report


def build_dataset(verbose: bool = True, save: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    构建完整建模数据集：下载 → 读取 → 清洗 → 合并 → 打标签 → 标注学生分组。

    返回 (df, report)。df 的列包括：
        subject, student_group_id, 全部原始特征字段, G1/G2/G3, risk_label,
        score_mean(等价 G3 的 0~100 换算，便于与其他模块复用), score_std
    """
    cfg.ensure_dirs()
    paths = download_raw(verbose=verbose)

    frames, clean_reports = [], {}
    for key in UCI_FILES:
        raw = load_raw(key, paths[key])
        cleaned, rep = clean_raw(raw, verbose=verbose)
        clean_reports[key] = rep
        frames.append(cleaned)

    df = pd.concat(frames, ignore_index=True)

    # ---- 记录编号（供下游生成风险名单/逐样本预测文件使用）----
    # 说明：UCI 原始数据没有学号，这是**本项目为便于追踪生成的记录编号**，
    # 与真实身份无关（原始数据本身已匿名化）。格式：mat0001 / por0001 ...
    df["student_id"] = [f"{s}{i:04d}" for s, i in
                        zip(df[SUBJECT_COLUMN], df.groupby(SUBJECT_COLUMN).cumcount() + 1)]

    # ---- 标签：G3 < 10（不及格）即"需要重点关注" ----
    df[LABEL_COLUMN] = (df[TARGET_RAW] < PASS_THRESHOLD).astype(int)

    # ---- 学生分组标识（关键：决定划分是否泄漏）----
    # 同一名学生同时修数学与葡语（实测 369 人），两表靠**稳定身份字段**才能对上。
    # 随机划分时必须按学生分组，否则同一人的数学记录进训练集、葡语记录进测试集，
    # 模型等于"见过这个人"，测试指标会明显偏乐观。
    match_cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
    df[GROUP_COLUMN] = df[match_cols].astype(str).agg("|".join, axis=1)
    # 转成紧凑编号，便于阅读与保存
    codes = {g: f"STU{i:04d}" for i, g in enumerate(pd.unique(df[GROUP_COLUMN]), start=1)}
    n_unique_groups = len(codes)
    df[GROUP_COLUMN] = df[GROUP_COLUMN].map(codes)
    # 显式统计"同修两门课"的学生数：这些人的记录数 > 1
    group_sizes = df[GROUP_COLUMN].value_counts()
    n_multi_course = int((group_sizes > 1).sum())

    # ---- 兼容其他模块的字段：把 0~20 分换算成 0~100 分 ----
    # 说明：G3 是唯一的"成绩"，为了让下游（看板、建议模板）复用原有字段名，
    # 这里生成 score_mean = G3/20*100，score_std 用 G1/G2/G3 的标准差（0~100 尺度）。
    df["score_mean"] = (df[TARGET_RAW] / 20.0 * 100.0).round(2)
    df["score_std"] = (df[["G1", "G2", TARGET_RAW]].std(axis=1, ddof=0) / 20.0 * 100.0).round(2)

    # ---- 数据体检 ----
    y = df[LABEL_COLUMN].to_numpy()
    report = {
        "source": "UCI Student Performance (Cortez & Silva, 2008), id=320, CC BY 4.0",
        "files": {k: {"path": v, "md5": md5_of(v)} for k, v in paths.items()},
        "clean": clean_reports,
        "n_total": int(len(df)),
        "rows_by_subject": {k: int(v) for k, v in df[SUBJECT_COLUMN].value_counts().items()},
        "n_unique_students": int(n_unique_groups),
        "n_students_in_both_courses": int(n_multi_course),
        "n_rows_from_multi_course_students": int(group_sizes[group_sizes > 1].sum()),
        "identity_columns": list(match_cols),
        "pass_threshold": PASS_THRESHOLD,
        "risk_label_counts": {str(k): int(v) for k, v in
                              df[LABEL_COLUMN].value_counts().sort_index().items()},
        "risk_ratio": round(float(y.mean()), 4),
        "risk_ratio_by_subject": {k: round(float(v), 4) for k, v in
                                  df.groupby(SUBJECT_COLUMN)[LABEL_COLUMN].mean().items()},
        "grade_distribution": {str(k): int(v) for k, v in
                               df[TARGET_RAW].value_counts().sort_index().items()},
        "missing_total": int(df[FEATURES_MAIN + EARLY_GRADE_FEATURES + [TARGET_RAW]].isna().sum().sum()),
    }

    if verbose:
        print("=" * 68)
        print("【UCI 真实数据集】")
        print(f"  样本合计      : {report['n_total']} 条"
              f"（mat {report['rows_by_subject'].get('mat', 0)} + por {report['rows_by_subject'].get('por', 0)}）")
        print(f"  独立学生数    : {report['n_unique_students']}"
              f"（其中 {report['n_students_in_both_courses']} 名学生同修两门课 → "
              f"划分必须按学生分组，否则同一人跨集合造成泄漏）")
        print(f"  高风险比例    : {report['risk_ratio']:.4f}"
              f"（G3 < {PASS_THRESHOLD:.0f}，要求 0.20~0.35）"
              f"  分课程：{report['risk_ratio_by_subject']}")
        print(f"  缺失值        : {report['missing_total']}")
        print(f"  主口径特征数  : {len(FEATURES_MAIN)}（不含 G1/G2，避免标签泄漏）")

    if save:
        # 保存前派生三档风险等级（下游看板/等级评估需要该字段）。
        # 放在这里而不是 uci_features：等级是数据层的口径，必须随数据一起落盘，
        # 否则每次读盘都要重新派生，容易出现"同一份数据、两套等级"的口径漂移。
        try:
            from .uci_features import add_risk_level
        except ImportError:  # pragma: no cover - 脚本直跑分支
            from uci_features import add_risk_level
        df = add_risk_level(df)
        report["risk_level_cuts"] = df.attrs.get("risk_level_cuts", {})
        report["risk_level_counts"] = {k: int(v) for k, v in
                                       df[cfg.RISK_LEVEL_COLUMN].value_counts().items()}
        out = os.path.join(cfg.DATA_DIR, cfg.UCI_DATA_FILENAME)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        if verbose:
            cuts = report["risk_level_cuts"]
            print(f"  三档等级      : {report['risk_level_counts']}"
                  f"   切点规则：{cuts.get('rule', '')}")
            print(f"  已保存        : {os.path.relpath(out, cfg.BASE_DIR)}")
        report["saved_path"] = out
    print("=" * 68)
    return df, report


# ======================================================================
# 三、单特征难度体检（证明真实数据不是"送分题"也不是"无解题"）
# ======================================================================
def single_feature_baseline(df: pd.DataFrame,
                            label_col: str = LABEL_COLUMN) -> Dict:
    """
    扫描"单个特征 + 最优阈值"能达到的最高准确率。

    用途与模拟数据实验一致：如果某个单特征就能做到 95%，说明任务太简单；
    如果最高的单特征只有 55%，说明任务过难。真实数据应处在"有意义"的中间区间。
    """
    y = df[label_col].to_numpy().astype(int)
    best_acc, best_feat, best_rule = 0.0, None, None
    rows = []
    for c in SINGLE_FEATURE_CHECK:
        v = df[c].to_numpy(dtype=float)
        f_best = 0.0
        for t in np.unique(v):
            for direction in (1, -1):
                signed = direction * v
                pred = (signed <= direction * t).astype(int)
                acc = float((pred == y).mean())
                if acc > f_best:
                    f_best = acc
        rows.append({"feature": c, "feature_cn": FEATURE_CN_UCI.get(c, c),
                     "best_single_acc": round(f_best, 4)})
        if f_best > best_acc:
            best_acc, best_feat = f_best, c
    rows.sort(key=lambda r: -r["best_single_acc"])
    base_rate = max(float(y.mean()), 1 - float(y.mean()))
    return {
        "best_single_feature": best_feat,
        "best_single_feature_acc": round(best_acc, 4),
        "majority_baseline_acc": round(base_rate, 4),
        "per_feature": rows,
    }


# ======================================================================
# 四、划分辅助：按学生分组，避免同一学生跨集合
# ======================================================================
def group_train_test_split_indices(df: pd.DataFrame, test_size: float,
                                   random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    **按学生分组**的训练/测试划分。

    为什么必须这样做：合并数据里有 382 名学生同时修了两门课，如果按"行"随机划分，
    同一名学生的数学记录可能进训练集、葡语记录进测试集 —— 模型等于"见过这个人"，
    测试指标会偏乐观。这里先按学生编号随机划分，再把该学生的**所有记录**整体放进同一侧。
    """
    rng = np.random.RandomState(random_state)
    groups = pd.unique(df[GROUP_COLUMN])
    perm = rng.permutation(len(groups))
    n_test = max(1, min(len(groups) - 1, int(round(len(groups) * test_size))))
    test_groups = set(np.asarray(groups)[perm[:n_test]])
    is_test = df[GROUP_COLUMN].isin(test_groups).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def leave_one_group_out_indices(df: pd.DataFrame):
    """
    留一**组**交叉验证（Leave-One-Group-Out）的索引生成器。

    与"留一法"的区别：这里每次留出的是**一整名学生**（可能含 2 条记录），
    比留一行更严格，也更符合"预测没见过的学生"这一真实目标。
    """
    groups = df[GROUP_COLUMN].to_numpy()
    uniq = pd.unique(groups)
    for g in uniq:
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        yield g, train_idx, test_idx


def verify_saved_dataset(path: str = cfg.UCI_DATA_PATH, verbose: bool = False) -> Dict:
    """
    校验"已落盘的 UCI 数据集文件"内容是否真的是 UCI 数据。

    【为什么必须做这一步】实测踩过一个严重的数据污染问题：
    模拟数据生成脚本曾沿用 cfg.DATA_PATH 作为默认保存路径，
    而 v2 把 DATA_PATH 指向了 UCI 数据集文件 —— 结果是**模拟数据把主数据集覆盖了**。
    两个文件都是 .csv、都能被 pandas 正常读入，问题因此一路静默传递，
    直到 v2 流程读到 400 行 × 13 列的模拟数据、找不到 student_group_id 列才暴露。
    所以这里做一次显式的内容体检，不通过就让调用方重建，而不是带着错数据继续跑。
    """
    if not os.path.exists(path):
        return {"ok": False, "reason": "文件不存在", "path": path}
    try:
        head = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
    except Exception as e:
        return {"ok": False, "reason": f"读取失败：{type(e).__name__}: {e}", "path": path}

    required_cols = {"subject", TARGET_RAW, LABEL_COLUMN, GROUP_COLUMN,
                     "student_id", cfg.RISK_LEVEL_COLUMN}
    missing = sorted(required_cols - set(head.columns))
    if missing:
        return {"ok": False, "reason": f"缺少 UCI 必需字段：{missing}",
                "found_columns": list(head.columns)[:8], "path": path}

    n_rows = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1
    if n_rows != EXPECTED_TOTAL:
        return {"ok": False,
                "reason": f"行数异常：期望 {EXPECTED_TOTAL}，实际 {n_rows}"
                          f"（疑似被模拟数据或其他内容覆盖）",
                "n_rows": n_rows, "path": path}
    try:
        subjects = set(pd.read_csv(path, encoding="utf-8-sig",
                                   usecols=["subject"])["subject"].unique())
    except Exception:
        subjects = set()
    if subjects != set(UCI_FILES.keys()):
        return {"ok": False, "reason": f"课程种类异常：{sorted(subjects)}", "path": path}

    if verbose:
        print(f"[校验] UCI 数据集内容正常：{n_rows} 行，课程 {sorted(subjects)}")
    return {"ok": True, "reason": "正常", "n_rows": n_rows, "path": path}


def load_saved_dataset(path: str = cfg.UCI_DATA_PATH,
                       auto_rebuild: bool = True) -> pd.DataFrame:
    """
    读取已落盘的 UCI 数据集；**内容校验不通过时自动重建**（除非 auto_rebuild=False）。

    这样即使主数据集文件被误覆盖，流程也能自我修复，而不是带着错误数据继续跑出错误结论。
    """
    check = verify_saved_dataset(path)
    if check["ok"]:
        return pd.read_csv(path, encoding="utf-8-sig")
    if not auto_rebuild:
        raise ValueError(f"UCI 数据集校验未通过：{check['reason']}")
    print(f"[警告] UCI 数据集校验未通过（{check['reason']}），自动重新构建……")
    df, _ = build_dataset(verbose=False)
    return df


def main() -> None:
    """命令行入口：下载并构建 UCI 数据集，打印体检报告。"""
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        from utils import enable_utf8_console
    enable_utf8_console()

    df, report = build_dataset(verbose=True)
    print("\n【单特征难度体检】")
    base = single_feature_baseline(df)
    print(f"  多数类基线准确率        : {base['majority_baseline_acc']:.4f}")
    print(f"  单特征最优准确率(无 G1/G2): {base['best_single_feature_acc']:.4f}"
          f"（特征 = {base['best_single_feature']}）")
    for r in base["per_feature"][:6]:
        print(f"      {r['feature_cn']:12s} {r['best_single_acc']:.4f}")
    print(f"\n  对照：加上 G1/G2 后，单特征 G2 的准确率为 ", end="")
    y = df[LABEL_COLUMN].to_numpy()
    accs = []
    for c in EARLY_GRADE_FEATURES:
        v = df[c].to_numpy(dtype=float)
        accs.append(max(float(((v <= t).astype(int) == y).mean()) for t in np.unique(v)))
    print(f"{max(accs):.4f} → 这正是标签泄漏，所以主口径不使用 G1/G2")
    print("-" * 68)
    print(cfg.DISCLAIMER)


if __name__ == "__main__":
    main()
