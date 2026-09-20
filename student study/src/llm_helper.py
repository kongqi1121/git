# -*- coding: utf-8 -*-
"""
llm_helper.py —— 学业帮扶建议生成（LLM 只做文字组织）
=========================================================
辅助功能 4：**依据真实预测结果生成学业帮扶建议**。

============= 本模块的边界（非常重要，答辩必讲） =============
LLM 允许做：
    · 把结构化预测结果整理成通顺的中文段落；
    · 依据"模型已给出的风险等级和关键特征"生成帮扶建议文字；
    · 生成实验报告初稿、回答项目使用方法。
LLM 绝对不允许做：
    · 修改预测类别、概率、风险等级；
    · 把高风险改成低风险（或反之）；
    · 编造不存在的实验数据、指标、学生；
    · 替代 KNN 这个核心 AI 功能。
本项目的工程化保障手段（不是靠"提示词里写一句不许改"就算数）：
    1. 概率、等级、近邻投票等**数值一律由模型产出后在本地拼接**，
       只有"建议文字"这一段允许 LLM 参与；
    2. 提供 `validate_no_conclusion_change()` 做**事后校验**：
       检查生成文本里的概率数字与风险等级是否与事实一致，
       只要出现"高风险被写成低风险""概率被改写"，立即回退到本地模板；
    3. 无网络 / 无 API Key 时（本机即为此情况）自动使用本地模板，
       功能完全可用，**不联网也能答辩**。
============================================================

【本机实测】未配置任何 LLM API Key，因此默认走本地模板分支。
若需联网调用，设置环境变量即可，代码会自动切换（见 call_llm_api 说明）。
"""

from __future__ import annotations

import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - 脚本直跑分支
    sys_path = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(0, sys_path)
    import config as cfg


# ======================================================================
# 一、从预测结果构造"事实包"（facts）
# ======================================================================
def build_facts(student_id: str,
                proba_risk: float,
                risk_level: str,
                top_contributions: Optional[Sequence[Dict]] = None,
                neighbors: Optional[Sequence[Dict]] = None,
                raw_features: Optional[Dict] = None,
                thresholds: Optional[Dict[str, float]] = None,
                k: Optional[int] = None,
                prediction_source: Optional[str] = None) -> Dict:
    """
    把一次预测的**全部事实**整理成结构化字典。

    这是本模块唯一的"数据入口"：建议文字只能引用这里已有的字段，
    从源头上避免编造（例如没有的特征值在 facts 里就是 None，模板不会去写它）。

    参数 prediction_source 由调用方给出（v2 主模型是 sklearn，若仍默认写
    "手写 KNN" 就会误导读者），不传时才回退到旧描述。
    """
    th = dict(cfg.RISK_LEVEL_THRESHOLDS if thresholds is None else thresholds)
    facts = {
        "student_id": str(student_id),
        "proba_risk": round(float(proba_risk), 4),
        "risk_level": str(risk_level),
        "thresholds": {"低": f"< {th['low']}", "中": f"{th['low']} ~ {th['high']}", "高": f"> {th['high']}"},
        "k": None if k is None else int(k),
        "n_neighbors_risk": None if not neighbors else int(sum(1 for n in neighbors if n["label"] == 1)),
        "n_neighbors": None if not neighbors else int(len(neighbors)),
        "neighbor_ids": [] if not neighbors else [str(n["student_id"]) for n in neighbors
                                                  if n.get("student_id")],
        "top_contributions": [],
        "raw_features": raw_features or {},
        "prediction_source": prediction_source or
        "numpy 手写 KNN（欧氏距离 + 多数投票 + 近邻高风险占比概率）",
        "disclaimer": cfg.DISCLAIMER,
    }
    for it in (top_contributions or []):
        # 原始特征值：优先用贡献项里带的；没有就从 raw_features 里按特征名补齐
        v = it.get("value_raw")
        if v is None and raw_features:
            v = raw_features.get(it["feature"])
        facts["top_contributions"].append({
            "feature": it["feature"],
            "feature_cn": it["feature_cn"],
            "value_raw": v,
            "risk_gap": it.get("risk_gap"),
            "direction": it.get("direction") or ("推向高风险" if (it.get("risk_gap") or 0) > 0
                                                 else "相对优势"),
        })
    return facts


# ======================================================================
# 二、本地模板生成（无 API 时的默认分支，功能完整可用）
# ======================================================================
# 依据"关键特征"给出具体、可执行的帮扶动作。
# 注意：这些建议由**特征阈值规则**触发，触发条件是模型给出的贡献特征，
#      因此建议与预测结论严格对应，不存在"随口推荐"。
ACTION_LIBRARY: Dict[str, Dict[str, str]] = {
    "score_mean": {
        "trigger": "三科平均分偏低",
        "actions": "① 与任课教师核对知识薄弱章节，制定 2 周补强清单；"
                   "② 安排同班学业优秀学生结对，每周固定 2 次答疑；"
                   "③ 下次小测前提交一份自测错题整理。",
    },
    "score_std": {
        "trigger": "三科成绩波动大（存在明显偏科）",
        "actions": "① 确认最弱科目的具体卡点（环境配置/数学基础/编程实践）；"
                   "② 为该科目单独安排实验课加练；"
                   "③ 允许用强科项目带动弱科，例如用 Python 实现机器学习作业以巩固基础。",
    },
    "python": {
        "trigger": "Python 程序设计成绩偏低",
        "actions": "① 补练基础语法与调试技能（建议完成 10 道循环/函数/文件读写练习）；"
                   "② 课后跟做课堂示例代码，逐行加注释提交。",
    },
    "ml": {
        "trigger": "机器学习成绩偏低",
        "actions": "① 复习数据预处理与模型评估指标（准确率/精确率/召回率的含义与区别）；"
                   "② 用本项目的手写 KNN 代码做一次完整跑通练习，理解距离与投票机制。",
    },
    "cv": {
        "trigger": "计算机视觉成绩偏低",
        "actions": "① 补齐图像基础操作（读取、尺寸变换、通道）练习；"
                   "② 结合课程实验报告模板，重做一次图像分类小实验。",
    },
    "attendance_rate": {
        "trigger": "出勤率偏低",
        "actions": "① 先了解缺勤原因（作息、兼职、健康、心理状态），必要时转介辅导员或心理中心；"
                   "② 与本人约定到课率目标（如两周内不低于 90%），并做签到跟踪。",
    },
    "homework_submit_rate": {
        "trigger": "作业提交率偏低",
        "actions": "① 明确每次作业的截止时间与提交方式，采用“课前提醒 + 课后确认”；"
                   "② 对欠交作业设置补交窗口，避免学生因欠账过多而放弃。",
    },
    "self_study_hours": {
        "trigger": "每周自习时长不足",
        "actions": "① 协助制定周计划表，把自习时间固定到具体时段与地点；"
                   "② 建议参加晚自习或学习小组，用同伴监督提高执行力。",
    },
}


def _fmt_feature_value(feature: str, v) -> str:
    """
    把特征值格式化成人类可读形式。

    同时覆盖两套数据的字段名：
      · 模拟数据：attendance_rate/homework_submit_rate（比率）、self_study_hours（小时）、各科成绩（分）
      · UCI 真实数据：yes/no 二元特征、分类指示特征（如 school_MS）、等级量表（studytime 等）
    """
    if v is None:
        return "（数据缺失）"
    if feature in ("attendance_rate", "homework_submit_rate"):
        return f"{float(v) * 100:.1f}%"
    if feature == "self_study_hours":
        return f"{float(v)} 小时/周"
    # UCI 分类指示特征：值为 0/1
    if "_" in feature and feature.split("_", 1)[0] in cfg.FEATURE_CN_UCI:
        return "是" if float(v) >= 0.5 else "否"
    if feature in cfg.FEATURE_CN_UCI:
        # UCI 数值特征：等级量表与计数分开写，避免把"缺课 12 次"写成"12 分"
        scale = {"studytime", "traveltime", "famrel", "freetime", "goout",
                 "Dalc", "Walc", "health", "Medu", "Fedu"}
        if feature in scale:
            return f"等级 {float(v):g}"
        if feature == "absences":
            return f"{float(v):g} 次"
        if feature == "failures":
            return f"{float(v):g} 次"
        if feature == "age":
            return f"{float(v):g} 岁"
        return f"{float(v):g}"
    try:
        return f"{float(v)} 分"
    except (TypeError, ValueError):
        return str(v)


def generate_summary_text(facts: Dict) -> str:
    """
    事实摘要：把预测结果整理成通顺中文（**数值全部来自 facts，不做任何加工**）。
    这段文本可以直接展示给辅导员，也可以作为 LLM 的输入素材。
    """
    lines: List[str] = []
    sid = facts["student_id"]
    p = facts["proba_risk"]
    lvl = facts["risk_level"]
    lines.append(f"【预测结论】学生 {sid} 的高风险概率为 {p:.4f}，风险等级判定为“{lvl}”。")
    if facts.get("n_neighbors"):
        lines.append(
            f"【判定依据】在其最近的 {facts['n_neighbors']} 位相似同学（k 近邻）中，"
            f"有 {facts['n_neighbors_risk']} 位属于“需要重点关注”，"
            f"因此高风险概率为 {facts['n_neighbors_risk']}/{facts['n_neighbors']} ≈ {p:.4f}。"
        )
    else:
        lines.append("【判定依据】本次预测未附带近邻明细。")

    top = facts.get("top_contributions") or []
    if top:
        lines.append("【关键特征】按“与高风险组的接近程度（贡献差距 = 到低风险组距离 - 到高风险组距离）”"
                     "从大到小排列，前 "
                     f"{len(top)} 项为（差距 > 0 表示推向高风险，< 0 表示相对优势）：")
        for i, it in enumerate(top, 1):
            raw = _fmt_feature_value(it["feature"], it.get("value_raw"))
            lines.append(f"    {i}. {it['feature_cn']} = {raw}"
                         f"（贡献差距 {it['risk_gap']:+.4f}，{it['direction']}）")
    else:
        lines.append("【关键特征】本次预测未附带特征贡献明细。")

    lines.append(f"【等级口径】低：概率 {facts['thresholds']['低']}；"
                 f"中：{facts['thresholds']['中']}；高：{facts['thresholds']['高']}（阈值可在配置中调整）。")
    lines.append(f"【模型来源】{facts['prediction_source']}。")
    lines.append(f"【使用声明】{facts['disclaimer']}")
    return "\n".join(lines)


def generate_suggestions_local(facts: Dict, top_n: int = 4) -> List[str]:
    """
    本地模板生成帮扶建议：**建议只由模型给出的关键特征触发**。

    输出是一个字符串列表，每条对应一项可执行动作，方便辅导员逐条勾选落实。
    """
    lvl = facts["risk_level"]
    p = facts["proba_risk"]
    suggestions: List[str] = []

    # 1) 等级对应的总体处置节奏（规则来自 knn_numpy.risk_level_to_action）
    overall = {
        "低": "总体处置：常规跟进。保持现有学习状态，无需额外干预；如后续成绩或出勤下降，重新评估。",
        "中": "总体处置：建议关注。两周内安排一次 15 分钟沟通，了解具体课程难点与个人困难。",
        "高": "总体处置：重点帮扶。纳入重点关注名单，指定固定跟进周期（建议每两周一次），并同步任课教师。",
    }.get(lvl)
    if overall:
        suggestions.append(overall)

    # 2) 由"贡献最大的几个特征"生成具体动作
    #    关键：**只对"推向高风险"的特征给补强动作**。
    #    如果某特征其实是在"拉向正常"（例如某科成绩已经很高），还去建议"补基础语法"
    #    就自相矛盾了 —— 早期版本犯过这个错，这里显式过滤，并单独说明。
    triggered: List[str] = []
    all_top = list(facts.get("top_contributions") or [])
    risk_pushing = [it for it in all_top if (it.get("risk_gap") or 0) > 0]
    protective = [it for it in all_top if (it.get("risk_gap") or 0) <= 0]

    if lvl == "低":
        # 低风险学生不安排"补强动作"：个别特征相对偏弱只是群体内的相对短板，
        # 不足以构成帮扶理由（这属于"相对差距"与"整体结论"的区分，答辩会问到）。
        suggestions.append(
            "具体帮扶动作：该生已被判定为低风险，不安排针对性补强动作。"
            "如任课教师另有观察（例如课堂表现、作业质量），再单独沟通。"
        )
        if risk_pushing:
            names = "、".join(it["feature_cn"] for it in risk_pushing[:top_n])
            suggestions.append(
                f"说明：模型给出的相对短板是 {names}，但该生整体风险仍为低，"
                f"这些只是群体内的相对差距，不作为干预依据。"
            )
    else:
        for it in risk_pushing[:top_n]:
            lib = ACTION_LIBRARY.get(it["feature"])
            if lib is None:
                continue
            v = _fmt_feature_value(it["feature"], it.get("value_raw"))
            triggered.append(
                f"针对 {it['feature_cn']}（当前 {v}，方向：{it['direction']}）："
                f"触发条件“{lib['trigger']}”。建议动作：{lib['actions']}"
            )
        if triggered:
            suggestions.append("具体帮扶动作（依据模型给出的、确实在推高风险的关键特征生成，非通用套话）：")
            suggestions.extend("    " + t for t in triggered)
        else:
            suggestions.append("具体帮扶动作：本次没有识别出“正在推高风险”的特征"
                               "（即该生各项特征相对其近邻并不处于劣势），"
                               "建议按上面的总体处置节奏跟进，并核查是否存在成绩之外的因素"
                               "（健康、家庭、心理状态等）。")

    if protective:
        names = "、".join(it["feature_cn"] for it in protective[:top_n])
        suggestions.append(
            f"相对优势（供沟通时参考，不要当作问题去纠正）：{names} 等特征相对该生所属群体"
            f"并不处于高风险方向，可与本人确认这些方面是否可以继续保持。"
        )

    # 3) 与"高风险概率"直接相关的复核提示（避免使用者把概率当精确值）
    k_val = facts.get("k")
    if k_val is not None and int(k_val) <= 2:
        # k 很小时概率只能取 0 / 1 这类极端值，必须明确提示，否则使用者会误读为"确定会出事"
        suggestions.append(
            f"复核提示：本模型 k={k_val}，因此概率只能取 "
            f"{'/'.join(['0'] + [f'{i}/{k_val}' for i in range(1, int(k_val) + 1)])} "
            f"这几种离散值，**没有中间档位**。这意味着“概率 1.0”只代表"
            f"“最近的那 {k_val} 位同学恰好都属于需要重点关注”，"
            f"不代表该生有 100% 的概率出问题，必须结合人工了解再定性。"
        )
    elif p > cfg.RISK_LEVEL_THRESHOLDS["high"]:
        suggestions.append(
            f"复核提示：概率 {p:.4f} 高于高等级阈值 {cfg.RISK_LEVEL_THRESHOLDS['high']}，"
            f"但在 KNN 中概率只能是 k 的分数形式（分母为 k），"
            f"因此它反映的是“与其最相似的 k 位同学的构成比例”，请结合人工了解后再定性。"
        )
    elif p >= cfg.RISK_LEVEL_THRESHOLDS["low"]:
        suggestions.append(
            f"复核提示：概率 {p:.4f} 落在中风险区间，属于“证据不足以直接定性”的边界情形，"
            f"建议以沟通了解为主，不要直接按高风险处理。"
        )

    suggestions.append(f"边界声明：{facts['disclaimer']}本建议由规则模板根据模型输出生成，"
                       f"不构成任何学籍、评奖或处分依据。")
    return suggestions


# ======================================================================
# 三、可选：联网调用 LLM（有 Key 时才用，只传事实）
# ======================================================================
SYSTEM_PROMPT = (
    "你是学业帮扶文字助手。你只允许做一件事：把给定的预测事实整理成通顺的中文段落，"
    "并依据这些事实写帮扶建议。\n"
    "硬性约束：\n"
    "1. 不得修改、四舍五入之外的任何数值：概率、近邻人数、风险等级必须原文照抄；\n"
    "2. 不得把高风险写成中/低风险，不得新增未给出的学生信息；\n"
    "3. 不得编造任何实验数据、指标或学生；\n"
    "4. 不得输出与预测结论矛盾的判断；\n"
    "5. 只输出正文，不要解释你的工作过程。"
)


def call_llm_api(facts: Dict, user_extra: str = "", timeout: int = 20) -> Tuple[bool, str]:
    """
    可选的联网 LLM 调用（仅在配置了环境变量时启用，本机默认不启用）。

    环境变量：
        DSH_LLM_API_KEY   必需，API Key
        DSH_LLM_BASE_URL  可选，默认 https://api.openai.com/v1/chat/completions
        DSH_LLM_MODEL     可选，默认 gpt-4o-mini

    工程约束：**只传 facts（结构化事实）+ 用户补充说明**，
    不传原始自由文本、不允许模型访问数据文件，从源头消除"模型自己造数"的可能。
    返回值 (是否成功, 文本)。
    """
    api_key = os.environ.get("DSH_LLM_API_KEY", "").strip()
    if not api_key:
        return False, "未配置 DSH_LLM_API_KEY，已自动使用本地模板生成（功能不受影响）。"

    base_url = os.environ.get("DSH_LLM_BASE_URL",
                              "https://api.openai.com/v1/chat/completions").strip()
    model = os.environ.get("DSH_LLM_MODEL", "gpt-4o-mini").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "以下是模型的预测事实（JSON，禁止修改其中任何数值）：\n"
                + json.dumps(facts, ensure_ascii=False, indent=2)
                + (f"\n\n补充要求：{user_extra}" if user_extra else "")
                + "\n\n请输出：① 一段预测结论说明；② 3~5 条具体帮扶建议。"},
        ],
        "temperature": 0.3,   # 低温：减少自由发挥，降低改写数值的风险
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return True, text
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError,
            json.JSONDecodeError, TimeoutError) as e:
        # 网络/接口异常绝不能影响主流程：直接回退本地模板
        return False, f"调用 LLM 失败（{type(e).__name__}: {e}），已回退本地模板生成。"


def validate_no_conclusion_change(text: str, facts: Dict) -> Dict:
    """
    **结论防篡改校验**（本项目对 LLM 边界的工程化落实）。

    校验三件事：
      1. 文本中若出现概率数字，必须与 facts 中的概率一致（容差 1e-4）；
      2. 文本中不得把该生风险等级说成与 facts 不一致的等级；
      3. 文本中不得出现"低风险"这类与"高风险"结论矛盾的改写
         （当 facts.risk_level == "高" 时，禁止出现"低风险""风险不高"等表述）。
    返回 {passed, issues}；不通过时调用方应回退本地模板。
    """
    issues: List[str] = []
    p = float(facts["proba_risk"])
    lvl = facts["risk_level"]
    th = facts.get("thresholds", {})

    # ---- 校验 1：概率数字是否被改写 ----
    # 思路：文本里任何"长得像概率"的数字（0.xxx），要么等于模型给出的概率，
    #       要么是等级阈值（配置里公开的口径，允许出现），否则一律判为疑似篡改。
    # 允许的阈值数字（低/中/高阈值本身，用于说明等级口径，不算篡改）
    allowed_numbers = {0.0, 0.5, 1.0, round(float(cfg.RISK_LEVEL_THRESHOLDS["low"]), 4),
                       round(float(cfg.RISK_LEVEL_THRESHOLDS["high"]), 4)}
    prob_like = re.compile(r"(?<![\d.])([01]\.\d{2,6})(?![\d])")
    p4 = round(p, 4)
    for m in prob_like.finditer(text):
        val = round(float(m.group(1)), 4)
        if abs(val - p4) <= 5e-5:      # 与模型输出一致 → 正常引用
            continue
        if val in allowed_numbers:      # 等级阈值口径，允许出现
            continue
        issues.append(f"文本出现疑似被改写的概率 {m.group(1)}，"
                      f"与模型输出 {p:.4f} 不一致（这是禁止行为）")
        break
    # 概率被完全抹掉的情况（例如只说"风险不高"而不给数字）另行由等级校验兜住

    # ---- 校验 2 & 3：等级是否被改写 ----
    level_conflict = {
        "高": ["低风险", "风险较低", "风险不高", "不属于高风险"],
        "中": ["高风险判定", "确认为高风险"],
        "低": ["需要重点关注", "重点帮扶"],
    }
    for bad in level_conflict.get(lvl, []):
        if bad in text:
            issues.append(f"文本出现与预测等级“{lvl}”矛盾的表述：{bad}")

    # 若文本明确写出了等级词，必须与 facts 一致
    mentioned = [x for x in ("高风险", "中风险", "低风险") if x in text]
    mapping = {"高": "高风险", "中": "中风险", "低": "低风险"}
    for m in mentioned:
        if mapping.get(lvl) and m != mapping[lvl] and m in text:
            # 允许同时出现"高风险概率"这类描述性用法，因此只在"判定为/等级为"语境下判定冲突
            for ctx in ("判定为", "等级为", "等级是", "结论为"):
                if f"{ctx}{m}" in text:
                    issues.append(f"文本将等级改写为“{m}”，与模型结论“{lvl}”不一致")
                    break
    return {"passed": len(issues) == 0, "issues": issues, "checked_text_len": len(text)}


def generate_assistance(facts: Dict,
                        use_api: Optional[bool] = None,
                        verbose: bool = True) -> Dict:
    """
    统一的帮扶建议入口（对外只需调用这一个函数）。

    流程：
      1. 先用本地模板生成一份"兜底文本"（永远可用、永远与结论一致）；
      2. 若启用 API 且调用成功 → 用 LLM 文本替换"叙述段"，并做结论防篡改校验；
      3. 校验不通过 → 丢弃 LLM 文本，回退本地模板，并记录原因。

    返回 dict：{facts, summary, suggestions, narrative, backend, validation}
    """
    if use_api is None:
        use_api = bool(os.environ.get("DSH_LLM_API_KEY", "").strip())

    local_summary = generate_summary_text(facts)
    local_suggestions = generate_suggestions_local(facts)

    backend = "本地模板"
    narrative = local_summary
    validation: Dict = {"passed": True, "issues": [], "checked_text_len": len(local_summary)}
    note = "未启用 LLM（无 API Key）；全部文字由本地模板基于模型输出生成。"

    if use_api:
        ok, text = call_llm_api(facts)
        if ok:
            validation = validate_no_conclusion_change(text, facts)
            if validation["passed"]:
                narrative = text
                backend = "LLM（已通过结论防篡改校验）"
                note = "LLM 仅做文字组织，概率、等级、近邻等数值均由本地模型产出后拼接。"
            else:
                note = ("LLM 输出未通过结论一致性校验，已按安全策略回退本地模板。"
                        f"原因：{validation['issues']}")
        else:
            note = text  # call_llm_api 返回的失败说明

    return {
        "facts": facts,
        "summary": local_summary,          # 结构化事实摘要（永远本地生成）
        "suggestions": local_suggestions,  # 规则化建议（永远与结论一致）
        "narrative": narrative,            # 叙述段（可能是 LLM 文本，已校验）
        "backend": backend,
        "note": note,
        "validation": validation,
    }


def format_assistance(result: Dict, width: int = 68) -> str:
    """把帮扶建议结果排版成可直接打印/粘贴的中文文本。"""
    out: List[str] = []
    out.append("=" * width)
    out.append("学业帮扶建议（预测结论由模型给出，LLM 仅做文字组织）")
    out.append("=" * width)
    out.append(result["summary"])
    out.append("-" * width)
    out.append("【帮扶建议】")
    for s in result["suggestions"]:
        out.append(s if s.startswith("    ") else "  " + s)
    out.append("-" * width)
    out.append(f"【文本生成后端】{result['backend']}")
    out.append(f"【说明】{result['note']}")
    out.append(f"【结论一致性校验】通过 = {result['validation']['passed']}"
               + ("" if result["validation"]["passed"]
                  else f"，问题：{result['validation']['issues']}"))
    out.append("=" * width)
    return "\n".join(out)


def _self_test() -> None:
    """自检：验证"只传事实"时模板结果正确，并且防篡改校验能抓出被改写的结论。"""
    facts = build_facts(
        student_id="SIM0001", proba_risk=5.0 / 7.0, risk_level="高",
        top_contributions=[
            {"feature": "score_mean", "feature_cn": "三科平均分", "value_raw": 52.3,
             "risk_gap": 0.91},
            {"feature": "attendance_rate", "feature_cn": "出勤率", "value_raw": 0.62,
             "risk_gap": 0.55},
        ],
        neighbors=[{"student_id": "SIM0009", "label": 1}, {"student_id": "SIM0012", "label": 1},
                   {"student_id": "SIM0031", "label": 0}],
        k=7)
    res = generate_assistance(facts, use_api=False)
    print(format_assistance(res))

    print("—— 防篡改校验测试 ——")
    good = "该生高风险概率为 0.7143，风险等级为高风险，建议重点帮扶。"
    bad_prob = "该生高风险概率为 0.2143，风险等级为高风险。"
    bad_level = "该生高风险概率为 0.7143，判定为低风险，无需关注。"
    for name, text in (("正常文本", good), ("概率被改写", bad_prob), ("等级被改写", bad_level)):
        v = validate_no_conclusion_change(text, facts)
        print(f"  {name}: passed={v['passed']} issues={v['issues']}")
        assert (v["passed"] is True) == (name == "正常文本"), f"{name} 校验结果不符合预期"
    print("自检通过 ✅ LLM 无法改结论（概率/等级改动均被拦截）")


if __name__ == "__main__":
    try:
        from .utils import enable_utf8_console
    except ImportError:  # pragma: no cover
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import enable_utf8_console
    enable_utf8_console()
    _self_test()
    print(cfg.DISCLAIMER)
