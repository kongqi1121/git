# -*- coding: utf-8 -*-
"""
utils.py —— 公共小工具
=========================================================
目前主要解决两个"演示体验"问题：

1. **Windows 控制台中文乱码**
   Windows 默认控制台编码常为 GBK（代码页 936），而本项目源码与输出都是 UTF-8，
   直接 print 中文会出现"锟斤拷"式乱码。这里统一在程序启动时把标准输出/错误
   重新配置为 UTF-8（Python 3.7+ 提供 reconfigure 接口），保证答辩现场输出可读。

2. **统一的终端排版**
   提供标题、小节、键值对的打印函数，让 20 多个文件、上千行输出保持一致的观感，
   也方便把终端输出直接截图进实验报告。
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, Optional


def enable_utf8_console() -> None:
    """
    把 stdout/stderr 切换为 UTF-8 编码。

    幂等且安全：任何异常都被吞掉（例如某些 IDE 的控制台不支持 reconfigure），
    绝不让"编码问题"导致主流程崩溃——这是教学项目里最容易浪费时间的坑。
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def print_title(title: str, width: int = 68, char: str = "=") -> None:
    """打印一级标题（带等号线框）。"""
    print(char * width)
    print(title)
    print(char * width)


def print_section(title: str, width: int = 68) -> None:
    """打印二级小节标题。"""
    print("-" * width)
    print(title)
    print("-" * width)


def print_kv(pairs: Dict[str, Any], indent: int = 2, nd: Optional[int] = None) -> None:
    """对齐打印"键 : 值"，数字默认保留 4 位小数，便于阅读。"""
    pad = " " * indent
    if not pairs:
        return
    width = max(len(str(k)) for k in pairs)
    for k, v in pairs.items():
        if isinstance(v, float):
            v = f"{v:.{4 if nd is None else nd}f}"
        print(f"{pad}{str(k).ljust(width)} : {v}")


def print_list(items: Iterable[str], indent: int = 2, bullet: str = "·") -> None:
    """逐行打印要点。"""
    pad = " " * indent
    for it in items:
        print(f"{pad}{bullet} {it}")


def fmt_pct(x: float, nd: int = 2) -> str:
    """把 0~1 的比例格式化成百分数字符串。"""
    return f"{100.0 * float(x):.{nd}f}%"


def bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    """
    生成终端文本进度条，用于直观展示概率/贡献度。
    例如 p=0.71 → "██████████████░░░░░░"（14/20）。
    """
    v = max(0.0, min(1.0, float(value)))
    n = int(round(v * width))
    return filled * n + empty * (width - n)
