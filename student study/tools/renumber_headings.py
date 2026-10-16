# -*- coding: utf-8 -*-
"""
renumber_headings.py —— 规范化文档的一级标题编号

【背景】本次按改进建议向 README.md 插入了新章节，导致一级标题出现重号与跳号：
    十三、验收自检清单
    十四、业务落地建议（新插入）
    十四、已知局限（原十四）
    十四之二、关于时间维度验证（新插入，沿用旧编号习惯）
    十五、快速上手
本脚本把一级标题按文档顺序重新连续编号，并**保留原标题正文**。

【设计要点】
  · 只处理形如 `## <中文数字>、<标题>` 的一级标题，其它行一律不动；
  · 中文数字按序生成（支持到二十以上），不依赖任何库；
  · 先打印"改名前 → 改名后"的对照表，便于人工核对；
  · 幂等：若编号已连续，则不产生任何改动。

【用法】python tools/renumber_headings.py README.md
        python tools/renumber_headings.py README.md --check   # 只检查不写
"""
from __future__ import annotations

import io
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CN_DIGITS = "零一二三四五六七八九"


def cn_number(n: int) -> str:
    """把整数转成中文数字（覆盖 1~99，够本项目使用）。"""
    if n < 0:
        raise ValueError("不支持负数")
    if n < 10:
        return CN_DIGITS[n]
    if n < 20:
        return "十" + (CN_DIGITS[n % 10] if n % 10 else "")
    tens, ones = divmod(n, 10)
    return CN_DIGITS[tens] + "十" + (CN_DIGITS[ones] if ones else "")


# 匹配 `## 一、xxx` / `## 十四之二、xxx` / `## 十五、xxx`
HEAD_RE = re.compile(r"^(##\s+)([一二三四五六七八九十]+)(之二)?(、)(.*)$")


def process(path: str, check_only: bool = False) -> int:
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    changes = []
    counter = 0
    out_lines = []
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        m = HEAD_RE.match(stripped)
        if not m:
            out_lines.append(line)
            continue
        counter += 1
        prefix, old_cn, suffix, sep, title = m.groups()
        new_cn = cn_number(counter)
        if old_cn != new_cn or suffix:
            changes.append((i + 1, stripped, "%s%s%s%s%s" % (prefix, new_cn, "", sep, title)))
            new_line = "%s%s%s%s%s" % (prefix, new_cn, "", sep, title)
            # 保留原行结尾符
            ending = line[len(stripped):]
            out_lines.append(new_line + ending)
        else:
            out_lines.append(line)

    name = os.path.basename(path)
    print("=" * 70)
    print("文件：%s（一级标题共 %d 个）" % (name, counter))
    print("=" * 70)
    if not changes:
        print("编号已连续，无需改动。")
        return 0
    for lineno, old, new in changes:
        print("  行 %-5d %s" % (lineno, old))
        print("        → %s" % new)
    if check_only:
        print("\n[--check] 仅检查，未写入。")
        return 0

    backup = path + ".bak-before-renumber"
    if not os.path.exists(backup):
        with io.open(path, "r", encoding="utf-8") as f:
            original = f.read()
        with io.open(backup, "w", encoding="utf-8") as f:
            f.write(original)
        print("\n已备份原文件 → %s" % os.path.basename(backup))
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("".join(out_lines))
    print("已写入 %d 处编号调整。" % len(changes))
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    if not args:
        print(__doc__)
        return 2
    targets = args if len(args) > 1 else [os.path.join(BASE, args[0])]
    rc = 0
    for t in targets:
        p = t if os.path.isabs(t) else os.path.join(BASE, t)
        if not os.path.exists(p):
            print("文件不存在：%s" % p)
            rc = 2
            continue
        rc |= process(p, check_only=check)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
