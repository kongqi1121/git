# -*- coding: utf-8 -*-
"""
reconcile_and_push.py —— 把本地仓库状态完整同步到远端（含删除）

【为什么需要"对账"而不是简单上传】
远端有若干提交是本会话之外/通过 API 创建的，其文件集合与本地未必一致。
只上传"新增与修改"会留下远端的多余文件；只上传本地全部文件则可能漏删。
因此这里做三向比对：
    应存在（本地已跟踪） vs 实际存在（远端 tree） →  需要新增 / 更新 / 删除

【删除策略（保守）】
只删除"远端存在、本地已不应存在，且路径在本轮变更范围内"的文件，
避免误删仓库里其它与本项目无关的内容。

【用法】
    python tools/reconcile_and_push.py            # 只对账，不写远端
    python tools/reconcile_and_push.py --apply    # 执行同步（创建一次提交）
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = r"E:\git"
REPO = "kongqi1121/git"
BRANCH = "main"
PROJECT = "student study/"

sys.path.insert(0, os.path.join(REPO_ROOT, PROJECT, "src"))
from push_via_api import API, api, load_token  # noqa: E402


def git(*args: str) -> str:
    r = subprocess.run(["git"] + list(args), cwd=REPO_ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    return (r.stdout or "").strip()


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def main() -> int:
    apply = "--apply" in sys.argv
    token = load_token()

    # 本地已跟踪文件（以 git 索引为准，即"应当存在于远端"的集合）
    tracked = [p for p in git("ls-files").splitlines() if p.strip()]
    local = {}
    for rel in tracked:
        full = os.path.join(REPO_ROOT, rel.replace("/", os.sep))
        if not os.path.exists(full):
            continue          # 索引里有但磁盘已删 → 视为应删除
        local[rel] = blob_sha(open(full, "rb").read())

    ref = api("GET", f"/repos/{REPO}/git/ref/heads/{BRANCH}", token)
    parent_sha = ref["object"]["sha"]
    tree = api("GET", f"/repos/{REPO}/git/trees/{parent_sha}?recursive=1", token)
    remote = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}

    to_add = [p for p in local if p not in remote]
    to_update = [p for p in local if p in remote and remote[p] != local[p]]
    # 只考虑本轮变更涉及的范围：本项目目录 + 仓库根的 .gitignore
    in_scope = lambda p: p.startswith(PROJECT) or p == ".gitignore"
    to_delete = [p for p in remote if in_scope(p) and p not in local]

    print("=" * 74)
    print("远端对账结果")
    print("=" * 74)
    print(f"本地已跟踪文件      : {len(local)}")
    print(f"远端文件            : {len(remote)}")
    print(f"需新增              : {len(to_add)}")
    print(f"需更新              : {len(to_update)}")
    print(f"需删除（范围内）    : {len(to_delete)}")
    for p in to_delete:
        print(f"    删除  {p}")
    print()
    for p in sorted(to_add):
        print(f"    新增  {p}")
    for p in sorted(to_update):
        print(f"    更新  {p}")

    if not apply:
        print("\n[对账模式] 未写远端。加 --apply 执行同步。")
        return 0

    # 建 blob（新增与更新）
    items = []
    for rel in sorted(set(to_add) | set(to_update)):
        raw = open(os.path.join(REPO_ROOT, rel.replace("/", os.sep)), "rb").read()
        blob = api("POST", f"/repos/{REPO}/git/blobs", token, {
            "content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"})
        items.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    # 删除用 sha=null 表示
    for rel in to_delete:
        items.append({"path": rel, "mode": "100644", "type": "blob", "sha": None})
    print(f"\n已创建 {len([i for i in items if i['sha']])} 个 blob，"
          f"{len(to_delete)} 个删除项")

    new_tree = api("POST", f"/repos/{REPO}/git/trees", token,
                   {"base_tree": tree["sha"], "tree": items})
    msg = git("log", "-1", "--pretty=%B") or "chore: 同步改进成果"
    commit = api("POST", f"/repos/{REPO}/git/commits", token,
                 {"message": msg, "tree": new_tree["sha"], "parents": [parent_sha]})
    api("PATCH", f"/repos/{REPO}/git/refs/heads/{BRANCH}", token,
        {"sha": commit["sha"], "force": False})
    print(f"同步完成：{parent_sha[:8]} → {commit['sha'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
