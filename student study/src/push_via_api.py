# -*- coding: utf-8 -*-
"""
push_via_api.py —— 当 `git push` 被网络阻断时，改用 GitHub REST API 推送提交
================================================================================
【背景】本机实测：`github.com:443` 的出站连接被重置（`git push` 报
"Connection was reset" / "Could not connect to server"），但 `api.github.com`
完全可达（HTTP 200）。因此改用 GitHub Git Data API 推送。

【为什么用 Git Data API 而不是逐个文件 Contents API】
Contents API 每推一个文件就产生一次提交。本次要更新 12 个文件，
会产生 12 个零散提交、历史很脏。
Git Data API 允许：建 blob → 一次性组装 tree → 建 commit → 移动 ref，
从而把多个文件合并成**一个语义完整的提交**。

【步骤】
  1. 读 gh 的凭据（~/.config 或 %APPDATA%\\GitHub CLI\\hosts.yml）取 token
  2. 取当前 main 指向的 commit 与 tree
  3. 对每个待更新文件创建 blob（base64）
  4. 组装新 tree（base_tree = 原 tree，只覆盖变更文件）
  5. 创建 commit（parent = 原 commit，message 由调用方给出）
  6. 更新 refs/heads/main 指向新 commit

【安全】token 只在内存中使用，不打印、不写盘、不进入任何输出。
"""

from __future__ import annotations

import os
import sys
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

REPO = "kongqi1121/git"
API = "https://api.github.com"
BRANCH = "main"


# ----------------------------------------------------------------------
# 凭据读取
# ----------------------------------------------------------------------
def load_token() -> str:
    """
    从 gh CLI 的凭据文件读取 token。

    只提取 gho_/ghp_ 开头的 token 字段，不做任何输出，避免泄露。
    找不到时抛错并提示改用 `gh auth login`。
    """
    candidates = [
        os.path.join(os.environ.get("APPDATA", ""), "GitHub CLI", "hosts.yml"),
        os.path.join(os.path.expanduser("~"), ".config", "gh", "hosts.yml"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("oauth_token:"):
                        tok = s.split(":", 1)[1].strip().strip('"').strip("'")
                        if tok:
                            return tok
    raise RuntimeError(
        "未能从 gh 凭据文件中读取 token。请先执行：gh auth login"
    )


# ----------------------------------------------------------------------
# 极简 API 客户端
# ----------------------------------------------------------------------
def api(method: str, path: str, token: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """调用 GitHub REST API，返回解析后的 JSON。"""
    url = path if path.startswith("http") else f"{API}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "dsh-push-script",
        "X-GitHub-Api-Version": "2022-11-28",
        **({"Content-Type": "application/json"} if data else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8")
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {url} 失败：HTTP {e.code} {detail}") from e


# ----------------------------------------------------------------------
# 推送
# ----------------------------------------------------------------------
def commit_files(repo_root: str, rel_paths: List[str], message: str,
                 branch: str = BRANCH, verbose: bool = True) -> Dict[str, Any]:
    """
    把若干文件合并成**一个**提交并推送到指定分支。

    参数
    ----
    repo_root : 本地仓库根目录（用于读取文件内容）
    rel_paths : 相对仓库根的路径列表（GitHub 的路径分隔符用 /）
    message   : 提交信息
    """
    token = load_token()
    if verbose:
        print(f"[1/6] 已读取凭据；仓库 {REPO}，分支 {branch}")

    ref = api("GET", f"/repos/{REPO}/git/ref/heads/{branch}", token)
    parent_sha = ref["object"]["sha"]
    parent = api("GET", f"/repos/{REPO}/git/commits/{parent_sha}", token)
    base_tree = parent["tree"]["sha"]
    if verbose:
        print(f"[2/6] 当前 HEAD = {parent_sha[:8]}，base tree = {base_tree[:8]}")

    # 建 blob
    tree_items = []
    for rel in rel_paths:
        local = os.path.join(repo_root, rel.replace("/", os.sep))
        if not os.path.exists(local):
            raise FileNotFoundError(f"本地文件不存在：{local}")
        with open(local, "rb") as f:
            content = f.read()
        blob = api("POST", f"/repos/{REPO}/git/blobs", token, {
            "content": base64.b64encode(content).decode("ascii"),
            "encoding": "base64",
        })
        tree_items.append({"path": rel, "mode": "100644", "type": "blob",
                           "sha": blob["sha"]})
        if verbose:
            print(f"      blob {blob['sha'][:8]}  {rel}  ({len(content)} 字节)")

    if verbose:
        print(f"[3/6] 已创建 {len(tree_items)} 个 blob")

    tree = api("POST", f"/repos/{REPO}/git/trees", token,
               {"base_tree": base_tree, "tree": tree_items})
    if verbose:
        print(f"[4/6] 新 tree = {tree['sha'][:8]}")

    commit = api("POST", f"/repos/{REPO}/git/commits", token, {
        "message": message,
        "tree": tree["sha"],
        "parents": [parent_sha],
    })
    if verbose:
        print(f"[5/6] 新 commit = {commit['sha'][:8]}")

    api("PATCH", f"/repos/{REPO}/git/refs/heads/{branch}", token,
        {"sha": commit["sha"], "force": False})
    if verbose:
        print(f"[6/6] 已更新 {branch} → {commit['sha'][:8]}")

    return {"commit": commit["sha"], "parent": parent_sha,
            "files": [t["path"] for t in tree_items]}


def main() -> int:
    """命令行：python push_via_api.py <仓库根> <提交信息文件> <路径1> [路径2 ...]"""
    if len(sys.argv) < 4:
        print(__doc__)
        print("用法：python push_via_api.py <repo_root> <message_file> <path1> [path2 ...]")
        return 2
    repo_root, msg_file = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:]
    with open(msg_file, "r", encoding="utf-8") as f:
        message = f.read().strip()
    result = commit_files(repo_root, paths, message)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
