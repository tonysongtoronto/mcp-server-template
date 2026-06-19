"""
cleanup_checkpoints.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checkpoint 数据库清理脚本

功能：
  调用 DELETE /db/cleanup 接口，保留最近 KEEP_THREADS 个会话，
  其余全部删除，释放 SQLite 磁盘空间。

排序依据：
  checkpoints 表的 rowid（SQLite 自增，越大越新）。
  取每个 thread_id 的 MAX(rowid) 代表最后活动顺序，前 N 个保留，其余删除。

流程：
  Step 1  dry_run=True  → 预演：列出将被删除的会话，不做任何修改
  Step 2  确认提示      → 用户输入 y/n（--yes 参数可跳过确认）
  Step 3  dry_run=False → 真正删除，打印删除行数

用法：
  # 默认保留 150 个会话，交互确认
  python scripts/cleanup_checkpoints.py

  # 自定义保留数量
  python scripts/cleanup_checkpoints.py --keep 200

  # 跳过交互确认，直接删除（适合 cron/计划任务）
  python scripts/cleanup_checkpoints.py --yes

  # 只预演，不删除（无论是否传 --yes）
  python scripts/cleanup_checkpoints.py --dry-run-only

  # 指定 API 地址
  python scripts/cleanup_checkpoints.py --url http://192.168.1.100:8000

依赖：
  纯标准库，无需额外安装。
  服务端已跑：uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

# ──────────────────────────────────────────────────────
# ★ 硬编码配置：以后要改保留数量，只改这一行 ★
# ──────────────────────────────────────────────────────
KEEP_THREADS = 150     # 默认保留最近 150 个会话
BASE_URL     = "http://127.0.0.1:8000"
TIMEOUT      = 30      # 清理接口超时（秒）


# ══════════════════════════════════════════════════════
# 底层 HTTP（纯标准库，与 test_multiuser_memory.py 风格一致）
# ══════════════════════════════════════════════════════

def _delete(path: str, timeout: int = TIMEOUT) -> dict:
    """DELETE {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e


def _get(path: str, timeout: int = 10) -> dict:
    """GET {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e


def call_cleanup(keep_threads: int, dry_run: bool) -> dict:
    """调用 DELETE /db/cleanup，返回 CleanupResponse dict。"""
    path = f"/db/cleanup?keep_threads={keep_threads}&dry_run={'true' if dry_run else 'false'}"
    return _delete(path)


# ══════════════════════════════════════════════════════
# 打印工具
# ══════════════════════════════════════════════════════

def print_dry_run_result(resp: dict) -> None:
    to_delete = resp.get("threads_to_delete", [])
    kept      = resp.get("threads_kept", 0)
    total     = resp.get("total_threads", 0)

    print(f"\n{'─' * 60}")
    print(f"  📋 预演结果（不会真正删除）")
    print(f"{'─' * 60}")
    print(f"  当前会话总数 : {total} 个")
    print(f"  保留最近     : {resp['keep_threads']} 个")
    print(f"  将删除       : {len(to_delete)} 个")
    print(f"  将保留       : {kept} 个")

    if to_delete:
        print(f"\n  将被删除的 thread_id（共 {len(to_delete)} 个）：")
        for tid in to_delete:
            print(f"    • {tid}")
    else:
        print(f"\n  ✅ 会话数未超过 {resp['keep_threads']} 个，无需清理。")
    print()


def print_delete_result(resp: dict) -> None:
    to_delete      = resp.get("threads_to_delete", [])
    kept           = resp.get("threads_kept", 0)
    total          = resp.get("total_threads", 0)
    rows_deleted   = resp.get("rows_deleted", 0)
    writes_deleted = resp.get("writes_deleted", 0)

    print(f"\n{'─' * 60}")
    print(f"  🗑️  清理完成")
    print(f"{'─' * 60}")
    print(f"  清理前会话总数  : {total} 个")
    print(f"  保留最近        : {resp['keep_threads']} 个")
    print(f"  已删除会话      : {len(to_delete)} 个")
    print(f"  保留会话        : {kept} 个")
    print(f"  checkpoints 行 : -{rows_deleted} 行")
    print(f"  writes 行       : -{writes_deleted} 行")

    if to_delete:
        print(f"\n  已删除的 thread_id：")
        for tid in to_delete:
            print(f"    • {tid}")
    print()


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

def check_api(url: str) -> bool:
    try:
        resp   = _get("/health")
        status = resp.get("status", "unknown")
        db     = resp.get("checkpoint_db", "?")
        print(f"🔌 API 连接正常  status={status}  checkpoint_db={db}")
        if status not in ("ok", "degraded"):
            print("⚠️  服务处于 initializing 状态，请稍后再试")
            return False
        return True
    except Exception as e:
        print(f"❌ 无法连接到 API ({url})：{e}")
        print("   请确认已启动：uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1")
        return False


def main() -> None:
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Checkpoint 数据库清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/cleanup_checkpoints.py               # 保留 150 个，交互确认\n"
            "  python scripts/cleanup_checkpoints.py --keep 200    # 保留 200 个\n"
            "  python scripts/cleanup_checkpoints.py --yes         # 跳过确认直接删除\n"
            "  python scripts/cleanup_checkpoints.py --dry-run-only  # 只预演不删除\n"
        ),
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=KEEP_THREADS,
        help=f"保留最近几个会话（默认 {KEEP_THREADS} 个）",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过交互确认，直接执行删除（适合计划任务）",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="只执行预演，不删除任何数据（无视 --yes）",
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"API 地址（默认 {BASE_URL}）",
    )
    args = parser.parse_args()

    BASE_URL     = args.url.rstrip("/")
    keep_threads = args.keep

    print(f"\n🧹 Checkpoint 数据库清理工具")
    print(f"   目标 API  : {BASE_URL}")
    print(f"   保留会话  : {keep_threads} 个")
    print(f"   模式      : {'只预演（--dry-run-only）' if args.dry_run_only else ('跳过确认（--yes）' if args.yes else '交互模式')}")

    if not check_api(BASE_URL):
        sys.exit(1)

    # ── Step 1：dry_run 预演 ────────────────────────────────────────
    print(f"\n⏳ Step 1 / 2  执行预演（dry_run=True）...")
    try:
        dry_resp = call_cleanup(keep_threads=keep_threads, dry_run=True)
    except Exception as e:
        print(f"❌ 预演请求失败：{e}")
        sys.exit(1)

    print_dry_run_result(dry_resp)

    if not dry_resp.get("threads_to_delete"):
        print("✅ 会话数量未超限，无需清理。")
        sys.exit(0)

    if args.dry_run_only:
        print("ℹ️  --dry-run-only 模式，跳过实际删除。")
        sys.exit(0)

    # ── Step 2：确认 ───────────────────────────────────────────────
    if not args.yes:
        n = len(dry_resp["threads_to_delete"])
        try:
            ans = input(f"⚠️  确认删除以上 {n} 个会话的 checkpoint？此操作不可逆。[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n取消。")
            sys.exit(0)

        if ans not in ("y", "yes"):
            print("取消，未做任何修改。")
            sys.exit(0)

    # ── Step 3：真正删除 ───────────────────────────────────────────
    print(f"\n⏳ Step 2 / 2  执行删除（dry_run=False）...")
    try:
        del_resp = call_cleanup(keep_threads=keep_threads, dry_run=False)
    except Exception as e:
        print(f"❌ 删除请求失败：{e}")
        sys.exit(1)

    print_delete_result(del_resp)
    print("✅ 清理完成。")
    sys.exit(0)


if __name__ == "__main__":
    main()
    
#     # 默认保留 150 个
# uv run python scripts/cleanup_checkpoints.py

# # 改数量
# uv run python scripts/cleanup_checkpoints.py --keep 200

# # 先看看有几个会话，不删
# uv run python scripts/cleanup_checkpoints.py --dry-run-only