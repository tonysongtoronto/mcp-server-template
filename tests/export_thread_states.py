import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph_sdk import get_client

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ── 配置区 ────────────────────────────────────────────────────
LANGGRAPH_URL = "http://127.0.0.1:2024"

# ── 在这里手动添加要导出 state 的 Thread ID ────────────────────
# （从 LangSmith Studio / LangGraph Studio -> Threads 视图顶部复制）
THREAD_IDS = [
    "019fbaf5-3cca-7f90-a5a3-e47ca74aa8d3",
    # "aa5d5ea2-c120-40f3-9d53-fc5faef3b81a",
    # 继续添加更多 Thread ID...
]

# 是否同时拉取完整 checkpoint 历史（每一步的 state 快照）
# False -> 只拉当前最新 state，速度更快
INCLUDE_HISTORY = True
HISTORY_LIMIT = 1000  # get_history 的 limit 上限

# 输出目录（通过 EXPORT_OUTPUT_DIR 覆盖，默认 trace_status_exports/）
OUTPUT_DIR = Path(
    os.getenv("EXPORT_OUTPUT_DIR", str(Path(__file__).parent.parent / "trace_status_exports"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────


def _serialize(obj):
    """兜底序列化：datetime / 其他不可 JSON 化对象一律转 str"""
    return str(obj)


async def export_single(lg, thread_id: str, index: int, total: int) -> dict:
    """导出单个 Thread 的完整 state 信息"""

    print(f"\n{'─' * 58}")
    print(f"  [{index}/{total}] Thread ID : {thread_id}")
    print(f"{'─' * 58}")

    result = {
        "thread_id": thread_id,
        "status": "success",
        "thread_meta": None,
        "current_state": None,
        "history": [],
        "error": None,
    }

    # ── Step 1：拿 Thread 本身的 metadata ────────────────────
    print("  📌 Step 1：读取 Thread metadata...")
    try:
        thread_meta = await lg.threads.get(thread_id)
        result["thread_meta"] = thread_meta
        print(f"     ✅ status: {thread_meta.get('status')}  "
              f"created_at: {thread_meta.get('created_at')}")
    except Exception as e:
        print(f"     ❌ 获取 Thread metadata 失败：{e}")
        result["status"] = "failed"
        result["error"] = f"get thread meta failed: {e}"
        return result

    # ── Step 2：拿当前最新 state ─────────────────────────────
    print("  📌 Step 2：读取当前最新 state...")
    try:
        current_state = await lg.threads.get_state(thread_id)
        result["current_state"] = current_state

        values = current_state.get("values", {}) if isinstance(current_state, dict) else {}
        next_nodes = current_state.get("next", []) if isinstance(current_state, dict) else []
        print(f"     ✅ state keys       : {list(values.keys())}")
        print(f"     ✅ next node(s)     : {next_nodes}")
        print(f"     ✅ values 预览      : {json.dumps(values, ensure_ascii=False, default=_serialize)[:200]}")
    except Exception as e:
        print(f"     ❌ 获取当前 state 失败：{e}")
        result["status"] = "failed"
        result["error"] = f"get_state failed: {e}"
        return result

    # ── Step 3（可选）：拿完整 checkpoint 历史 ────────────────
    if INCLUDE_HISTORY:
        print("  📌 Step 3：读取完整 checkpoint 历史...")
        try:
            history = await lg.threads.get_history(thread_id, limit=HISTORY_LIMIT)
            result["history"] = history
            print(f"     ✅ 共 {len(history)} 个 checkpoint")

            print(f"\n     {'#':<4} {'checkpoint_id':<38} {'next':<20}")
            print(f"     {'-' * 62}")
            for i, snap in enumerate(reversed(history)):
                ckpt_id = snap.get("checkpoint_id") or snap.get("checkpoint", {}).get("checkpoint_id", "-")
                next_n = snap.get("next", [])
                print(f"     {i:<4} {str(ckpt_id):<38} {str(next_n):<20}")
        except Exception as e:
            print(f"     ⚠️  获取历史失败（不影响当前 state 已导出）：{e}")
            result["error"] = f"get_history failed: {e}"
    else:
        print("  ⏭️  Step 3：跳过历史（INCLUDE_HISTORY=False）")

    return result


async def export_all():
    if not THREAD_IDS:
        print("❌  未指定任何 Thread ID，请在脚本顶部的 THREAD_IDS 列表中添加。")
        return

    lg = get_client(url=LANGGRAPH_URL)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(THREAD_IDS)

    print("=" * 58)
    print(f"  Thread State Export -- {total} Thread ID(s)")
    print(f"  LangGraph URL : {LANGGRAPH_URL}")
    print(f"  Include hist. : {INCLUDE_HISTORY}")
    print(f"  Output        : {OUTPUT_DIR}")
    print("=" * 58)

    results = []
    success_count = 0
    failed_count = 0

    for i, thread_id in enumerate(THREAD_IDS, start=1):
        result = await export_single(lg, thread_id, i, total)
        results.append(result)

        if result.get("status") == "success":
            success_count += 1
            single_path = OUTPUT_DIR / f"thread_state_{thread_id[:8]}_{timestamp}.json"
            single_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=_serialize),
                encoding="utf-8",
            )
            print(f"\n     💾 已保存：{single_path.name}")
        else:
            failed_count += 1

    # ── 保存批量汇总文件 ──────────────────────────────────────
    summary = {
        "exported_at": timestamp,
        "langgraph_url": LANGGRAPH_URL,
        "include_history": INCLUDE_HISTORY,
        "total": total,
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results,
    }
    batch_path = OUTPUT_DIR / f"batch_thread_state_export_{timestamp}.json"
    batch_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_serialize),
        encoding="utf-8",
    )

    # ── 最终汇总打印 ──────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              ✅  Thread State Export Done               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for r in results:
        status = r.get("status")
        icon = "✅" if status == "success" else "❌"
        tid = r.get("thread_id", "")[:36]
        print(f"║  {icon} thread_id : {tid}")
        if status == "success":
            print(f"║     checkpoints : {len(r.get('history', []))}")
        else:
            print(f"║     ❌ error: {r.get('error', '-')}")
        print("║")
    print(f"║  success: {success_count}  failed: {failed_count}")
    print(f"║  batch  : {batch_path.name}")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(export_all())

# ── 启动方式 ──────────────────────────────────────────────────
# 在脚本顶部 THREAD_IDS 列表里手动填写要导出的 Thread ID，然后：
#
#   uv run tests/export_thread_states.py
#
# 覆盖输出目录：
#   EXPORT_OUTPUT_DIR=/some/path uv run tests/export_thread_states.py

# 开一个新终端,在项目目录(有 langgraph.json 那个目录)下执行:
# powershell
#    langgraph dev

# 看到类似 Server started at http://127.0.0.1:2024 就说明起来了,让它保持运行,不要关。

# 再开一个终端(保持第一个终端别关),运行我之前给你的 export_thread_states.py。
