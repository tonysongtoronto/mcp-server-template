import asyncio
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from langsmith import Client
from langgraph_sdk import get_client

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ── 配置区：填入多个 TRACE_ID ────────────────────────────────
LANGGRAPH_URL = "http://127.0.0.1:2024"
ASSISTANT_ID  = "supervisorpalner"      # 来自 langgraph.json

# 项目名映射（与 .env 保持一致）
PROJECT_NAMES = {
    "dev":  "MCP_SERVER_TEMPLEATE",      # <- 暂时保留原拼写
    "qa":   "MCP_SERVER_TEMPLATE_QA",
    "prod": "MCP_SERVER_TEMPLATE_PROD",
}

# 当前运行环境：dev / qa / prod（通过环境变量或 .env 的 APP_ENV 切换，默认 qa）
CURRENT_ENV  = os.getenv("APP_ENV", "qa")
PROJECT_NAME = PROJECT_NAMES.get(CURRENT_ENV, PROJECT_NAMES["qa"])

# 项目过滤器：
#   默认跟随当前环境 PROJECT_NAME，只处理该项目的 trace_id，其余跳过
#   改为 None -> 全局模式，所有项目的 trace_id 都能处理
PROJECT_FILTER = PROJECT_NAME
# PROJECT_FILTER = None   # 全局模式，跨项目不过滤

TRACE_IDS = [
    "019dfcb7-0fc5-7002-bfc9-1a4d90f2d03f",
    # "019e046e-8a97-7e81-9c60-5a359b3fc284",
    # "019e0161-1a89-73b3-b332-44923aecc199",
    # 继续添加更多 Trace ID...
]

OUTPUT_DIR = Path(__file__).parent.parent / "RunTrace_ID_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

# session_id -> 项目名 缓存，避免重复查询
_session_cache: dict = {}
# ────────────────────────────────────────────────────────────


def get_source_project(ls: Client, trace) -> str:
    """
    通过 trace.session_id 查询所属项目名。
    session_id 是项目 UUID，需要用 list_projects 匹配。
    结果缓存避免重复请求。
    """
    session_id = getattr(trace, "session_id", None)
    if not session_id:
        return "未知项目"

    session_id = str(session_id)

    # 命中缓存直接返回
    if session_id in _session_cache:
        return _session_cache[session_id]

    # 遍历账号下所有项目，匹配 session_id
    try:
        for project in ls.list_projects():
            if str(project.id) == session_id:
                name = project.name
                _session_cache[session_id] = name
                return name
    except Exception as e:
        print(f"     ⚠️  查询项目名失败：{e}")

    _session_cache[session_id] = "未知项目"
    return "未知项目"


async def replay_single(lg, ls, trace_id: str, index: int, total: int) -> dict:
    """处理单个 Trace ID"""

    print(f"\n{'─' * 58}")
    print(f"  [{index}/{total}] Trace ID : {trace_id}")
    print(f"{'─' * 58}")

    # ── Step 1: 从 LangSmith 拿原始输入 ──────────────────────
    # read_run() 全局查询，trace_id 跨项目有效，不受 PROJECT_FILTER 影响
    print("  📌 Step 1：读取原始输入...")
    try:
        trace = ls.read_run(trace_id)
        original_input = trace.inputs
        source_project = get_source_project(ls, trace)
        print(f"     ✅ 来源项目：{source_project}")
        print(f"     ✅ 输入：{json.dumps(original_input, ensure_ascii=False)[:120]}")
    except Exception as e:
        print(f"     ❌ 获取失败：{e}")
        return {"trace_id": trace_id, "status": "failed", "error": str(e)}

    # ── 项目过滤判断 ──────────────────────────────────────────
    if PROJECT_FILTER and source_project != PROJECT_FILTER:
        print(f"     ⚠️  跳过：trace 属于 [{source_project}]，"
              f"不在过滤项目 [{PROJECT_FILTER}] 内")
        return {
            "trace_id":       trace_id,
            "status":         "skipped",
            "source_project": source_project,
            "reason":         f"project filter: {PROJECT_FILTER}",
        }

    # ── Step 2: 新建 Thread ───────────────────────────────────
    print("  📌 Step 2：新建 Thread...")
    new_thread = await lg.threads.create()
    new_thread_id = new_thread["thread_id"]
    print(f"     ✅ Thread ID : {new_thread_id}")

    # ── Step 3: 重新执行 Graph ────────────────────────────────
    print("  🚀 Step 3：重新执行 Graph...")
    stream_events = []
    new_run_id = ""

    try:
        async for chunk in lg.runs.stream(
            thread_id=new_thread_id,
            assistant_id=ASSISTANT_ID,
            input=original_input,
            stream_mode="updates",
        ):
            stream_events.append({"event": chunk.event, "data": str(chunk.data)[:300]})
            print(f"     [{chunk.event}] {str(chunk.data)[:80]}")

            if chunk.event == "metadata" and chunk.data:
                new_run_id = chunk.data.get("run_id", "")

        print("     ✅ 执行完成！")
    except Exception as e:
        print(f"     ❌ 执行失败：{e}")
        return {"trace_id": trace_id, "status": "failed", "error": str(e)}

    # ── Step 4: 等待 LangSmith 收录 ──────────────────────────
    print("  ⏳ Step 4：等待 LangSmith 收录（5秒）...")
    await asyncio.sleep(5)

    # ── Step 5: 拉取新 Trace（写入当前环境项目）────────────────
    print(f"  📊 Step 5：拉取新 Trace（项目：{PROJECT_NAME}）...")
    new_trace_id = new_run_id   # Step 3 已从 metadata 拿到 run_id，直接用
    nodes_data = []

    try:
        # 用 trace_id=new_run_id 精确查当次，避免累加历史节点
        all_runs = list(ls.list_runs(
            project_name=PROJECT_NAME,
            trace_id=new_run_id,
        ))
        all_runs.sort(key=lambda r: r.start_time or datetime.min)

        print(f"     ✅ 新 Trace ID : {new_trace_id}")
        print(f"     ✅ 节点数量   : {len(all_runs)}")

        print(f"\n     {'节点名':<26} {'状态':<10} {'耗时(s)':<10} {'Token'}")
        print(f"     {'-' * 58}")

        for r in all_runs:
            latency = (
                (r.end_time - r.start_time).total_seconds()
                if r.end_time and r.start_time else None
            )
            tokens = getattr(r, "total_tokens", None)
            print(f"     {r.name:<26} {str(r.status):<10} "
                  f"{f'{latency:.3f}' if latency else '-':<10} "
                  f"{tokens or '-'}")

            nodes_data.append({
                "id":              str(r.id),
                "name":            r.name,
                "run_type":        r.run_type,
                "status":          r.status,
                "parent_run_id":   str(r.parent_run_id) if r.parent_run_id else None,
                "start_time":      str(r.start_time),
                "end_time":        str(r.end_time),
                "latency_seconds": latency,
                "total_tokens":    tokens,
                "inputs":          r.inputs,
                "outputs":         r.outputs,
                "error":           r.error,
                "tags":            r.tags,
            })
    except Exception as e:
        print(f"     ❌ 拉取失败：{e}")

    return {
        "original_trace_id": trace_id,
        "source_project":    source_project,
        "target_project":    PROJECT_NAME,
        "status":            "success",
        "replayed": {
            "thread_id":   new_thread_id,
            "run_id":      new_run_id,
            "trace_id":    new_trace_id,
            "total_nodes": len(nodes_data),
            "nodes":       nodes_data,
        },
        "stream_events": stream_events,
    }


async def replay_all():
    lg = get_client(url=LANGGRAPH_URL)
    ls = Client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(TRACE_IDS)

    filter_info = f"filter: {PROJECT_FILTER}" if PROJECT_FILTER else "mode: global (cross-project)"

    print("=" * 58)
    print(f"  Batch Replay -- {total} Trace ID(s)")
    print(f"  Env: {CURRENT_ENV.upper()} -> write to: {PROJECT_NAME}")
    print(f"  Read: {filter_info}")
    print("=" * 58)

    results = []
    success_count = 0
    failed_count  = 0
    skipped_count = 0

    for i, trace_id in enumerate(TRACE_IDS, start=1):
        result = await replay_single(lg, ls, trace_id, i, total)
        results.append(result)

        status = result.get("status")
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

        if i < total:
            print("\n  ⏸️  pause 2s before next...")
            await asyncio.sleep(2)

    # ── 保存汇总结果 ──────────────────────────────────────────
    summary = {
        "exported_at":    timestamp,
        "env":            CURRENT_ENV,
        "target_project": PROJECT_NAME,
        "project_filter": PROJECT_FILTER,
        "total":          total,
        "success_count":  success_count,
        "skipped_count":  skipped_count,
        "failed_count":   failed_count,
        "results":        results,
    }

    output_file = OUTPUT_DIR / f"batch_replay_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # ── 最终汇总 ──────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  ✅  Batch Replay Done                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for r in results:
        status = r.get("status")
        icon   = "✅" if status == "success" else ("⏭️ " if status == "skipped" else "❌")
        orig   = r.get("original_trace_id", "")[:36]
        src    = r.get("source_project", "-")
        new    = r.get("replayed", {}).get("trace_id", "-")[:36]
        reason = r.get("reason", "")
        print(f"║  {icon} orig : {orig}")
        print(f"║     from : {src}")
        if status == "success":
            print(f"║     new  : {new}")
        elif status == "skipped":
            print(f"║     skip : {reason}")
        print("║")
    print(f"║  success: {success_count}  skipped: {skipped_count}  failed: {failed_count}")
    print(f"║  saved  : {output_file.name}")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  LangSmith : https://smith.langchain.com                 ║")
    print(f"║  env: {CURRENT_ENV.upper():<8}  project: {PROJECT_NAME}")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(replay_all())

# ── 启动方式 ─────────────────────────────────────────────────
# default qa, global mode:
#   uv run tests/run_trace_replay.py
#
# switch env:
#   APP_ENV=dev  uv run tests/run_trace_replay.py
#   APP_ENV=qa   uv run tests/run_trace_replay.py
#   APP_ENV=prod uv run tests/run_trace_replay.py
#
# project filter (edit PROJECT_FILTER above):
#   PROJECT_FILTER = "MCP_SERVER_TEMPLATE_QA"