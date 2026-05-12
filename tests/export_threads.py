import os
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from langsmith import Client

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ── 配置区 ────────────────────────────────────────────────────
# 项目名映射（与 .env 保持一致）
PROJECT_NAMES = {
    "dev":  "MCP_SERVER_TEMPLEATE",
    "qa":   "MCP_SERVER_TEMPLATE_QA",
    "prod": "MCP_SERVER_TEMPLATE_PROD",
}

# 当前运行环境（通过 APP_ENV 环境变量切换，默认 qa）
CURRENT_ENV  = os.getenv("APP_ENV", "qa")
PROJECT_NAME = PROJECT_NAMES.get(CURRENT_ENV, PROJECT_NAMES["qa"])

# ── 在这里手动添加要导出的 Thread ID ────────────────────────
# （从 LangSmith Studio -> Threads 视图顶部复制）
THREAD_IDS = [
    # "aa5d5ea2-c120-40f3-9d53-fc5faef3b81a",
    "c4bd2411-9e27-46a4-bd0b-3f6db770dfee",
    # "019e0161-1a89-73b3-b332-44923aecc199",
    # 继续添加更多 Thread ID...
]

# 输出目录（通过 EXPORT_OUTPUT_DIR 覆盖，默认 trace_exports/）
OUTPUT_DIR = Path(
    os.getenv("EXPORT_OUTPUT_DIR", str(Path(__file__).parent.parent / "trace_exports"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────


def format_run(r) -> dict:
    """把 Run 对象序列化成字典"""
    latency = (
        round((r.end_time - r.start_time).total_seconds(), 3)
        if r.end_time and r.start_time else None
    )
    return {
        "id":              str(r.id),
        "name":            r.name,
        "run_type":        r.run_type,
        "status":          r.status,
        "parent_run_id":   str(r.parent_run_id) if r.parent_run_id else None,
        "start_time":      str(r.start_time),
        "end_time":        str(r.end_time),
        "latency_seconds": latency,
        "total_tokens":    getattr(r, "total_tokens",      None),
        "prompt_tokens":   getattr(r, "prompt_tokens",     None),
        "completion_tokens": getattr(r, "completion_tokens", None),
        "inputs":          r.inputs,
        "outputs":         r.outputs,
        "error":           r.error,
        "tags":            r.tags,
        "metadata":        r.extra.get("metadata", {}) if r.extra else {},
    }


def export_single(ls: Client, thread_id: str, index: int, total: int) -> dict:
    """处理单个 Thread ID，导出该 thread 下所有 run"""

    print(f"\n{'─' * 58}")
    print(f"  [{index}/{total}] Thread ID : {thread_id}")
    print(f"{'─' * 58}")

    # ── Step 1: 通过 filter query language 按 thread_id 拉取所有 runs ──
    # list_runs 没有原生 thread_id 参数；LangSmith 的 Thread 是通过
    # metadata 里 session_id / conversation_id / thread_id 三个 key 分组的
    print("  📌 Step 1：通过 filter 按 thread_id 拉取所有 runs...")
    filter_string = (
        f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]),'
        f' eq(metadata_value, "{thread_id}"))'
    )
    try:
        all_runs = list(ls.list_runs(
            project_name=PROJECT_NAME,
            filter=filter_string,
        ))
        all_runs.sort(key=lambda r: r.start_time or datetime.min)
        print(f"     ✅ 共找到 {len(all_runs)} 个 run")
    except Exception as e:
        print(f"     ❌ 获取 runs 失败：{e}")
        return {"thread_id": thread_id, "status": "failed", "error": str(e)}

    if not all_runs:
        print("     ⚠️  没有找到任何 run，请确认 Thread ID 正确")
        return {
            "thread_id":  thread_id,
            "status":     "empty",
            "total_runs": 0,
            "runs":       [],
        }

    # ── Step 2: 找顶层 runs ───────────────────────────────────
    top_runs = [r for r in all_runs if r.parent_run_id is None]
    print(f"     ✅ 顶层 run: {len(top_runs)} 个")

    # ── Step 3: 逐个顶层 run 展开，打印节点摘要 ──────────────
    runs_data: list[dict] = []
    total_tokens_all = 0

    for top_run in top_runs:
        run_id = str(top_run.id)
        print(f"\n  🔍 处理 run : {run_id[:8]}... ({top_run.name})")

        # 递归收集属于该顶层 run 的所有后代节点
        # （覆盖直接子节点、孙子节点等所有深度，修复原始文件只排除顶层自身的 bug）
        all_ids_in_trace = {run_id}
        child_runs = []
        for r in sorted(all_runs, key=lambda x: x.start_time or datetime.min):
            if r.parent_run_id is not None and str(r.parent_run_id) in all_ids_in_trace:
                child_runs.append(r)
                all_ids_in_trace.add(str(r.id))

        print(f"     共 {len(child_runs)} 个子节点")

        node_counts = Counter(r.name for r in child_runs)
        for name, count in node_counts.items():
            print(f"     - {name}: {count} 次")

        root_dict      = format_run(top_run)
        children_dicts = [format_run(r) for r in child_runs]
        all_nodes      = [root_dict] + children_dicts

        # 节点摘要表格
        print(f"\n     {'节点名':<26} {'类型':<12} {'状态':<10} {'耗时(s)':<10} {'Token'}")
        print(f"     {'-' * 64}")
        for d in all_nodes:
            indent      = "  " if d["parent_run_id"] else ""
            latency_str = f"{d['latency_seconds']:.3f}" if d["latency_seconds"] else "-"
            print(f"     {indent}{d['name']:<24} {str(d['run_type']):<12} "
                  f"{str(d['status']):<10} "
                  f"{latency_str:<10} "
                  f"{d['total_tokens'] or '-'}")

        run_tokens    = sum(d["total_tokens"] or 0 for d in all_nodes)
        total_tokens_all += run_tokens
        llm_nodes     = [d for d in all_nodes if d["run_type"] == "llm"]

        print(f"\n     总 Token  : {run_tokens}")
        print(f"     根节点耗时 : {root_dict['latency_seconds']}s")
        print(f"     LLM 调用  : {len(llm_nodes)} 次")

        runs_data.append({
            "root":        root_dict,
            "total_nodes": len(child_runs),
            "llm_calls":   len(llm_nodes),
            "total_tokens": run_tokens,
            "nodes":       children_dicts,
        })

    return {
        "thread_id":       thread_id,
        "status":          "success",
        "project":         PROJECT_NAME,
        "total_runs":      len(runs_data),
        "total_tokens_all": total_tokens_all,
        "runs":            runs_data,
    }


def export_all():
    if not THREAD_IDS:
        print("❌  未指定任何 Thread ID，请在脚本顶部的 THREAD_IDS 列表中添加。")
        return

    ls        = Client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total     = len(THREAD_IDS)

    print("=" * 58)
    print(f"  Thread Export -- {total} Thread ID(s)")
    print(f"  Env   : {CURRENT_ENV.upper()} -> project: {PROJECT_NAME}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 58)

    results       = []
    success_count = 0
    failed_count  = 0
    empty_count   = 0

    for i, thread_id in enumerate(THREAD_IDS, start=1):
        result = export_single(ls, thread_id, i, total)
        results.append(result)

        status = result.get("status")
        if status == "success":
            success_count += 1
            single_path = OUTPUT_DIR / f"thread_{thread_id[:8]}_{timestamp}.json"
            single_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n     💾 已保存：{single_path.name}")
        elif status == "empty":
            empty_count += 1
        else:
            failed_count += 1

    # ── 保存批量汇总文件 ──────────────────────────────────────
    summary = {
        "exported_at":   timestamp,
        "env":           CURRENT_ENV,
        "project":       PROJECT_NAME,
        "total":         total,
        "success_count": success_count,
        "empty_count":   empty_count,
        "failed_count":  failed_count,
        "results":       results,
    }
    batch_path = OUTPUT_DIR / f"batch_thread_export_{timestamp}.json"
    batch_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # ── 最终汇总打印 ──────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               ✅  Thread Export Done                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for r in results:
        status = r.get("status")
        icon   = "✅" if status == "success" else ("⚠️ " if status == "empty" else "❌")
        tid    = r.get("thread_id", "")[:36]
        print(f"║  {icon} thread_id : {tid}")
        if status == "success":
            print(f"║     project : {r.get('project', '-')}")
            print(f"║     runs    : {r.get('total_runs')}  "
                  f"tokens(total): {r.get('total_tokens_all')}")
        elif status == "empty":
            print(f"║     ⚠️  no runs found")
        else:
            print(f"║     ❌  error: {r.get('error', '-')}")
        print("║")
    print(f"║  success: {success_count}  empty: {empty_count}  failed: {failed_count}")
    print(f"║  batch  : {batch_path.name}")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  LangSmith : https://smith.langchain.com                 ║")
    print(f"║  env: {CURRENT_ENV.upper():<8}  project: {PROJECT_NAME:<30}║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    export_all()

# ── 启动方式 ──────────────────────────────────────────────────
# 在脚本顶部 THREAD_IDS 列表里手动填写要导出的 Thread ID，然后：
#
# 默认 qa 环境：
#   uv run tests/export_threads.py
#
# 切换环境：
#   APP_ENV=dev  uv run tests/export_threads.py
#   APP_ENV=qa   uv run tests/export_threads.py
#   APP_ENV=prod uv run tests/export_threads.py