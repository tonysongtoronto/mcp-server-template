import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from langsmith import Client

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

client = Client()

# ── 配置区 ──────────────────────────────────────────────────
PROJECT_NAME = "MCP_SERVER_TEMPLEATE"

# 填 Thread ID（从 Studio 顶部复制）
THREAD_IDS = [
    "8cd92196-39b4-4d06-983e-2c4e68a4ffb6",
    # "另一个 thread_id",
]

OUTPUT_DIR = Path(__file__).parent.parent / "trace_exports"
OUTPUT_DIR.mkdir(exist_ok=True)
# ────────────────────────────────────────────────────────────


def format_run(r) -> dict:
    return {
        "id": str(r.id),
        "name": r.name,
        "run_type": r.run_type,
        "status": r.status,
        "parent_run_id": str(r.parent_run_id) if r.parent_run_id else None,
        "start_time": str(r.start_time),
        "end_time": str(r.end_time),
        "latency_seconds": (
            (r.end_time - r.start_time).total_seconds()
            if r.end_time and r.start_time else None
        ),
        "total_tokens": getattr(r, "total_tokens", None),
        "inputs": r.inputs,
        "outputs": r.outputs,
        "error": r.error,
        "tags": r.tags,
    }


def export_by_thread(thread_id: str) -> dict:
    print(f"\n📡 正在拉取 Thread: {thread_id}")

    # 通过 metadata 的 thread_id 字段找所有属于该 thread 的 runs
    try:
        all_runs = list(client.list_runs(
            project_name=PROJECT_NAME,
            metadata={"thread_id": thread_id},
        ))
        print(f"  ✅ 共找到 {len(all_runs)} 个 run")
    except Exception as e:
        print(f"  ❌ 获取 runs 失败：{e}")
        return {"thread_id": thread_id, "error": str(e), "runs": []}

    if not all_runs:
        print("  ⚠️ 没有找到任何 run，请确认 Thread ID 正确")
        return {"thread_id": thread_id, "total_runs": 0, "runs": []}

    # 找顶层 run（没有 parent_run_id 的）
    top_runs = [r for r in all_runs if r.parent_run_id is None]
    print(f"  ✅ 顶层 run: {len(top_runs)} 个")

    all_runs_data = []
    for top_run in top_runs:
        run_id = str(top_run.id)
        print(f"\n  🔍 处理 run: {run_id[:8]}... ({top_run.name})")

        # 该 run 下的所有子节点（从已拿到的 all_runs 里过滤，不需要再请求）
        child_runs = [r for r in all_runs if str(r.id) != run_id]

        print(f"     共 {len(child_runs)} 个子节点")
        node_counts = Counter(r.name for r in child_runs)
        for name, count in node_counts.items():
            print(f"     - {name}: {count} 次")

        all_runs_data.append({
            "root": format_run(top_run),
            "total_nodes": len(child_runs),
            "nodes": [format_run(r) for r in child_runs],
        })

    return {
        "thread_id": thread_id,
        "total_runs": len(all_runs_data),
        "runs": all_runs_data,
    }


def export_all(thread_ids: list[str]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for thread_id in thread_ids:
        result = export_by_thread(thread_id)

        output_file = OUTPUT_DIR / f"thread_{thread_id[:8]}_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n  💾 已保存：{output_file.name}")
        print(f"     包含 {result.get('total_runs', 0)} 个 run")

    print(f"\n✅ 全部完成！文件保存在：{OUTPUT_DIR}")


if __name__ == "__main__":
    export_all(THREAD_IDS)
    
    # uv run tests/export_trace.py