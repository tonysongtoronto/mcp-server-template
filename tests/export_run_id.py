import asyncio
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
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

# 项目过滤器：
#   默认跟随当前环境，只处理属于该项目的 Run ID，其余跳过
#   改为 None -> 全局模式，跨项目不过滤
PROJECT_FILTER = PROJECT_NAME
# PROJECT_FILTER = None

# ── 在这里手动添加要导出的 Run ID ────────────────────────────
RUN_IDS = [
    "019e0506-ef46-74f1-9f9c-0efefce6b369",
    # "019dfcb7-0fc5-7002-bfc9-1a4d90f2d03f",
    # "019e046e-8a97-7e81-9c60-5a359b3fc284",
    # 继续添加更多 Run ID...
]

# 输出目录（对应截图里的 Run_ID_exports 文件夹）
OUTPUT_DIR = Path(__file__).parent.parent / "Run_ID_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

# session_id -> 项目名 缓存，避免重复查询
_session_cache: dict = {}
# ─────────────────────────────────────────────────────────────


def get_source_project(ls: Client, run) -> str:
    """通过 run.session_id 查询所属项目名，结果缓存避免重复请求"""
    session_id = getattr(run, "session_id", None)
    if not session_id:
        return "未知项目"
    session_id = str(session_id)

    if session_id in _session_cache:
        return _session_cache[session_id]

    try:
        for project in ls.list_projects():
            if str(project.id) == session_id:
                _session_cache[session_id] = project.name
                return project.name
    except Exception as e:
        print(f"     ⚠️  查询项目名失败：{e}")

    _session_cache[session_id] = "未知项目"
    return "未知项目"


def run_to_dict(r) -> dict:
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


def build_tree(nodes: list[dict]) -> list[dict]:
    """
    把扁平 node 列表重建成树形结构。
    每个节点加 children 字段，返回根节点列表。
    """
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        n["children"] = []

    roots = []
    for n in nodes:
        pid = n["parent_run_id"]
        if pid and pid in by_id:
            by_id[pid]["children"].append(n)
        else:
            roots.append(n)
    return roots


def export_single(ls: Client, run_id: str, index: int, total: int) -> dict:
    """处理单个 Run ID，导出完整调用树"""

    print(f"\n{'─' * 58}")
    print(f"  [{index}/{total}] Run ID : {run_id}")
    print(f"{'─' * 58}")

    # ── Step 1: 读取顶层 Run ──────────────────────────────────
    print("  📌 Step 1：读取顶层 Run...")
    try:
        root_run = ls.read_run(run_id)
        source_project = get_source_project(ls, root_run)
        print(f"     ✅ 来源项目  : {source_project}")
        print(f"     ✅ Run 名称  : {root_run.name}")
        print(f"     ✅ 状态      : {root_run.status}")
    except Exception as e:
        print(f"     ❌ 读取失败：{e}")
        return {"run_id": run_id, "status": "failed", "error": str(e)}

    # ── 项目过滤判断 ──────────────────────────────────────────
    if PROJECT_FILTER and source_project != PROJECT_FILTER:
        print(f"     ⚠️  跳过：Run 属于 [{source_project}]，"
              f"不在过滤项目 [{PROJECT_FILTER}] 内")
        return {
            "run_id":         run_id,
            "status":         "skipped",
            "source_project": source_project,
            "reason":         f"project filter: {PROJECT_FILTER}",
        }

    # ── Step 2: 拉取完整调用树（所有子节点）──────────────────
    print("  📌 Step 2：拉取完整调用树（trace_id = run_id）...")
    try:
        # trace_id 等于顶层 run_id，可以一次拉出整棵树的所有节点
        all_runs = list(ls.list_runs(
            project_name=source_project,
            trace_id=run_id,
        ))
        all_runs.sort(key=lambda r: r.start_time or datetime.min)
        print(f"     ✅ 共 {len(all_runs)} 个节点")
    except Exception as e:
        print(f"     ❌ 拉取失败：{e}")
        return {"run_id": run_id, "status": "failed", "error": str(e)}

    # ── Step 3: 序列化 + 打印节点摘要 ────────────────────────
    print(f"\n     {'节点名':<26} {'类型':<12} {'状态':<10} {'耗时(s)':<10} {'Token'}")
    print(f"     {'-' * 64}")

    nodes_flat = []
    for r in all_runs:
        d = run_to_dict(r)
        nodes_flat.append(d)
        indent = "  " if d["parent_run_id"] else ""
        latency_str = f"{d['latency_seconds']:.3f}" if d["latency_seconds"] else "-"
        print(f"     {indent}{r.name:<24} {str(r.run_type):<12} "
              f"{str(r.status):<10} "
              f"{latency_str:<10} "
              f"{d['total_tokens'] or '-'}")

    # ── Step 4: 构建树形结构 ──────────────────────────────────
    tree = build_tree([n.copy() for n in nodes_flat])

    # ── Step 5: 汇总统计 ─────────────────────────────────────
    total_tokens  = sum(n["total_tokens"]    or 0 for n in nodes_flat)
    total_latency = sum(n["latency_seconds"] or 0 for n in nodes_flat
                        if not n["parent_run_id"] is None or True)
    root_latency  = run_to_dict(root_run)["latency_seconds"]
    llm_nodes     = [n for n in nodes_flat if n["run_type"] == "llm"]

    print(f"\n     总 Token  : {total_tokens}")
    print(f"     总耗时    : {root_latency}s（顶层 Run）")
    print(f"     LLM 调用  : {len(llm_nodes)} 次")

    return {
        "run_id":         run_id,
        "status":         "success",
        "source_project": source_project,
        "summary": {
            "root_run_name":  root_run.name,
            "root_status":    root_run.status,
            "total_nodes":    len(nodes_flat),
            "llm_calls":      len(llm_nodes),
            "total_tokens":   total_tokens,
            "root_latency_s": root_latency,
            "start_time":     str(root_run.start_time),
            "end_time":       str(root_run.end_time),
        },
        "tree":  tree,
        "nodes": nodes_flat,   # 扁平列表，方便后续分析
    }


def export_all():
    ls = Client()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    total      = len(RUN_IDS)
    filter_info = f"filter: {PROJECT_FILTER}" if PROJECT_FILTER else "mode: global (cross-project)"

    print("=" * 58)
    print(f"  Run ID Export -- {total} Run ID(s)")
    print(f"  Env   : {CURRENT_ENV.upper()} -> project: {PROJECT_NAME}")
    print(f"  Read  : {filter_info}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 58)

    results       = []
    success_count = 0
    failed_count  = 0
    skipped_count = 0

    for i, run_id in enumerate(RUN_IDS, start=1):
        result = export_single(ls, run_id, i, total)
        results.append(result)

        status = result.get("status")
        if status == "success":
            success_count += 1
            # 每个 Run ID 单独保存一个文件，方便逐条查阅
            single_path = OUTPUT_DIR / f"run_{run_id[:8]}_{timestamp}.json"
            single_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8"
            )
            print(f"\n     💾 已保存：{single_path.name}")
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

    # ── 保存批量汇总文件 ──────────────────────────────────────
    summary = {
        "exported_at":    timestamp,
        "env":            CURRENT_ENV,
        "project_filter": PROJECT_FILTER,
        "total":          total,
        "success_count":  success_count,
        "skipped_count":  skipped_count,
        "failed_count":   failed_count,
        "results":        results,
    }
    batch_path = OUTPUT_DIR / f"batch_export_{timestamp}.json"
    batch_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    # ── 最终汇总打印 ──────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  ✅  Run ID Export Done                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for r in results:
        status = r.get("status")
        icon   = "✅" if status == "success" else ("⏭️ " if status == "skipped" else "❌")
        rid    = r.get("run_id", "")[:36]
        src    = r.get("source_project", "-")
        reason = r.get("reason", "")
        sm     = r.get("summary", {})
        print(f"║  {icon} run_id : {rid}")
        print(f"║     project : {src}")
        if status == "success":
            print(f"║     nodes   : {sm.get('total_nodes')}  "
                  f"llm: {sm.get('llm_calls')}  "
                  f"tokens: {sm.get('total_tokens')}  "
                  f"latency: {sm.get('root_latency_s')}s")
        elif status == "skipped":
            print(f"║     skip    : {reason}")
        print("║")
    print(f"║  success: {success_count}  skipped: {skipped_count}  failed: {failed_count}")
    print(f"║  batch  : {batch_path.name}")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  LangSmith : https://smith.langchain.com                 ║")
    print(f"║  env: {CURRENT_ENV.upper():<8}  project: {PROJECT_NAME:<30}║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    export_all()

# ── 启动方式 ──────────────────────────────────────────────────
# 默认 qa 环境，按项目过滤：
#   uv run tests/export_run_id.py
#
# 切换环境：
#   APP_ENV=dev  uv run tests/export_run_id.py
#   APP_ENV=qa   uv run tests/export_run_id.py
#   APP_ENV=prod uv run tests/export_run_id.py
#
# 全局模式（跨项目不过滤）：
#   把脚本里 PROJECT_FILTER = PROJECT_NAME 改为 PROJECT_FILTER = None