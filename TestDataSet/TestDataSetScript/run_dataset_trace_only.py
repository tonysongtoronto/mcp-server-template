"""
run_dataset_trace_only.py — 方法二：批量跑数据集，只看全链路 Trace，不打分

功能：
  - 从 LangSmith 数据集中批量拉取所有 example
  - 依次调用 agent，自动上报 trace 到 LangSmith
  - 将每次运行结果（含 trace_url）保存到 TestDataSetResultWithEaluate/<timestamp>.json
  - 单独文件存放，每次运行生成一个新文件

运行方式：
    uv run python TestDataSetScript/run_dataset_trace_only.py

结果位置：
    TestDataSetResultWithEaluate/<YYYYMMDD_HHMMSS>_trace.json
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client
import uuid

load_dotenv()

# ── 路径设置 ──────────────────────────────────────────────────────────────────
# 本文件位于 TestDataSetScript/，结果输出到 TestDataSetResult/
THIS_DIR    = Path(__file__).parent                        # TestDataSetScript/
PROJECT_DIR = THIS_DIR.parent                              # 项目根目录
OUTPUT_DIR  = PROJECT_DIR / "TestDataSetResult"
SRC_DIR     = PROJECT_DIR / "src"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(SRC_DIR))

# ── LangSmith 追踪配置（确保 trace 自动上报）────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "dataset-trace-only")  # LangSmith 项目名

# ══════════════════════════════════════════════════════════════════════════════
# 配置区（按需修改）
# ══════════════════════════════════════════════════════════════════════════════
DATASET_NAME     = "backup"   # ← 手工改成你的数据集名称
MAX_CONCURRENCY  = 1          # 并发数，建议先用 1，稳定后可调大
TIMEOUT_SECONDS  = 120        # 单条 example 超时（秒）

# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def extract_question(inputs: dict) -> str:
    """
    兼容两种 input 格式：
      1. {"messages": [{"role": "human", "content": "..."}]}  ← dataset3 格式
      2. {"human": "..."}                                      ← backup 格式（图1所示）
    """
    # 格式1：messages 列表
    for msg in inputs.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "human":
            return msg.get("content", "").strip()

    # 格式2：直接 human 字段
    if "human" in inputs:
        return str(inputs["human"]).strip()

    # 格式3：content 字段
    if "content" in inputs:
        return str(inputs["content"]).strip()

    # 兜底：取第一个字符串值
    for v in inputs.values():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def extract_reference_answer(outputs: dict) -> str:
    """从 reference output 中提取期望答案文本"""
    if not outputs:
        return ""

    # 格式1：messages 列表
    for msg in reversed(outputs.get("messages", [])):
        msg_type = msg.get("type", "") or msg.get("role", "")
        if msg_type in ("ai", "assistant"):
            return msg.get("content", "").strip()

    # 格式2：直接字符串字段
    for key in ("output", "answer", "response", "ai", "assistant"):
        if key in outputs and isinstance(outputs[key], str):
            return outputs[key].strip()

    return ""


def build_output_filename(prefix: str = "trace") -> Path:
    """生成带时间戳的输出文件名"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_{prefix}.json"


# ══════════════════════════════════════════════════════════════════════════════
# 核心：调用 agent
# ══════════════════════════════════════════════════════════════════════════════

async def run_single_example(
    agent_module,
    question: str,
    example_id: str,
    idx: int,
) -> dict:
    """
    对单条 example 调用 agent，返回结构化结果。
    LangSmith tracing 由环境变量 LANGCHAIN_TRACING_V2=true 自动完成。
    """
    thread_id = f"trace_eval_{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}

    print(f"\n[{idx}] 🤖 [{time.strftime('%H:%M:%S')}] 问题: {question[:60]}")

    t0     = time.monotonic()
    error  = None
    output = ""
    task_plan: list = []
    all_messages: list = []

    try:
        result = await asyncio.wait_for(
            agent_module.graph.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            ),
            timeout=TIMEOUT_SECONDS,
        )

        task_plan    = result.get("task_plan", [])
        all_messages = result.get("messages", [])

        # 优先从 task_plan 拼完整结果
        task_results = [
            t.get("result", "") for t in task_plan
            if t.get("status") == "done" and t.get("result")
        ]
        if task_results:
            output = "\n\n".join(task_results)
        else:
            # fallback：最长 AIMessage
            ai_msgs = [
                m for m in all_messages
                if hasattr(m, "content") and m.content
                and type(m).__name__ == "AIMessage"
            ]
            output = max(ai_msgs, key=lambda m: len(str(m.content))).content if ai_msgs else ""

    except asyncio.TimeoutError:
        error = f"超时（>{TIMEOUT_SECONDS}s）"
        print(f"   ⚠️  {error}")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"   ❌ 错误: {error}")
        traceback.print_exc()

    latency_ms = int((time.monotonic() - t0) * 1000)
    agents_used = [t.get("agent", "") for t in task_plan]

    print(f"   ⏱️  耗时 {latency_ms/1000:.1f}s  |  "
          f"输出 {len(output)} 字  |  agents: {agents_used}")

    # 序列化 task_plan（去掉不可 JSON 化的对象）
    serializable_plan = []
    for t in task_plan:
        serializable_plan.append({
            "agent":       t.get("agent", ""),
            "description": t.get("description", ""),
            "status":      t.get("status", ""),
            "result":      str(t.get("result", ""))[:2000],   # 截断防止文件过大
            "task_id":     t.get("task_id", ""),
        })

    return {
        "example_id":  example_id,
        "question":    question,
        "output":      output,
        "task_plan":   serializable_plan,
        "latency_ms":  latency_ms,
        "agents_used": agents_used,
        "error":       error,
        "thread_id":   thread_id,
        "timestamp":   datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    # ── 动态 import agent（放在这里，避免顶层 import 失败阻断整个脚本）─────────
    try:
        import langgraph_parallel_agent as agent_module
        from langgraph_parallel_agent import (
            _start_mcp_sessions_stdio,
            _stop_mcp_sessions,
        )
    except ImportError as e:
        print(f"❌ 无法导入 agent 模块: {e}")
        print(f"   请确认 src/ 目录下存在 langgraph_parallel_agent.py")
        sys.exit(1)

    # ── 拉取数据集 ──────────────────────────────────────────────────────────
    client   = Client()
    examples = []
    try:
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        print(f"📦 数据集 [{DATASET_NAME}] 共 {len(examples)} 条 example")
    except Exception as e:
        print(f"❌ 拉取数据集失败: {e}")
        sys.exit(1)

    if not examples:
        print("⚠️  数据集为空，退出。")
        return

    # ── 启动 MCP ────────────────────────────────────────────────────────────
    print("\n🚀 启动 MCP sessions（stdio 模式）...")
    await _start_mcp_sessions_stdio()
    print(f"✅ MCP 初始化完成，agents: {agent_module._registry.agents}\n")

    # ── 运行结果容器 ─────────────────────────────────────────────────────────
    run_results   = []
    success_count = 0
    error_count   = 0
    run_start_ts  = datetime.now().isoformat()

    try:
        for idx, example in enumerate(examples, start=1):
            question = extract_question(example.inputs or {})
            ref_ans  = extract_reference_answer(example.outputs or {})

            if not question:
                print(f"[{idx}] ⚠️  example {example.id} 无法提取问题，跳过")
                continue

            result = await run_single_example(
                agent_module  = agent_module,
                question      = question,
                example_id    = str(example.id),
                idx           = idx,
            )
            result["reference_answer"] = ref_ans
            run_results.append(result)

            if result["error"]:
                error_count += 1
            else:
                success_count += 1

            # 并发控制（当前 MAX_CONCURRENCY=1，顺序执行）
            # 如需并发，可改用 asyncio.Semaphore + gather

    finally:
        await _stop_mcp_sessions()
        print("\n🛑 MCP sessions 已关闭")

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    avg_latency = (
        sum(r["latency_ms"] for r in run_results) / len(run_results)
        if run_results else 0
    )

    summary = {
        "meta": {
            "dataset_name":  DATASET_NAME,
            "run_type":      "trace_only",           # 方法二标识
            "langsmith_project": os.environ.get("LANGCHAIN_PROJECT", ""),
            "run_start":     run_start_ts,
            "run_end":       datetime.now().isoformat(),
            "total":         len(run_results),
            "success":       success_count,
            "error":         error_count,
            "avg_latency_ms": int(avg_latency),
        },
        "results": run_results,
    }

    # ── 保存文件 ─────────────────────────────────────────────────────────────
    out_file = build_output_filename(prefix="trace")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 控制台汇总 ───────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"📊 运行完成")
    print(f"   数据集:   {DATASET_NAME}")
    print(f"   总条数:   {len(run_results)}")
    print(f"   成功:     {success_count}")
    print(f"   失败:     {error_count}")
    print(f"   平均耗时: {avg_latency/1000:.1f}s")
    print(f"   结果文件: {out_file}")
    print(f"\n🔗 LangSmith UI → Projects → [{os.environ.get('LANGCHAIN_PROJECT')}]")
    print(f"   在 Traces 里查看每条请求的全链路调用信息")
    print("═" * 65)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
    
    
    # uv run python TestDataSet/TestDataSetScript/run_dataset_trace_only.py