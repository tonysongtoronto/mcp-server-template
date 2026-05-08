"""
eval_dataset3_v2.py — V2：工具调用验证 + 延迟统计 + 批量测试

新增功能（相比 V1）：
  ✅ V2  tool_called_judge   — 验证 agent 真的调了 MCP 工具（不是在瞎猜）
  ✅ V3  latency_judge       — 记录每条 example 的耗时，超时算失败
  ✅ V4  批量 dataset 支持   — dataset3 可随时添加新 example，自动全跑

target 返回格式（相比 V1 新增 task_plan / latency_ms）：
  {
    "output":      "<最终AI回答>",
    "task_plan":   [...],     ← agent 实际执行的任务列表（含 agent 类型）
    "latency_ms":  1234,      ← 本条 example 耗时（毫秒）
    "token_in":    350,       ← 输入 token 估算（字符数/4）
    "token_out":   210,       ← 输出 token 估算
  }

运行方式：
    uv run python tests/eval_dataset3_v2.py
"""

import asyncio
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import aevaluate

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import langgraph_parallel_agent as agent_module
from langgraph_parallel_agent import (
    _start_mcp_sessions_stdio,
    _stop_mcp_sessions,
)

# ══════════════════════════════════════════════════════
# Judge 模型
# ══════════════════════════════════════════════════════
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ══════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════
LATENCY_LIMIT_MS = 60_000   # 超过 60 秒视为超时失败
KEYWORD_THRESHOLD = 0.6     # 关键词命中率阈值


# ══════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════
def extract_question(inputs: dict) -> str:
    """dataset3 input 格式: {"messages": [{"role":"human","content":"..."}]}"""
    for msg in inputs.get("messages", []):
        if msg.get("role") == "human":
            return msg.get("content", "").strip()
    return ""


def extract_reference_answer(outputs: dict) -> str:
    """取 output messages 里最后一条 AI 消息"""
    for msg in reversed(outputs.get("messages", [])):
        msg_type = msg.get("type", "") or msg.get("role", "")
        if msg_type == "ai":
            return msg.get("content", "").strip()
    return ""


def extract_expected_agents(outputs: dict) -> list[str]:
    """从 reference output 的 task_plan 中取期望被调用的 agent 列表"""
    return [t.get("agent", "") for t in outputs.get("task_plan", []) if t.get("agent")]


def strip_emoji(text: str) -> str:
    return re.sub(r'[^\w\s@.\-|:，。]', '', text, flags=re.UNICODE).strip()


def extract_keywords(reference: str) -> list[str]:
    """从期望答案的 markdown 表格中提取关键词（用户名、年龄、日期）"""
    keywords = []
    for line in reference.split("\n"):
        line = strip_emoji(line.strip())
        if "|" in line and "@" in line:
            for part in line.split("|"):
                part = strip_emoji(part.strip())
                if not part or part.startswith("-") or "@" in part:
                    continue
                if len(part) < 20:
                    keywords.append(part)
    return keywords


# ══════════════════════════════════════════════════════
# Target：调用 agent，返回回答 + 元数据
# ══════════════════════════════════════════════════════
async def target(inputs: dict) -> dict:
    question = extract_question(inputs)
    if not question:
        return {"output": "", "task_plan": [], "latency_ms": 0, "token_in": 0, "token_out": 0}

    print(f"\n🤖 [{time.strftime('%H:%M:%S')}] 处理: {question[:50]}")

    thread_id = f"eval_{abs(hash(question)) % 100000}"
    config    = {"configurable": {"thread_id": thread_id}}

    t0 = time.monotonic()
    result = await agent_module.graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── 输出提取策略 ────────────────────────────────────────
    # final_answer_node 只输出简短总结（~70字）到 messages，
    # 完整表格在 task_plan[].result 里。
    # 策略：优先拼接 task_plan 里所有 done 任务的 result（完整数据）；
    #       fallback 到 AIMessage 最长内容。
    task_plan = result.get("task_plan", [])
    all_msgs  = result.get("messages", [])

    # 方案A：从 task_plan 拼接完整结果
    task_results = [
        t.get("result", "") for t in task_plan
        if t.get("status") == "done" and t.get("result")
    ]
    if task_results:
        output = "\n\n".join(task_results)
        print(f"   📨 取 task_plan result，共 {len(task_results)} 段，总长 {len(output)} 字")
    else:
        # 方案B：fallback 到最长 AIMessage
        ai_msgs = [m for m in all_msgs
                   if hasattr(m, "content") and m.content
                   and type(m).__name__ == "AIMessage"]
        output  = max(ai_msgs, key=lambda m: len(str(m.content))).content if ai_msgs else ""
        print(f"   📨 fallback AIMessage，最长 {len(output)} 字")


    # token 粗估（字符数 / 4）
    token_in  = sum(len(str(m.content)) for m in result.get("messages", [])
                    if hasattr(m, "content") and isinstance(m.content, str)) // 4
    token_out = len(output) // 4

    print(f"   ⏱️  耗时 {latency_ms/1000:.1f}s  |  输出 {len(output)} 字  |  "
          f"任务数 {len(task_plan)}  |  agents: {[t.get('agent') for t in task_plan]}")

    return {
        "output":     output,
        "task_plan":  task_plan,
        "latency_ms": latency_ms,
        "token_in":   token_in,
        "token_out":  token_out,
    }


# ══════════════════════════════════════════════════════
# Evaluator 1：关键词匹配（V1 沿用，无需 LLM）
# ══════════════════════════════════════════════════════
async def keyword_judge(run, example) -> dict:
    reference = extract_reference_answer(example.outputs or {})
    actual    = (run.outputs or {}).get("output", "")

    if not reference:
        return {"key": "keyword_match", "score": 0, "comment": "无期望答案"}
    if not actual:
        return {"key": "keyword_match", "score": 0, "comment": "Agent无输出"}

    keywords = extract_keywords(reference)
    if not keywords:
        score = 1 if len(actual) > 50 else 0
        return {"key": "keyword_match", "score": score, "comment": "无关键词，按长度判断"}

    matched = sum(1 for kw in keywords if kw and kw in actual)
    total   = len(keywords)
    ratio   = matched / total if total > 0 else 0
    score   = 1 if ratio >= KEYWORD_THRESHOLD else 0

    return {
        "key":     "keyword_match",
        "score":   score,
        "comment": f"关键词命中 {matched}/{total} ({ratio:.0%})"
    }


# ══════════════════════════════════════════════════════
# Evaluator 2：语义评估（V1 沿用）
# ══════════════════════════════════════════════════════
async def deepseek_judge(run, example) -> dict:
    question  = extract_question(example.inputs or {})
    reference = extract_reference_answer(example.outputs or {})
    actual    = (run.outputs or {}).get("output", "")

    if not reference:
        return {"key": "semantic_correctness", "score": 0, "comment": "无期望答案"}
    if not actual:
        return {"key": "semantic_correctness", "score": 0, "comment": "Agent无输出"}

    prompt = f"""请判断"实际答案"是否正确回答了用户问题，并与"期望答案"的核心内容一致。

用户问题：{question}
期望答案：{reference[:500]}
实际答案：{actual[:500]}

评估标准：数据是否基本吻合（数量、用户名等关键信息），不要求格式完全一样。

只返回 JSON：{{"correct": true或false, "reason": "一句话说明"}}"""

    try:
        resp   = await judge_llm.ainvoke([
            SystemMessage(content="你是评估专家，只返回 JSON。"),
            HumanMessage(content=prompt),
        ])
        raw    = resp.content.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        score  = 1 if result.get("correct") else 0
        return {"key": "semantic_correctness", "score": score, "comment": result.get("reason", "")}
    except Exception as e:
        return {"key": "semantic_correctness", "score": 0, "comment": f"解析失败: {e}"}


# ══════════════════════════════════════════════════════
# ✅ V2 新增 Evaluator 3：工具调用验证
# ══════════════════════════════════════════════════════
async def tool_called_judge(run, example) -> dict:
    """
    验证 agent 是否真的调用了期望的 MCP 工具，而不是凭空猜测。

    判断逻辑：
      1. 从 reference output 的 task_plan 取期望 agents（如 ["db_agent"]）
      2. 从 run.outputs 的 task_plan 取实际 agents
      3. 期望 agents 都出现在实际 agents 里 → score=1
      4. 实际 task_plan 为空（agent 没有规划任务，直接猜答案）→ score=0

    注意：direct agent 不算工具调用，只有 db_agent/math_agent 等才算。
    """
    expected_agents = extract_expected_agents(example.outputs or {})
    actual_plan     = (run.outputs or {}).get("task_plan", [])
    actual_agents   = [t.get("agent", "") for t in actual_plan]

    # 过滤掉 direct（直接回答不算工具调用）
    tool_agents = [a for a in expected_agents if a != "direct"]

    if not tool_agents:
        # 期望本来就不需要工具（direct 任务），直接给满分
        return {"key": "tool_called", "score": 1, "comment": "期望无工具调用（direct任务），跳过"}

    if not actual_plan:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"Agent未规划任何任务（task_plan为空），期望调用: {tool_agents}"
        }

    # 检查每个期望 agent 是否出现在实际执行里
    missing = [a for a in tool_agents if a not in actual_agents]
    if missing:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"缺少工具调用: {missing}，实际执行: {actual_agents}"
        }

    # 进一步验证：这些任务的 status 是否为 done
    done_agents = [t.get("agent") for t in actual_plan if t.get("status") == "done"]
    not_done    = [a for a in tool_agents if a not in done_agents]
    if not_done:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"工具调用未完成(status≠done): {not_done}"
        }

    return {
        "key":     "tool_called",
        "score":   1,
        "comment": f"工具调用正常 ✓ agents={actual_agents}"
    }


# ══════════════════════════════════════════════════════
# ✅ V3 新增 Evaluator 4：延迟评估
# ══════════════════════════════════════════════════════
async def latency_judge(run, example) -> dict:
    """
    记录耗时，超过 LATENCY_LIMIT_MS 视为失败。
    score 用连续值（0.0~1.0）表示速度，方便在 LangSmith 里画趋势图。

    计分公式：
      latency <= 20s  → 1.0（优秀）
      latency <= 40s  → 0.7（良好）
      latency <= 60s  → 0.4（勉强）
      latency >  60s  → 0.0（超时）
    """
    latency_ms = (run.outputs or {}).get("latency_ms", 0)
    token_in   = (run.outputs or {}).get("token_in",   0)
    token_out  = (run.outputs or {}).get("token_out",  0)

    if latency_ms <= 20_000:
        score, level = 1.0, "优秀"
    elif latency_ms <= 40_000:
        score, level = 0.7, "良好"
    elif latency_ms <= LATENCY_LIMIT_MS:
        score, level = 0.4, "勉强"
    else:
        score, level = 0.0, "超时"

    return {
        "key":     "latency",
        "score":   score,
        "comment": (f"{level} {latency_ms/1000:.1f}s  "
                    f"token≈in:{token_in} out:{token_out}")
    }


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════
async def main():
    print("🚀 启动 MCP sessions（stdio 模式）...")
    await _start_mcp_sessions_stdio()
    print(f"✅ MCP 初始化完成，agents: {agent_module._registry.agents}\n")

    try:
        print("📊 开始评估 dataset3（V2：工具调用 + 延迟）...")
        results = await aevaluate(
            target,
            data="dataset3",
            evaluators=[
                keyword_judge,       # E1: 关键词匹配
                deepseek_judge,      # E2: 语义正确性
                tool_called_judge,   # E3: ✅ 工具调用验证（V2新增）
                latency_judge,       # E4: ✅ 延迟评估（V3新增）
            ],
            experiment_prefix="dataset3-eval-v2",
            max_concurrency=1,
        )

        # ── 汇总统计 ──────────────────────────────────
        scores: dict[str, list] = {
            "keyword_match":       [],
            "semantic_correctness":[],
            "tool_called":         [],
            "latency":             [],
        }

        print("\n" + "═" * 65)
        async for r in results:
            question = extract_question((r.get("example", {}).inputs or {}))
            output   = (r.get("run", {}).outputs or {}).get("output", "")
            evals    = r.get("evaluation_results", {}).get("results", [])

            ev_map = {ev.key: ev for ev in evals}

            def get(key):
                ev = ev_map.get(key)
                return (ev.score if ev else "?"), (ev.comment if ev else "")

            k_s, k_c = get("keyword_match")
            s_s, s_c = get("semantic_correctness")
            t_s, t_c = get("tool_called")
            l_s, l_c = get("latency")

            for key, val in [("keyword_match", k_s), ("semantic_correctness", s_s),
                              ("tool_called", t_s),  ("latency", l_s)]:
                if isinstance(val, (int, float)):
                    scores[key].append(val)

            all_pass = all(v == 1 for v in [k_s, s_s, t_s] if isinstance(v, (int, float)))
            icon = "✅" if all_pass else "❌"

            print(f"{icon} {question[:45]}")
            print(f"   📝 回答:    {str(output)[:65]}...")
            print(f"   🔑 关键词:  [{k_s}] {k_c}")
            print(f"   🧠 语义:    [{s_s}] {s_c}")
            print(f"   🔧 工具调用:[{t_s}] {t_c}")
            print(f"   ⏱️  延迟:    [{l_s}] {l_c}")
            print("-" * 65)

        # ── 最终得分 ──────────────────────────────────
        print()
        labels = {
            "keyword_match":        "🔑 关键词匹配",
            "semantic_correctness": "🧠 语义正确性",
            "tool_called":          "🔧 工具调用验证",
            "latency":              "⏱️  延迟评分",
        }
        for key, label in labels.items():
            vals = scores[key]
            if vals:
                avg = sum(vals) / len(vals)
                passed = sum(1 for v in vals if v >= 1.0)
                print(f"{label}: {avg:.0%}  ({passed}/{len(vals)} 条通过)")

        print("\n🔗 LangSmith UI → dataset3 → Experiments 查看详细结果")
        print("   实验名前缀: dataset3-eval-v2")

    finally:
        await _stop_mcp_sessions()
        print("\n🛑 MCP sessions 已关闭")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

    # 运行方式：
    #   uv run python tests/eval_dataset3.py