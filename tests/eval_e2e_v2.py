"""
eval_e2e.py —— DeepSeek 双模型对比评估脚本

对比：deepseek-chat（聊天）vs deepseek-reasoner（推理）
数据集：dataset_e2e（或你在 DATASET_NAME 里指定的名称）

运行方式：
    uv run python tests/eval_e2e.py

流程：
    1. 启动 MCP session，初始化工具和图
    2. 用 deepseek-chat   跑一轮评估 → experiment A
    3. 用 deepseek-reasoner 跑一轮评估 → experiment B
    4. LLM-as-judge 打分（正确性 + 完整性）
    5. 结果自动上传 LangSmith，终端打印对比汇总
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import aevaluate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# 把 src 加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from langgraph_stdio_agent import (
    load_tools,
    _init_registry,
    _rebuild_graph,
    graph as agent_graph,
    mcp_params,
)

# ══════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════

DATASET_NAME = "dataset_e2e"   # ← 改成你的数据集名称

# 两个被测模型
MODEL_A = "deepseek-chat"       # 聊天模型
MODEL_B = "deepseek-reasoner"   # 推理模型

# 裁判模型（用 chat 模型即可，推理模型做 judge 太慢）
JUDGE_MODEL = "deepseek-chat"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ══════════════════════════════════════════════════════
# 1. 构建 LLM 实例的工厂函数
# ══════════════════════════════════════════════════════

def make_llm(model_name: str) -> ChatOpenAI:
    """根据模型名称创建 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
        temperature=0,
    )

# 裁判 LLM（全局固定，不随对比切换）
judge_llm = make_llm(JUDGE_MODEL)

# ══════════════════════════════════════════════════════
# 2. 脏数据清洗（原样保留）
# ══════════════════════════════════════════════════════

ROLE_MAP = {
    "human": "human", "user": "human",
    "ai": "ai", "assistant": "ai",
    "system": "system", "tool": "tool",
}

def _normalize_role(msg: dict) -> str:
    role = msg.get("role") or msg.get("type") or ""
    return ROLE_MAP.get(role.lower(), role.lower())

def _normalize_content(msg: dict) -> str:
    return (
        msg.get("content") or msg.get("text") or msg.get("output") or ""
    )

def _parse_lc_serialized(msg: dict) -> dict | None:
    if msg.get("type") != "constructor":
        return None
    id_list = msg.get("id", [])
    kwargs  = msg.get("kwargs", {})
    content = kwargs.get("content", "")
    if not content or not id_list:
        return None
    class_name = id_list[-1] if id_list else ""
    role_map = {
        "SystemMessage": "system", "HumanMessage": "human",
        "AIMessage": "ai", "ToolMessage": "tool",
    }
    role = role_map.get(class_name) or ROLE_MAP.get(kwargs.get("type", "").lower(), "human")
    return {"role": role, "content": content}

def _flatten_messages(messages: list) -> list:
    if messages and isinstance(messages[0], list):
        flat = []
        for item in messages:
            flat.extend(item) if isinstance(item, list) else flat.append(item)
        return flat
    return messages

def extract_messages_as_lc(inputs: dict) -> list:
    lc_messages = []
    messages = _flatten_messages(inputs.get("messages", []))
    for msg in messages:
        if hasattr(msg, "content"):
            lc_messages.append(msg)
            continue
        if not isinstance(msg, dict):
            continue
        parsed = _parse_lc_serialized(msg)
        if parsed:
            msg = parsed
        role    = _normalize_role(msg)
        content = _normalize_content(msg)
        if not content:
            continue
        if role == "human":
            lc_messages.append(HumanMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "ai":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(ToolMessage(content=content, tool_call_id=""))
        else:
            lc_messages.append(HumanMessage(content=content))
    if not lc_messages:
        fallback = (
            inputs.get("question") or inputs.get("input") or inputs.get("content") or ""
        )
        if fallback:
            lc_messages.append(HumanMessage(content=str(fallback)))
    return lc_messages

def extract_human_question(inputs: dict) -> str:
    messages = _flatten_messages(inputs.get("messages", []))
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict):
            parsed = _parse_lc_serialized(msg)
            if parsed:
                msg = parsed
            if _normalize_role(msg) == "human":
                return _normalize_content(msg)
    return inputs.get("question") or inputs.get("input") or inputs.get("content") or ""

def extract_reference(example_outputs: dict) -> str:
    clean = (
        example_outputs.get("answer")
        or example_outputs.get("output")
        or example_outputs.get("content")
    )
    if clean:
        return clean
    messages = example_outputs.get("messages", [])
    ai_contents = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg, AIMessage):
            if msg.content:
                ai_contents.append(msg.content)
            continue
        if not isinstance(msg, dict):
            continue
        parsed = _parse_lc_serialized(msg)
        if parsed:
            msg = parsed
        if _normalize_role(msg) in ("ai", "assistant"):
            content = _normalize_content(msg)
            if content:
                ai_contents.append(content)
    if ai_contents:
        return ai_contents[-1]
    generations = example_outputs.get("generations", [])
    for gen in generations:
        if isinstance(gen, list):
            gen = gen[0] if gen else {}
        if not isinstance(gen, dict):
            continue
        message = gen.get("message", {})
        if isinstance(message, dict):
            kwargs     = message.get("kwargs", message)
            content    = kwargs.get("content", "") or _normalize_content(message)
            tool_calls = kwargs.get("tool_calls", [])
            if not content and tool_calls:
                print("  ⚠️ Reference Output 是中间节点（tool_calls），跳过")
                return ""
            if content:
                return content
        content = _normalize_content(gen)
        if content:
            return content
    return ""

def _inject_system_into_human(lc_messages: list) -> list:
    system_msgs = [m for m in lc_messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in lc_messages if not isinstance(m, SystemMessage)]
    if not system_msgs:
        return lc_messages
    style_hint = system_msgs[0].content
    for i, msg in enumerate(non_system):
        if isinstance(msg, HumanMessage):
            non_system[i] = HumanMessage(
                content=f"{msg.content}\n\n【回复风格要求】{style_hint}"
            )
            break
    return non_system


# ══════════════════════════════════════════════════════
# 3. target 工厂函数 —— 核心改动！
#    每次调用传入不同模型名，动态替换 agent 的 llm
# ══════════════════════════════════════════════════════

def make_target(model_name: str):
    """
    返回一个绑定了指定模型的 target 函数。
    每次调用前临时替换 agent_module.llm，调用结束后自动恢复。

    为什么要替换 agent_module.llm？
    → langgraph_stdio_agent.py 里的 planner_node / run_agent /
      final_answer_node 都直接用模块级 llm 变量，
      替换它就能让整条链路换模型，不需要重建 graph。
    """
    import langgraph_stdio_agent as agent_module

    async def target(inputs: dict) -> dict:
        # ── 临时换模型 ──────────────────────────────────
        original_llm    = agent_module.llm
        agent_module.llm = make_llm(model_name)
        print(f"\n  🤖 使用模型：{model_name}")

        try:
            lc_messages = extract_messages_as_lc(inputs)
            if not lc_messages:
                print("  ⚠️ 无法提取消息，跳过该条")
                return {"answer": ""}

            lc_messages = _inject_system_into_human(lc_messages)

            # 打印完整输入（新增）
            human_q = extract_human_question(inputs)
            print(f"  📥 输入任务：{human_q}")

            result = await agent_module.graph.ainvoke({
                "messages":           lc_messages,
                "task_plan":          [],
                "current_task_index": 0,
                "next_agent":         "",
            })

            result_messages = result.get("messages", [])
            for msg in reversed(result_messages):
                if hasattr(msg, "content") and msg.content:
                    return {"answer": msg.content}
                if isinstance(msg, dict) and _normalize_content(msg):
                    return {"answer": _normalize_content(msg)}

            return {"answer": ""}

        finally:
            # ── 无论是否出错，都恢复原来的 llm ──────────
            agent_module.llm = original_llm

    return target


# ══════════════════════════════════════════════════════
# 4. LLM-as-judge Evaluator（同原版，裁判不变）
# ══════════════════════════════════════════════════════

def make_judge_evaluator():
    async def llm_as_judge(run, example) -> dict:
        question  = extract_human_question(example.inputs)
        reference = extract_reference(example.outputs)
        actual    = run.outputs.get("answer", "") if run.outputs else ""

        if not reference:
            print("  ⚠️ reference 为空，跳过 judge 评分")
            return [
                {"key": "correctness",  "score": 0, "comment": "reference 为空，无法评估"},
                {"key": "completeness", "score": 0, "comment": "reference 为空，无法评估"},
            ]

        prompt = f"""你是一个严格的评估专家。请根据以下信息对实际答案进行评估。

用户问题：
{question}

期望答案（参考）：
{reference}

实际答案：
{actual}

请从以下两个维度评估实际答案，每个维度打分 0 或 1：

1. 正确性（correctness）：实际答案的核心结论/数值是否与期望答案一致？
2. 完整性（completeness）：实际答案是否覆盖了用户问题要求的所有内容？

只返回 JSON，不要有任何其他文字：
{{
  "correctness": 0或1,
  "completeness": 0或1,
  "reason": "简短说明"
}}"""

        response = await judge_llm.ainvoke([
            SystemMessage(content="你是一个评估专家，只返回 JSON 格式的评估结果。"),
            HumanMessage(content=prompt),
        ])

        try:
            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"  ⚠️ judge 解析失败：{e}")
            result = {"correctness": 0, "completeness": 0, "reason": "解析失败"}

        return [
            {"key": "correctness",  "score": result.get("correctness", 0),  "comment": result.get("reason", "")},
            {"key": "completeness", "score": result.get("completeness", 0), "comment": result.get("reason", "")},
        ]

    return llm_as_judge


# ══════════════════════════════════════════════════════
# 5. 单模型评估函数（可复用）
# ══════════════════════════════════════════════════════

async def run_single_eval(model_name: str, experiment_prefix: str) -> dict:
    """
    跑一次完整评估，返回 {"correctness": avg, "completeness": avg} 汇总结果。
    """
    print(f"\n{'═' * 60}")
    print(f"  📊 评估模型：{model_name}")
    print(f"  🏷  Experiment：{experiment_prefix}-xxxxx")
    print(f"{'═' * 60}")

    results = await aevaluate(
        make_target(model_name),
        data=DATASET_NAME,
        evaluators=[make_judge_evaluator()],
        experiment_prefix=experiment_prefix,
        max_concurrency=1,   # 串行跑，稳定后可以调大
    )

    correctness_scores  = []
    completeness_scores = []

    async for r in results:
        eval_results = r.get("evaluation_results", {}).get("results", [])
        for er in eval_results:
            if er.key == "correctness":
                correctness_scores.append(er.score)
            elif er.key == "completeness":
                completeness_scores.append(er.score)

    avg_c = sum(correctness_scores)  / len(correctness_scores)  if correctness_scores  else 0
    avg_p = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    print(f"\n  ✅ {model_name} 评估完成")
    print(f"     正确性：{avg_c:.2f}  ({sum(correctness_scores)}/{len(correctness_scores)})")
    print(f"     完整性：{avg_p:.2f}  ({sum(completeness_scores)}/{len(completeness_scores)})")

    return {"correctness": avg_c, "completeness": avg_p}


# ══════════════════════════════════════════════════════
# 6. 主流程：初始化 MCP → 跑 A → 跑 B → 打印对比
# ══════════════════════════════════════════════════════

async def run_evaluation():

    print("🚀 启动 MCP session，初始化工具...")
    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await load_tools(session)
            _init_registry(loaded)
            _rebuild_graph()
            print(f"✅ 工具初始化完成，共 {len(loaded)} 个工具")

            # ── 实验 A：deepseek-chat ─────────────────────
            scores_A = await run_single_eval(
                model_name=MODEL_A,
                experiment_prefix="deepseek-chat",
            )

            # ── 实验 B：deepseek-reasoner ─────────────────
            scores_B = await run_single_eval(
                model_name=MODEL_B,
                experiment_prefix="deepseek-reasoner",
            )

            # ── 终端对比汇总 ──────────────────────────────
            print("\n" + "═" * 60)
            print("🏆  模型对比汇总")
            print("═" * 60)
            print(f"{'指标':<12} {'deepseek-chat':>16} {'deepseek-reasoner':>20}  {'胜出'}")
            print("-" * 60)

            for key, label in [("correctness", "正确性"), ("completeness", "完整性")]:
                a, b = scores_A[key], scores_B[key]
                winner = (
                    "🔵 chat"     if a > b  else
                    "🟢 reasoner" if b > a  else
                    "🤝 平局"
                )
                print(f"{label:<12} {a:>16.2f} {b:>20.2f}  {winner}")

            print("═" * 60)
            print(f"\n🔗 去 LangSmith → Datasets & Experiments → {DATASET_NAME}")
            print("   → Pairwise Experiments 可以做更细粒度的逐条对比")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_evaluation())

    # uv run python tests/eval_e2e_v2.py