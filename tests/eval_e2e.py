"""
eval_e2e.py

dataset_e2e 端到端评估脚本

运行方式：
    uv run python tests/eval_e2e.py

流程：
    1. 启动 MCP session，初始化工具和图
    2. 从 dataset_e2e 拉取所有 example
    3. 清洗脏数据（提取干净的 input/output）
    4. aevaluate() 跑评估，LLM-as-judge 打分
    5. 结果自动上传到 LangSmith Experiments
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
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
# 1. LLM-as-judge 用的模型（同样用 deepseek）
# ══════════════════════════════════════════════════════
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ══════════════════════════════════════════════════════
# 2. 脏数据清洗函数
# ══════════════════════════════════════════════════════

# role 字段的各种写法映射到标准类型
ROLE_MAP = {
    "human":     "human",
    "user":      "human",
    "ai":        "ai",
    "assistant": "ai",
    "system":    "system",
    "tool":      "tool",
}

def _normalize_role(msg: dict) -> str:
    """统一 role 字段，兼容各种写法"""
    role = msg.get("role") or msg.get("type") or ""
    return ROLE_MAP.get(role.lower(), role.lower())


def _normalize_content(msg: dict) -> str:
    """统一 content 字段，兼容各种写法"""
    return (
        msg.get("content")
        or msg.get("text")
        or msg.get("output")
        or ""
    )


def _parse_lc_serialized(msg: dict) -> dict | None:
    """
    解析 LangChain 序列化格式：
    {
      "id": ["langchain", "schema", "messages", "SystemMessage"],
      "kwargs": {"content": "...", "type": "system"},
      "type": "constructor"
    }
    返回标准化的 {"role": "...", "content": "..."} dict，解析失败返回 None。
    """
    if msg.get("type") != "constructor":
        return None

    id_list = msg.get("id", [])
    kwargs  = msg.get("kwargs", {})
    content = kwargs.get("content", "")

    if not content or not id_list:
        return None

    # 从 id 列表最后一项推断 role
    class_name = id_list[-1] if id_list else ""
    role_map = {
        "SystemMessage": "system",
        "HumanMessage":  "human",
        "AIMessage":     "ai",
        "ToolMessage":   "tool",
    }
    role = role_map.get(class_name)
    if not role:
        # 兜底：从 kwargs.type 字段推断
        role = ROLE_MAP.get(kwargs.get("type", "").lower(), "human")

    return {"role": role, "content": content}


def _flatten_messages(messages: list) -> list:
    """
    展平嵌套数组，兼容 messages: [[msg1, msg2]] 的情况。
    LangSmith 有时会把消息列表多包一层。
    """
    if messages and isinstance(messages[0], list):
        flat = []
        for item in messages:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return messages


def extract_messages_as_lc(inputs: dict) -> list:
    """
    从任意格式的 inputs 里提取消息列表，转成 LangChain 消息对象。

    兼容以下情况：
      1. {"messages": [{"role": "human", "content": "..."}]}            ← 标准格式
      2. {"messages": [{"role": "user", "content": "..."}]}             ← user 写法
      3. {"messages": [{"type": "human", ...}]}                         ← type 字段
      4. {"messages": [system消息, human消息, ...]}                     ← 带 system
      5. {"messages": [[msg1, msg2]]}                                    ← 嵌套数组（展平）
      6. {"messages": [{"id": [..., "SystemMessage"], "kwargs": {...}}]} ← LangChain 序列化格式
      7. {"question": "..."}                                             ← 直接字符串
      8. {"input": "..."}                                                ← input 字段
      9. LangChain 对象（已经是 BaseMessage）                            ← 直接用
    """
    lc_messages = []
    raw_messages = inputs.get("messages", [])

    # ★ 展平嵌套数组，兼容 [[msg1, msg2]] 情况
    messages = _flatten_messages(raw_messages)

    for msg in messages:
        # 已经是 LangChain 消息对象，直接用
        if hasattr(msg, "content"):
            lc_messages.append(msg)
            continue

        if not isinstance(msg, dict):
            continue

        # ★ 兼容 LangChain 序列化格式（type=constructor）
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
            # 未知 role，当作 human 处理
            lc_messages.append(HumanMessage(content=content))

    # 兜底：如果 messages 为空，尝试其他字段
    if not lc_messages:
        fallback = (
            inputs.get("question")
            or inputs.get("input")
            or inputs.get("content")
            or ""
        )
        if fallback:
            lc_messages.append(HumanMessage(content=str(fallback)))

    return lc_messages


def extract_human_question(inputs: dict) -> str:
    """只提取用户问题文本，用于 judge prompt 展示"""
    messages = _flatten_messages(inputs.get("messages", []))
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict):
            # 兼容 LangChain 序列化格式
            parsed = _parse_lc_serialized(msg)
            if parsed:
                msg = parsed
            if _normalize_role(msg) == "human":
                return _normalize_content(msg)
    return (
        inputs.get("question")
        or inputs.get("input")
        or inputs.get("content")
        or ""
    )


def extract_reference(example_outputs: dict) -> str:
    """
    从脏 output 里提取期望答案。
    兼容：
      1. {"answer": "..."}                         ← 手动填写的干净格式（最优先）
      2. {"output": "..."}                         ← 其他干净格式
      3. {"messages": [...ai消息...]}              ← 顶层 trace 保存
      4. {"generations": [[{"message": {...}}]]}   ← ChatOpenAI 子节点保存
         - message.content 有值则直接用
         - content 为空但有 tool_calls，说明是中间节点，返回空字符串
    """
    # 情况零：手动填写的干净格式（最优先）
    clean = (
        example_outputs.get("answer")
        or example_outputs.get("output")
        or example_outputs.get("content")
    )
    if clean:
        return clean

    # 情况一：messages 结构，取最后一条 AI 消息
    messages = example_outputs.get("messages", [])
    ai_contents = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg, AIMessage):
            if msg.content:
                ai_contents.append(msg.content)
            continue
        if not isinstance(msg, dict):
            continue
        # 兼容 LangChain 序列化格式
        parsed = _parse_lc_serialized(msg)
        if parsed:
            msg = parsed
        if _normalize_role(msg) in ("ai", "assistant"):
            content = _normalize_content(msg)
            if content:
                ai_contents.append(content)
    if ai_contents:
        return ai_contents[-1]

    # 情况二：generations 结构（ChatOpenAI 子节点自动保存的格式）
    generations = example_outputs.get("generations", [])
    for gen in generations:
        if isinstance(gen, list):
            gen = gen[0] if gen else {}
        if not isinstance(gen, dict):
            continue

        message = gen.get("message", {})
        if isinstance(message, dict):
            kwargs  = message.get("kwargs", message)  # 兼容序列化格式
            content = kwargs.get("content", "") or _normalize_content(message)
            # ★ content 为空且有 tool_calls，说明是中间节点，跳过
            tool_calls = kwargs.get("tool_calls", [])
            if not content and tool_calls:
                print("  ⚠️ Reference Output 是中间节点（tool_calls），无法用于 e2e 评估，跳过")
                return ""
            if content:
                return content

        # 直接从 gen 取
        content = _normalize_content(gen)
        if content:
            return content

    return ""


def _inject_system_into_human(lc_messages: list) -> list:
    """
    把 SystemMessage 的内容注入到第一条 HumanMessage 的末尾。
    SystemMessage 可有可无：有就注入，没有原样返回。
    注入后去掉原 SystemMessage，只保留 human/ai/tool 消息。
    """
    system_msgs = [m for m in lc_messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in lc_messages if not isinstance(m, SystemMessage)]

    if not system_msgs:
        return lc_messages

    style_hint = system_msgs[0].content  # 只取第一条 system 消息

    # 找到第一条 HumanMessage，把风格/提示词内容拼进去
    for i, msg in enumerate(non_system):
        if isinstance(msg, HumanMessage):
            non_system[i] = HumanMessage(
                content=f"{msg.content}\n\n【回复风格要求】{style_hint}"
            )
            break

    return non_system


# ══════════════════════════════════════════════════════
# 3. target 函数（跑你的 agent）
# ══════════════════════════════════════════════════════

async def target(inputs: dict) -> dict:
    """
    LangSmith 把每条 example 的 inputs 传进来。
    支持多种输入格式，清洗后喂给 agent，返回最终答案。

    ★ SystemMessage（如风格要求）可有可无：
      有 → 注入到第一条 HumanMessage 末尾，确保 agent 能看到
      无 → 正常传递，不做任何处理
    """
    import langgraph_stdio_agent as agent_module

    # 提取完整消息列表（包含 system、human 等）
    lc_messages = extract_messages_as_lc(inputs)
    if not lc_messages:
        print("⚠️ 无法提取消息，跳过该条")
        return {"answer": ""}

    # ★ 把 SystemMessage 注入到 HumanMessage（agent 内部会忽略 SystemMessage）
    lc_messages = _inject_system_into_human(lc_messages)

    result = await agent_module.graph.ainvoke({
        "messages": lc_messages,
        "task_plan": [],
        "next_agent": "",
    })

    # 取最后一条 AI 消息作为最终答案
    result_messages = result.get("messages", [])
    for msg in reversed(result_messages):
        if hasattr(msg, "content") and msg.content:
            return {"answer": msg.content}
        if isinstance(msg, dict) and _normalize_content(msg):
            return {"answer": _normalize_content(msg)}

    return {"answer": ""}


# ══════════════════════════════════════════════════════
# 4. LLM-as-judge Evaluator
# ══════════════════════════════════════════════════════

def make_judge_evaluator():
    """返回 LLM-as-judge 评估函数"""

    async def llm_as_judge(run, example) -> dict:
        question  = extract_human_question(example.inputs)
        reference = extract_reference(example.outputs)
        actual    = run.outputs.get("answer", "") if run.outputs else ""

        # reference 为空说明是脏数据（中间节点），跳过评分
        if not reference:
            print("  ⚠️ reference 为空，跳过 judge 评分（example 数据不完整）")
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
            # 去掉可能的 markdown 代码块
            if "```" in raw:
                raw = raw.split("```")[1]
                raw = raw.lstrip("json").strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"⚠️ judge 解析失败：{e}，原始输出：{response.content}")
            result = {"correctness": 0, "completeness": 0, "reason": "解析失败"}

        # LangSmith 支持返回多个评分
        return [
            {
                "key": "correctness",
                "score": result.get("correctness", 0),
                "comment": result.get("reason", ""),
            },
            {
                "key": "completeness",
                "score": result.get("completeness", 0),
                "comment": result.get("reason", ""),
            },
        ]

    return llm_as_judge


# ══════════════════════════════════════════════════════
# 5. 主评估流程
# ══════════════════════════════════════════════════════

async def run_evaluation():

    # ── 第一步：初始化 MCP + 工具 + 图 ──────────────────
    print("🚀 启动 MCP session，初始化工具...")
    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await load_tools(session)
            _init_registry(loaded)
            _rebuild_graph()
            print(f"✅ 工具初始化完成，共 {len(loaded)} 个工具")

            # ── 第二步：跑评估 ───────────────────────────
            print("\n📊 开始评估 dataset_e2e ...")
            results = await aevaluate(
                target,
                data="dataset_e2e",
                evaluators=[make_judge_evaluator()],
                experiment_prefix="e2e-deepseek-judge",
                max_concurrency=1,      # 先串行跑，稳定后可以调大
            )

            # ── 第三步：打印汇总 ─────────────────────────
            print("\n✅ 评估完成！结果已上传到 LangSmith Experiments。")
            print("=" * 60)

            correctness_scores  = []
            completeness_scores = []

            async for r in results:
                eval_results = r.get("evaluation_results", {}).get("results", [])
                for er in eval_results:
                    if er.key == "correctness":
                        correctness_scores.append(er.score)
                    elif er.key == "completeness":
                        completeness_scores.append(er.score)

            if correctness_scores:
                avg_c = sum(correctness_scores) / len(correctness_scores)
                avg_p = sum(completeness_scores) / len(completeness_scores)
                print(f"📈 正确性平均分：{avg_c:.2f}  ({sum(correctness_scores)}/{len(correctness_scores)})")
                print(f"📈 完整性平均分：{avg_p:.2f}  ({sum(completeness_scores)}/{len(completeness_scores)})")

            print("\n🔗 在 LangSmith UI 的 Experiments 标签页查看详细结果")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run_evaluation())

    # uv run python tests/eval_e2e.py