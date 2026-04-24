"""
eval_simple.py — 本地 DeepSeek judge 版

运行方式：
    uv run python tests/eval_simple.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import aevaluate
from mcp import ClientSession
from mcp.client.stdio import stdio_client

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from langgraph_stdio_agent import (
    load_tools,
    _init_registry,
    _rebuild_graph,
    mcp_params,
)

# ══════════════════════════════════════════════════════
# DeepSeek judge 模型
# ══════════════════════════════════════════════════════
judge_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ══════════════════════════════════════════════════════
# target：把每条 example 喂给 agent，返回回答
# ══════════════════════════════════════════════════════
async def target(inputs: dict) -> dict:
    import langgraph_stdio_agent as agent_module

    question = inputs.get("question", "")
    if not question:
        return {"output": ""}

    result = await agent_module.graph.ainvoke({
        "messages":   [HumanMessage(content=question)],
        "task_plan":  [],
        "next_agent": "",
    })

    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            return {"output": msg.content}

    return {"output": ""}


# ══════════════════════════════════════════════════════
# judge：用 DeepSeek 在本地打分
# ══════════════════════════════════════════════════════
async def deepseek_judge(run, example) -> dict:
    question  = (example.inputs  or {}).get("question", "")
    reference = (example.outputs or {}).get("expected_answer", "")
    actual    = (run.outputs     or {}).get("output", "")

    if not reference:
        return {"key": "correctness", "score": 0, "comment": "无期望答案"}

    prompt = f"""请判断"实际答案"是否正确回答了用户问题，并与"期望答案"语义一致。

用户问题：{question}
期望答案：{reference}
实际答案：{actual}

只返回 JSON，不要其他文字：
{{"correct": true或false, "reason": "一句话说明"}}"""

    response = await judge_llm.ainvoke([
        SystemMessage(content="你是评估专家，只返回 JSON。"),
        HumanMessage(content=prompt),
    ])

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        score = 1 if result.get("correct") else 0
        return {"key": "correctness", "score": score, "comment": result.get("reason", "")}
    except Exception:
        return {"key": "correctness", "score": 0, "comment": "解析失败"}


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════
async def main():
    print("🚀 启动 MCP session，初始化工具...")

    async with stdio_client(mcp_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await load_tools(session)
            _init_registry(loaded)
            _rebuild_graph()
            print(f"✅ 工具初始化完成，共 {len(loaded)} 个工具\n")

            print("📊 开始评估 my-agent-eval ...")
            results = await aevaluate(
                target,
                data="my-agent-eval",
                evaluators=[deepseek_judge],
                experiment_prefix="simple-eval-v2",
                max_concurrency=1,
            )

            # 打印汇总
            scores = []
            print("\n" + "=" * 55)
            async for r in results:
                question = (r.get("example", {}).inputs or {}).get("question", "?")
                output   = (r.get("run",     {}).outputs or {}).get("output", "")
                evals    = r.get("evaluation_results", {}).get("results", [])
                score    = evals[0].score if evals else "?"
                comment  = evals[0].comment if evals else ""
                scores.append(score if isinstance(score, (int, float)) else 0)
                icon = "✅" if score == 1 else "❌"
                print(f"{icon} [{score}] {question[:30]}")
                print(f"   回答：{str(output)[:60]}")
                print(f"   理由：{comment}")
                print("-" * 55)

            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n📈 正确率：{avg:.0%}  ({sum(scores)}/{len(scores)})")

            print("\n🔗 LangSmith UI → my-agent-eval → Experiments 查看详细结果")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
    
    #   uv run python tests/eval_e2e_v3.py