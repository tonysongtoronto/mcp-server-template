from DBAgent.agent import run


def sql_agent_tool(question: str):
    """
    MCP Tool:
    自然语言 → SQL → 执行 → 结果解释
    """
    return run(question)