from DB.schema import get_schema_text
import os
import re
from dotenv import load_dotenv

from dotenv import load_dotenv
from pathlib import Path

# 明确指定 .env 路径（项目根目录）
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# =========================
# 🧹 SQL 清洗
# =========================
def clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^sql:\s*", "", sql, flags=re.IGNORECASE)
    return sql.strip()


# =========================
# 🧠 懒加载 LLM 和 Optimizer（调用时才初始化，不在 import 时执行）
# =========================
_llm = None
_optimizer = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
        )
    return _llm

def get_optimizer():
    global _optimizer
    if _optimizer is None:
        from DBAgent.optimizer import SQLOptimizer
        _optimizer = SQLOptimizer()
    return _optimizer


# =========================
# 🧠 NL → SQL
# =========================
def nl_to_sql(question: str) -> str:
    try:
        schema = get_schema_text()
        prompt = f"""
数据库：
{schema}

问题：
{question}

只输出SQL
"""
        response = get_llm().invoke(prompt)
        return clean_sql(response.content)

    except Exception as e:
        return f"-- Error: {e}\nSELECT * FROM users"


# =========================
# 🧠 结果解释
# =========================
def explain_result(question: str, result):
    prompt = f"""
用户问题：
{question}

查询结果：
{result}

用一句话总结结果。
"""
    response = get_llm().invoke(prompt)
    return response.content.strip()


# =========================
# 🚀 MCP 核心入口
# =========================
def run(question: str):
    from DBAgent.tools import query_db, execute_db

    # 1. NL → SQL
    sql = nl_to_sql(question)

    # 2. 优化 SQL
    try:
        optimized = get_optimizer().optimize(sql)
    except ValueError as e:
        return {"error": str(e), "sql": sql, "result": None}

    # 3. 执行 SQL
    try:
        if optimized.action == "query_db":
            result = query_db(optimized.sql)
        else:
            result = execute_db(optimized.sql)
    except Exception as e:
        return {"error": f"执行失败: {str(e)}", "sql": optimized.sql, "result": None}

    # 4. 返回结果
    return {
        "sql": optimized.sql,
        "action": optimized.action,
        "result": result
    }