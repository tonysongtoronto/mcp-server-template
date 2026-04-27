from DBAgent.optimizer import SQLOptimizer
from DBAgent.tools import query_db, execute_db
from DB.schema import get_schema_text
from langchain_openai import ChatOpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

# =========================
# 🧠 LLM
# =========================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# =========================
# 🧹 SQL 清洗
# =========================
def clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^sql:\s*", "", sql, flags=re.IGNORECASE)
    return sql.strip()


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

        response = llm.invoke(prompt)
        return clean_sql(response.content)

    except Exception as e:
        return "SELECT * FROM users"


# =========================
# 🧠 SQL Optimizer
# =========================
optimizer = SQLOptimizer()


# =========================
# 🧠 结果解释（可选扩展用）
# =========================
def explain_result(question: str, result):
    prompt = f"""
用户问题：
{question}

查询结果：
{result}

用一句话总结结果。
"""

    response = llm.invoke(prompt)
    return response.content.strip()


# =========================
# 🚀 MCP 核心入口
# =========================
def run(question: str):
    # 1. NL → SQL
    sql = nl_to_sql(question)

    # 2. 优化 SQL（安全检查 + 补 LIMIT + 展开 SELECT * 等）
    try:
        optimized = optimizer.optimize(sql)
    except ValueError as e:
        return {
            "error": str(e),
            "sql": sql,
            "result": None
        }

    # 3. 执行 SQL
    try:
        if optimized.action == "query_db":
            result = query_db(optimized.sql)
        else:
            result = execute_db(optimized.sql)
    except Exception as e:
        return {
            "error": f"执行失败: {str(e)}",
            "sql": optimized.sql,
            "result": None
        }

    # 4. 返回结果（JSON 友好格式）
    return {
        "sql": optimized.sql,
        "action": optimized.action,
        "result": result
    }