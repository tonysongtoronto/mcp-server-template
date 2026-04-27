"""
src/DBAgent/agent.py

核心 Agent：自然语言 → SQL → 执行 → 返回结果
对外暴露：run(question) → dict
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


# ── 日志 ──────────────────────────────────────────────────────────────
def _log(msg: str):
    print(f"[DBAgent] {msg}", file=sys.stderr, flush=True)


# ── SQL 清洗 ──────────────────────────────────────────────────────────
def _clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^sql[:\s]+", "", sql, flags=re.IGNORECASE)
    return sql.strip()


# ── 懒加载 LLM / Optimizer ────────────────────────────────────────────
_llm = None
_optimizer = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        _log("LLM initialized ✅")
    return _llm


def _get_optimizer():
    global _optimizer
    if _optimizer is None:
        from DBAgent.optimizer import SQLOptimizer
        _optimizer = SQLOptimizer()
        _log("Optimizer initialized ✅")
    return _optimizer


# ── NL → SQL ──────────────────────────────────────────────────────────
def nl_to_sql(question: str) -> str:
    from DB.schema import get_schema_text
    schema = get_schema_text()
    prompt = f"""你是一个专业的 SQL 生成器。

数据库 schema（SQLite 语法）：
{schema}

用户问题：
{question}

要求：
- 只输出 SQL 语句，不要任何解释或 markdown 代码块
- 使用标准 SQLite 语法
- 需要 JOIN 时使用表别名提高可读性
- 日期字段类型为 TEXT（格式 YYYY-MM-DD），比较时直接用字符串
"""
    try:
        response = _get_llm().invoke(prompt)
        sql = _clean_sql(response.content)
        _log(f"NL→SQL: {sql}")
        return sql
    except Exception as e:
        _log(f"NL→SQL error: {e}")
        return f"-- Error: {e}\nSELECT 'error' as msg"


# ── 核心入口 ──────────────────────────────────────────────────────────
def run(question: str) -> dict:
    """
    完整流程：NL → SQL → 优化 → 执行
    返回：
    {
      "sql":    str,            # 最终执行的 SQL
      "action": str,            # "query_db" | "execute_db"
      "result": list[dict] | dict | None,
      "error":  str | None
    }
    """
    from DBAgent.tools import query_db, execute_db

    # Step 1: NL → SQL
    _log(f"Question: {question}")
    sql = nl_to_sql(question)

    # Step 2: 安全检查 + 优化
    try:
        optimized = _get_optimizer().optimize(sql)
        _log(f"Optimized SQL: {optimized.sql}  action={optimized.action}")
    except ValueError as e:
        _log(f"Optimizer blocked: {e}")
        return {"sql": sql, "action": None, "result": None, "error": str(e)}

    # Step 3: 执行
    try:
        if optimized.action == "query_db":
            result = query_db(optimized.sql)
        else:
            result = execute_db(optimized.sql)
        _log(f"Executed OK, result rows: {len(result) if isinstance(result, list) else result}")
        return {"sql": optimized.sql, "action": optimized.action, "result": result, "error": None}
    except Exception as e:
        _log(f"Execute error: {e}")
        return {"sql": optimized.sql, "action": optimized.action, "result": None, "error": str(e)}