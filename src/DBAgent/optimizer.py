"""
src/DBAgent/optimizer.py

SQL 安全检查 + 优化：
  - 拦截危险语句（DROP / TRUNCATE）
  - UPDATE 必须有 WHERE
  - SELECT * 自动展开为具名列
  - SELECT 自动加 LIMIT 保护
  - 判断 action 类型（query_db / execute_db）
"""

import re
from dataclasses import dataclass
from DB.schema import get_schema_dict


@dataclass
class OptimizedResult:
    sql: str
    action: str   # "query_db" | "execute_db"


class SQLOptimizer:

    # ── 公共入口 ─────────────────────────────────────────────────────
    def optimize(self, sql: str) -> OptimizedResult:
        sql = sql.strip()

        # 1. 拦截危险语句
        if not self._block_dangerous(sql):
            raise ValueError(f"Dangerous SQL blocked: {sql[:80]}")

        sql_type = self._classify(sql)

        if sql_type == "SELECT":
            sql = self._expand_star(sql)
            sql = self._add_limit(sql)
            return OptimizedResult(sql=sql, action="query_db")

        if sql_type == "UPDATE":
            if not self._has_where(sql):
                raise ValueError("Unsafe UPDATE: missing WHERE clause")
            return OptimizedResult(sql=sql, action="execute_db")

        if sql_type in ("INSERT", "DELETE"):
            return OptimizedResult(sql=sql, action="execute_db")

        # 未知类型：放行但标记为 query_db（兜底）
        return OptimizedResult(sql=sql, action="query_db")

    # ── 私有方法 ─────────────────────────────────────────────────────
    def _classify(self, sql: str) -> str:
        first = sql.strip().split()[0].upper() if sql.strip() else ""
        return first if first in ("SELECT", "INSERT", "UPDATE", "DELETE") else "UNKNOWN"

    def _block_dangerous(self, sql: str) -> bool:
        lower = sql.lower()
        for kw in ("drop", "truncate", "alter", "create", "attach", "detach"):
            if re.search(rf"\b{kw}\b", lower):
                return False
        return True

    def _has_where(self, sql: str) -> bool:
        return bool(re.search(r"\bwhere\b", sql, re.IGNORECASE))

    def _expand_star(self, sql: str) -> str:
        """把 SELECT * FROM <table> 展开为具名列"""
        match = re.search(r"select\s+\*\s+from\s+(\w+)", sql, re.IGNORECASE)
        if not match:
            return sql
        table = match.group(1).lower()
        schema = get_schema_dict()
        if table not in schema:
            return sql
        columns = ", ".join(schema[table]["columns"].keys())
        return re.sub(
            r"select\s+\*\s+from",
            f"SELECT {columns} FROM",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

    def _add_limit(self, sql: str, limit: int = 100) -> str:
        """SELECT 语句没有 LIMIT 时自动添加"""
        if re.search(r"\blimit\b", sql, re.IGNORECASE):
            return sql
        sql = sql.rstrip(";").rstrip()
        return f"{sql} LIMIT {limit}"