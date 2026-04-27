

from DB.schema import SCHEMA, get_schema_text  
import re

from dataclasses import dataclass

@dataclass
class OptimizedResult:
    sql: str
    action: str   # query_db / execute_db




class SQLOptimizer:
    def __init__(self):
        self.schema_text = get_schema_text()


    def optimize(self, sql: str) -> OptimizedResult:
        if not self._block_dangerous_sql(sql):
            raise ValueError("Dangerous SQL blocked")

        sql_type = self.classify_sql(sql)

        action = "query_db"

        if sql_type == "SELECT":
            sql = self._rewrite_select_star(sql)
            sql = self._add_limit(sql)
            action = "query_db"

        elif sql_type in ["INSERT", "UPDATE"]:
            action = "execute_db"

            if sql_type == "UPDATE":
                if not self._check_update_safe(sql):
                    raise ValueError("Unsafe UPDATE: missing WHERE clause")

        return OptimizedResult(sql=sql, action=action)
    
    
    
  
    def _rewrite_select_star(self, sql: str) -> str:
        match = re.search(r"select\s+\*\s+from\s+(\w+)", sql, re.IGNORECASE)
        if not match:
            return sql

        table = match.group(1)

        if table not in SCHEMA:
            return sql

        columns = ", ".join(SCHEMA[table]["columns"].keys())

        return re.sub(
            r"select\s+\*\s+from",
            f"SELECT {columns} FROM",
            sql,
            flags=re.IGNORECASE
        )
        
    def _add_limit(self, sql: str, limit: int = 100) -> str:
        if "limit" in sql.lower():
            return sql

        sql = sql.rstrip(";")
        return f"{sql} LIMIT {limit}"
    

    
    def _check_update_safe(self, sql: str) -> bool:
        sql_lower = sql.lower()
        if "update" in sql_lower and "where" not in sql_lower:
            return False
        return True
    
    def _block_dangerous_sql(self, sql: str) -> bool:
        sql_lower = sql.lower()
        forbidden = ["drop", "truncate"]
        
        for kw in forbidden:
            if kw in sql_lower:
                return False

        return True
    
    def classify_sql(self, sql: str) -> str:
        sql = sql.strip().lower()

        if sql.startswith("select"):
            return "SELECT"
        if sql.startswith("insert"):
            return "INSERT"
        if sql.startswith("update"):
            return "UPDATE"
        if sql.startswith("delete"):
            return "DELETE"

        return "UNKNOWN"