"""
src/DB/schema.py

从 SQLite 数据库自动读取 schema，不再手动维护。
提供两种格式：
  - get_schema_dict()  → dict，供代码逻辑使用
  - get_schema_text()  → str，供 LLM prompt 使用
"""

import sqlite3
from pathlib import Path
from functools import lru_cache

DB_PATH = Path(__file__).parent / "ecommerce.db"


def get_schema_dict() -> dict:
    """
    从 DB 自动读取所有表的结构。
    返回格式：
    {
      "users": {
        "columns": {"id": "INTEGER", "name": "TEXT", ...},
        "primary_keys": ["id"],
        "foreign_keys": [{"from": "user_id", "to_table": "users", "to_col": "id"}]
      },
      ...
    }
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 获取所有用户表
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    schema = {}
    for table in tables:
        # 列信息
        cur.execute(f"PRAGMA table_info({table})")
        cols_info = cur.fetchall()
        # (cid, name, type, notnull, dflt_value, pk)
        columns = {row[1]: row[2] for row in cols_info}
        primary_keys = [row[1] for row in cols_info if row[5] > 0]

        # 外键信息
        cur.execute(f"PRAGMA foreign_key_list({table})")
        fk_info = cur.fetchall()
        # (id, seq, table, from, to, on_update, on_delete, match)
        foreign_keys = [
            {"from": row[3], "to_table": row[2], "to_col": row[4]}
            for row in fk_info
        ]

        # 行数（给 LLM 提示数据规模）
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cur.fetchone()[0]

        schema[table] = {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "row_count": row_count,
        }

    conn.close()
    return schema


def get_schema_text() -> str:
    """
    生成 LLM 友好的 schema 描述文本。

    示例输出：
      Table: users (20 rows)
        Columns: id(INTEGER), name(TEXT), email(TEXT), ...
        PK: id
        FK: (none)

      Table: orders (20 rows)
        Columns: id(INTEGER), user_id(INTEGER), ...
        PK: id
        FK: user_id → users.id
    """
    schema = get_schema_dict()
    lines = []

    for table, info in schema.items():
        lines.append(f"Table: {table} ({info['row_count']} rows)")

        col_str = ", ".join(
            f"{col}({dtype})" for col, dtype in info["columns"].items()
        )
        lines.append(f"  Columns: {col_str}")

        pk_str = ", ".join(info["primary_keys"]) if info["primary_keys"] else "none"
        lines.append(f"  PK: {pk_str}")

        if info["foreign_keys"]:
            for fk in info["foreign_keys"]:
                lines.append(f"  FK: {fk['from']} → {fk['to_table']}.{fk['to_col']}")
        else:
            lines.append("  FK: (none)")

        lines.append("")  # 空行分隔

    return "\n".join(lines)


if __name__ == "__main__":
    print(get_schema_text())