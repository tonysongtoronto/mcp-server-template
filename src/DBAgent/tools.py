"""
src/DBAgent/tools.py

底层数据库执行工具。
- query_db()   → SELECT，返回 [{"col": val, ...}] 字典列表
- execute_db() → INSERT / UPDATE，返回状态字典
"""

import sqlite3
from DB.init_db import DB_PATH


def query_db(sql: str) -> list[dict]:
    """
    执行 SELECT 语句，返回带列名的字典列表。
    例：[{"id": 1, "name": "Alice", "city": "Toronto"}, ...]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # ← 关键：让 cursor 返回带列名的行
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # sqlite3.Row 支持 keys()，转成普通 dict
        return [dict(row) for row in rows]
    finally:
        conn.close()


def execute_db(sql: str) -> dict:
    """
    执行 INSERT / UPDATE 语句，返回影响行数。
    例：{"status": "ok", "rows_affected": 1}
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        return {"status": "ok", "rows_affected": cur.rowcount}
    finally:
        conn.close()