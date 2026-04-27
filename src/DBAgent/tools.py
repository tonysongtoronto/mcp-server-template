import sqlite3
from DB.init_db import DB_PATH


def query_db(sql: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def execute_db(sql: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()