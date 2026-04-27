import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "test.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        city TEXT
    )
    """)

    # orders
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        amount REAL,
        created_at TEXT
    )
    """)

    # 避免重复插入（关键优化）
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:

        cur.executescript("""
        INSERT INTO users VALUES
        (1, 'Alice', 25, 'Toronto'),
        (2, 'Bob', 35, 'Toronto'),
        (3, 'Charlie', 40, 'Vancouver');

        INSERT INTO orders VALUES
        (1, 1, 120.5, '2024-01-01'),
        (2, 1, 80.0, '2024-01-02'),
        (3, 2, 200.0, '2024-01-03'),
        (4, 3, 50.0, '2024-01-04');
        """)

    conn.commit()
    conn.close()

    print("DB initialized at:", DB_PATH)


# 允许直接运行
if __name__ == "__main__":
    init_db()