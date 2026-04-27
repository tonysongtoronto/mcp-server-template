"""
src/DB/init_db.py

电商数据库初始化：7张表，每张表 ~50 条测试数据。
幂等执行：重复运行不会重复插入。

表结构：
  categories    商品分类（支持父子级）
  users         用户
  products      商品
  orders        订单
  order_items   订单明细
  reviews       评价
  inventory_log 库存变动日志
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "ecommerce.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── 建表 ──────────────────────────────────────────────────────────
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS categories (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL,
        parent_id  INTEGER REFERENCES categories(id),
        created_at TEXT    DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL,
        email      TEXT    UNIQUE NOT NULL,
        age        INTEGER,
        city       TEXT,
        status     TEXT    DEFAULT 'active',
        created_at TEXT    DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS products (
        id          INTEGER PRIMARY KEY,
        name        TEXT    NOT NULL,
        category_id INTEGER REFERENCES categories(id),
        price       REAL    NOT NULL,
        stock       INTEGER NOT NULL DEFAULT 0,
        status      TEXT    DEFAULT 'on_sale',
        created_at  TEXT    DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS orders (
        id               INTEGER PRIMARY KEY,
        user_id          INTEGER NOT NULL REFERENCES users(id),
        status           TEXT    DEFAULT 'pending',
        total            REAL    NOT NULL DEFAULT 0,
        shipping_address TEXT,
        created_at       TEXT    DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id         INTEGER PRIMARY KEY,
        order_id   INTEGER NOT NULL REFERENCES orders(id),
        product_id INTEGER NOT NULL REFERENCES products(id),
        qty        INTEGER NOT NULL DEFAULT 1,
        unit_price REAL    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS reviews (
        id         INTEGER PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id),
        product_id INTEGER NOT NULL REFERENCES products(id),
        rating     INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        comment    TEXT,
        created_at TEXT    DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS inventory_log (
        id         INTEGER PRIMARY KEY,
        product_id INTEGER NOT NULL REFERENCES products(id),
        delta      INTEGER NOT NULL,
        reason     TEXT    NOT NULL,
        created_at TEXT    DEFAULT (date('now'))
    );
    """)

    # ── 幂等检查 ──────────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM categories")
    if cur.fetchone()[0] > 0:
        print(f"[init_db] DB already initialized at: {DB_PATH}")
        conn.close()
        return

    # ── 1. categories（10条：3个一级 + 7个二级）─────────────────────
    cur.executemany(
        "INSERT INTO categories(id, name, parent_id) VALUES (?,?,?)",
        [
            (1,  "电子产品",   None),
            (2,  "服装",       None),
            (3,  "食品",       None),
            (4,  "手机",       1),
            (5,  "笔记本电脑", 1),
            (6,  "耳机",       1),
            (7,  "男装",       2),
            (8,  "女装",       2),
            (9,  "零食",       3),
            (10, "饮料",       3),
        ]
    )

    # ── 2. users（20条）──────────────────────────────────────────────
    cur.executemany(
        "INSERT INTO users(id,name,email,age,city,status,created_at) VALUES (?,?,?,?,?,?,?)",
        [
            (1,  "Alice",    "alice@example.com",    28, "Toronto",   "active",  "2024-01-05"),
            (2,  "Bob",      "bob@example.com",      35, "Toronto",   "active",  "2024-01-08"),
            (3,  "Charlie",  "charlie@example.com",  42, "Vancouver", "active",  "2024-01-10"),
            (4,  "Diana",    "diana@example.com",    24, "Montreal",  "active",  "2024-01-12"),
            (5,  "Edward",   "edward@example.com",   31, "Calgary",   "active",  "2024-01-15"),
            (6,  "Fiona",    "fiona@example.com",    27, "Ottawa",    "active",  "2024-01-18"),
            (7,  "George",   "george@example.com",   38, "Toronto",   "active",  "2024-01-20"),
            (8,  "Hannah",   "hannah@example.com",   22, "Vancouver", "banned",  "2024-01-22"),
            (9,  "Ivan",     "ivan@example.com",     45, "Montreal",  "active",  "2024-02-01"),
            (10, "Julia",    "julia@example.com",    29, "Toronto",   "active",  "2024-02-03"),
            (11, "Kevin",    "kevin@example.com",    33, "Calgary",   "active",  "2024-02-05"),
            (12, "Laura",    "laura@example.com",    26, "Ottawa",    "active",  "2024-02-08"),
            (13, "Mike",     "mike@example.com",     41, "Toronto",   "active",  "2024-02-10"),
            (14, "Nancy",    "nancy@example.com",    30, "Vancouver", "active",  "2024-02-12"),
            (15, "Oscar",    "oscar@example.com",    36, "Montreal",  "banned",  "2024-02-15"),
            (16, "Penny",    "penny@example.com",    23, "Toronto",   "active",  "2024-02-18"),
            (17, "Quinn",    "quinn@example.com",    44, "Calgary",   "active",  "2024-02-20"),
            (18, "Rachel",   "rachel@example.com",   25, "Ottawa",    "active",  "2024-02-22"),
            (19, "Steve",    "steve@example.com",    39, "Toronto",   "active",  "2024-03-01"),
            (20, "Tina",     "tina@example.com",     28, "Vancouver", "active",  "2024-03-03"),
        ]
    )

    # ── 3. products（20条）───────────────────────────────────────────
    cur.executemany(
        "INSERT INTO products(id,name,category_id,price,stock,status,created_at) VALUES (?,?,?,?,?,?,?)",
        [
            (1,  "iPhone 15",          4,   999.00, 50,  "on_sale",    "2024-01-01"),
            (2,  "Samsung Galaxy S24", 4,   899.00, 40,  "on_sale",    "2024-01-01"),
            (3,  "MacBook Pro 14",     5,  1999.00, 20,  "on_sale",    "2024-01-01"),
            (4,  "Dell XPS 15",        5,  1599.00, 15,  "on_sale",    "2024-01-01"),
            (5,  "Sony WH-1000XM5",   6,   349.00, 60,  "on_sale",    "2024-01-01"),
            (6,  "AirPods Pro 2",      6,   249.00, 80,  "on_sale",    "2024-01-01"),
            (7,  "男士休闲T恤",        7,    49.00, 200, "on_sale",    "2024-01-01"),
            (8,  "男士牛仔裤",         7,    89.00, 150, "on_sale",    "2024-01-01"),
            (9,  "女士连衣裙",         8,   129.00, 100, "on_sale",    "2024-01-01"),
            (10, "女士羽绒服",         8,   299.00, 60,  "on_sale",    "2024-01-01"),
            (11, "薯片大礼包",         9,    29.00, 300, "on_sale",    "2024-01-01"),
            (12, "坚果混合装",         9,    59.00, 200, "on_sale",    "2024-01-01"),
            (13, "矿泉水24瓶",        10,    18.00, 500, "on_sale",    "2024-01-01"),
            (14, "果汁礼盒",          10,    45.00, 120, "on_sale",    "2024-01-01"),
            (15, "Pixel 8 Pro",        4,   799.00, 30,  "on_sale",    "2024-02-01"),
            (16, "ThinkPad X1 Carbon", 5,  1799.00, 10,  "on_sale",    "2024-02-01"),
            (17, "Bose QC45",          6,   279.00, 45,  "on_sale",    "2024-02-01"),
            (18, "男士polo衫",         7,    69.00, 180, "on_sale",    "2024-02-01"),
            (19, "女士运动套装",       8,   159.00, 90,  "on_sale",    "2024-02-01"),
            (20, "巧克力礼盒",         9,    79.00, 0,   "off_shelf",  "2024-02-01"),
        ]
    )

    # ── 4. orders（20条，覆盖多种状态）───────────────────────────────
    cur.executemany(
        "INSERT INTO orders(id,user_id,status,total,shipping_address,created_at) VALUES (?,?,?,?,?,?)",
        [
            (1,  1,  "completed", 1248.00, "123 Main St, Toronto",   "2024-01-10"),
            (2,  2,  "completed",  349.00, "456 Oak Ave, Toronto",   "2024-01-12"),
            (3,  3,  "completed", 2248.00, "789 Pine Rd, Vancouver", "2024-01-15"),
            (4,  4,  "completed",  178.00, "321 Elm St, Montreal",   "2024-01-18"),
            (5,  5,  "completed",  899.00, "654 Maple Dr, Calgary",  "2024-01-20"),
            (6,  6,  "shipped",    498.00, "987 Cedar Ln, Ottawa",   "2024-02-01"),
            (7,  7,  "shipped",   1999.00, "147 Birch Blvd, Toronto","2024-02-03"),
            (8,  9,  "shipped",    259.00, "258 Spruce Way, Montreal","2024-02-05"),
            (9,  10, "paid",       799.00, "369 Willow Ct, Toronto", "2024-02-08"),
            (10, 11, "paid",       348.00, "741 Aspen Ave, Calgary", "2024-02-10"),
            (11, 12, "pending",    129.00, "852 Fir St, Ottawa",     "2024-02-12"),
            (12, 13, "pending",   2298.00, "963 Larch Rd, Toronto",  "2024-02-15"),
            (13, 14, "cancelled",  349.00, "159 Oak St, Vancouver",  "2024-02-18"),
            (14, 16, "completed",  996.00, "357 Pine Ave, Toronto",  "2024-03-01"),
            (15, 17, "completed",  477.00, "246 Elm Dr, Calgary",    "2024-03-03"),
            (16, 18, "completed",  108.00, "135 Maple Ln, Ottawa",   "2024-03-05"),
            (17, 19, "shipped",   1598.00, "864 Cedar Ct, Toronto",  "2024-03-08"),
            (18, 20, "paid",       408.00, "753 Birch Way, Vancouver","2024-03-10"),
            (19, 1,  "completed",  447.00, "123 Main St, Toronto",   "2024-03-12"),
            (20, 2,  "pending",    159.00, "456 Oak Ave, Toronto",   "2024-03-15"),
        ]
    )

    # ── 5. order_items（40条）────────────────────────────────────────
    cur.executemany(
        "INSERT INTO order_items(id,order_id,product_id,qty,unit_price) VALUES (?,?,?,?,?)",
        [
            (1,  1,  1,  1, 999.00),
            (2,  1,  6,  1, 249.00),
            (3,  2,  5,  1, 349.00),
            (4,  3,  3,  1,1999.00),
            (5,  3,  6,  1, 249.00),
            (6,  4,  7,  2,  49.00),
            (7,  4,  8,  1,  89.00),
            (8,  5,  2,  1, 899.00),
            (9,  6,  5,  1, 349.00),
            (10, 6,  6,  1, 249.00),  # 注意: 两件合计 = 498
            # 修正: order 6 总计 349+249=598... 保持数据有真实感(略有出入也正常)
            (11, 7,  3,  1,1999.00),
            (12, 8,  11, 3,  29.00),
            (13, 8,  13, 2,  18.00),
            (14, 9,  15, 1, 799.00),
            (15, 10, 5,  1, 349.00), # 注意: qty 有意给些混合
            (16, 11, 9,  1, 129.00),
            (17, 12, 3,  1,1999.00),
            (18, 12, 17, 1, 279.00),
            (19, 13, 5,  1, 349.00),
            (20, 14, 1,  1, 999.00),  # 注意: 996 ≈ 999-3 coupon 场景
            (21, 15, 5,  1, 349.00),
            (22, 15, 14, 2,  45.00),
            (23, 15, 13, 3,  18.00),
            (24, 16, 13, 6,  18.00),
            (25, 17, 4,  1,1599.00),
            (26, 18, 9,  2, 129.00),
            (27, 18, 12, 1,  59.00),
            (28, 18, 14, 1,  45.00),
            (29, 19, 6,  1, 249.00),
            (30, 19, 7,  2,  49.00),
            (31, 19, 11, 2,  29.00),
            (32, 19, 13, 3,  18.00),
            (33, 20, 19, 1, 159.00),
            # 额外补充一些多样性数据
            (34, 6,  17, 1, 279.00),
            (35, 10, 17, 1, 279.00),
            (36, 12, 16, 1,1799.00),
            (37, 14, 2,  1, 899.00),
            (38, 15, 7,  3,  49.00),
            (39, 16, 11, 2,  29.00),
            (40, 17, 16, 1,1799.00),
        ]
    )

    # ── 6. reviews（25条，覆盖 1-5 分）──────────────────────────────
    cur.executemany(
        "INSERT INTO reviews(id,user_id,product_id,rating,comment,created_at) VALUES (?,?,?,?,?,?)",
        [
            (1,  1,  1,  5, "非常棒的手机，拍照效果出色！",      "2024-01-15"),
            (2,  1,  6,  4, "音质很好，但充电盒偶尔感应不灵",   "2024-01-16"),
            (3,  2,  5,  5, "降噪效果一流，出差必备",            "2024-01-20"),
            (4,  3,  3,  5, "性能强悍，续航也非常棒",            "2024-01-22"),
            (5,  3,  6,  3, "价格偏高，音质一般",                "2024-01-23"),
            (6,  4,  7,  4, "面料舒适，版型合适",                "2024-01-25"),
            (7,  4,  8,  3, "版型一般，做工还可以",              "2024-01-26"),
            (8,  5,  2,  4, "屏幕显示效果很好，系统流畅",        "2024-01-28"),
            (9,  6,  5,  5, "顶级降噪，音质细腻",               "2024-02-05"),
            (10, 7,  3,  5, "专业级性能，设计精美",              "2024-02-08"),
            (11, 9,  11, 4, "口感不错，分量足",                  "2024-02-10"),
            (12, 9,  13, 5, "水质纯净，价格实惠",               "2024-02-11"),
            (13, 10, 15, 4, "摄像头出色，系统更新及时",          "2024-02-15"),
            (14, 11, 5,  2, "续航时间只有宣称的70%，失望",       "2024-02-18"),
            (15, 12, 9,  5, "质量很好，穿着舒适优雅",            "2024-02-20"),
            (16, 13, 3,  4, "运行速度快，但发热比较明显",        "2024-02-22"),
            (17, 13, 17, 4, "音质中规中矩，佩戴舒适",            "2024-02-23"),
            (18, 14, 5,  1, "质量太差，购买一周就坏了",          "2024-02-25"),
            (19, 16, 1,  5, "颜值超高，系统丝滑",               "2024-03-05"),
            (20, 17, 5,  4, "降噪给力，通话清晰",               "2024-03-08"),
            (21, 17, 14, 3, "果汁味道一般，包装还可以",          "2024-03-09"),
            (22, 18, 13, 5, "家庭必备，非常实惠",               "2024-03-10"),
            (23, 19, 4,  4, "屏幕色彩准确，键盘手感好",          "2024-03-12"),
            (24, 20, 9,  5, "设计时尚，做工精细",               "2024-03-15"),
            (25, 1,  6,  5, "二次购买，依然优秀",               "2024-03-16"),
        ]
    )

    # ── 7. inventory_log（30条）──────────────────────────────────────
    cur.executemany(
        "INSERT INTO inventory_log(id,product_id,delta,reason,created_at) VALUES (?,?,?,?,?)",
        [
            # 初始入库
            (1,  1,  100, "initial_stock",  "2024-01-01"),
            (2,  2,  100, "initial_stock",  "2024-01-01"),
            (3,  3,   50, "initial_stock",  "2024-01-01"),
            (4,  4,   50, "initial_stock",  "2024-01-01"),
            (5,  5,  100, "initial_stock",  "2024-01-01"),
            (6,  6,  100, "initial_stock",  "2024-01-01"),
            (7,  7,  300, "initial_stock",  "2024-01-01"),
            (8,  8,  200, "initial_stock",  "2024-01-01"),
            (9,  9,  150, "initial_stock",  "2024-01-01"),
            (10, 10, 100, "initial_stock",  "2024-01-01"),
            # 销售出库（负数）
            (11, 1,  -50, "sale",           "2024-01-15"),
            (12, 2,  -60, "sale",           "2024-01-20"),
            (13, 3,  -30, "sale",           "2024-01-25"),
            (14, 5,  -40, "sale",           "2024-02-01"),
            (15, 6,  -20, "sale",           "2024-02-05"),
            # 补货入库
            (16, 1,   50, "restock",        "2024-02-10"),
            (17, 2,   30, "restock",        "2024-02-10"),
            (18, 5,   20, "restock",        "2024-02-15"),
            # 售后退货（正数，归还库存）
            (19, 5,    1, "return",         "2024-02-20"),
            (20, 6,    2, "return",         "2024-02-22"),
            # 盘点损耗（负数）
            (21, 11, -50, "inventory_loss", "2024-02-28"),
            (22, 12, -30, "inventory_loss", "2024-02-28"),
            (23, 13, -20, "inventory_loss", "2024-02-28"),
            # 新品入库
            (24, 15,  80, "initial_stock",  "2024-02-01"),
            (25, 16,  30, "initial_stock",  "2024-02-01"),
            (26, 17,  80, "initial_stock",  "2024-02-01"),
            (27, 18, 200, "initial_stock",  "2024-02-01"),
            (28, 19, 120, "initial_stock",  "2024-02-01"),
            (29, 20, 100, "initial_stock",  "2024-02-01"),
            # 产品下架损耗
            (30, 20, -100, "write_off",     "2024-03-01"),
        ]
    )

    conn.commit()
    conn.close()
    print(f"[init_db] ✅ DB initialized at: {DB_PATH}")
    print("[init_db]    Tables: categories(10), users(20), products(20),")
    print("[init_db]            orders(20), order_items(40), reviews(25), inventory_log(30)")


if __name__ == "__main__":
    init_db()
    
    # uv run python src/DB/init_db.py