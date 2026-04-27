# db/schema.py

SCHEMA = {
    "users": {
        "columns": {
            "id": "INTEGER",
            "name": "TEXT",
            "age": "INTEGER",
            "city": "TEXT"
        },
        "primary_key": "id"
    },

    "orders": {
        "columns": {
            "id": "INTEGER",
            "user_id": "INTEGER",
            "amount": "REAL",
            "created_at": "TEXT"
        },
        "primary_key": "id",
        "foreign_keys": {
            "user_id": "users.id"
        }
    }
}


# =========================
# 🧠 给 optimizer 用的简化版
# =========================
def get_schema_text() -> str:
    """转成 LLM 友好的描述"""
    lines = []

    for table, info in SCHEMA.items():
        cols = ", ".join([f"{k}:{v}" for k, v in info["columns"].items()])
        lines.append(f"{table}({cols})")

        if "foreign_keys" in info:
            for k, v in info["foreign_keys"].items():
                lines.append(f"  {table}.{k} → {v}")

    return "\n".join(lines)