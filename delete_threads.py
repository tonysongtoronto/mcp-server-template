import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LANGSMITH_API_KEY")
BASE_URL = "http://127.0.0.1:2024"
headers = {"x-api-key": api_key}

# 第一步：查询所有 threads
resp = requests.post(
    f"{BASE_URL}/threads/search",
    headers=headers,
    json={"limit": 100}  # 每次最多取100条
)

if resp.status_code != 200:
    print(f"查询失败: {resp.status_code} {resp.text}")
    exit()

threads = resp.json()
print(f"共找到 {len(threads)} 个 thread")

if not threads:
    print("没有需要删除的 thread")
    exit()

# 确认提示
confirm = input(f"确认删除全部 {len(threads)} 个 thread？(y/n): ")
if confirm.lower() != "y":
    print("已取消")
    exit()
    
# thread_ids = [
#     "9f21f6b0-7ecf-4b96-98b2-46269f5faa26",
#     "6c437ce7-1952-4c9e-bec0-bfbdfe2b08ee",
#     "8262f34b-5ac2-40cb-b78a-7b0a2c1496bd",
#     "16a49d56-d1bf-47da-823f-1c378f0934c4",
#     "af02e70d-97f9-46e8-9242-ffab5d6afd3f",
#     "e1f3e572-408e-41ac-a527-c541731d7b14",
#     "a2e762f2-9210-4e41-900e-4e7c497bb793",
# ]

# 第二步：逐个删除
success = 0
failed = 0

for thread in threads:
    tid = thread["thread_id"]
    resp = requests.delete(f"{BASE_URL}/threads/{tid}", headers=headers)
    if resp.status_code in (200, 204):
        print(f"✅ 已删除: {tid}")
        success += 1
    else:
        print(f"❌ 失败: {tid} - {resp.status_code} {resp.text}")
        failed += 1

print(f"\n完成！成功: {success}，失败: {failed}")

# uv run delete_threads.py