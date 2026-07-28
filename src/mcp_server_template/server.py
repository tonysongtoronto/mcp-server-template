import asyncio
import json as _json
import logging
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8000")

logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

mcp = FastMCP("MCP Server Template", host="0.0.0.0")
# ★ 文件系统工具已移除，改由 mcp-server-filesystem 独立进程提供。
#   见 webapp.py lifespan（SSE 模式）和 langgraph_stdio_agent.py（stdio 模式）。
# ★ 数学工具已移除（add_numbers / multiply_numbers / divide_numbers），
#   改由 math-mcp（Node.js）独立进程提供，见 webapp.py lifespan @ 8004。

# ──────────────────────────────────────────
# 🌐 HTTP 工具（依赖 httpx）
# ──────────────────────────────────────────


@mcp.tool()
async def fetch_url(url: str, timeout: float = 10.0) -> str:
    """
    用 GET 请求获取指定 URL 的响应内容（纯文本/JSON）。
    参数:
        url     - 目标网址，例如 https://api.github.com/zen
        timeout - 超时秒数，默认 10 秒
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                import json
                return json.dumps(resp.json(), ensure_ascii=False, indent=2)[:2000]
            return resp.text[:2000]
    except httpx.TimeoutException:
        return f"❌ 请求超时（>{timeout}s）：{url}"
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP {e.response.status_code}：{url}"
    except Exception as e:
        return f"❌ 请求失败：{e}"


@mcp.tool()
async def post_json(url: str, payload: str) -> str:
    """
    向指定 URL 发送 JSON POST 请求，返回响应内容。
    参数:
        url     - 目标接口地址
        payload - 请求体 JSON 字符串，可以是 dict 或 list，例如：
                  '{"key":"value"}' 或 '[{"item":"a"},{"item":"b"}]'
    ★ payload 必须是 JSON 字符串，不是 Python 对象！
    """
    import json
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as e:
        return f"❌ payload JSON 解析失败：{e}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=body, follow_redirects=True)
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)[:2000]
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP {e.response.status_code}：{e.response.text[:200]}"
    except Exception as e:
        return f"❌ 请求失败：{e}"


# ──────────────────────────────────────────
# 📊 数据处理工具（依赖 pandas）
# ──────────────────────────────────────────


def _dataframe_summary_sync(records_json: str) -> str:
    """真正的 pandas 计算逻辑，跑在线程池里，不占事件循环。"""
    data = _json.loads(records_json)
    if not isinstance(data, list):
        return "❌ 输入必须是 JSON 数组（list of dict）"
    df = pd.DataFrame(data)
    lines = [
        f"行数: {len(df)}，列数: {len(df.columns)}",
        f"列名: {list(df.columns)}",
        "",
        "── 数值列统计 ──",
        df.describe().round(2).to_string(),
    ]
    return "\n".join(lines)


@mcp.tool()
async def dataframe_summary(records_json: str) -> str:
    """
    对一组 JSON 记录做统计摘要（行数、列名、数值列的 describe）。
    ★ 数据直接从用户消息中提取，不需要访问任何 URL 或文件！
    参数:
        records_json - JSON 字符串，格式为 list[dict]，
                       例如 '[{"name":"Alice","score":90},{"name":"Bob","score":75}]'

    ★ Fix：改为 async + asyncio.to_thread 包裹 pandas 计算，
      防止大 DataFrame 的 describe() 阻塞事件循环、卡住同一 server
      进程上其他并发的 MCP 请求（做法与 mcp_db_server/server.py 对齐）。
    """
    try:
        return await asyncio.to_thread(_dataframe_summary_sync, records_json)
    except _json.JSONDecodeError as e:
        return f"❌ JSON 解析失败：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


def _group_and_aggregate_sync(
    records_json: str, group_by: str, agg_col: str, agg_func: str
) -> str:
    """真正的 pandas 计算逻辑，跑在线程池里，不占事件循环。"""
    data = _json.loads(records_json)
    df = pd.DataFrame(data)
    result = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_by, f"{agg_col}_{agg_func}"]
    return result.to_string(index=False)


@mcp.tool()
async def group_and_aggregate(
    records_json: str, group_by: str, agg_col: str, agg_func: str = "sum"
) -> str:
    """
    对 JSON 记录按指定列分组并聚合。
    参数:
        records_json - JSON 字符串，格式为 list[dict]
        group_by     - 分组列名，例如 "department"
        agg_col      - 聚合列名，例如 "salary"
        agg_func     - 聚合函数：sum / mean / max / min / count，默认 sum

    ★ Fix：改为 async + asyncio.to_thread 包裹 pandas 计算，
      防止大 DataFrame 的 groupby/agg 阻塞事件循环、卡住同一 server
      进程上其他并发的 MCP 请求（做法与 mcp_db_server/server.py 对齐）。
    """
    allowed = {"sum", "mean", "max", "min", "count"}
    if agg_func not in allowed:
        return f"❌ agg_func 只支持：{allowed}"
    try:
        return await asyncio.to_thread(
            _group_and_aggregate_sync, records_json, group_by, agg_col, agg_func
        )
    except KeyError as e:
        return f"❌ 找不到列：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


def _filter_rows_sync(records_json: str, column: str, operator: str, value: str) -> str:
    """真正的 pandas 计算逻辑，跑在线程池里，不占事件循环。"""
    data = _json.loads(records_json)
    df = pd.DataFrame(data)

    col = df[column]
    # value 从 JSON 字符串传入，尝试转成和列一致的数值类型，转不了就按字符串比较
    cmp_value: object = value
    if pd.api.types.is_numeric_dtype(col):
        try:
            cmp_value = float(value)
        except ValueError:
            pass

    ops = {
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
        ">":  lambda s, v: s > v,
        ">=": lambda s, v: s >= v,
        "<":  lambda s, v: s < v,
        "<=": lambda s, v: s <= v,
        "contains": lambda s, v: s.astype(str).str.contains(str(v), na=False),
    }
    mask = ops[operator](col, cmp_value)
    result = df[mask]
    if result.empty:
        return "（筛选结果为空，没有符合条件的行）"
    return result.to_string(index=False)


@mcp.tool()
async def filter_rows(records_json: str, column: str, operator: str, value: str) -> str:
    """
    按条件筛选 JSON 记录中的行。
    参数:
        records_json - JSON 字符串，格式为 list[dict]
        column       - 要筛选的列名，例如 "score"
        operator     - 比较符：== / != / > / >= / < / <= / contains
        value        - 比较值（字符串形式传入，数值列会自动转换）
    示例：filter_rows(records_json='...', column="score", operator=">", value="80")

    ★ Fix：改为 async + asyncio.to_thread 包裹 pandas 计算，避免阻塞事件循环
      （做法与 dataframe_summary / group_and_aggregate / mcp_db_server 保持一致）。
    """
    allowed_ops = {"==", "!=", ">", ">=", "<", "<=", "contains"}
    if operator not in allowed_ops:
        return f"❌ operator 只支持：{allowed_ops}"
    try:
        return await asyncio.to_thread(_filter_rows_sync, records_json, column, operator, value)
    except _json.JSONDecodeError as e:
        return f"❌ JSON 解析失败：{e}"
    except KeyError as e:
        return f"❌ 找不到列：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


def _sort_dataframe_sync(records_json: str, sort_by: str, ascending: bool) -> str:
    """真正的 pandas 计算逻辑，跑在线程池里，不占事件循环。"""
    data = _json.loads(records_json)
    df = pd.DataFrame(data)
    result = df.sort_values(by=sort_by, ascending=ascending)
    return result.to_string(index=False)


@mcp.tool()
async def sort_dataframe(records_json: str, sort_by: str, ascending: bool = True) -> str:
    """
    按指定列对 JSON 记录排序。
    参数:
        records_json - JSON 字符串，格式为 list[dict]
        sort_by      - 排序列名，例如 "score"
        ascending    - 是否升序，默认 True（False 为降序）

    ★ Fix：改为 async + asyncio.to_thread 包裹 pandas 计算，避免阻塞事件循环
      （做法与 dataframe_summary / group_and_aggregate / mcp_db_server 保持一致）。
    """
    try:
        return await asyncio.to_thread(_sort_dataframe_sync, records_json, sort_by, ascending)
    except _json.JSONDecodeError as e:
        return f"❌ JSON 解析失败：{e}"
    except KeyError as e:
        return f"❌ 找不到列：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


def _pivot_table_sync(
    records_json: str, index: str, columns: str, values: str, agg_func: str
) -> str:
    """真正的 pandas 计算逻辑，跑在线程池里，不占事件循环。"""
    data = _json.loads(records_json)
    df = pd.DataFrame(data)
    result = pd.pivot_table(
        df, index=index, columns=columns, values=values, aggfunc=agg_func, fill_value=0
    )
    return result.to_string()


@mcp.tool()
async def pivot_table(
    records_json: str, index: str, columns: str, values: str, agg_func: str = "sum"
) -> str:
    """
    对 JSON 记录做透视表。
    参数:
        records_json - JSON 字符串，格式为 list[dict]
        index        - 透视表的行索引列名，例如 "dept"
        columns      - 透视表的列索引列名，例如 "month"
        values       - 要聚合的值列名，例如 "revenue"
        agg_func     - 聚合函数：sum / mean / max / min / count，默认 sum
    ★ agg_func 只允许 sum/mean/max/min/count，与 group_and_aggregate 保持一致，
      传入其他值会直接报错，不会自行替换。

    ★ Fix：改为 async + asyncio.to_thread 包裹 pandas 计算，避免阻塞事件循环
      （做法与 dataframe_summary / group_and_aggregate / mcp_db_server 保持一致）。
    """
    allowed = {"sum", "mean", "max", "min", "count"}
    if agg_func not in allowed:
        return f"❌ agg_func 只支持：{allowed}"
    try:
        return await asyncio.to_thread(
            _pivot_table_sync, records_json, index, columns, values, agg_func
        )
    except _json.JSONDecodeError as e:
        return f"❌ JSON 解析失败：{e}"
    except KeyError as e:
        return f"❌ 找不到列：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


@mcp.tool()
def get_server_info() -> str:
    """返回服务器信息"""
    return "MCP Server Template 运行中，平台: {}, Python: {}".format(
        sys.platform, sys.version.split()[0]
    )


@mcp.resource("welcome://message")
def welcome_message() -> str:
    """欢迎资源"""
    return "欢迎使用企业级 MCP Server 模板"


@mcp.resource("info://server")
def server_info() -> str:
    """服务器信息资源"""
    return f"运行在 {sys.platform} 平台，Python {sys.version}"


# ──────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # ★ --sse / --dev 模式：给 langgraph dev（通过 webapp.py lifespan）调用
    if "--sse" in sys.argv or "--dev" in sys.argv:
        port = int(os.environ.get("PORT", "8001"))
        print(f"🚀 Streamable HTTP 模式启动，监听 http://0.0.0.0:{port}", file=sys.stderr)
        print(f"🚀 Endpoint: http://localhost:{port}/mcp", file=sys.stderr)

        from starlette.middleware.cors import CORSMiddleware
        import uvicorn

        app = mcp.streamable_http_app()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        uvicorn.run(app, host="0.0.0.0", port=port)

    else:
        # ★ 默认 stdio 模式：给后端测试（__main__）和 Claude Desktop 等 MCP 客户端用
        #   后端测试命令：uv run python src/langgraph_stdio_agent.py
        #   langgraph_stdio_agent.py 的 __main__ 直接用 stdio_client spawn 本进程
        print("🚀 stdio 模式启动（后端测试 / MCP 客户端）", file=sys.stderr)
        mcp.run(transport="stdio")
        
        
# npx @modelcontextprotocol/inspector uv run python src/mcp_server_template/server.py --dev

# npx @modelcontextprotocol/inspector uv run python src/mcp_server_template/server.py 

  
        # uv run python -m debugpy --listen 5678 --wait-for-client src/mcp_server_template/server.py --sse
        
        # F5      "name": "Attach MCP Server",
        # npx @modelcontextprotocol/inspector
        
        # 第一步：server.py 里设好断点
        # 第二步：启动 Inspector
        # bash  npx @modelcontextprotocol/inspector
        # 第三步：Inspector UI 里填

        # Transport Type → STDIO
        # Command → uv
        # Arguments → run python -m debugpy --listen 5678 --wait-for-client src/mcp_server_template/server.py 

        # 点 Connect → 进程启动，挂起等待 attach
        # 第四步：VS Code 下拉选 Attach MCP Server → F5
        # 进程开始运行 ✅
        # 第五步：Inspector UI 里调用 tool → VS Code 命中断点 → 暂停 ✅
        
        # uv run python src/mcp_server_template/server.py --sse
        # uv run python src/mcp_server_template/server.py
        
        # 第一步,单独开一个终端,自己手动起 HTTP 服务:

        # uv run python src/mcp_server_template/server.py --dev

        # 确认终端里打印出:

        # 🚀 Streamable HTTP 模式启动，监听 http://0.0.0.0:8000
        # 🚀 Endpoint: http://localhost:8000/mcp

        # 第二步,另开一个终端,不带任何命令参数,单独启动 Inspector:

        # npx @modelcontextprotocol/inspector

        # 浏览器打开后,在 UI 里手动填:

        # Transport Type: Streamable HTTP
        # URL: http://localhost:8000/mcp

        # 点击 Connect