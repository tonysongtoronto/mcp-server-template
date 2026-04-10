import asyncio
import json as _json
import logging
import sys

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

mcp = FastMCP("MCP Server Template")

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
            # 如果是 JSON，直接返回格式化字符串；否则返回纯文本（截断 2000 字）
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
async def post_json(url: str, payload: dict) -> str:
    """
    向指定 URL 发送 JSON POST 请求，返回响应内容。
    参数:
        url     - 目标接口地址
        payload - 请求体（会被序列化为 JSON）
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, follow_redirects=True)
            resp.raise_for_status()
            import json
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)[:2000]
    except httpx.HTTPStatusError as e:
        return f"❌ HTTP {e.response.status_code}：{e.response.text[:200]}"
    except Exception as e:
        return f"❌ 请求失败：{e}"


# ──────────────────────────────────────────
# 📊 数据处理工具（依赖 pandas）
# ──────────────────────────────────────────


@mcp.tool()
def dataframe_summary(records_json: str) -> str:
    """
    对一组 JSON 记录做统计摘要（行数、列名、数值列的 describe）。
    参数:
        records_json - JSON 字符串，格式为 list[dict]，
                       例如 '[{"name":"Alice","score":90},{"name":"Bob","score":75}]'
    """
    try:
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
    except _json.JSONDecodeError as e:
        return f"❌ JSON 解析失败：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


@mcp.tool()
def group_and_aggregate(
    records_json: str, group_by: str, agg_col: str, agg_func: str = "sum"
) -> str:
    """
    对 JSON 记录按指定列分组并聚合。
    参数:
        records_json - JSON 字符串，格式为 list[dict]
        group_by     - 分组列名，例如 "department"
        agg_col      - 聚合列名，例如 "salary"
        agg_func     - 聚合函数：sum / mean / max / min / count，默认 sum
    """
    allowed = {"sum", "mean", "max", "min", "count"}
    if agg_func not in allowed:
        return f"❌ agg_func 只支持：{allowed}"
    try:
        data = _json.loads(records_json)
        df = pd.DataFrame(data)
        result = df.groupby(group_by)[agg_col].agg(agg_func).reset_index()
        result.columns = [group_by, f"{agg_col}_{agg_func}"]
        return result.to_string(index=False)
    except KeyError as e:
        return f"❌ 找不到列：{e}"
    except Exception as e:
        return f"❌ 处理失败：{e}"


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b


@mcp.tool()
def multiply_numbers(a: int, b: int) -> int:
    """两个数字相乘"""
    return a * b


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


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
 

    # 单独运行时用 --dev 启动 SSE 模式，可以在浏览器测试
    if "--dev" in sys.argv:
        print("🚀 开发模式启动，访问 http://127.0.0.1:6274", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        # 默认 stdio 模式，给 MCP 客户端（Claude Desktop 等）用
        print("🚀 后台模式启动", file=sys.stderr)
        mcp.run(transport="stdio")