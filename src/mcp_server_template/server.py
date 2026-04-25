import asyncio
import json as _json
import logging
import os
import shutil
import sys
from pathlib import Path

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP

os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8000")

logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)

mcp = FastMCP("MCP Server Template", host="0.0.0.0")

# ★ Filesystem 工具的根目录，从环境变量读取，默认 ./File_Agent
_FS_BASE = Path(os.environ.get("MCP_FS_BASE_DIR", "./File_Agent")).resolve()


def _safe_path(relative_path: str) -> Path:
    """
    把相对路径解析为绝对路径，并确保在 _FS_BASE 目录内（防止路径穿越）。
    """
    target = (_FS_BASE / relative_path).resolve()
    if not str(target).startswith(str(_FS_BASE)):
        raise PermissionError(f"禁止访问授权目录以外的路径：{target}")
    return target

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
def divide_numbers(a: float, b: float) -> float:
    """两个数字相除"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


@mcp.tool()
def get_server_info() -> str:
    """返回服务器信息"""
    return "MCP Server Template 运行中，平台: {}, Python: {}".format(
        sys.platform, sys.version.split()[0]
    )


# ──────────────────────────────────────────
# 📁 Filesystem 工具（替代 mcp-proxy + @modelcontextprotocol/server-filesystem）
# 所有操作限制在 MCP_FS_BASE_DIR 目录内
# ──────────────────────────────────────────

@mcp.tool()
def list_directory(path: str = "") -> str:
    """
    列出目录内容。
    参数:
        path - 相对于授权根目录的路径，默认为根目录（""）
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"❌ 路径不存在：{path or '/'}"
        if not target.is_dir():
            return f"❌ 不是目录：{path}"
        items = []
        for item in sorted(target.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            size = f"  ({item.stat().st_size} bytes)" if item.is_file() else ""
            items.append(f"{kind} {item.name}{size}")
        return f"目录：{path or '/'}\n" + ("\n".join(items) if items else "（空目录）")
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 错误：{e}"


@mcp.tool()
def read_file(path: str) -> str:
    """
    读取文件内容。
    参数:
        path - 相对于授权根目录的文件路径，例如 "hello.txt"
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"❌ 文件不存在：{path}"
        if not target.is_file():
            return f"❌ 不是文件：{path}"
        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > 5000:
            content = content[:5000] + f"\n\n…（已截断，原文件 {target.stat().st_size} bytes）"
        return content
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 读取失败：{e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    写入文件内容（覆盖已有文件）。
    参数:
        path    - 相对于授权根目录的文件路径，例如 "hello.txt"
        content - 要写入的文本内容
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"✅ 已写入：{path}（{len(content.encode())} bytes）"
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 写入失败：{e}"


@mcp.tool()
def create_directory(path: str) -> str:
    """
    创建目录（含所有父目录）。
    参数:
        path - 相对于授权根目录的目录路径
    """
    try:
        target = _safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"✅ 目录已创建：{path}"
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 创建失败：{e}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """
    移动或重命名文件/目录。
    参数:
        source      - 源路径（相对于授权根目录）
        destination - 目标路径（相对于授权根目录）
    """
    try:
        src = _safe_path(source)
        dst = _safe_path(destination)
        if not src.exists():
            return f"❌ 源路径不存在：{source}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"✅ 已移动：{source} → {destination}"
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 移动失败：{e}"


@mcp.tool()
def search_files(pattern: str, path: str = "") -> str:
    """
    在授权目录内搜索匹配的文件（支持 glob 模式）。
    参数:
        pattern - 文件名模式，例如 "*.txt"、"hello*"
        path    - 搜索起始目录（相对于授权根目录），默认为根目录
    """
    try:
        base = _safe_path(path)
        if not base.is_dir():
            return f"❌ 不是目录：{path or '/'}"
        matches = list(base.rglob(pattern))
        if not matches:
            return f"未找到匹配 '{pattern}' 的文件"
        lines = [f"找到 {len(matches)} 个匹配："]
        for m in matches[:50]:
            rel = m.relative_to(_FS_BASE)
            lines.append(f"  {'📁' if m.is_dir() else '📄'} {rel}")
        if len(matches) > 50:
            lines.append(f"  …（仅显示前 50 条，共 {len(matches)} 条）")
        return "\n".join(lines)
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 搜索失败：{e}"


@mcp.tool()
def get_file_info(path: str) -> str:
    """
    获取文件或目录的详细信息。
    参数:
        path - 相对于授权根目录的路径
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"❌ 路径不存在：{path}"
        stat = target.stat()
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        kind  = "目录" if target.is_dir() else "文件"
        lines = [
            f"路径：{path}",
            f"类型：{kind}",
            f"大小：{stat.st_size} bytes",
            f"修改时间：{mtime}",
        ]
        return "\n".join(lines)
    except PermissionError as e:
        return f"❌ 权限错误：{e}"
    except Exception as e:
        return f"❌ 获取信息失败：{e}"


@mcp.tool()
def list_allowed_directories() -> str:
    """列出所有允许访问的根目录"""
    exists = _FS_BASE.exists()
    return f"授权根目录：{_FS_BASE}（{'存在' if exists else '不存在'}）"


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

    # ★ --sse 模式：给 langgraph dev（通过 webapp.py lifespan）调用
    if "--sse" in sys.argv or "--dev" in sys.argv:
        port = int(os.environ.get("PORT", "8001"))
        print(f"🚀 SSE 模式启动，监听 http://0.0.0.0:{port}", file=sys.stderr)
        # FastMCP.run() 不支持 port 参数，通过 settings 注入端口
        try:
            mcp.settings.port = port
            mcp.settings.host = "0.0.0.0"
        except Exception:
            pass
        mcp.run(transport="sse")

    else:
        # ★ 默认 stdio 模式：给后端测试（__main__）和 Claude Desktop 等 MCP 客户端用
        #   后端测试命令：uv run python src/langgraph_stdio_agent.py
        #   langgraph_stdio_agent.py 的 __main__ 直接用 stdio_client spawn 本进程
        print("🚀 stdio 模式启动（后端测试 / MCP 客户端）", file=sys.stderr)
        mcp.run(transport="stdio")