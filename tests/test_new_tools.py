import json

from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server_template.server import mcp

# ── HTTP 工具测试（用真实公网接口，可换成 mock）──

async def test_fetch_url_success():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("fetch_url", {"url": "https://httpbin.org/get"})
        assert not result.isError
        assert "httpbin" in result.content[0].text.lower() or "{" in result.content[0].text


async def test_fetch_url_timeout():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        # 用极短超时，触发超时分支
        result = await client.call_tool(
            "fetch_url", 
            {
                "url": "https://httpbin.org/delay/5", 
                "timeout": 0.001
            }
        )
        
        assert not result.isError
        assert "超时" in result.content[0].text


# ── 数据处理工具测试 ──

SAMPLE = json.dumps([
    {"dept": "eng",  "salary": 20000},
    {"dept": "eng",  "salary": 25000},
    {"dept": "biz",  "salary": 18000},
    {"dept": "biz",  "salary": 22000},
])

async def test_dataframe_summary():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("dataframe_summary", {"records_json": SAMPLE})
        assert not result.isError
        text = result.content[0].text
        assert "行数: 4" in text
        assert "salary" in text


async def test_group_and_aggregate():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "group_and_aggregate",
            {"records_json": SAMPLE, "group_by": "dept", "agg_col": "salary", "agg_func": "sum"}
        )
        assert not result.isError
        text = result.content[0].text
        assert "45000" in text  # eng: 20000+25000
        assert "40000" in text  # biz: 18000+22000


async def test_invalid_agg_func():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool(
            "group_and_aggregate",
            {"records_json": SAMPLE, "group_by": "dept", "agg_col": "salary", "agg_func": "median"}
        )
        assert not result.isError
        assert "只支持" in result.content[0].text