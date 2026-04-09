import asyncio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp_server_template.server import mcp


async def test_add_numbers():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("add_numbers", {"a": 5, "b": 7})
        assert not result.isError
        assert result.content[0].text == "12"


async def test_get_server_info():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        result = await client.call_tool("get_server_info", {})
        assert not result.isError
        assert "MCP Server Template" in result.content[0].text


async def test_welcome_resource():
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        contents = await client.read_resource("welcome://message")
        assert len(contents.contents) > 0
        assert "欢迎使用企业级 MCP Server 模板" in contents.contents[0].text