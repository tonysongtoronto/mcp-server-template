import pytest
from fastmcp import Client
from src.mcp_server_template.server import mcp 

@pytest.mark.asyncio
async def test_add_numbers():
    async with Client(mcp) as client:
        result = await client.call_tool("add_numbers", {"a": 5, "b": 7})
        
        # 修正：使用 is_error 而不是 isError
        assert not result.is_error
        
        assert len(result.content) > 0
        assert result.content[0].text == "12"

@pytest.mark.asyncio
async def test_get_server_info():
    async with Client(mcp) as client:
        result = await client.call_tool("get_server_info", {})
        
        # 修正：使用 is_error
        assert not result.is_error
        assert isinstance(result.content[0].text, str)

@pytest.mark.asyncio
async def test_welcome_resource():
    async with Client(mcp) as client:
        content = await client.read_resource("welcome://message")
        assert "欢迎使用企业级 MCP Server 模板" in str(content)