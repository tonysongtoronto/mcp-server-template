# MCP Server 企业级模板

使用 uv + pyproject.toml + Docker 构建的生产级 MCP Server。

## 快速开始

1. `uv sync`
2. `uv run python -m mcp_server_template.server`
3. 在 Claude Desktop / Cursor 等客户端添加 MCP Server（stdio 或 SSE）。

## Docker 运行

```bash
docker compose up --build