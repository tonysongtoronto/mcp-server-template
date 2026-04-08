# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# 环境变量优化构建
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存（依赖不变时不重装）
COPY pyproject.toml uv.lock ./

# 安装生产依赖（不安装 dev 组）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 复制源码
COPY . .

# 最终运行阶段（轻量）
FROM python:3.12-slim

WORKDIR /app

# 从 builder 复制虚拟环境和源码（只复制必要文件）
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# 非 root 用户运行（安全最佳实践）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# 暴露端口（如果使用 SSE 传输）
EXPOSE 8000

# 启动 MCP Server
CMD ["uv", "run", "--with-editable", ".", "python", "-m", "mcp_server_template.server"]