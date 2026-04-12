# syntax=docker/dockerfile:1

# ══════════════════════════════════════════════════════════════════
# 阶段 1: builder  —— 用 uv 安装依赖
# 基础镜像已内置 uv + Python 3.12，专为构建设计
# ══════════════════════════════════════════════════════════════════
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV 优化选项：
#   UV_COMPILE_BYTECODE=1    → 预编译 .pyc，让容器启动更快
#   UV_LINK_MODE=copy        → 用复制代替硬链接，跨层安全
#   UV_PYTHON_DOWNLOADS=never → 禁止 uv 自动下载 Python（镜像已内置）
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# ── 先只复制依赖描述文件 ───────────────────────────────────────────
# 好处：依赖没变时，Docker 缓存此层，不重新下载安装，节省大量时间
COPY pyproject.toml uv.lock ./

# 安装生产依赖
#   --frozen         → 严格按 uv.lock 版本，保证可复现
#   --no-dev         → 跳过开发工具（pytest/ruff等），减小体积
#   --no-install-project → 先只装依赖，源码后面再复制
#   --mount=type=cache  → uv 下载缓存跨构建复用，加速二次构建
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── 再复制源码（源码变动不会让上面的依赖缓存失效）─────────────────
COPY src/ ./src/

# ══════════════════════════════════════════════════════════════════
# 阶段 2: runtime  —— 最终运行镜像（精简，不含构建工具）
# ══════════════════════════════════════════════════════════════════
FROM python:3.12-slim

WORKDIR /app

# 从 builder 只拷贝必要内容，不把 uv / 编译器 带进最终镜像
COPY --from=builder /app/.venv  /app/.venv
COPY --from=builder /app/src    /app/src

# ── 安全：非 root 用户运行 ────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# PATH 加入虚拟环境，让 python 命令直接可用
# PYTHONPATH 让 `import mcp_server_template` 能找到 src/ 下的包
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# 声明 SSE 端口
EXPOSE 8000

# 默认启动：SSE 模式运行 MCP Server
CMD ["python", "src/mcp_server_template/server.py", "--dev"]