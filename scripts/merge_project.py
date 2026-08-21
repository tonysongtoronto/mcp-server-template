# merge_project.py
# -*- coding: utf-8 -*-

# py merge_project.py
# python merge_project.py

from pathlib import Path
from datetime import datetime


# ============================================================
# 配置
# ============================================================

# 项目根目录 = 当前 Python 脚本所在目录
ROOT = Path(__file__).resolve().parent

# 输出目录
OUTPUT_DIR = ROOT / "ai_export"

# 输出文件
OUTPUT_FILE = OUTPUT_DIR / "project-all.txt"

# 单个文件超过这个大小（MB）就不写入正文内容，只记录一条提示
# （避免 uv.lock / 数据库文件之类的东西把 txt 撑爆）
MAX_TEXT_FILE_MB = 2


# ============================================================
# 需要包含的源码 / 文本文件扩展名
# （相比原版大幅扩充：补上了 Python / Shell / PowerShell / SQL /
#   配置文件等等，这些正是之前丢失的类型）
# ============================================================

EXTENSIONS = {
    # 前端 / Web
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".css", ".scss", ".sass", ".less",
    ".html", ".htm",
    ".vue", ".svelte",

    # 数据 / 配置
    ".json", ".jsonc", ".yml", ".yaml", ".xml", ".toml",
    ".ini", ".cfg", ".conf", ".env.example",

    # 文档
    ".md", ".mdx", ".rst", ".txt",

    # Python
    ".py", ".pyi", ".pyx",

    # Shell / 脚本
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",

    # 其他常见后端语言（按需删减）
    ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".sql", ".graphql", ".proto",

    # Docker / CI
    ".dockerfile",

    # ORM / Schema / 其他常见文本格式
    ".prisma", ".graphqls", ".proto3",
    ".env", ".envrc",
    ".gql",
    ".vue", ".astro",
    ".tf", ".tfvars",  # Terraform
    ".liquid", ".hbs", ".ejs", ".pug",
}

# 没有扩展名、或者是点开头文件名的“特殊文件”，靠文件名精确匹配来收录
# （这些文件用 Path.suffix 判断不到，原脚本因此把它们全部漏掉了，
#   例如 Dockerfile、.gitignore、.python-version）
SPECIAL_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
    "LICENSE",
    "LICENCE",
    "CHANGELOG",
    "AUTHORS",
    "CODEOWNERS",
    ".gitignore",
    ".dockerignore",
    ".gitattributes",
    ".editorconfig",
    ".env.example",
    ".env.sample",
    ".npmrc",
    ".nvmrc",
    ".python-version",
    ".flake8",
    ".pylintrc",
    ".prettierrc",
    ".eslintrc",
}

# 前缀匹配（例如 Dockerfile.qa, Dockerfile.dev 这种命名）
SPECIAL_PREFIXES = (
    "Dockerfile.",
    "docker-compose",
)


# ============================================================
# 明确是二进制 / 不适合当文本读的扩展名
# 这些文件**仍然会被列出来**（文件名 + 大小），只是不把内容当文本塞进去，
# 这样才能做到“信息不丢失”——至少让 AI 知道这个文件存在。
# ============================================================

BINARY_EXTENSIONS = {
    ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp", ".svg",
    ".pdf",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo",
    ".lock",  # uv.lock / poetry.lock 等，内容对 AI 阅读价值低且体积大
}


# ============================================================
# 需要排除的目录
# ============================================================

EXCLUDED_DIRS = {
    "node_modules",
    ".next",
    ".git",
    "dist",
    "build",
    "coverage",
    "out",
    ".cache",
    ".turbo",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "egg-info",
}


# ============================================================
# 需要排除的文件（内容价值极低 / 体积大 / 纯生成物）
# 注意：这些文件依然会出现在“文件清单”里，只是不塞正文，
# 从而尽量做到不遗漏信息，只是不灌水。
# ============================================================

EXCLUDED_CONTENT_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "poetry.lock",
    "Cargo.lock",
}


# ============================================================
# 判断逻辑
# ============================================================

def is_excluded_dir(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


def is_special_filename(name: str) -> bool:
    if name in SPECIAL_FILENAMES:
        return True
    for prefix in SPECIAL_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def looks_like_text(path: Path, sniff_bytes: int = 8192) -> bool:
    """
    对“扩展名不认识”的文件做内容探测，判断它究竟是文本还是二进制。
    规则很简单也很保守：
      1) 读取文件开头一小段字节
      2) 如果里面出现了 NUL(\\x00) 字节 -> 基本可以断定是二进制
      3) 尝试用 UTF-8 解码，能解码成功就当文本，否则当二进制
    这样即使遇到 .prisma / .astro / .liquid 这类没写进白名单的新格式，
    也能被正确识别为文本而不是被误伤成“仅登记”。
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return False

    if not chunk:
        # 空文件，当文本处理没问题
        return True

    if b"\x00" in chunk:
        return False

    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def classify(path: Path) -> str:
    """
    返回该文件的处理方式：
      "text"     -> 当作文本读入正文
      "binary"   -> 只登记文件名+大小，不读内容
      "skip"     -> 不予理会（不出现在清单里，例如 node_modules 内的文件）
    """

    if is_excluded_dir(path):
        return "skip"

    name = path.name
    suffix = path.suffix.lower()

    # 明确的二进制类型（即使内容探测觉得像文本，也强制当二进制，
    # 比如某些 .db 文件头恰好没有 NUL 字节，也不该被当文本塞进去）
    if suffix in BINARY_EXTENSIONS:
        return "binary"

    # 已知会造成信息灌水的大文件（lock 文件等），仍登记但不塞正文
    if name in EXCLUDED_CONTENT_FILES:
        return "binary"

    # 特殊文件名（无扩展名的配置文件）
    if is_special_filename(name):
        return "text"

    # 常规扩展名白名单
    if suffix in EXTENSIONS:
        return "text"

    # 兜底：既不在白名单也不是已知二进制类型的“陌生”文件
    # 不再直接判死刑为二进制，而是先探测内容——像文本就照收不误，
    # 保证不会因为一个扩展名没被写进白名单就悄悄漏掉一个源码文件
    if looks_like_text(path):
        return "text"

    return "binary"


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AI PROJECT SOURCE MERGER")
    print("=" * 70)
    print()

    print("项目根目录:")
    print(ROOT)
    print()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 扫描并分类所有文件
    # --------------------------------------------------------

    text_files = []
    binary_files = []  # (path, size_bytes, reason)

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        # 不把自己生成的输出目录再次扫描进去
        if OUTPUT_DIR in path.parents:
            continue

        kind = classify(path)

        if kind == "skip":
            continue

        elif kind == "text":
            # 文件太大的话，也降级为只登记，不读正文，避免撑爆 txt
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = 0

            if size_bytes > MAX_TEXT_FILE_MB * 1024 * 1024:
                binary_files.append(
                    (path, size_bytes, f"超过 {MAX_TEXT_FILE_MB}MB，仅登记未读取正文")
                )
            else:
                text_files.append(path)

        elif kind == "binary":
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = 0
            binary_files.append((path, size_bytes, "二进制/生成文件，仅登记"))

    text_files.sort(key=lambda p: str(p).lower())
    binary_files.sort(key=lambda item: str(item[0]).lower())

    total_files = len(text_files) + len(binary_files)

    print(f"发现文件总数: {total_files}")
    print(f"  -> 将写入正文: {len(text_files)}")
    print(f"  -> 仅登记(二进制/超限): {len(binary_files)}")
    print()

    # --------------------------------------------------------
    # 开始生成
    # --------------------------------------------------------

    total_lines = 0
    success_count = 0
    failed_count = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as output:

        # ----------------------------------------------------
        # 文件头
        # ----------------------------------------------------

        output.write("=" * 70 + "\n")
        output.write("AI PROJECT SOURCE EXPORT\n")
        output.write("=" * 70 + "\n\n")

        output.write(f"Project: {ROOT}\n")
        output.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        output.write(f"Total Files Discovered: {total_files}\n")
        output.write(f"Files With Full Content: {len(text_files)}\n")
        output.write(f"Files Registered Only (binary/oversized): {len(binary_files)}\n\n")

        # ----------------------------------------------------
        # 完整文件清单（目录），保证即使某个文件内容没写进来，
        # AI 也至少知道它存在，不会“凭空消失”
        # ----------------------------------------------------

        output.write("=" * 70 + "\n")
        output.write("FILE INDEX (all files discovered)\n")
        output.write("=" * 70 + "\n\n")

        for p in text_files:
            rel = p.relative_to(ROOT)
            output.write(f"[TEXT]    {rel}\n")

        for p, size_bytes, reason in binary_files:
            rel = p.relative_to(ROOT)
            size_kb = size_bytes / 1024
            output.write(f"[BINARY]  {rel}  ({size_kb:.1f} KB, {reason})\n")

        output.write("\n" + "=" * 70 + "\n\n")

        # ----------------------------------------------------
        # 写入每个文本文件的正文
        # ----------------------------------------------------

        for index, file_path in enumerate(text_files, start=1):

            relative_path = file_path.relative_to(ROOT)

            print(f"[{index}/{len(text_files)}] {relative_path}")

            output.write("\n")
            output.write("=" * 70 + "\n")
            output.write(f"FILE: {relative_path}\n")
            output.write("=" * 70 + "\n\n")

            try:
                # errors=replace 防止某个文件编码异常导致整个程序停止
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                output.write(text)

                if not text.endswith("\n"):
                    output.write("\n")

                output.write("\n")

                total_lines += text.count("\n") + (
                    1 if text and not text.endswith("\n") else 0
                )

                success_count += 1

            except Exception as e:

                failed_count += 1

                print(f"  WARNING: 读取失败: {e}")

                output.write(
                    f"[WARNING: Unable to read this file: {e}]\n\n"
                )

        # ----------------------------------------------------
        # 二进制/超限文件的登记区（不含正文，只有元信息）
        # ----------------------------------------------------

        if binary_files:
            output.write("\n" + "=" * 70 + "\n")
            output.write("REGISTERED FILES WITHOUT CONTENT (binary / oversized)\n")
            output.write("=" * 70 + "\n\n")

            for p, size_bytes, reason in binary_files:
                rel = p.relative_to(ROOT)
                size_kb = size_bytes / 1024
                output.write(f"- {rel}  ({size_kb:.1f} KB) — {reason}\n")

            output.write("\n")

        # ----------------------------------------------------
        # 文件尾部
        # ----------------------------------------------------

        output.write("=" * 70 + "\n")
        output.write("END OF PROJECT\n")
        output.write("=" * 70 + "\n\n")

        output.write(f"Total Files Discovered: {total_files}\n")
        output.write(f"Files With Full Content: {len(text_files)}\n")
        output.write(f"  Successful: {success_count}\n")
        output.write(f"  Failed: {failed_count}\n")
        output.write(f"Files Registered Only: {len(binary_files)}\n")
        output.write(f"Total Lines (content files): {total_lines}\n")
        output.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    # --------------------------------------------------------
    # 文件大小
    # --------------------------------------------------------

    size_bytes = OUTPUT_FILE.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # --------------------------------------------------------
    # 最终结果
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("完成！")
    print("=" * 70)
    print()

    print(f"发现文件总数: {total_files}")
    print(f"写入正文: {len(text_files)}  (成功 {success_count} / 失败 {failed_count})")
    print(f"仅登记(未读正文): {len(binary_files)}")
    print(f"代码行数: {total_lines:,}")
    print(f"TXT 大小: {size_mb:.2f} MB")
    print()

    if binary_files:
        print("以下文件只登记了名字和大小，没有把内容写进 txt：")
        for p, size_bytes, reason in binary_files[:20]:
            rel = p.relative_to(ROOT)
            print(f"  - {rel} ({reason})")
        if len(binary_files) > 20:
            print(f"  ... 以及另外 {len(binary_files) - 20} 个文件，详见 txt 内的登记区")
        print()

    print("输出文件:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("你可以把 project-all.txt 上传给 AI")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()