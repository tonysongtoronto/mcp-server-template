"""
test_multiuser_memory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
多用户 · 多轮 · 长期记忆 三合一测试套件

测试目标：
  TEST_1  用户隔离    —— alice 和 bob 的记忆互不干扰
  TEST_2  多轮记忆    —— 同一会话内，早期信息在多轮对话后仍可回忆
  TEST_3  长期记忆    —— 重新运行脚本（模拟进程重启），SQLite 持久化仍有效

用法：
  # 首次运行（TEST_3 的"写入"阶段会创建 thread_id 文件）
  python test_multiuser_memory.py

  # 二次运行（TEST_3 的"读取"阶段验证跨进程持久化）
  python test_multiuser_memory.py --rerun

依赖：
  pip install httpx          # 或 requests，这里用标准库 urllib 避免额外依赖
  服务端已跑：uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

注意：
  - 每道题都会真正走一次 LLM，耗时视网络/模型速度而定（预计每题 5~30s）
  - TEST_3 必须运行两次才能验证跨进程持久化，第一次只建档，第二次才断言
  - 测试题故意用自然语言，不用指令式口吻，贴近真实用户场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT  = 120        # 单题超时（秒），parallel agent 较慢，留充裕时间
# TEST_3 用这个文件保存第一次运行的 thread_id，供第二次运行读取
PERSIST_FILE = Path(__file__).parent / ".test3_thread_id.json"


# ══════════════════════════════════════════════
# 底层 HTTP 工具（纯标准库）
# ══════════════════════════════════════════════

def _post(path: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    """POST {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url  = f"{BASE_URL}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = data,
        headers = {"Content-Type": "application/json; charset=utf-8"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e


def _get(path: str, timeout: int = 10) -> dict:
    """GET {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e


def chat(question: str, user_id: str, thread_id: str = "") -> dict:
    """
    调 POST /chat，返回完整响应 dict。
    字段：answer / user_id / thread_id / message_count / duration_ms
    """
    return _post("/chat", {
        "question":  question,
        "user_id":   user_id,
        "thread_id": thread_id,
    })


# ══════════════════════════════════════════════
# 断言 & 输出工具
# ══════════════════════════════════════════════

PASS = "✅ PASS"
FAIL = "❌ FAIL"

_results: list[dict] = []   # 全局收集所有断言结果
_log:     list[dict] = []   # 全局收集所有输出事件，最终写入 JSON


def _emit(event_type: str, **kwargs) -> None:
    """把一条结构化事件追加到 _log，同时记录时间戳。"""
    _log.append({
        "ts":   round(time.time(), 3),
        "type": event_type,
        **kwargs,
    })


def assert_contains(
    answer:   str,
    keywords: list[str],
    label:    str,
    *,
    mode: str = "any",   # "any"=至少一个关键词命中  "all"=所有关键词都命中
) -> bool:
    """
    检查 answer 里是否包含期望关键词，打印结果，返回 bool。
    mode="any"：keywords 中有任意一个出现即为通过（用于同义词/多种表达方式）。
    mode="all"：keywords 全部出现才通过（用于必须同时包含多个信息点）。
    """
    lower = answer.lower()
    hits  = [kw for kw in keywords if kw.lower() in lower]

    passed = (len(hits) > 0) if mode == "any" else (len(hits) == len(keywords))
    status = PASS if passed else FAIL

    print(f"  {status}  {label}")
    if not passed:
        print(f"         期望关键词（mode={mode}）: {keywords}")
        print(f"         实际回答片段: {answer[:200]}")

    _results.append({"label": label, "passed": passed})
    _emit("assert", kind="contains", label=label, passed=passed,
          mode=mode, keywords=keywords, hits=hits,
          answer_snippet=answer[:200] if not passed else None)
    return passed


def assert_not_contains(answer: str, keywords: list[str], label: str) -> bool:
    """检查 answer 里不应出现的关键词（用于隔离测试）。"""
    lower = answer.lower()
    hits  = [kw for kw in keywords if kw.lower() in lower]

    passed = len(hits) == 0
    status = PASS if passed else FAIL

    print(f"  {status}  {label}")
    if not passed:
        print(f"         不应出现的关键词: {hits}")
        print(f"         实际回答片段: {answer[:200]}")

    _results.append({"label": label, "passed": passed})
    _emit("assert", kind="not_contains", label=label, passed=passed,
          keywords=keywords, hits=hits,
          answer_snippet=answer[:200] if not passed else None)
    return passed


def section(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")
    _emit("section", title=title)


def step(n: int, question: str) -> None:
    print(f"\n  [轮{n}] ❓ {question}")
    _emit("step", turn=n, question=question)


def show_answer(resp: dict) -> str:
    answer = resp.get("answer", "")
    mc     = resp.get("message_count", "?")
    ms     = resp.get("duration_ms", "?")
    print(f"       💬 {answer[:300]}{'...' if len(answer) > 300 else ''}")
    print(f"       📊 消息数={mc}  耗时={ms}ms")
    _emit("answer", answer=answer, message_count=mc, duration_ms=ms)
    return answer


def check_api() -> bool:
    """启动前检查 API 是否可达。"""
    try:
        resp = _get("/health")
        status = resp.get("status", "unknown")
        tools  = resp.get("tool_count", 0)
        print(f"🔌 API 连接正常  status={status}  tool_count={tools}")
        if status not in ("ok", "degraded"):
            print("⚠️  服务处于 initializing 状态，部分工具可能不可用")
        _emit("api_check", status=status, tool_count=tools, ok=True)
        return True
    except Exception as e:
        print(f"❌ 无法连接到 API ({BASE_URL})：{e}")
        print("   请确认已启动：uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1")
        _emit("api_check", ok=False, error=str(e))
        return False


# ══════════════════════════════════════════════
# TEST_1：多用户隔离
# ══════════════════════════════════════════════

def test_1_user_isolation() -> None:
    """
    验证：alice 和 bob 的记忆完全隔离，各自只记得自己的信息。

    测试逻辑：
      alice 自我介绍（姓名 + 城市 + 职业）
      bob   自我介绍（不同的姓名 + 城市 + 职业）
      alice 问自己的信息 → 应只含 alice 的答案，不含 bob 的信息
      bob   问自己的信息 → 应只含 bob   的答案，不含 alice 的信息
      alice 问 bob 有没有介绍过自己 → 应回答"不知道"或"没有"
    """
    section("TEST 1：多用户隔离（alice vs bob）")

    tid_alice = f"alice_isolation_{int(time.time())}"
    tid_bob   = f"bob_isolation_{int(time.time())}"

    # ── Alice 的对话 ────────────────────────────────────
    step(1, "Alice 自我介绍")
    r = chat(
        "你好！我叫 Alice，今年30岁，住在北京，是一名 UI 设计师。我养了一只叫「拿铁」的猫。",
        user_id="alice", thread_id=tid_alice,
    )
    show_answer(r)

    step(2, "Alice 再补充喜好")
    r = chat(
        "对了，我最喜欢的颜色是珊瑚红，最喜欢的设计工具是 Figma。",
        user_id="alice", thread_id=tid_alice,
    )
    show_answer(r)

    # ── Bob 的对话 ─────────────────────────────────────
    step(3, "Bob 自我介绍")
    r = chat(
        "嗨，我叫 Bob，33岁，住在深圳，是一名后端工程师，主要用 Go 语言。我有一条叫「黄豆」的狗。",
        user_id="bob", thread_id=tid_bob,
    )
    show_answer(r)

    step(4, "Bob 再补充喜好")
    r = chat(
        "我最喜欢的框架是 Gin，业余时间喜欢打羽毛球。",
        user_id="bob", thread_id=tid_bob,
    )
    show_answer(r)

    # ── 隔离验证：Alice 查自己的信息 ────────────────────
    step(5, "Alice 问：我的名字、城市、职业、猫叫什么？")
    r = chat("你好，我叫什么名字？我住哪里？我是做什么的？我的猫叫什么？",
             user_id="alice", thread_id=tid_alice)
    a = show_answer(r)

    assert_contains(a, ["Alice", "alice"],         "alice 记忆：姓名正确")
    assert_contains(a, ["北京", "beijing"],         "alice 记忆：城市正确")
    assert_contains(a, ["设计", "design"],          "alice 记忆：职业正确")
    assert_contains(a, ["拿铁"],                    "alice 记忆：猫的名字正确")
    assert_not_contains(a, ["Bob", "bob", "深圳", "Go", "黄豆"], "alice 记忆：未混入 bob 的信息")

    # ── 隔离验证：Bob 查自己的信息 ─────────────────────
    step(6, "Bob 问：我的名字、城市、职业、狗叫什么？")
    r = chat("我叫什么名字？住哪里？我是做什么工作的？我的狗叫什么？",
             user_id="bob", thread_id=tid_bob)
    b = show_answer(r)

    assert_contains(b, ["Bob", "bob"],              "bob 记忆：姓名正确")
    assert_contains(b, ["深圳", "shenzhen"],         "bob 记忆：城市正确")
    assert_contains(b, ["工程师", "engineer", "Go", "go"], "bob 记忆：职业/语言正确")
    assert_contains(b, ["黄豆"],                    "bob 记忆：狗的名字正确")
    assert_not_contains(b, ["Alice", "alice", "北京", "Figma", "拿铁"], "bob 记忆：未混入 alice 的信息")

    # ── 跨用户盲区验证：Alice 问 Bob 的信息 ────────────
    step(7, "Alice 问：你知道 Bob 住在哪里吗？（Alice 的会话里从未提过 Bob）")
    r = chat("你知道 Bob 住在哪里吗？他是做什么工作的？",
             user_id="alice", thread_id=tid_alice)
    c = show_answer(r)
    assert_not_contains(c, ["深圳", "Go", "后端", "工程师"],
                        "alice 会话中不应出现 bob 的城市/职业（隔离验证）")


# ══════════════════════════════════════════════
# TEST_2：多轮对话记忆
# ══════════════════════════════════════════════

def test_2_multiturn() -> None:
    """
    验证：同一会话内，早期信息在经历多轮干扰后仍可正确回忆。

    测试逻辑：
      轮1  建立身份（姓名 + 城市 + 爱好）
      轮2  建立身份补充（宠物信息）
      轮3  数学干扰（把早期消息推出近期窗口）
      轮4  数学干扰（继续）
      轮5  身份更新（搬家 + 换工作）—— 旧信息应被覆盖
      轮6  身份追问 —— 新信息优先，旧城市不应再出现
      轮7  早期细节追问 —— 摘要应保留猫的名字
      轮8  消息数递增验证 —— 每轮 +2 条，8轮应有 16 条
    """
    section("TEST 2：多轮对话记忆（单用户，8轮）")

    uid = "charlie"
    tid = f"charlie_multiturn_{int(time.time())}"

    step(1, "建立初始身份")
    r = chat(
        "你好！我叫 Charlie，今年25岁，住在成都，是一名前端工程师，主要写 React 和 TypeScript。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(2, "补充宠物信息")
    r = chat(
        "我有一只叫「可可」的边牧，还有一株叫「小绿」的龟背竹。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(3, "数学干扰（把早期信息挤出近期窗口）")
    r = chat(
        "帮我计算一下 999 × 888 是多少？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["887112", "88 7112", "887,112"], "数学干扰：乘法结果正确")

    step(4, "再次数学干扰")
    r = chat(
        "上面那个结果再加上 12345，等于多少？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["899457", "899,457"], "跨轮数值引用：加法结果正确")

    step(5, "身份更新（搬家 + 换工作）")
    r = chat(
        "顺便告诉你，我刚搬到杭州了，而且跳槽了，现在是全栈工程师，主要用 Next.js 和 Python。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(6, "追问更新后的身份")
    r = chat(
        "我现在住哪里？我现在的职业和技术栈是什么？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["杭州"],                            "身份更新：新城市正确")
    assert_contains(a, ["全栈", "full", "Next", "Python"],  "身份更新：新职业/技术栈正确")
    assert_not_contains(a, ["成都"],                        "身份更新：旧城市已被覆盖（不应出现）")

    step(7, "回忆早期细节（测试摘要对宠物信息的保留）")
    r = chat(
        "我的狗叫什么名字？我的植物叫什么？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["可可"],   "早期细节：狗的名字保留")
    assert_contains(a, ["小绿"],   "早期细节：植物名字保留")

    step(8, "验证消息数递增（checkpoint 是否正常追加）")
    r = chat(
        "好的，谢谢！",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)
    mc = r.get("message_count", 0)
    # 8 轮对话 = 16 条消息（每轮 Human + AI）
    # 但 agent 内部会产生 planner/executor 额外消息，实际数可能更多
    # 这里只断言"至少 14 条"（给 2 条容差），核心是"消息数随轮次递增"
    passed = mc >= 14
    status = PASS if passed else FAIL
    print(f"  {status}  消息数验证：{mc} 条（期望 ≥ 14，实际={mc}）")
    _results.append({"label": "消息数随轮次递增", "passed": passed})


# ══════════════════════════════════════════════
# TEST_3：跨进程长期记忆（SQLite 持久化）
# ══════════════════════════════════════════════

def test_3_persistence_write() -> None:
    """
    第一次运行：建立用户档案，把 thread_id 写入本地文件。
    下次用 --rerun 运行时，从文件读取 thread_id，验证记忆仍在。
    """
    section("TEST 3：长期记忆写入（首次运行）")

    uid = "diana"
    tid = f"diana_persist_{int(time.time())}"

    step(1, "建立完整用户档案")
    r = chat(
        "你好！我叫 Diana，35岁，住在广州，是一名产品经理。"
        "我喜欢喝手冲咖啡，收藏黑胶唱片，最喜欢的乐队是 Pink Floyd。"
        "我的车牌是粤A·88888（当然这是编的）。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(2, "补充工作信息")
    r = chat(
        "我在一家叫「未来科技」的公司工作，负责 B 端 SaaS 产品线，"
        "手下带 3 个产品同学。我们的旗舰产品叫「智云 Pro」。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(3, "即时验证（确认本次会话记忆正常）")
    r = chat(
        "我叫什么名字？我在哪家公司？我们的产品叫什么？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["Diana", "diana"],  "持久化写入前验证：姓名")
    assert_contains(a, ["未来科技"],         "持久化写入前验证：公司")
    assert_contains(a, ["智云", "Pro"],      "持久化写入前验证：产品名")

    # 把 thread_id 存下来，供第二次运行读取
    PERSIST_FILE.write_text(
        json.dumps({"user_id": uid, "thread_id": tid}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  📁 thread_id 已保存到 {PERSIST_FILE}")
    print(f"     user_id={uid}  thread_id={tid}")
    print("\n  ⚠️  现在请重启服务（或等待下一次运行），然后用 --rerun 参数再次执行此脚本，")
    print("     验证 SQLite 持久化是否真正有效。")


def test_3_persistence_read() -> None:
    """
    第二次运行（--rerun）：从文件读取 thread_id，直接问问题，
    验证即使进程重启过，SQLite 里的记忆仍然存在。
    """
    section("TEST 3：长期记忆读取（二次运行，验证跨进程持久化）")

    if not PERSIST_FILE.exists():
        print("  ❌ 找不到上次运行的 thread_id 文件，请先不带 --rerun 运行一次。")
        _results.append({"label": "长期记忆：读取 thread_id 文件", "passed": False})
        return

    saved   = json.loads(PERSIST_FILE.read_text(encoding="utf-8"))
    uid     = saved["user_id"]
    tid     = saved["thread_id"]
    print(f"  📂 读取到：user_id={uid}  thread_id={tid}")

    step(1, "跨进程记忆：基础身份（姓名 + 城市 + 职业）")
    r = chat(
        "你还记得我吗？我叫什么名字？我住哪里？我是做什么的？",
        user_id=uid, thread_id=tid,
    )
    a = show_answer(r)
    assert_contains(a, ["Diana", "diana"],              "跨进程记忆：姓名")
    assert_contains(a, ["广州", "guangzhou"],            "跨进程记忆：城市")
    assert_contains(a, ["产品", "product", "经理", "PM"], "跨进程记忆：职业")

    step(2, "跨进程记忆：工作细节（公司 + 产品）")
    r = chat(
        "我在哪家公司上班？我们的产品叫什么？我带几个人？",
        user_id=uid, thread_id=tid,
    )
    b = show_answer(r)
    assert_contains(b, ["未来科技"],           "跨进程记忆：公司名")
    assert_contains(b, ["智云", "Pro"],        "跨进程记忆：产品名")
    assert_contains(b, ["3", "三"],            "跨进程记忆：团队人数")

    step(3, "跨进程记忆：兴趣爱好（浓缩在摘要里的细节）")
    r = chat(
        "我喜欢喝什么咖啡？我收藏什么？我最喜欢哪个乐队？",
        user_id=uid, thread_id=tid,
    )
    c = show_answer(r)
    assert_contains(c, ["手冲", "手冲咖啡"],   "跨进程记忆：咖啡偏好")
    assert_contains(c, ["黑胶", "唱片"],       "跨进程记忆：收藏爱好")
    assert_contains(c, ["Pink Floyd", "floyd"], "跨进程记忆：喜欢的乐队")

    # 清理持久化文件（可选，注释掉保留以便反复测试）
    # PERSIST_FILE.unlink(missing_ok=True)


# ══════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════

def print_summary(mode: str, elapsed: float, api_url: str) -> int:
    """打印最终汇总，保存 JSON 报告，返回失败数（0=全通过）。"""
    total  = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed

    print(f"\n{'═' * 60}")
    print(f"  测试汇总：{passed}/{total} 通过   {'🎉 全部通过！' if failed == 0 else f'❌ {failed} 项失败'}")
    print(f"{'═' * 60}")

    if failed > 0:
        print("\n  失败项明细：")
        for r in _results:
            if not r["passed"]:
                print(f"    ❌ {r['label']}")

    # ── 保存 JSON 报告 ──────────────────────────────────────
    # 文件名：test_result_<日期>_<时间>_<模式>.json
    # 例如：test_result_20260618_143022_first.json
    #       test_result_20260618_143022_rerun.json
    import datetime
    ts_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "rerun" if mode == "rerun" else "first"
    report_path = Path(__file__).parent / f"test_result_{ts_str}_{mode_tag}.json"

    report = {
        "meta": {
            "run_at":     datetime.datetime.now().isoformat(timespec="seconds"),
            "mode":       mode_tag,
            "api_url":    api_url,
            "elapsed_s":  elapsed,
            "total":      total,
            "passed":     passed,
            "failed":     failed,
            "all_passed": failed == 0,
        },
        "summary": _results,
        "events":  _log,
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  📄 测试报告已保存：{report_path.name}")

    return failed


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

def main() -> None:
    global BASE_URL   # 声明必须在函数内第一次引用 BASE_URL 之前

    parser = argparse.ArgumentParser(
        description="多用户 · 多轮 · 长期记忆测试套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="二次运行模式：跳过 TEST_3 写入，执行 TEST_3 读取（验证跨进程持久化）",
    )
    parser.add_argument(
        "--only",
        choices=["1", "2", "3"],
        help="只跑指定测试（1/2/3）",
    )
    parser.add_argument(
        "--url",
        default=BASE_URL,
        help=f"API 地址（默认 {BASE_URL}）",
    )
    args = parser.parse_args()

    BASE_URL = args.url.rstrip("/")

    print(f"\n🚀 多用户记忆测试套件")
    print(f"   目标 API : {BASE_URL}")
    print(f"   模式     : {'二次运行（--rerun）' if args.rerun else '首次运行'}")
    print(f"   单题超时 : {TIMEOUT}s")

    # 连通性检查
    if not check_api():
        sys.exit(1)

    start = time.time()
    mode  = "rerun" if args.rerun else "first"

    try:
        only = args.only

        if only is None or only == "1":
            test_1_user_isolation()

        if only is None or only == "2":
            test_2_multiturn()

        if only is None or only == "3":
            if args.rerun:
                test_3_persistence_read()
            else:
                test_3_persistence_write()

    except Exception as e:
        import traceback
        elapsed = round(time.time() - start, 1)
        msg = f"{type(e).__name__}: {e}"
        print(f"\n❌ 意外错误（已运行 {elapsed}s）：{msg}")
        print(traceback.format_exc())
        _emit("abort", error=msg, elapsed_s=elapsed)
        print_summary(mode=mode, elapsed=elapsed, api_url=BASE_URL)
        sys.exit(1)

    elapsed = round(time.time() - start, 1)
    print(f"\n  ⏱  总耗时：{elapsed}s")

    failed = print_summary(mode=mode, elapsed=elapsed, api_url=BASE_URL)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
    
        # # 在项目根目录（mcp-server-template/）下
# uv run python scripts/test_multiuser_memory.py

# # 二次运行（验证跨进程持久化）
# uv run python scripts/test_multiuser_memory.py --rerun

# # 只跑某一组
# uv run python scripts/test_multiuser_memory.py --only 1

# 完整操作流程（从零开始）
# 终端 1 — 启动 API（保持运行，不要关）：
# powershellcd mcp-server-template
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
# 等出现这行才算好：
# INFO:     Application startup complete.
# 终端 2 — 运行测试：
# powershellcd mcp-server-template
# uv run python scripts/test_multiuser_memory.py
# TEST 3 跨进程验证（可选，验证 SQLite 重启后记忆仍在）：
# powershell# 终端1 按 Ctrl+C 停掉 api，再重启
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

# # 等 startup complete 后，终端2 执行
# uv run python scripts/test_multiuser_memory.py --rerun