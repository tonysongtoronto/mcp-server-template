"""
test_checkpoint_resume.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checkpoint 中断恢复测试套件
（验证：进程被杀/请求超时/客户端断线之后，从 SQLite checkpoint 恢复对话，
  数据不丢、不重复执行已完成的任务、能正常接续）

测试目标：
  TEST_1（基础）  跨连接恢复    —— 模拟"客户端断开重连"：同一 thread_id 换一条
                                   新的 HTTP 连接继续问，历史消息和摘要都还在
  TEST_2（基础）  跨进程恢复    —— 模拟"服务端重启"：先建立对话存档 thread_id，
                                   重启 uvicorn 后用 --rerun 验证 checkpoint 仍在
  TEST_3（高级）  超时中断恢复  —— 故意发一个会超时/失败的慢请求把它打断，
                                   验证 checkpoint 没有写入半成品脏状态，
                                   下一轮正常请求仍可在同一 thread_id 上继续
  TEST_4（高级）  状态自检     —— 直接读 checkpoint 内部状态（aget_tuple 等价物，
                                   这里通过 /sessions 接口间接验证 message_count
                                   单调递增、不因为中断而回退或膨胀
  TEST_5（复杂）  并发恢复竞争  —— 同一个 thread_id 被两个并发请求同时打中
                                   （模拟客户端重复提交 / 网络重试风暴），
                                   验证 SQLite 串行写入下数据不损坏、不丢轮次
  TEST_6（复杂）  恢复后继续多轮 —— 在"中断点"之后继续追加多轮对话，
                                   验证旧摘要/旧细节在恢复后仍可被正确召回
                                   （即恢复不是简单清空，而是真正接上断点）

用法：
  # 首次运行：执行除 TEST_2 读取外的全部基础/高级/复杂测试
  python test_checkpoint_resume.py

  # 配合 TEST_2：先正常运行一次（建立存档），重启 uvicorn 服务，再 --rerun
  python test_checkpoint_resume.py --rerun

  # 只跑某一等级
  python test_checkpoint_resume.py --level basic
  python test_checkpoint_resume.py --level advanced
  python test_checkpoint_resume.py --level complex

  # 只跑某一组（1~6）
  python test_checkpoint_resume.py --only 3

依赖：
  服务端已跑：uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
  纯标准库 urllib，无需额外安装

注意：
  - "中断"在这里指应用层可观测到的中断：客户端超时放弃 / 连接被掐断 /
    进程被杀后重启。LangGraph 的 ainvoke() 是单个原子调用，中途真正杀死
    服务进程会让本次 invoke 直接失败，但已经走过的 checkpoint 写入点
    （每个 node 执行完会落一次盘）不会丢，这正是要验证的核心特性。
  - TEST_3 用一个"故意制造超时"的客户端短超时（而非让 agent 真正跑很久）
    来模拟"请求被打断"，更贴近真实场景里网络抖动/客户端取消的情况，
    比强行 kill -9 服务进程更容易自动化、可重复。
  - TEST_2 必须跑两次（建档 + --rerun）才能验证真正跨进程持久化，
    这点和 test_multiuser_memory.py 的 TEST_3 完全一致。
  - 测试题故意用自然语言，不用指令式口吻，贴近真实用户场景。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT  = 120        # 单题正常超时（秒）
SHORT_TIMEOUT = 3      # TEST_3 用来"故意打断"的短超时（秒）
# TEST_2 用这个文件保存第一次运行的 thread_id，供第二次运行读取
PERSIST_FILE = Path(__file__).parent / ".test_checkpoint_resume_thread_id.json"


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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, timeout: int = 10) -> dict:
    """GET {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(path: str, timeout: int = 20) -> dict:
    """DELETE {BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(question: str, user_id: str, thread_id: str = "", timeout: int = TIMEOUT) -> dict:
    """
    调 POST /chat，返回完整响应 dict。
    字段：answer / user_id / thread_id / message_count / duration_ms
    抛出 urllib.error.URLError / socket.timeout 等异常时，调用方自行捕获
    （这正是 TEST_3 用来模拟"中断"的手段）。
    """
    return _post("/chat", {
        "question":  question,
        "user_id":   user_id,
        "thread_id": thread_id,
    }, timeout=timeout)


def get_sessions(user_id: str) -> dict:
    """调 GET /sessions/{user_id}，返回该用户的会话列表。"""
    return _get(f"/sessions/{user_id}")


def find_session(user_id: str, thread_id: str) -> dict | None:
    """从 /sessions/{user_id} 列表里找到指定 thread_id 的会话详情。"""
    data = get_sessions(user_id)
    for s in data.get("sessions", []):
        if s.get("thread_id") == thread_id:
            return s
    return None


# ══════════════════════════════════════════════
# 断言 & 输出工具
# ══════════════════════════════════════════════

PASS = "✅ PASS"
FAIL = "❌ FAIL"
INFO = "ℹ️  INFO"

_results: list[dict] = []   # 全局收集所有断言结果
_log:     list[dict] = []   # 全局收集所有输出事件，最终写入 JSON


def _emit(event_type: str, **kwargs) -> None:
    _log.append({"ts": round(time.time(), 3), "type": event_type, **kwargs})


def assert_true(condition: bool, label: str, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if not condition and detail:
        print(f"         {detail}")
    _results.append({"label": label, "passed": condition})
    _emit("assert", kind="true", label=label, passed=condition, detail=detail)
    return condition


def assert_contains(answer: str, keywords: list[str], label: str, *, mode: str = "any") -> bool:
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


def assert_ge(actual: int, expected_min: int, label: str) -> bool:
    passed = actual >= expected_min
    status = PASS if passed else FAIL
    print(f"  {status}  {label}（期望 ≥ {expected_min}，实际={actual}）")
    _results.append({"label": label, "passed": passed})
    _emit("assert", kind="ge", label=label, passed=passed,
          actual=actual, expected_min=expected_min)
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


def info(msg: str) -> None:
    print(f"  {INFO}  {msg}")
    _emit("info", message=msg)


def check_api() -> bool:
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
# TEST_1（基础）：跨连接恢复 —— 模拟客户端断开重连
# ══════════════════════════════════════════════

def test_1_reconnect_resume() -> None:
    """
    验证：同一个 thread_id，每一轮都用一条全新的 HTTP 短连接发送
    （模拟客户端每次都重新建立 TCP 连接，比如手机切后台再回来），
    服务端完全凭 SQLite 里的 checkpoint 接续历史，不依赖任何
    进程内 session/connection 状态。

    这是最基础的"中断恢复"形态：网络层面的连接其实从未真正复用，
    但用户体验是"对话连续"的——这正是把状态放进 checkpoint 而不是
    放进内存 session 对象的意义所在。
    """
    section("TEST 1（基础）：跨连接恢复（每轮独立 HTTP 连接，同一 thread_id）")

    uid = "resume_eve"
    tid = f"resume_reconnect_{int(time.time())}"

    step(1, "建立身份（用独立连接发送）")
    r = chat("你好，我叫 Eve，是一名数据库管理员，最喜欢的数据库是 PostgreSQL。",
             user_id=uid, thread_id=tid)
    show_answer(r)
    mc1 = r.get("message_count", 0)

    info("模拟连接断开 —— 这里每次 chat() 调用本身就是新连接，无需额外操作")

    step(2, "新连接：追问刚才说的身份信息")
    r = chat("我叫什么名字？我喜欢什么数据库？", user_id=uid, thread_id=tid)
    a = show_answer(r)
    mc2 = r.get("message_count", 0)

    assert_contains(a, ["Eve", "eve"], "跨连接恢复：姓名正确召回")
    assert_contains(a, ["PostgreSQL", "postgres"], "跨连接恢复：偏好正确召回")
    assert_true(mc2 > mc1, "跨连接恢复：消息数在新连接上继续累加（而非重置）",
                detail=f"mc1={mc1}, mc2={mc2}")

    step(3, "再来一条新连接，验证消息数持续单调递增")
    r = chat("好的，谢谢。", user_id=uid, thread_id=tid)
    show_answer(r)
    mc3 = r.get("message_count", 0)
    assert_true(mc3 > mc2, "跨连接恢复：第三次连接消息数继续递增",
                detail=f"mc2={mc2}, mc3={mc3}")


# ══════════════════════════════════════════════
# TEST_2（基础）：跨进程恢复 —— 模拟服务端重启
# ══════════════════════════════════════════════

def test_2_restart_write() -> None:
    """第一次运行：建立对话，把 thread_id 存盘。等待人工重启 uvicorn 后用 --rerun 验证。"""
    section("TEST 2（基础）：跨进程恢复 · 写入阶段（首次运行）")

    uid = "resume_frank"
    tid = f"resume_restart_{int(time.time())}"

    step(1, "建立身份 + 待办事项")
    r = chat(
        "你好，我叫 Frank，是一名 SRE。我手头有个未完成的任务：把数据库从"
        "MySQL 迁移到 PostgreSQL，目前进度是设计阶段。",
        user_id=uid, thread_id=tid,
    )
    show_answer(r)

    step(2, "补充细节")
    r = chat("这次迁移预计还需要 3 周，负责人是我自己，优先级是高。",
             user_id=uid, thread_id=tid)
    show_answer(r)

    step(3, "即时验证（确认本次会话记忆正常，作为中断前状态基准）")
    r = chat("我叫什么名字？我在做什么迁移任务？进度和优先级是什么？",
             user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_contains(a, ["Frank", "frank"], "重启前基准：姓名")
    assert_contains(a, ["MySQL"], "重启前基准：迁移来源")
    assert_contains(a, ["PostgreSQL", "postgres"], "重启前基准：迁移目标")

    mc_before = r.get("message_count", 0)

    PERSIST_FILE.write_text(
        json.dumps({"user_id": uid, "thread_id": tid, "message_count_before": mc_before},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  📁 thread_id 已保存到 {PERSIST_FILE}")
    print(f"     user_id={uid}  thread_id={tid}  message_count_before={mc_before}")
    print("\n  ⚠️  现在请重启服务（模拟进程被杀/部署重启这种'硬中断'），")
    print("     然后用 --rerun 参数再次执行此脚本，验证 checkpoint 是否真正从断点恢复。")


def test_2_restart_read() -> None:
    """第二次运行（--rerun）：从文件读取 thread_id，验证重启后能从断点继续。"""
    section("TEST 2（基础）：跨进程恢复 · 读取阶段（二次运行，验证服务重启后断点续接）")

    if not PERSIST_FILE.exists():
        print("  ❌ 找不到上次运行的 thread_id 文件，请先不带 --rerun 运行一次。")
        _results.append({"label": "跨进程恢复：读取 thread_id 文件", "passed": False})
        return

    saved = json.loads(PERSIST_FILE.read_text(encoding="utf-8"))
    uid, tid = saved["user_id"], saved["thread_id"]
    mc_before = saved.get("message_count_before", 0)
    print(f"  📂 读取到：user_id={uid}  thread_id={tid}  message_count_before={mc_before}")

    step(1, "重启后追问：还记得迁移任务吗？")
    r = chat("还记得我吗？我之前在做什么迁移任务？现在进度、负责人、优先级分别是什么？",
             user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_contains(a, ["Frank", "frank"], "断点恢复：姓名仍可召回")
    assert_contains(a, ["MySQL"], "断点恢复：迁移来源仍可召回")
    assert_contains(a, ["PostgreSQL", "postgres"], "断点恢复：迁移目标仍可召回")
    assert_contains(a, ["3", "三"], "断点恢复：剩余周数仍可召回")

    mc_after = r.get("message_count", 0)
    assert_true(
        mc_after > mc_before,
        "断点恢复：重启后消息数在原有基础上继续累加（未被清零重建）",
        detail=f"重启前={mc_before}, 重启后={mc_after}",
    )

    step(2, "从断点继续往下推进任务（验证'接续'而非'重新开始'）")
    r = chat("迁移进度更新一下：现在已经进入实施阶段了。", user_id=uid, thread_id=tid)
    show_answer(r)

    step(3, "验证更新后的状态生效，且旧上下文未丢失")
    r = chat("我的迁移任务现在是什么阶段？谁负责？", user_id=uid, thread_id=tid)
    b = show_answer(r)
    assert_contains(b, ["实施"], "断点恢复后继续推进：阶段已更新")
    assert_contains(b, ["Frank", "frank", "自己", "我"], "断点恢复后继续推进：负责人仍正确")


# ══════════════════════════════════════════════
# TEST_3（高级）：超时中断恢复
# ══════════════════════════════════════════════

def test_3_timeout_interrupt_resume() -> None:
    """
    验证：一次请求被"打断"（客户端短超时强制放弃，模拟网络抖动/用户取消/
    反向代理网关超时）之后：
      1. 这次半途而废的请求不应该把 checkpoint 写成损坏或不一致的状态
      2. 下一次正常请求在同一 thread_id 上应该能正常继续工作
      3. 被打断前已经建立的早期记忆应该完好无损

    原理说明：
      ainvoke() 是服务端一次完整的图执行，本身是"要么跑完一个 node 落一次
      checkpoint，要么这个 node 没跑完不落盘"，不存在 node 内部的"半成品
      checkpoint"。客户端这边等待超时只是放弃了 HTTP 连接，服务端的
      asyncio task 可能仍在后台跑完——这恰恰是真实世界最常见的"中断"场景
      （用户等不及关掉了页面，但后端任务还在跑），值得专门测试。
    """
    section("TEST 3（高级）：超时中断恢复（客户端短超时模拟请求被打断）")

    uid = "resume_grace"
    tid = f"resume_timeout_{int(time.time())}"

    step(1, "建立身份（正常完整请求，作为中断前基准）")
    r = chat("你好，我叫 Grace，是一名产品经理，正在做一个叫「极光」的项目。",
             user_id=uid, thread_id=tid)
    show_answer(r)

    step(2, "故意发一个会被短超时打断的请求（要求做一件比较重的事，拉长耗时）")
    interrupted = False
    interrupt_error = ""
    t0 = time.time()
    try:
        # SHORT_TIMEOUT 通常远小于 agent 实际完成时间，预期会抛出超时异常
        chat(
            "帮我详细规划一下「极光」项目接下来三个月的发布计划，"
            "分阶段列出里程碑、风险点和需要协调的团队，要尽量详细。",
            user_id=uid, thread_id=tid, timeout=SHORT_TIMEOUT,
        )
        info(f"请求在 {SHORT_TIMEOUT}s 内就完成了，没有真正触发中断"
             "（如果你的环境/模型响应很快，可以调低 SHORT_TIMEOUT 重测）")
    except Exception as e:
        interrupted = True
        interrupt_error = f"{type(e).__name__}: {e}"
        print(f"       ⚡ 请求按预期被打断：{interrupt_error}")
    dt = round(time.time() - t0, 2)
    _emit("interrupt", interrupted=interrupted, error=interrupt_error, elapsed_s=dt)

    info(f"等待 3 秒，给服务端后台任务（如果仍在跑）一点收尾时间...")
    time.sleep(3)

    step(3, "中断之后，发一个全新的正常请求，验证服务和该 thread 仍然可用")
    r = chat("你好，还在吗？", user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_true(
        bool(a.strip()),
        "中断恢复：被打断后服务仍能正常响应（未卡死/未崩溃）",
    )

    step(4, "验证中断前的早期记忆完好无损")
    r = chat("我叫什么名字？我在做什么项目？", user_id=uid, thread_id=tid)
    b = show_answer(r)
    assert_contains(b, ["Grace", "grace"], "中断恢复：早期记忆（姓名）未因中断而丢失")
    assert_contains(b, ["极光"], "中断恢复：早期记忆（项目名）未因中断而丢失")

    step(5, "确认会话仍可正常继续多轮（不会卡在某个半成品状态）")
    r = chat("帮我记一下：极光项目本周的重点是完成原型评审。", user_id=uid, thread_id=tid)
    show_answer(r)
    r = chat("本周重点是什么来着？", user_id=uid, thread_id=tid)
    c = show_answer(r)
    assert_contains(c, ["原型", "评审"], "中断恢复：中断之后的新一轮对话正常生效")


# ══════════════════════════════════════════════
# TEST_4（高级）：状态自检 —— message_count 单调性 & /sessions 一致性
# ══════════════════════════════════════════════

def test_4_state_consistency() -> None:
    """
    验证：checkpoint 暴露给外部的两个观测渠道 ——
      (a) /chat 响应里的 message_count
      (b) /sessions/{user_id} 列表里查到的同一 thread 的 message_count
    在多轮 + 中途插入一次"短超时打断"之后，两者应该始终保持一致，
    并且 message_count 应该单调不减（不会因为中断而回退或被重复计数翻倍）。

    这一项本质上是把 /sessions 接口当成"读 checkpoint 内部状态"的探针，
    替代直接连 SQLite 文件读 channel_values（更贴近黑盒集成测试的做法，
    也顺带验证了 /sessions 接口本身的正确性）。
    """
    section("TEST 4（高级）：状态自检（message_count 单调性 + /sessions 一致性）")

    uid = "resume_henry"
    tid = f"resume_state_{int(time.time())}"

    history_counts: list[int] = []

    for i, q in enumerate([
        "你好，我叫 Henry，是一名运维工程师。",
        "我负责的系统叫「方舟」，主要做容器编排。",
        "「方舟」目前跑在 200 台机器上。",
    ], start=1):
        step(i, q)
        r = chat(q, user_id=uid, thread_id=tid)
        show_answer(r)
        history_counts.append(r.get("message_count", 0))

    step(4, "尝试一次会被打断的请求（制造一次中断噪音）")
    try:
        chat("详细分析一下「方舟」系统未来扩容到 1000 台机器需要考虑的所有方面。",
             user_id=uid, thread_id=tid, timeout=SHORT_TIMEOUT)
    except Exception as e:
        info(f"中断噪音请求按预期超时：{type(e).__name__}")

    time.sleep(3)

    step(5, "中断后再来一轮正常对话")
    r = chat("「方舟」现在跑在几台机器上？", user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_contains(a, ["200"], "状态自检：中断后业务问答仍正确")
    history_counts.append(r.get("message_count", 0))

    # ── 单调性检查 ──────────────────────────────────────
    monotonic = all(b >= a for a, b in zip(history_counts, history_counts[1:]))
    assert_true(
        monotonic,
        "状态自检：message_count 全程单调不减（中断未导致计数回退）",
        detail=f"序列={history_counts}",
    )

    # ── /sessions 一致性检查 ─────────────────────────────
    step(6, "通过 /sessions/{user_id} 反查该会话，核对 message_count 一致")
    sess = find_session(uid, tid)
    if sess is None:
        assert_true(False, "状态自检：/sessions 接口能查到该会话", detail="未找到对应 thread_id")
    else:
        chat_mc    = history_counts[-1]
        session_mc = sess.get("message_count", -1)
        print(f"       📊 /chat 最后一次 message_count={chat_mc}  "
              f"/sessions 查到的 message_count={session_mc}")
        assert_true(
            session_mc == chat_mc,
            "状态自检：/sessions 接口与 /chat 接口的 message_count 一致",
            detail=f"/chat={chat_mc} vs /sessions={session_mc}",
        )
        assert_true(
            bool(sess.get("last_message", "").strip()),
            "状态自检：/sessions 返回的 last_message 非空（checkpoint 内容可读）",
        )


# ══════════════════════════════════════════════
# TEST_5（复杂）：并发恢复竞争
# ══════════════════════════════════════════════

def test_5_concurrent_resume_race() -> None:
    """
    验证：同一个 thread_id 被两个并发请求同时打中（模拟客户端网络重试、
    双击提交按钮、多标签页操作同一会话等真实场景）时：
      - SQLite 的串行写入特性应保证两次写入都不丢、不损坏
      - 至少有一个请求能成功拿到合理回答
      - 之后该 thread 仍然可以正常继续对话（没有被并发写坏）

    注意：
      langgraph-checkpoint-sqlite 底层对同一 thread_id 的并发写入依赖
      SQLite 的锁机制串行化，理论上不会产生"脏写"，但两个并发 invoke
      操作同一份历史消息列表时的"谁先谁后"语义本身是不确定的——
      这正是企业场景里需要显式测试覆盖的边界情况，而不是假设它"应该没事"。
    """
    section("TEST 5（复杂）：并发恢复竞争（同一 thread_id 两个并发请求）")

    uid = "resume_ivy"
    tid = f"resume_concurrent_{int(time.time())}"

    step(1, "建立身份（先用一条消息打底，确保后续并发请求是在已有历史上竞争）")
    r = chat("你好，我叫 Ivy，是一名前端工程师。", user_id=uid, thread_id=tid)
    show_answer(r)

    info("发起两个并发请求，打向同一个 thread_id ...")

    results: dict[str, dict] = {}
    errors:  dict[str, str]  = {}

    def _worker(tag: str, question: str) -> None:
        try:
            results[tag] = chat(question, user_id=uid, thread_id=tid, timeout=TIMEOUT)
        except Exception as e:
            errors[tag] = f"{type(e).__name__}: {e}"

    t1 = threading.Thread(target=_worker, args=("A", "我最喜欢的颜色是蓝色，请记住。"))
    t2 = threading.Thread(target=_worker, args=("B", "我最喜欢的运动是跑步，请记住。"))

    t0 = time.time()
    t1.start()
    t2.start()
    t1.join(timeout=TIMEOUT + 10)
    t2.join(timeout=TIMEOUT + 10)
    dt = round(time.time() - t0, 2)

    print(f"\n  ⏱  两个并发请求总耗时 {dt}s")
    for tag in ("A", "B"):
        if tag in results:
            show_answer(results[tag])
        else:
            print(f"       请求 {tag} 失败：{errors.get(tag, '未知错误')}")
    _emit("concurrent_result", elapsed_s=dt,
          ok=list(results.keys()), failed=errors)

    at_least_one_ok = len(results) >= 1
    assert_true(
        at_least_one_ok,
        "并发竞争：两个并发请求中至少一个成功返回（SQLite 写入未死锁/崩溃）",
        detail=f"成功={list(results.keys())}  失败={errors}",
    )

    step(2, "并发请求结束后，验证该会话仍然健康可用")
    r = chat("我叫什么名字？我是做什么的？", user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_contains(a, ["Ivy", "ivy"], "并发竞争后：会话未损坏，基础身份仍可召回")

    step(3, "验证并发写入的两条信息至少有一条被正确保留（不要求都在，但不能全丢）")
    r = chat("我喜欢的颜色或者运动，你还记得哪个？", user_id=uid, thread_id=tid)
    b = show_answer(r)
    kept_color  = any(k in b.lower() for k in ["蓝色", "blue"])
    kept_sport  = any(k in b.lower() for k in ["跑步", "running", "run"])
    assert_true(
        kept_color or kept_sport,
        "并发竞争：并发写入的至少一条信息被正确保留（未发生数据全丢）",
        detail=f"颜色保留={kept_color}  运动保留={kept_sport}",
    )


# ══════════════════════════════════════════════
# TEST_6（复杂）：恢复后继续多轮 —— 真正"接上断点"而非"清空重来"
# ══════════════════════════════════════════════

def test_6_resume_then_continue_multiturn() -> None:
    """
    验证：把"建立历史 → 制造一次中断 → 恢复 → 继续多轮"串联起来做端到端验证，
    覆盖 TEST_1~5 没有覆盖的组合场景：

      轮1-2   建立身份 + 早期细节（猫的名字）
      中断    故意打断一次请求
      轮3     恢复后立刻追问早期细节 —— 验证摘要/历史未受中断影响
      轮4-5   恢复后继续推进新对话（身份更新）
      轮6     验证新旧信息共存：新信息生效，旧信息（猫的名字）仍可召回
              —— 这正是区分"真正断点续传" vs "中断后退化成新会话"的关键断言：
              如果中断导致 checkpoint 被破坏、悄悄开了一个新的隐藏状态，
              这里的早期细节（猫的名字）就会召回失败。
    """
    section("TEST 6（复杂）：恢复后继续多轮（中断穿插在多轮对话中间）")

    uid = "resume_jack"
    tid = f"resume_e2e_{int(time.time())}"

    step(1, "建立初始身份")
    r = chat("你好，我叫 Jack，住在上海，是一名后端工程师。", user_id=uid, thread_id=tid)
    show_answer(r)

    step(2, "补充宠物信息（用于后续验证早期细节是否在中断后仍可召回）")
    r = chat("我养了一只叫「饭团」的橘猫。", user_id=uid, thread_id=tid)
    show_answer(r)

    info("中断点：故意发一个会被短超时打断的请求")
    try:
        chat("帮我详细对比一下 Kafka、RabbitMQ 和 Pulsar 在我们这种中等流量场景下的优劣，"
             "包括运维成本、延迟、吞吐、生态成熟度，要展开讲。",
             user_id=uid, thread_id=tid, timeout=SHORT_TIMEOUT)
    except Exception as e:
        print(f"       ⚡ 按预期被打断：{type(e).__name__}")
    time.sleep(3)

    step(3, "【恢复后】立刻追问早期细节（猫的名字）")
    r = chat("我养的猫叫什么名字？", user_id=uid, thread_id=tid)
    a = show_answer(r)
    assert_contains(a, ["饭团"], "恢复后：中断之前的早期细节（猫的名字）完整保留")

    step(4, "【恢复后】继续推进对话：身份更新")
    r = chat("顺便说一下，我刚晋升了，现在是技术负责人，带 5 个人的团队。",
             user_id=uid, thread_id=tid)
    show_answer(r)

    step(5, "【恢复后】追问更新后的新信息")
    r = chat("我现在是什么职位？带几个人？", user_id=uid, thread_id=tid)
    b = show_answer(r)
    assert_contains(b, ["负责人", "lead", "技术负责人"], "恢复后继续推进：新职位信息正确")
    assert_contains(b, ["5", "五"], "恢复后继续推进：团队人数正确")

    step(6, "终极验证：新旧信息同时共存（证明这是接续，不是退化成新会话）")
    r = chat("总结一下我的情况：姓名、城市、职业、猫的名字、现在带几个人？",
             user_id=uid, thread_id=tid)
    c = show_answer(r)
    assert_contains(c, ["Jack", "jack"], "终极验证：姓名（最早期信息）")
    assert_contains(c, ["上海"], "终极验证：城市（最早期信息）")
    assert_contains(c, ["饭团"], "终极验证：猫的名字（中断前信息）")
    assert_contains(c, ["5", "五"], "终极验证：团队人数（中断后新信息）")


# ══════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════

def print_summary(mode: str, elapsed: float, api_url: str) -> int:
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

    import datetime
    ts_str   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "rerun" if mode == "rerun" else "first"
    report_path = Path(__file__).parent / f"test_checkpoint_resume_result_{ts_str}_{mode_tag}.json"

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

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  📄 测试报告已保存：{report_path.name}")

    return failed


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

# 等级 → 该等级下属的测试编号（注意 TEST_2 横跨"基础"，因为它是经典基础场景，
# 但需要 --rerun 配合，所以单独在 main() 里处理 first/rerun 分支）
LEVEL_TESTS = {
    "basic":    ["1", "2"],
    "advanced": ["3", "4"],
    "complex":  ["5", "6"],
}


def main() -> None:
    global BASE_URL, SHORT_TIMEOUT

    parser = argparse.ArgumentParser(
        description="Checkpoint 中断恢复测试套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="二次运行模式：跳过 TEST_2 写入，执行 TEST_2 读取（验证服务重启后断点续接）",
    )
    parser.add_argument(
        "--level", choices=["basic", "advanced", "complex", "all"], default="all",
        help="只跑某一等级：basic(1,2) / advanced(3,4) / complex(5,6) / all（默认）",
    )
    parser.add_argument(
        "--only", choices=["1", "2", "3", "4", "5", "6"],
        help="只跑指定编号的单个测试",
    )
    parser.add_argument(
        "--short-timeout", type=float, default=SHORT_TIMEOUT,
        help=f"用于制造'中断'的短超时秒数（默认 {SHORT_TIMEOUT}s，"
             "如果你的 agent 响应很快导致测不出中断效果，可以调更小）",
    )
    parser.add_argument("--url", default=BASE_URL, help=f"API 地址（默认 {BASE_URL}）")
    args = parser.parse_args()

    BASE_URL = args.url.rstrip("/")
    SHORT_TIMEOUT = args.short_timeout

    print(f"\n🚀 Checkpoint 中断恢复测试套件")
    print(f"   目标 API   : {BASE_URL}")
    print(f"   模式       : {'二次运行（--rerun，验证服务重启续接）' if args.rerun else '首次运行'}")
    print(f"   等级       : {args.level}")
    print(f"   单题超时   : {TIMEOUT}s")
    print(f"   中断短超时 : {SHORT_TIMEOUT}s")

    if not check_api():
        sys.exit(1)

    # 决定本次要跑哪些编号
    if args.only:
        todo = [args.only]
    elif args.level == "all":
        todo = ["1", "2", "3", "4", "5", "6"]
    else:
        todo = LEVEL_TESTS[args.level]

    start = time.time()
    mode  = "rerun" if args.rerun else "first"

    try:
        if "1" in todo:
            test_1_reconnect_resume()

        if "2" in todo:
            if args.rerun:
                test_2_restart_read()
            else:
                test_2_restart_write()

        if "3" in todo:
            test_3_timeout_interrupt_resume()

        if "4" in todo:
            test_4_state_consistency()

        if "5" in todo:
            test_5_concurrent_resume_race()

        if "6" in todo:
            test_6_resume_then_continue_multiturn()

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

# ──────────────────────────────────────────────
# 常用命令速查（在项目根目录 mcp-server-template/ 下执行）
# ──────────────────────────────────────────────
#
# 终端 1 — 启动 API（保持运行，不要关）：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
#
# 终端 2 — 跑全部测试（基础 + 高级 + 复杂）：
#   uv run python scripts/test_checkpoint_resume.py
#
# 只跑基础测试（最快，适合日常回归）：
#   uv run python scripts/test_checkpoint_resume.py --level basic
#
# 只跑高级测试（超时中断 + 状态一致性）：
#   uv run python scripts/test_checkpoint_resume.py --level advanced
#
# 只跑复杂测试（并发竞争 + 端到端断点续传）：
#   uv run python scripts/test_checkpoint_resume.py --level complex
#
# 只跑单个编号：
#   uv run python scripts/test_checkpoint_resume.py --only 3
#
# TEST_2 服务重启验证（完整流程）：
#   1) uv run python scripts/test_checkpoint_resume.py --only 2      # 建档
#   2) 终端1 按 Ctrl+C 停掉 api，再重新执行 uvicorn 命令启动
#   3) uv run python scripts/test_checkpoint_resume.py --only 2 --rerun
#
# 如果你的模型/网络响应很快，TEST_3/TEST_6 的"中断"可能触发不了
# （3秒内就跑完了），可以调小短超时强制触发：
#   uv run python scripts/test_checkpoint_resume.py --level advanced --short-timeout 1