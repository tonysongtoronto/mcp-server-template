"""
test_checkpoint_resume.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checkpoint 中断恢复测试套件
（验证：进程被杀/请求超时/客户端断线/图执行被强制中止之后，从 SQLite
  checkpoint 恢复对话，数据不丢、不产生脏写、能正常接续）

设计原则（工业界对"中断恢复"测试的通行分层）：
  1. 黑盒层（通过 HTTP API 观测）  —— 这是用户真正能感知到的行为，
     大部分用例应该走这一层，因为它不依赖具体存储引擎实现细节。
  2. 白盒层（直连 SQLite 文件做行级校验） —— 用于在黑盒断言"看起来恢复正常"
     之外，进一步确认底层数据没有产生孤儿行 / 断链 / 重复写等隐患。
     黑盒测试可能因为应用层做了容错而"看起来正常"，但底层已经埋雷，
     白盒校验是专门用来戳破这种假象的。
  3. 故障注入层（真正杀死被测进程） —— 模拟生产环境最暴力也最真实的故障：
     容器被 OOM Killer 杀掉、Kubernetes 滚动发布时 SIGKILL、宿主机断电。
     这一层脚本需要自己接管目标服务的生命周期（启动/等待就绪/SIGKILL/
     重启），因此复杂度和"侵入性"都明显更高，默认不随基础/高级/复杂三档
     一起跑，必须显式加 --enable-process-kill 才会执行。

测试目标一览：

  ── 基础（black-box，对已运行的服务做黑盒验证）──────────────
  TEST_1   跨连接恢复    —— 模拟"客户端断开重连"：同一 thread_id 换一条
                            新的 HTTP 连接继续问，历史消息和摘要都还在
  TEST_2   跨进程恢复    —— 模拟"服务端正常重启"（非强杀，比如 systemctl
                            restart）：先建立对话存档 thread_id，
                            重启服务后用 --rerun 验证 checkpoint 仍在

  ── 高级（black-box，注入网络/超时类中断）────────────────────
  TEST_3   超时中断恢复  —— 故意发一个会超时/失败的慢请求把它打断，
                            验证 checkpoint 没有写入半成品脏状态，
                            下一轮正常请求仍可在同一 thread_id 上继续
  TEST_4   状态自检      —— 通过 /sessions 接口间接验证 message_count
                            单调递增、不因为中断而回退或膨胀

  ── 复杂（black-box，并发与端到端组合场景）───────────────────
  TEST_5   并发恢复竞争   —— 同一个 thread_id 被两个并发请求同时打中
                            （模拟客户端重复提交 / 网络重试风暴），
                            验证 SQLite 串行写入下数据不损坏、不丢轮次
  TEST_6   恢复后继续多轮 —— 在"中断点"之后继续追加多轮对话，
                            验证旧摘要/旧细节在恢复后仍可被正确召回
                            （即恢复不是简单清空，而是真正接上断点）

  ── 故障注入（white-box + 真实 kill -9，默认不跑，需显式开启）──
  TEST_7   kill -9 基础恢复     —— 脚本自己拉起一个独立的 uvicorn 子进程，
                                   发一条正常请求建立历史，在【空闲期】
                                   对该子进程发 SIGKILL（不是优雅关闭），
                                   重新拉起进程后验证历史完整、可继续对话。
                                   这是"强杀"测试里最基础的一档：杀的时机
                                   保证不会卡在某个 node 执行中间。
  TEST_8   图执行中 kill -9（最难也最有价值）——
                                   故意在【LangGraph 正在执行某个 node】
                                   的时间窗口内发 SIGKILL，制造"truly half
                                   -done"的中断：这次 invoke 永远不会再有
                                   返回了，进程直接消失。重启后：
                                     (a) 黑盒验证服务能正常处理新请求，
                                         且被打断前的历史完整无损；
                                     (b) 白盒验证 SQLite 里有没有产生
                                         "writes 表有行但 checkpoints 表
                                         没有对应行"的孤儿写入 ——
                                         这正是"kill 发生在 aput_writes()
                                         已提交、但下一次 aput() 还没提交"
                                         这个时间缝隙里时的特征信号。
  TEST_9   Checkpoint 行级完整性审计（纯白盒，SQL 级）——
                                   不依赖前面任何测试制造中断，直接对现有
                                   checkpoints.db 做一次"体检"：
                                     - parent_checkpoint_id 链条是否有断裂
                                       （子节点引用了一个表里不存在的父节点）
                                     - writes 表是否存在孤儿行
                                     - 每个 thread 内 metadata.step 是否单调
                                     - 主键冲突 / 重复 checkpoint_id 等异常
                                   可以单独跑，作为"运维巡检脚本"长期使用，
                                   不一定要紧跟在故障注入测试后面。

用法：
  # 默认：只跑黑盒的基础 + 高级 + 复杂（1~6），安全、无侵入性，CI 友好
  python test_checkpoint_resume.py

  # 配合 TEST_2：先正常运行一次（建立存档），重启服务，再 --rerun
  python test_checkpoint_resume.py --rerun --only 2

  # 只跑某一等级
  python test_checkpoint_resume.py --level basic
  python test_checkpoint_resume.py --level advanced
  python test_checkpoint_resume.py --level complex

  # 故障注入测试（TEST_7/8）：脚本会自己拉起/杀死一个独立的 uvicorn 子进程，
  # 必须显式开启，并且建议在隔离环境（独立的 checkpoints.db、独立端口）跑，
  # 不要对着你正在用的开发服务器跑，因为它会被真的杀掉。
  python test_checkpoint_resume.py --level chaos --enable-process-kill

  # 纯白盒审计（不需要服务在跑，只需要 SQLite 文件存在）：
  python test_checkpoint_resume.py --only 9 --db-path data/checkpoints.db

  # 只跑某一组（1~9）
  python test_checkpoint_resume.py --only 3

依赖：
  服务端已跑（TEST_1~6）：uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
  （注：本说明按 src 布局项目给出；如果你的服务入口是根目录的 api.py，
  则改回 uvicorn api:app 即可——以你实际能跑通的启动命令为准）
  纯标准库 urllib/sqlite3/subprocess，无需额外安装

注意：
  - "中断"分三个粒度，由弱到强：
      (a) 应用层可观测的网络/超时中断（TEST_3/4/5/6 覆盖）——最容易自动化
      (b) 进程被正常关闭后重启（TEST_2 覆盖）—— 模拟运维正常发布
      (c) 进程被 SIGKILL 强杀，包括杀在图执行中途（TEST_7/8 覆盖）——
          最贴近"容器被 OOM kill / 宿主机断电"这类真实生产事故，
          但因为需要脚本接管进程生命周期，复杂度和"杀错进程"的风险都更高，
          所以单独分一档，默认不自动跑。
  - LangGraph 的单次 ainvoke() 内部按 node 粒度落盘：每个 node 跑完，
    LangGraph 会先把这个 node 的输出写入 writes 表（aput_writes），
    决定好下一步要跑哪个 node 之后，再把新的完整 checkpoint 写入
    checkpoints 表（aput）。如果 kill 发生在"writes 已落盘，但新
    checkpoint 还没来得及写"这个缝隙里，理论上会在 writes 表留下一条
    "孤儿行"（在 checkpoints 表里找不到对应的 checkpoint_id）。
    TEST_8 + TEST_9 就是用来捕获这种缝隙的。
  - TEST_2 必须跑两次（建档 + --rerun）才能验证真正跨进程持久化，
    这点和 test_multiuser_memory.py 的 TEST_3 完全一致。
  - 测试题故意用自然语言，不用指令式口吻，贴近真实用户场景。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT  = 120        # 单题正常超时（秒）
SHORT_TIMEOUT = 3      # TEST_3 用来"故意打断"的短超时（秒）
# TEST_2 用这个文件保存第一次运行的 thread_id，供第二次运行读取
PERSIST_FILE = Path(__file__).parent / ".test_checkpoint_resume_thread_id.json"

# ── 故障注入（TEST_7/8）专用配置 ────────────────────────────
# 这些测试会自己拉起一个独立的 uvicorn 子进程，刻意和"日常开发服务器"
# 使用不同的端口 + 不同的数据库文件，避免误杀正在用的服务、
# 也避免污染日常开发用的 checkpoints.db。
CHAOS_PORT       = 8099
CHAOS_HOST       = "127.0.0.1"
CHAOS_BASE_URL   = f"http://{CHAOS_HOST}:{CHAOS_PORT}"
CHAOS_APP_MODULE = "src.api:app"  # 按你项目的 src 布局设为默认值；
                                   # 如果服务入口不是 src/api.py，用 --app-module 覆盖
CHAOS_DB_DIR     = Path(__file__).parent / ".chaos_test_db"
CHAOS_READY_TIMEOUT = 30   # 等待子进程 /health 就绪的最长秒数


# ══════════════════════════════════════════════
# 底层 HTTP 工具（纯标准库）
# ══════════════════════════════════════════════

def _post(path: str, payload: dict, timeout: int = TIMEOUT, base_url: str | None = None) -> dict:
    """POST {base_url or BASE_URL}{path}，返回解析后的 JSON dict。"""
    url  = f"{base_url or BASE_URL}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = data,
        headers = {"Content-Type": "application/json; charset=utf-8"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, timeout: int = 10, base_url: str | None = None) -> dict:
    """GET {base_url or BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{base_url or BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(path: str, timeout: int = 20, base_url: str | None = None) -> dict:
    """DELETE {base_url or BASE_URL}{path}，返回解析后的 JSON dict。"""
    url = f"{base_url or BASE_URL}{path}"
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(question: str, user_id: str, thread_id: str = "", timeout: int = TIMEOUT,
         base_url: str | None = None) -> dict:
    """
    调 POST /chat，返回完整响应 dict。
    字段：answer / user_id / thread_id / message_count / duration_ms
    抛出 urllib.error.URLError / socket.timeout 等异常时，调用方自行捕获
    （这正是 TEST_3 用来模拟"中断"的手段）。

    base_url：默认 None 时打全局 BASE_URL；TEST_7/8（故障注入）会传入
    CHAOS_BASE_URL，指向脚本自己拉起的独立子进程，避免和全局状态混用。
    """
    return _post("/chat", {
        "question":  question,
        "user_id":   user_id,
        "thread_id": thread_id,
    }, timeout=timeout, base_url=base_url)


def get_sessions(user_id: str, base_url: str | None = None) -> dict:
    """调 GET /sessions/{user_id}，返回该用户的会话列表。"""
    return _get(f"/sessions/{user_id}", base_url=base_url)


def find_session(user_id: str, thread_id: str, base_url: str | None = None) -> dict | None:
    """从 /sessions/{user_id} 列表里找到指定 thread_id 的会话详情。"""
    data = get_sessions(user_id, base_url=base_url)
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


def check_api(base_url: str | None = None, quiet: bool = False) -> bool:
    url = base_url or BASE_URL
    try:
        resp = _get("/health", base_url=url)
        status = resp.get("status", "unknown")
        tools  = resp.get("tool_count", 0)
        if not quiet:
            print(f"🔌 API 连接正常  status={status}  tool_count={tools}")
            if status not in ("ok", "degraded"):
                print("⚠️  服务处于 initializing 状态，部分工具可能不可用")
        _emit("api_check", status=status, tool_count=tools, ok=True, base_url=url)
        return True
    except Exception as e:
        if not quiet:
            print(f"❌ 无法连接到 API ({url})：{e}")
            print("   请确认已启动：uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1")
        _emit("api_check", ok=False, error=str(e), base_url=url)
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


# ════════════════════════════════════════════════════════════
# 白盒层：Checkpoint SQLite 行级校验工具
# ════════════════════════════════════════════════════════════
#
# 这一段不依赖 agent_module / langgraph 的 Python 包，纯用标准库 sqlite3
# 直接读 checkpoints.db 文件。这样设计的考虑：
#   1. 测试脚本和被测服务进程解耦——即使被测进程已经被我们 kill -9 了，
#      仍然可以读盘上的 SQLite 文件做校验（这正是 TEST_8 需要的能力）。
#   2. 用只读模式打开（sqlite3.connect 加 ?mode=ro），避免测试脚本本身
#      意外把数据库写坏，也避免和正在跑的服务进程抢锁。
#   3. schema 来自 langgraph-checkpoint-sqlite（aio.py / __init__.py 里的
#      setup()）：
#         checkpoints(thread_id, checkpoint_ns, checkpoint_id,
#                     parent_checkpoint_id, type, checkpoint, metadata)
#         主键 (thread_id, checkpoint_ns, checkpoint_id)
#         writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
#                channel, type, value)
#         主键 (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
#      metadata 字段是明文 JSON（json.dumps 写入），可以直接 json.loads，
#      不需要引入 langgraph 的 JsonPlusSerializer 反序列化 checkpoint 本体
#      （checkpoint/value 是序列化后的二进制 blob，校验"行级完整性"
#      用不到这部分内容，所以故意不解码，降低对 langgraph 内部实现的依赖）。
#
# 维护提示：如果未来升级 langgraph-checkpoint-sqlite 版本导致表结构变化，
# 这里的 SQL 是第一个需要同步检查的地方。

@dataclass
class DbAuditResult:
    """一次 checkpoints.db 行级审计的结构化结果，方便写进 JSON 报告。"""
    db_path:               str
    table_exists:          bool
    total_checkpoints:     int = 0
    total_writes:          int = 0
    threads_checked:       int = 0
    orphan_writes:         list[dict] = field(default_factory=list)   # writes 表里找不到对应 checkpoint 的行
    broken_parent_links:   list[dict] = field(default_factory=list)   # parent_checkpoint_id 指向不存在的父节点
    non_monotonic_steps:   list[dict] = field(default_factory=list)   # 同一 thread 内 metadata.step 出现回退
    duplicate_primary_keys: list[dict] = field(default_factory=list)  # 理论上不会出现（PK 约束），兜底检查
    error:                 str | None = None

    @property
    def is_clean(self) -> bool:
        return (
            self.table_exists
            and not self.orphan_writes
            and not self.broken_parent_links
            and not self.non_monotonic_steps
            and not self.duplicate_primary_keys
            and self.error is None
        )


def _open_readonly_sqlite(db_path: str) -> sqlite3.Connection:
    """
    以只读模式打开 SQLite 文件。
    用 URI 形式 file:...?mode=ro，这样即使被测服务进程仍持有写连接
    （WAL 模式下读写可以并发），我们这边也绝不会意外写入或加写锁。
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5)


def audit_checkpoint_db(db_path: str, thread_id_filter: str | None = None) -> DbAuditResult:
    """
    对 checkpoints.db 做一次行级完整性审计。

    检查项：
      1. orphan_writes：writes 表中的某一行，其 (thread_id, checkpoint_ns,
         checkpoint_id) 组合在 checkpoints 表里完全找不到对应记录。
         —— 这是"kill 发生在 aput_writes() 已提交、但后续 aput() 还没
         提交"这个时间窗口的直接证据，是 TEST_8 最关心的信号。
      2. broken_parent_links：某个 checkpoint 的 parent_checkpoint_id
         不为空，但在同一 thread 内找不到这个 parent_checkpoint_id
         对应的行。—— 说明这个 thread 的"checkpoint 链"断裂了，
         理论上不应该出现（除非配合 TEST_2 的 /db/cleanup 做过部分清理，
         这种情况下出现属于预期内，调用方需要自行判断是否要排除被清理过
         的 thread）。
      3. non_monotonic_steps：同一 thread 内，按 checkpoint 落盘顺序
         （这里用 rowid 近似"写入顺序"）取出 metadata.step，如果出现
         "后写入的 step 比之前还小"，说明可能发生了乱序写入或并发覆盖。
      4. duplicate_primary_keys：理论上 SQLite 主键约束会阻止重复，
         这里只是个兜底体检项（万一未来 schema 变成非强约束的存储引擎）。

    thread_id_filter：只审计指定 thread_id（内部复合 key，含 user_id__
    前缀也可以，用 LIKE 模糊匹配），用于 TEST_8 只关注本次测试自己造的
    那个 thread，避免被历史遗留的脏数据干扰断言。
    """
    result = DbAuditResult(db_path=db_path, table_exists=False)

    if not Path(db_path).exists():
        result.error = f"数据库文件不存在：{db_path}"
        return result

    try:
        conn = _open_readonly_sqlite(db_path)
    except Exception as exc:
        result.error = f"无法以只读模式打开数据库：{exc}"
        return result

    try:
        cur = conn.cursor()

        # ── 确认表存在 ──────────────────────────────────────────
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('checkpoints', 'writes')"
        )
        existing_tables = {row[0] for row in cur.fetchall()}
        if not {"checkpoints", "writes"}.issubset(existing_tables):
            result.error = f"缺少必要的表，现有表：{existing_tables}"
            return result
        result.table_exists = True

        thread_clause = ""
        thread_params: tuple = ()
        if thread_id_filter:
            thread_clause = " WHERE thread_id LIKE ?"
            thread_params = (f"%{thread_id_filter}%",)

        # ── 基础计数 ────────────────────────────────────────────
        cur.execute(f"SELECT COUNT(*) FROM checkpoints{thread_clause}", thread_params)
        result.total_checkpoints = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM writes{thread_clause}", thread_params)
        result.total_writes = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(DISTINCT thread_id) FROM checkpoints{thread_clause}",
            thread_params,
        )
        result.threads_checked = cur.fetchone()[0]

        # ── 检查 1：orphan writes（writes 有行，checkpoints 没有对应行）──
        # 用 LEFT JOIN 找 checkpoints 侧为 NULL 的行，这是 SQL 里判断
        # "孤儿外键"最直接的写法，不需要遍历 Python 侧两个大 list 再比对。
        cur.execute(
            f"""
            SELECT w.thread_id, w.checkpoint_ns, w.checkpoint_id, w.task_id, w.channel
            FROM writes w
            LEFT JOIN checkpoints c
              ON w.thread_id = c.thread_id
             AND w.checkpoint_ns = c.checkpoint_ns
             AND w.checkpoint_id = c.checkpoint_id
            WHERE c.checkpoint_id IS NULL
            {('AND w.thread_id LIKE ?' if thread_id_filter else '')}
            """,
            thread_params,
        )
        for row in cur.fetchall():
            result.orphan_writes.append({
                "thread_id": row[0], "checkpoint_ns": row[1],
                "checkpoint_id": row[2], "task_id": row[3], "channel": row[4],
            })

        # ── 检查 2：parent_checkpoint_id 断链 ─────────────────────
        cur.execute(
            f"""
            SELECT c.thread_id, c.checkpoint_ns, c.checkpoint_id, c.parent_checkpoint_id
            FROM checkpoints c
            WHERE c.parent_checkpoint_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM checkpoints p
                  WHERE p.thread_id = c.thread_id
                    AND p.checkpoint_ns = c.checkpoint_ns
                    AND p.checkpoint_id = c.parent_checkpoint_id
              )
            {('AND c.thread_id LIKE ?' if thread_id_filter else '')}
            """,
            thread_params,
        )
        for row in cur.fetchall():
            result.broken_parent_links.append({
                "thread_id": row[0], "checkpoint_ns": row[1],
                "checkpoint_id": row[2], "missing_parent_id": row[3],
            })

        # ── 检查 3：metadata.step 单调性（按 rowid 近似写入顺序） ───
        cur.execute(
            f"""
            SELECT thread_id, rowid, metadata
            FROM checkpoints
            {thread_clause}
            ORDER BY thread_id, rowid ASC
            """,
            thread_params,
        )
        last_step_by_thread: dict[str, int] = {}
        for thread_id, rowid, metadata_blob in cur.fetchall():
            try:
                meta = json.loads(metadata_blob) if metadata_blob else {}
                step = meta.get("step")
            except (json.JSONDecodeError, TypeError):
                step = None
            if step is None:
                continue
            prev = last_step_by_thread.get(thread_id)
            if prev is not None and step < prev:
                result.non_monotonic_steps.append({
                    "thread_id": thread_id, "rowid": rowid,
                    "prev_step": prev, "current_step": step,
                })
            last_step_by_thread[thread_id] = step

        # ── 检查 4：主键重复兜底体检（正常情况下不可能触发） ────────
        cur.execute(
            f"""
            SELECT thread_id, checkpoint_ns, checkpoint_id, COUNT(*) AS cnt
            FROM checkpoints
            {thread_clause}
            GROUP BY thread_id, checkpoint_ns, checkpoint_id
            HAVING cnt > 1
            """,
            thread_params,
        )
        for row in cur.fetchall():
            result.duplicate_primary_keys.append({
                "thread_id": row[0], "checkpoint_ns": row[1],
                "checkpoint_id": row[2], "count": row[3],
            })

    except Exception as exc:
        result.error = f"审计过程出错：{exc}"
    finally:
        conn.close()

    return result


def print_audit_result(result: DbAuditResult, label: str) -> None:
    """把 DbAuditResult 打印成人类可读的报告，并落一份 _log 事件。"""
    print(f"\n  📋 {label}")
    print(f"     数据库文件   : {result.db_path}")
    if result.error:
        print(f"     ❌ 审计出错  : {result.error}")
    else:
        print(f"     checkpoints 行数 : {result.total_checkpoints}")
        print(f"     writes 行数      : {result.total_writes}")
        print(f"     涉及 thread 数   : {result.threads_checked}")
        print(f"     孤儿 writes 行   : {len(result.orphan_writes)}")
        print(f"     断裂的父子链     : {len(result.broken_parent_links)}")
        print(f"     step 非单调记录  : {len(result.non_monotonic_steps)}")
        print(f"     重复主键         : {len(result.duplicate_primary_keys)}")

    _emit("db_audit", label=label, db_path=result.db_path,
          table_exists=result.table_exists, error=result.error,
          total_checkpoints=result.total_checkpoints,
          total_writes=result.total_writes,
          threads_checked=result.threads_checked,
          orphan_writes=result.orphan_writes,
          broken_parent_links=result.broken_parent_links,
          non_monotonic_steps=result.non_monotonic_steps,
          duplicate_primary_keys=result.duplicate_primary_keys,
          is_clean=result.is_clean)


# ════════════════════════════════════════════════════════════
# 故障注入层：独立 uvicorn 子进程的生命周期管理
# ════════════════════════════════════════════════════════════
#
# 设计要点：
#   - 用 subprocess.Popen 而不是 os.system/os.spawn，是因为我们需要
#     拿到子进程的 pid 以便后续精确 SIGKILL，也需要能读它的 stdout/stderr
#     方便调试（写入日志文件，而不是让它和测试脚本自己的输出混在一起）。
#   - 用 SIGKILL（signal.SIGKILL / Windows 下 proc.kill()）而不是
#     SIGTERM：SIGTERM 会触发 FastAPI 的 lifespan 优雅关闭逻辑
#     （flush、关数据库连接等），那是"正常重启"，TEST_2 已经覆盖了。
#     这里要测的是"没有任何收尾机会的强杀"，必须是 SIGKILL/kill -9
#     语义，进程没有任何执行用户态清理代码的机会。
#   - 子进程使用独立端口（CHAOS_PORT）和独立数据库目录（CHAOS_DB_DIR），
#     通过环境变量 CHECKPOINT_DB / STORE_DB 注入，这样无论杀多少次、
#     杀得多脏，都不会碰到开发者自己正在用的 8000 端口和 checkpoints.db。
#   - 每次"重启"（包括第一次启动）都会等待 /health 返回 200，避免在
#     MCP 子进程还没就绪时就发请求导致误判为"恢复失败"。

class ChaosProcessManager:
    """
    管理一个独立的、专供故障注入测试使用的 uvicorn 子进程。

    典型用法：
        mgr = ChaosProcessManager(app_module="api:app", port=CHAOS_PORT,
                                   db_dir=CHAOS_DB_DIR)
        mgr.start()                 # 启动并等待就绪
        ... 发请求 ...
        mgr.kill(graceful=False)    # SIGKILL 强杀
        mgr.start()                 # 重新拉起（复用同一个 db_dir，模拟"重启后恢复"）
        ... 验证 ...
        mgr.kill(graceful=True)     # 测试结束，优雅关闭，避免残留进程
    """

    def __init__(self, app_module: str, port: int, host: str, db_dir: Path,
                 cwd: str | None = None, ready_timeout: float = CHAOS_READY_TIMEOUT):
        self.app_module    = app_module
        self.port           = port
        self.host           = host
        self.db_dir         = db_dir
        self.cwd             = cwd or str(Path.cwd())
        self.ready_timeout  = ready_timeout
        self.base_url        = f"http://{host}:{port}"
        self.proc: subprocess.Popen | None = None
        self.log_path        = self.db_dir / "uvicorn_chaos.log"
        self._log_fh          = None

    # ── 端口探测：避免端口被占用导致"启动失败"误判为"恢复失败" ────
    def _port_is_listening(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def start(self) -> None:
        """启动（或重启）uvicorn 子进程，并阻塞等待 /health 就绪。"""
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("子进程已经在运行，请先 kill() 再 start()")

        self.db_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CHECKPOINT_DB"] = str(self.db_dir / "checkpoints.db")
        env["STORE_DB"]      = str(self.db_dir / "memory_store.db")

        cmd = [
            sys.executable, "-m", "uvicorn", self.app_module,
            "--host", self.host, "--port", str(self.port),
            "--workers", "1",
            "--log-level", "warning",   # 减少子进程日志噪音，关键信息仍写进 log 文件
        ]

        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self._log_fh.write(f"\n{'=' * 60}\n[{time.strftime('%H:%M:%S')}] 启动: {' '.join(cmd)}\n")
        self._log_fh.flush()

        self.proc = subprocess.Popen(
            cmd, cwd=self.cwd, env=env,
            stdout=self._log_fh, stderr=subprocess.STDOUT,
            # 独立进程组：保证后面 kill 的时候只杀这一棵进程树，
            # 不会误伤测试脚本自己（POSIX 下用 start_new_session）。
            start_new_session=(os.name != "nt"),
        )

        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.ready_timeout
        last_err = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._dump_log_tail()
                raise RuntimeError(
                    f"子进程提前退出（returncode={self.proc.returncode}），"
                    f"详见日志：{self.log_path}"
                )
            try:
                if check_api(base_url=self.base_url, quiet=True):
                    return
            except Exception as e:
                last_err = str(e)
            time.sleep(0.5)
        self._dump_log_tail()
        raise RuntimeError(
            f"等待子进程就绪超时（{self.ready_timeout}s），最后一次错误：{last_err}，"
            f"详见日志：{self.log_path}"
        )

    def _dump_log_tail(self, n_lines: int = 30) -> None:
        try:
            if self._log_fh:
                self._log_fh.flush()
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"\n  📄 子进程日志最后 {n_lines} 行（{self.log_path}）：")
            for ln in lines[-n_lines:]:
                print(f"       {ln}")
        except Exception:
            pass

    def kill(self, graceful: bool = False) -> float:
        """
        终止子进程，返回从发信号到进程确认退出所耗的秒数。

        graceful=False（默认）：直接 SIGKILL（POSIX）/ proc.kill()
            —— 这是 TEST_7/8 真正要测的"强杀"路径，进程没有任何
            执行 Python finally / lifespan 收尾代码的机会。
        graceful=True：先 SIGTERM 等待退出，超时再 SIGKILL 兜底
            —— 仅用于测试结束后的清理，不代表"正常关闭"测试场景
            （那个由 TEST_2 通过人工重启覆盖）。
        """
        if self.proc is None or self.proc.poll() is not None:
            return 0.0

        t0 = time.time()
        pid = self.proc.pid

        if graceful:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        else:
            if os.name == "nt":
                # Windows 没有 SIGKILL 语义，proc.kill() 本身就是强制终止
                self.proc.kill()
            else:
                # 杀整个进程组，防止 uvicorn 启动的任何子进程（如 MCP
                # stdio 子进程）变成孤儿继续占用端口/资源
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        dt = time.time() - t0
        if self._log_fh:
            self._log_fh.write(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{'优雅终止' if graceful else 'SIGKILL 强杀'} pid={pid}，"
                f"耗时 {dt:.2f}s\n"
            )
            self._log_fh.flush()
            self._log_fh.close()
            self._log_fh = None

        self.proc = None
        return round(dt, 2)

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _preflight_chaos_env() -> bool:
    """
    TEST_7/8 开跑前的环境自检：
      - 确认目标端口没有被别的进程占用（避免"以为杀的是我们自己的子进程，
        其实杀到了别的服务"这种事故）
      - 确认 uvicorn 可执行（用 -m uvicorn --version 探测，而不是真的
        启动一次，避免重复逻辑）
    任何一项失败都直接跳过 TEST_7/8 并打印清晰原因，而不是让 subprocess
    抛出一堆不好理解的底层异常。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((CHAOS_HOST, CHAOS_PORT)) == 0:
            print(f"  ❌ 端口 {CHAOS_PORT} 已被占用，跳过故障注入测试。")
            print(f"     请确认没有遗留的 chaos 测试子进程（可手动检查/结束占用该端口的进程），")
            print(f"     或用 --chaos-port 换一个端口重试。")
            return False

    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "--version"],
            capture_output=True, timeout=10, check=True,
        )
    except Exception as e:
        print(f"  ❌ 找不到可用的 uvicorn（{e}），跳过故障注入测试。")
        print(f"     请确认已 pip install uvicorn，且当前 Python 环境与服务端一致。")
        return False

    return True


# ══════════════════════════════════════════════
# TEST_7（故障注入 · 基础）：kill -9 + 重启恢复
# ══════════════════════════════════════════════

def test_7_kill9_idle_resume(app_module: str, app_cwd: str | None = None) -> None:
    """
    验证：进程在【空闲期】被 SIGKILL 强杀（没有正在处理任何请求），
    重新拉起后历史完整、可以正常继续对话。

    这是故障注入测试里最基础的一档：因为杀的时机保证不落在某个请求
    的处理过程中，所以理论上和"正常重启"（TEST_2）在数据层面应该是
    等价的——做这组测试的意义在于：用真实的 SIGKILL 而不是 Ctrl+C/
    SIGTERM 来验证"没有走任何优雅关闭代码路径"时数据依然完好，
    这是 TEST_2 没有覆盖到的差异点（uvicorn 收到 SIGTERM 后，FastAPI
    的 lifespan 会执行收尾代码；SIGKILL 则完全没有这个机会，
    内核直接回收进程，任何"还没来得及 flush"的东西都没机会 flush 了）。
    """
    section("TEST 7（故障注入 · 基础）：kill -9 强杀（空闲期）+ 重启恢复")

    if not _preflight_chaos_env():
        _results.append({"label": "TEST_7：环境预检", "passed": False})
        return

    mgr = ChaosProcessManager(app_module=app_module, port=CHAOS_PORT,
                              host=CHAOS_HOST, db_dir=CHAOS_DB_DIR, cwd=app_cwd)
    uid = "chaos_kevin"
    tid = f"chaos_idle_kill_{int(time.time())}"

    try:
        info(f"启动独立子进程：{app_module} @ {mgr.base_url}（数据库目录：{mgr.db_dir}）")
        mgr.start()
        assert_true(mgr.is_alive(), "TEST_7：子进程启动并通过 /health 就绪检查")

        step(1, "建立身份（正常请求，子进程存活期间完成）")
        r = chat("你好，我叫 Kevin，是一名 DBA，专长是 MySQL 性能调优。",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        show_answer(r)
        mc_before = r.get("message_count", 0)

        info("等待 1 秒确保上一次请求的 checkpoint 已完全落盘（进入真正空闲态）")
        time.sleep(1)

        info(f"对子进程（pid={mgr.proc.pid if mgr.proc else '?'}）发送 SIGKILL ...")
        kill_dt = mgr.kill(graceful=False)
        assert_true(not mgr.is_alive(), "TEST_7：子进程已被 SIGKILL 终止",
                    detail=f"信号发送到确认退出耗时 {kill_dt}s")

        info("重新拉起子进程（沿用同一个数据库目录，模拟容器重启/调度器重新拉起）")
        mgr.start()
        assert_true(mgr.is_alive(), "TEST_7：重启后子进程再次通过 /health 就绪检查")

        step(2, "重启后追问：还记得我吗？")
        r = chat("你还记得我吗？我叫什么名字？我是做什么的？",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        a = show_answer(r)
        assert_contains(a, ["Kevin", "kevin"], "TEST_7：kill -9 重启后姓名仍可召回")
        assert_contains(a, ["DBA", "MySQL"], "TEST_7：kill -9 重启后职业/专长仍可召回")

        mc_after = r.get("message_count", 0)
        assert_true(
            mc_after > mc_before,
            "TEST_7：重启后消息数在原有基础上继续累加（未被强杀清零）",
            detail=f"强杀前={mc_before}, 重启后={mc_after}",
        )

        step(3, "重启后继续推进对话，确认会话完全可用")
        r = chat("好的，那帮我记一下：下周要做一次慢查询日志分析。",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        show_answer(r)
        r = chat("我下周要做什么？", user_id=uid, thread_id=tid, base_url=mgr.base_url)
        b = show_answer(r)
        assert_contains(b, ["慢查询", "日志"], "TEST_7：强杀重启后新一轮对话正常生效")

        # ── 白盒层加固：顺手做一次 DB 审计，确认空闲期强杀没有留下任何孤儿行 ──
        db_path = str(mgr.db_dir / "checkpoints.db")
        audit = audit_checkpoint_db(db_path, thread_id_filter=tid)
        print_audit_result(audit, "TEST_7 白盒校验：空闲期强杀后的 checkpoints.db")
        assert_true(
            audit.is_clean,
            "TEST_7：空闲期强杀未在 SQLite 中留下孤儿写入/断链/乱序记录",
            detail=f"orphan_writes={len(audit.orphan_writes)}, "
                   f"broken_parent_links={len(audit.broken_parent_links)}, "
                   f"non_monotonic_steps={len(audit.non_monotonic_steps)}",
        )

    finally:
        # 无论测试成功/失败，都要确保子进程被清理掉，不留下僵尸进程占用端口
        if mgr.is_alive():
            info("清理：关闭本次测试拉起的子进程")
            mgr.kill(graceful=True)


# ══════════════════════════════════════════════
# TEST_8（故障注入 · 最高难度）：图执行中 kill -9
# ══════════════════════════════════════════════

def test_8_kill9_mid_execution(app_module: str, kill_delay: float, app_cwd: str | None = None) -> None:
    """
    验证：在 LangGraph 正在执行某个 node（planner / parallel_executor /
    final_answer 任一阶段）的【时间窗口内】发送 SIGKILL，制造一次
    "真正执行到一半被打断、且这次 invoke 永远不会再返回任何结果"的中断。

    实现思路：
      1. 在后台线程发起一个【刻意设计得比较慢】的请求（要求 agent
         调用工具、做多步规划，从而拉长 planner/parallel_executor 的
         执行时间），这个请求本身预期会因为子进程被杀而直接失败
         （连接被重置/对端关闭），这是预期内的，不算测试失败。
      2. 主线程 sleep(kill_delay) 后发送 SIGKILL —— kill_delay 需要
         小于这个慢请求的预期总耗时，从而保证杀的时候 graph 大概率
         还在执行中（而不是已经跑完在等返回）。这个时机本质上无法
         100% 精确控制（毕竟我们不知道 node 内部执行到了第几步），
         所以这是一个"概率性"的故障注入——多跑几次、配合 TEST_9 的
         事后审计，比单次"严格证明杀在某一行代码"更现实也更通用。
      3. 重启子进程后做两层验证：
         (a) 黑盒：服务恢复可用，且被打断之前已确认完成的历史
             （第 1 步建立的身份信息）完整无损。
         (b) 白盒：直接审计 SQLite，重点看 orphan_writes —— 如果
             kill 真的命中了"writes 已落盘但新 checkpoint 还没写"
             这个窗口，这里就会观测到非零的孤儿行。命中与否本身
             是概率事件，所以这一项不强制断言"必须为 0"，而是
             如实记录现象、并断言"即使出现孤儿行，重启后的会话
             依然可用、早期历史依然完整"——这才是用户真正关心的
             结果（数据库里一两条不会再被读到的孤儿 writes 行，
             不影响功能正确性；图状态不一致、历史丢失才是真正的事故）。

    这是全套测试里最难、也最有价值的一组：它是唯一一组真正触达
    "LangGraph 单步执行被物理中断"这个故障域的用例，其余测试要么是
    "两次完整 invoke 之间"的中断（TEST_1/2/7），要么是"客户端层面的
    放弃等待，但服务端 invoke 可能仍在后台跑完"（TEST_3/4/5/6）。
    """
    section("TEST 8（故障注入 · 最高难度）：图执行过程中 kill -9")

    if not _preflight_chaos_env():
        _results.append({"label": "TEST_8：环境预检", "passed": False})
        return

    mgr = ChaosProcessManager(app_module=app_module, port=CHAOS_PORT,
                              host=CHAOS_HOST, db_dir=CHAOS_DB_DIR, cwd=app_cwd)
    uid = "chaos_luna"
    tid = f"chaos_midexec_kill_{int(time.time())}"

    try:
        info(f"启动独立子进程：{app_module} @ {mgr.base_url}")
        mgr.start()
        assert_true(mgr.is_alive(), "TEST_8：子进程启动并通过 /health 就绪检查")

        step(1, "建立身份（正常完整请求，作为'被打断前确认完成'的基准历史）")
        r = chat("你好，我叫 Luna，是一名机器学习工程师，专注推荐系统。",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        show_answer(r)
        mc_before = r.get("message_count", 0)
        info("等待 1 秒确保第 1 步的 checkpoint 已完全落盘")
        time.sleep(1)

        step(2, f"后台发起一个较慢的请求，{kill_delay}s 后对子进程发送 SIGKILL")

        slow_question = (
            "帮我详细设计一套推荐系统的离线评估方案，"
            "分别说明召回、排序、重排三个阶段分别要看哪些离线指标，"
            "每个指标的计算方式、适用场景和局限性都要展开讲清楚，"
            "并给出一个完整的实验对照设计。"
        )

        bg_result: dict = {}
        bg_error:  dict = {}

        def _slow_request_worker() -> None:
            try:
                bg_result["resp"] = chat(
                    slow_question, user_id=uid, thread_id=tid,
                    timeout=TIMEOUT, base_url=mgr.base_url,
                )
            except Exception as e:
                bg_error["err"] = f"{type(e).__name__}: {e}"

        bg_thread = threading.Thread(target=_slow_request_worker, daemon=True)
        t_request_start = time.time()
        bg_thread.start()

        time.sleep(kill_delay)

        # 这里不再检查 mgr.is_alive()——只要请求线程已经发出去了，
        # 不论 graph 实际跑到哪个 node，我们都按计划强杀，这正是
        # "概率性命中执行中间状态"的设计本意。
        info(f"已等待 {kill_delay}s（预期此时慢请求大概率仍在 planner/parallel_executor 阶段），"
             f"对子进程（pid={mgr.proc.pid if mgr.proc else '?'}）发送 SIGKILL ...")
        kill_dt = mgr.kill(graceful=False)
        t_killed = time.time()
        assert_true(not mgr.is_alive(), "TEST_8：子进程已在执行过程中被 SIGKILL 终止",
                    detail=f"信号发送到确认退出耗时 {kill_dt}s")

        # 给后台线程一点时间感知到连接被对端重置并退出（不强制等待太久，
        # 它大概率会在几秒内因为 ConnectionResetError / RemoteDisconnected
        # 之类的异常而结束）。
        bg_thread.join(timeout=10)
        if "resp" in bg_result:
            info("意外：慢请求在被 kill 之前就已经完整返回了——说明 kill_delay 设得太长，"
                 "本次没有真正命中'执行中'窗口，可以用 --kill-delay 调小重测。")
            _emit("test8_timing", hit_mid_execution=False)
        else:
            err_msg = bg_error.get("err", "（线程仍在等待，可能已成为孤儿线程）")
            print(f"       ⚡ 慢请求按预期未能正常完成：{err_msg}")
            _emit("test8_timing", hit_mid_execution=True, error=err_msg,
                  elapsed_before_kill_s=round(t_killed - t_request_start, 2))

        info("重新拉起子进程（沿用同一个数据库目录）")
        mgr.start()
        assert_true(mgr.is_alive(), "TEST_8：重启后子进程再次通过 /health 就绪检查")

        # ── 黑盒验证 (a)：被打断前已确认完成的历史必须完整 ──────────
        step(3, "【黑盒】重启后追问：被打断前确认过的身份信息是否完整")
        r = chat("我叫什么名字？我是做什么方向的？",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        a = show_answer(r)
        assert_contains(a, ["Luna", "luna"], "TEST_8：图执行中被强杀后，更早期的历史姓名仍完整")
        assert_contains(a, ["推荐系统", "机器学习", "ML"],
                        "TEST_8：图执行中被强杀后，更早期的历史职业方向仍完整")

        mc_after = r.get("message_count", 0)
        assert_true(
            mc_after >= mc_before,
            "TEST_8：图执行中被强杀后，消息数未发生回退（至少不低于打断前的基准）",
            detail=f"打断前基准={mc_before}, 重启后={mc_after}",
        )

        step(4, "【黑盒】验证服务在'最难的中断点'之后仍能正常处理新请求")
        r = chat("没事，我们重新开始这个话题：先帮我列一个推荐系统离线评估要看哪几类指标就行，简单说。",
                 user_id=uid, thread_id=tid, base_url=mgr.base_url)
        c = show_answer(r)
        assert_true(bool(c.strip()), "TEST_8：图执行中被强杀后，服务对新请求仍能正常给出回答")

        # ── 白盒验证 (b)：审计 SQLite，如实记录孤儿写入现象 ──────────
        db_path = str(mgr.db_dir / "checkpoints.db")
        audit = audit_checkpoint_db(db_path, thread_id_filter=tid)
        print_audit_result(audit, "TEST_8 白盒校验：图执行中强杀后的 checkpoints.db")

        if audit.orphan_writes:
            info(
                f"检测到 {len(audit.orphan_writes)} 条孤儿 writes 行——"
                "这正是'kill 命中 aput_writes() 已提交但 aput() 还没提交'这个"
                "缝隙的直接证据，符合预期的故障特征，不代表数据丢失"
                "（这些行不会再被任何正常读路径引用到）。"
            )
        else:
            info("本次未观测到孤儿 writes 行——说明这次 kill 没有精确命中那个极窄的"
                 "写入缝隙（这本来就是概率事件），可以多跑几次 / 调整 --kill-delay 复测。")

        # 真正的硬性要求不是"数据库必须 0 孤儿行"，而是"断链不能发生"——
        # parent_checkpoint_id 断链意味着这个 thread 的状态机本身读不回来了，
        # 这才是会真正影响用户体验的硬伤。
        assert_true(
            not audit.broken_parent_links,
            "TEST_8：即使可能存在孤儿 writes 行，checkpoint 主链路（parent 链）不能断裂",
            detail=f"broken_parent_links={audit.broken_parent_links}",
        )
        assert_true(
            not audit.duplicate_primary_keys,
            "TEST_8：未出现重复主键（SQLite 主键约束 + WAL 串行写入应保证这一点）",
        )

    finally:
        if mgr.is_alive():
            info("清理：关闭本次测试拉起的子进程")
            mgr.kill(graceful=True)


# ══════════════════════════════════════════════
# TEST_9（白盒巡检）：Checkpoint 行级完整性审计
# ══════════════════════════════════════════════

def test_9_db_integrity_audit(db_path: str) -> None:
    """
    独立的、纯 SQL 级别的"数据库体检"，不依赖前面任何测试制造中断，
    可以单独对一个已经跑了一段时间的 checkpoints.db 做巡检。

    典型使用场景：
      - 作为运维巡检脚本，定期对生产环境的 checkpoints.db 跑一遍
        （注意：生产环境建议先 cp 一份只读快照再跑，不要直接对线上
        文件做长查询，避免和 WAL checkpoint 机制产生不必要的 IO 争用）。
      - 在 TEST_7/TEST_8 这类故障注入测试之后，作为补充复查手段。
      - 怀疑某次线上事故和 checkpoint 持久化有关时，第一时间跑一遍
        定位是否存在孤儿行/断链，比逐条人工翻 SQLite 文件快得多。
    """
    section("TEST 9（白盒巡检）：Checkpoint 行级完整性审计")

    if not Path(db_path).exists():
        print(f"  ❌ 数据库文件不存在：{db_path}")
        print(f"     请用 --db-path 指定正确路径，或先跑过 TEST_1~6 让它产生数据。")
        _results.append({"label": "TEST_9：数据库文件存在性检查", "passed": False})
        return

    audit = audit_checkpoint_db(db_path)
    print_audit_result(audit, "TEST_9 全库审计")

    assert_true(audit.table_exists, "TEST_9：checkpoints / writes 表结构存在且可读")
    assert_true(
        len(audit.orphan_writes) == 0,
        "TEST_9：全库范围内不存在孤儿 writes 行",
        detail=(f"发现 {len(audit.orphan_writes)} 条，示例："
                f"{audit.orphan_writes[:3]}" if audit.orphan_writes else ""),
    )
    assert_true(
        len(audit.broken_parent_links) == 0,
        "TEST_9：全库范围内不存在断裂的 parent_checkpoint_id 链",
        detail=(f"发现 {len(audit.broken_parent_links)} 条，示例："
                f"{audit.broken_parent_links[:3]}" if audit.broken_parent_links else ""),
    )
    assert_true(
        len(audit.non_monotonic_steps) == 0,
        "TEST_9：全库范围内每个 thread 的 metadata.step 均单调不减",
        detail=(f"发现 {len(audit.non_monotonic_steps)} 条，示例："
                f"{audit.non_monotonic_steps[:3]}" if audit.non_monotonic_steps else ""),
    )
    assert_true(
        len(audit.duplicate_primary_keys) == 0,
        "TEST_9：未发现重复主键（SQLite 主键约束应天然保证，属兜底体检）",
    )

    if audit.is_clean:
        print(f"\n  ✅ 数据库体检结论：{db_path} 行级完整性良好，无可疑记录。")
    else:
        print(f"\n  ⚠️  数据库体检结论：{db_path} 存在需要关注的记录，详见上方明细，"
              f"也已完整写入 JSON 报告的 events 部分（type=db_audit）。")


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

# 等级 → 该等级下属的测试编号。
#   basic/advanced/complex 三档全部是黑盒、无侵入性，CI 里可以放心跑。
#   chaos 档（TEST_7/8）会真的杀进程，必须配合 --enable-process-kill
#   显式开启，且默认不包含在 --level all 里。
#   TEST_9（纯白盒巡检）不属于任何"等级"，只能通过 --only 9 单独调用，
#   因为它不依赖一个正在运行的服务，语义上更接近独立工具而非测试用例。
LEVEL_TESTS = {
    "basic":    ["1", "2"],
    "advanced": ["3", "4"],
    "complex":  ["5", "6"],
    "chaos":    ["7", "8"],
}

# "all" 默认只包含黑盒三档，故障注入（7/8）和纯白盒巡检（9）需要显式指定，
# 这是有意为之的安全默认值：不应该让一条不加任何参数的命令意外杀掉
# 别人正在用的服务进程。
ALL_SAFE_TESTS = ["1", "2", "3", "4", "5", "6"]


def main() -> None:
    global BASE_URL, SHORT_TIMEOUT, CHAOS_PORT, CHAOS_HOST, CHAOS_BASE_URL

    parser = argparse.ArgumentParser(
        description="Checkpoint 中断恢复测试套件（黑盒 + 白盒 + 故障注入三层）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 基础运行控制 ──────────────────────────────────────────
    parser.add_argument(
        "--rerun", action="store_true",
        help="二次运行模式：跳过 TEST_2 写入，执行 TEST_2 读取（验证服务重启后断点续接）",
    )
    parser.add_argument(
        "--level", choices=["basic", "advanced", "complex", "chaos", "all"], default="all",
        help="只跑某一等级：basic(1,2) / advanced(3,4) / complex(5,6) / "
             "chaos(7,8，需配合 --enable-process-kill) / all（默认，"
             "等价于 basic+advanced+complex，不含 chaos 和 TEST_9）",
    )
    parser.add_argument(
        "--only", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        help="只跑指定编号的单个测试（7/8 需配合 --enable-process-kill；"
             "9 是独立的白盒巡检，配合 --db-path 使用）",
    )
    parser.add_argument(
        "--short-timeout", type=float, default=SHORT_TIMEOUT,
        help=f"用于制造网络层'中断'的客户端短超时秒数（默认 {SHORT_TIMEOUT}s，"
             "TEST_3/TEST_6 使用；如果你的 agent 响应很快导致测不出"
             "中断效果，可以调更小）",
    )
    parser.add_argument("--url", default=BASE_URL, help=f"被测 API 地址（默认 {BASE_URL}，"
                         "供 TEST_1~6 黑盒测试使用，要求该服务已在运行）")

    # ── 故障注入（TEST_7/8）专用参数 ──────────────────────────
    chaos_group = parser.add_argument_group(
        "故障注入参数（TEST_7/8 专用）",
        "这组测试会自己拉起/强杀一个独立的 uvicorn 子进程，默认不会执行，"
        "必须显式加 --enable-process-kill 才会真正运行。",
    )
    chaos_group.add_argument(
        "--enable-process-kill", action="store_true",
        help="显式确认允许本次运行执行 kill -9 强杀测试（TEST_7/8）。"
             "不加这个参数时，即使用 --level chaos 或 --only 7/8 指定，"
             "也只会打印安全提示并跳过，不会真的启动/杀死任何进程。",
    )
    chaos_group.add_argument(
        "--app-module", default=CHAOS_APP_MODULE,
        help=f"TEST_7/8 用来拉起子进程的 uvicorn 'module:attribute'"
             f"（默认 {CHAOS_APP_MODULE}，已按 src 布局设置；"
             "如果你的服务入口不是 src/api.py，用这个参数覆盖，"
             "比如根目录 api.py 项目要传 --app-module api:app）",
    )
    chaos_group.add_argument(
        "--app-cwd", default=None,
        help="启动 TEST_7/8 子进程时使用的工作目录（默认当前目录；"
             "需要能在该目录下 import 到 --app-module 指定的模块）",
    )
    chaos_group.add_argument(
        "--chaos-port", type=int, default=CHAOS_PORT,
        help=f"TEST_7/8 子进程监听的端口（默认 {CHAOS_PORT}，"
             "刻意和常规开发端口 8000 区分开）",
    )
    chaos_group.add_argument(
        "--kill-delay", type=float, default=2.0,
        help="TEST_8 专用：发出慢请求后等待多少秒再发送 SIGKILL（默认 2.0s）。"
             "这个值需要小于 agent 处理该慢请求的实际耗时，才能命中"
             "'图执行中'这个窗口；如果你的环境响应很快，调小这个值；"
             "如果总是'提前完成'命中不了，也可以调小。",
    )

    # ── 白盒巡检（TEST_9）专用参数 ────────────────────────────
    audit_group = parser.add_argument_group(
        "白盒巡检参数（TEST_9 专用）",
        "TEST_9 直接读 SQLite 文件，不需要服务在运行。",
    )
    audit_group.add_argument(
        "--db-path", default=None,
        help="TEST_9 要审计的 checkpoints.db 路径（默认尝试 data/checkpoints.db，"
             "找不到则提示手动指定）",
    )

    args = parser.parse_args()

    BASE_URL      = args.url.rstrip("/")
    SHORT_TIMEOUT = args.short_timeout
    CHAOS_PORT    = args.chaos_port
    CHAOS_BASE_URL = f"http://{CHAOS_HOST}:{CHAOS_PORT}"

    # ── 决定本次要跑哪些编号 ──────────────────────────────────
    if args.only:
        todo = [args.only]
    elif args.level == "all":
        todo = list(ALL_SAFE_TESTS)
    else:
        todo = list(LEVEL_TESTS[args.level])

    needs_chaos = any(t in ("7", "8") for t in todo)
    needs_audit_only = todo == ["9"]
    needs_live_api = any(t in ("1", "2", "3", "4", "5", "6") for t in todo)

    print(f"\n🚀 Checkpoint 中断恢复测试套件")
    if needs_live_api:
        print(f"   黑盒目标 API : {BASE_URL}")
    print(f"   模式         : {'二次运行（--rerun，验证服务重启续接）' if args.rerun else '首次运行'}")
    print(f"   本次计划执行 : TEST {', '.join(todo)}")
    print(f"   单题超时     : {TIMEOUT}s")
    print(f"   中断短超时   : {SHORT_TIMEOUT}s")
    if needs_chaos:
        print(f"   故障注入端口 : {CHAOS_BASE_URL}")
        print(f"   故障注入开关 : {'已启用 --enable-process-kill' if args.enable_process_kill else '⚠️  未启用，TEST_7/8 将被跳过'}")

    # ── 安全闸门：故障注入测试必须显式确认 ────────────────────
    if needs_chaos and not args.enable_process_kill:
        print(f"\n  ⚠️  检测到本次计划包含 TEST_7/8（kill -9 故障注入），")
        print(f"     但未加 --enable-process-kill，出于安全考虑自动跳过这两项。")
        print(f"     这两项测试会真实启动并强杀一个独立的 uvicorn 子进程")
        print(f"     （监听 {CHAOS_BASE_URL}，独立数据库目录，不会碰你正在用的服务），")
        print(f"     确认要执行的话请加上 --enable-process-kill 重新运行。")
        for t in ("7", "8"):
            if t in todo:
                todo.remove(t)
                _results.append({"label": f"TEST_{t}：因未加 --enable-process-kill 而跳过", "passed": True})
                _emit("skip", test=t, reason="--enable-process-kill not set")

    # ── 黑盒测试需要先确认目标服务可达，再开始计时 ────────────
    if needs_live_api and not check_api():
        sys.exit(1)

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

        if "7" in todo:
            test_7_kill9_idle_resume(app_module=args.app_module, app_cwd=args.app_cwd)

        if "8" in todo:
            test_8_kill9_mid_execution(app_module=args.app_module, kill_delay=args.kill_delay, app_cwd=args.app_cwd)

        if "9" in todo:
            db_path = args.db_path
            if db_path is None:
                # 沿用 langgraph_parallel_agent.py 里 _CHECKPOINT_DB 的默认约定：
                # 项目根目录 / data / checkpoints.db。这里只是个便利兜底，
                # 强烈建议显式传 --db-path 避免猜错路径。
                default_guess = Path("data") / "checkpoints.db"
                db_path = str(default_guess)
                info(f"未指定 --db-path，按约定尝试默认路径：{db_path}")
            test_9_db_integrity_audit(db_path=db_path)

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

# ──────────────────────────────────────────────────────────────────
# 常用命令速查（按你的项目结构：mcp-server-template/ 为根目录，
#               服务入口是 src/api.py，启动方式 uvicorn src.api:app）
# ──────────────────────────────────────────────────────────────────
#
# ⚠️ 重要：你的项目是 src 布局（src/api.py），不是根目录 api.py！
#    本脚本的 --app-module 默认值已经按你的项目设为 src.api:app，
#    所以下面 TEST_7/8 的命令理论上可以不写 --app-module 也能跑对；
#    但仍然建议显式带上 --app-module src.api:app --app-cwd .，
#    一是更清楚地表明意图，二是防止以后你把这个脚本复制到别的、
#    布局不同的项目里时，默认值悄悄帮你指向了错误的模块。
#    黑盒测试（TEST_1~6）和白盒巡检（TEST_9）不受这个布局差异影响，
#    因为它们要么只打 HTTP 接口，要么只读 SQLite 文件，跟 uvicorn
#    怎么启动无关。
#
# 以下命令均在 PowerShell 下，项目根目录
#   C:\Users\tonysong\Desktop\AI_Python\mcp-server-template
# 执行（和你截图里的终端路径一致）。
#
#
# ════ 黑盒测试（TEST_1~6，需要服务已在运行，安全、无侵入性）════
#
# 终端 1 — 启动 API（保持运行，不要关）：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
#
# 终端 2 — 跑全部黑盒测试（基础 + 高级 + 复杂，默认行为）：
#   uv run python scripts/test_checkpoint_resume.py
#
# 只跑基础测试（最快，适合日常回归 / CI 冒烟）：
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
# TEST_2 服务重启验证（完整流程，模拟"正常重启"而非强杀）：
#   1) uv run python scripts/test_checkpoint_resume.py --only 2      # 建档
#   2) 终端1 按 Ctrl+C 停掉 api，再重新执行 uvicorn 命令启动
#   3) uv run python scripts/test_checkpoint_resume.py --only 2 --rerun
#
# 如果你的模型/网络响应很快，TEST_3/TEST_6 的"中断"可能触发不了
# （3秒内就跑完了），可以调小短超时强制触发：
#   uv run python scripts/test_checkpoint_resume.py --level advanced --short-timeout 1
#
#
# ════ 故障注入测试（TEST_7/8，会真实 kill -9 一个子进程）════
#
# 不需要你手动起服务——脚本自己管生命周期，用独立端口 8099 +
# 独立数据库目录 scripts/.chaos_test_db/，不会碰你正在用的开发服务器
# 和 data/checkpoints.db。必须显式加 --enable-process-kill，否则
# 会被安全跳过（不会真的启动/杀死任何进程）。
#
# 下面命令里仍然显式写出 --app-module 和 --app-cwd（习惯性写法，
#    理由见上方提示）：
#
# 跑 TEST_7（空闲期强杀，基础档）：
#   uv run python scripts/test_checkpoint_resume.py --only 7 `
#       --enable-process-kill --app-module src.api:app --app-cwd .
#
# 跑 TEST_8（图执行中强杀，最高难度档）：
#   uv run python scripts/test_checkpoint_resume.py --only 8 `
#       --enable-process-kill --app-module src.api:app --app-cwd .
#
# 一次跑完 7+8：
#   uv run python scripts/test_checkpoint_resume.py --level chaos `
#       --enable-process-kill --app-module src.api:app --app-cwd .
#
# （注：PowerShell 里多行命令的续行符是反引号 ` ，不是 Linux/Mac 的
#  反斜杠 \ ；如果你想写在一行里，把 ` 去掉、所有参数接在同一行即可）
#
# TEST_8 如果总是"提前完成命中不了执行中窗口"（脚本会打印提示），
# 适当调小 --kill-delay（比如网络/模型很快的环境）：
#   uv run python scripts/test_checkpoint_resume.py --only 8 `
#       --enable-process-kill --app-module src.api:app --app-cwd . `
#       --kill-delay 0.8
#
# TEST_8 是概率性测试，建议至少跑 3~5 次观察孤儿写入命中率，
# 单次跑不出孤儿行不代表"没问题"，跑出来孤儿行也不代表"有问题"——
# 真正要看的硬性指标是 broken_parent_links 和 duplicate_primary_keys
# 是否始终为 0（这两项是断言失败会让脚本退出码非 0 的硬性要求）。
#
#
# ════ 白盒巡检（TEST_9，纯 SQL，不需要服务在运行）════
#
# 对你项目的真实数据库 data/checkpoints.db 做一次体检
# （这是你项目里 _CHECKPOINT_DB 默认指向的路径，见 langgraph_parallel_agent.py）：
#   uv run python scripts/test_checkpoint_resume.py --only 9 `
#       --db-path data/checkpoints.db
#
# 也可以单独对 TEST_7/8 自己产生的 chaos 测试数据库做体检：
#   uv run python scripts/test_checkpoint_resume.py --only 9 `
#       --db-path scripts/.chaos_test_db/checkpoints.db
#
#
# ════ 退出码约定（便于接入 CI / 巡检脚本）════
#
#   0 = 本次执行的所有断言全部通过
#   1 = 至少一项断言失败，或执行过程中出现未捕获异常
#
# 每次运行都会在 scripts/ 目录下生成一份
#   test_checkpoint_resume_result_<时间戳>_<first|rerun>.json
# 包含完整的 meta / summary（断言列表）/ events（含 db_audit 事件，
# 可以从中提取 orphan_writes 等明细做进一步分析）。
#
#
# ──────────────────────────────────────────────────────────────────
# 完整操作步骤（按你的项目结构、从零开始，照抄即可）
# ──────────────────────────────────────────────────────────────────
#
# 【准备工作】打开两个 PowerShell 终端，都 cd 到项目根目录：
#   cd C:\Users\tonysong\Desktop\AI_Python\mcp-server-template
#
#
# 【第一轮】黑盒测试（TEST_1~6）—— 日常回归，最常用，最安全
#
#   终端 1（保持运行，不要关）：
#     uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
#     等到看见 "Uvicorn running on http://0.0.0.0:8000" 再去终端 2。
#
#   终端 2：
#     uv run python scripts/test_checkpoint_resume.py
#     跑完会在屏幕上看到 PASS/FAIL 列表和总耗时，同时在 scripts/ 目录下
#     生成一份 test_checkpoint_resume_result_<时间戳>_first.json。
#
#   如果想单独验证"服务重启后还能不能接上对话"（TEST_2，模拟正常重启）：
#     终端 2：uv run python scripts/test_checkpoint_resume.py --only 2
#     终端 1：按 Ctrl+C 停掉，再重新跑一遍 uvicorn 那条启动命令
#     终端 2：uv run python scripts/test_checkpoint_resume.py --only 2 --rerun
#
#
# 【第二轮】故障注入测试（TEST_7/8）—— 真实 kill -9，建议单独空出一段时间跑
#
#   不需要你手动起服务，也不需要终端 1 继续开着（这组测试会自己拉起
#   一个完全独立的 uvicorn 子进程，端口 8099，不会跟终端 1 的 8000
#   服务冲突，也不会用到 data/checkpoints.db）。
#
#   终端 2（或新开一个终端）：
#     uv run python scripts/test_checkpoint_resume.py --only 7 `
#         --enable-process-kill --app-module src.api:app --app-cwd .
#
#   屏幕上应该依次看到：
#     ✅ 子进程启动并通过 /health 就绪检查
#     [轮1] 建立身份...
#     ✅ 子进程已被 SIGKILL 终止
#     ✅ 重启后子进程再次通过 /health 就绪检查
#     [轮2] 重启后追问：还记得我吗？...
#     ✅ kill -9 重启后姓名仍可召回
#     ...
#     📋 TEST_7 白盒校验：空闲期强杀后的 checkpoints.db
#
#   确认 TEST_7 全过之后，再跑最难的 TEST_8（图执行中强杀）：
#     uv run python scripts/test_checkpoint_resume.py --only 8 `
#         --enable-process-kill --app-module src.api:app --app-cwd .
#
#   TEST_8 屏幕上会明确告诉你这次"有没有真的命中执行中窗口"：
#     - 如果看到 "⚡ 慢请求按预期未能正常完成：..." → 命中了，是最有价值的一次
#     - 如果看到 "意外：慢请求在被 kill 之前就已经完整返回了" → 没命中，
#       说明你的 agent 处理那道"详细规划"题比预期快，可以加 --kill-delay
#       调小重跑一次，比如 --kill-delay 1（默认是 2 秒）
#   建议这条命令多跑 3~5 次，分别看一下白盒校验部分的"孤儿 writes 行"
#   是不是出现过，出现了也不算失败（脚本会解释这是正常的概率事件），
#   真正要盯的是"断裂的父子链"和"重复主键"这两项必须一直是 0。
#
#
# 【第三轮】白盒巡检（TEST_9）—— 单独检查 data/checkpoints.db 是否健康
#
#   这一步不需要任何服务在跑，随时可以执行，建议在跑完第一轮 + 第二轮
#   之后都各跑一次，确认你的真实数据库没有被前面的测试间接搞坏：
#
#     uv run python scripts/test_checkpoint_resume.py --only 9 `
#         --db-path data/checkpoints.db
#
#   全绿即代表 data/checkpoints.db 行级完整性良好。
#
#
# 【看结果】每次运行后，去 scripts/ 目录下找最新的
#   test_checkpoint_resume_result_<时间戳>_<first|rerun>.json
# 这份文件里 meta.all_passed 是 true/false 的总开关，summary 是每一条
# 断言的明细，events 里能看到完整的回答内容和（如果跑了 TEST_7/8/9）
# db_audit 审计的原始数据，方便你追查具体是哪一条没过、为什么没过。