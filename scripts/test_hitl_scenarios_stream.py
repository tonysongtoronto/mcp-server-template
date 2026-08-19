#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1



uv run python scripts/test_hitl_scenarios_stream.py

HITL（人工介入断点续传）功能 —— 流式（SSE）自动化回归测试脚本
────────────────────────────────────────────────────────
这是 scripts/test_hitl_scenarios.py 的"流式版姊妹篇"：场景编排完全复用
src/hitl_test_scenarios.py（"/hitl_test <scenario_name>" 旁路钩子），
断言目标也和非流式版一致（task_plan 各任务最终状态、pending_gate_items、
plan_status），但请求路径改成走 api.py 里的三个 SSE 接口：

    GET  /chat/stream
    POST /session/{user_id}/{thread_id}/resume/stream
    POST /session/{user_id}/{thread_id}/abort/stream

而不是非流式版用的 /chat、/resume、/abort。

★ 关于"流式模式下收不到 answer 文本"的说明（读代码得出的重要结论，
   不是本脚本的 bug）：
   final_answer_node 命中 "/hitl_test " 前缀时会直接调用
   hitl_test_scenarios.build_final_answer() 拼文本并 return，这条早退路径
   在真正的 "q is not None → llm.astream() 逐 token 推流" 分支之前，
   所以 SSE 连接里根本不会有任何 `data: <token>` 事件，只会有：
     - （可能有）event: interrupted + data: {...}
     - 收尾的 `data: [WAITING_HUMAN:<tid>]` 或 `data: [DONE:<tid>]`
   因此本脚本的断言全部基于"收尾标记 + interrupted 事件 payload +
   事后查询 /session/.../state"，不依赖流里的文本 token（那本来就是空的）。
   如果想验证"真的在流式吐 token"，需要发一条不带 "/hitl_test " 前缀、
   会触发真实 LLM 汇总的请求，但那样就需要真实可用的 LLM API Key，
   不适合放进这份追求"无 LLM 依赖、可重复"的回归测试里。

依赖：pip install requests

用法：
    python scripts/test_hitl_scenarios_stream.py
    API_BASE=http://localhost:8000 python scripts/test_hitl_scenarios_stream.py

    # 自定义日志输出目录（默认 scripts/logs_stream/，跟脚本同目录下的子文件夹）
    HITL_STREAM_TEST_LOG_DIR=/tmp/my_logs python scripts/test_hitl_scenarios_stream.py

    # 不想生成日志文件时可以关掉
    HITL_STREAM_TEST_LOG_ENABLED=0 python scripts/test_hitl_scenarios_stream.py

前置条件：
    - api.py 已经在跑（默认 http://localhost:8000，见 /health）
    - src/hitl_test_scenarios.py 存在（HITL 测试专用模块，独立文件）
    - langgraph_parallel_agent.py 里的三处 "★ HITL 测试专用钩子" 还在
      （没有被删掉），"/hitl_test <scenario>" 才能正常触发测试场景
────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
USER_ID  = "hitl_stream_test_user"

# requests 的 (connect_timeout, read_timeout)：
# read_timeout 要比服务端 _stream_graph_run 里的 120s 等待超时更长一点，
# 避免服务端还没来得及自己吐出 "[ERROR] 等待超时" 就被客户端先掐断连接。
_REQUEST_TIMEOUT = (5, 130)

PASS, FAIL = [], []


# ══════════════════════════════════════════════════════
# 日志子系统（结构与 test_hitl_scenarios.py 保持一致，独立的文件名/目录，
# 避免跟非流式版的日志互相覆盖）
# ══════════════════════════════════════════════════════

LOG_ENABLED = os.environ.get("HITL_STREAM_TEST_LOG_ENABLED", "1") != "0"
LOG_DIR = os.environ.get(
    "HITL_STREAM_TEST_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_stream"),
)

_RUN_STARTED_AT = datetime.now(timezone.utc)
_RUN_ID = _RUN_STARTED_AT.strftime("%Y%m%d_%H%M%S")

_CURRENT_SCENARIO = {"name": None}
_CALL_LOG: list[dict] = []
_CHECK_LOG: list[dict] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _enter_scenario(name: str) -> None:
    _CURRENT_SCENARIO["name"] = name


def _record_call(method: str, url: str, request_body, status_code, response_body, elapsed_ms: float) -> None:
    _CALL_LOG.append({
        "seq":          len(_CALL_LOG) + 1,
        "scenario":     _CURRENT_SCENARIO["name"],
        "timestamp":    _now_iso(),
        "method":       method,
        "url":          url,
        "request":      request_body,
        "status_code":  status_code,
        "response":     response_body,
        "elapsed_ms":   round(elapsed_ms, 1),
    })


def _record_check(scenario: str, desc: str, passed: bool, extra: str) -> None:
    _CHECK_LOG.append({
        "seq":       len(_CHECK_LOG) + 1,
        "scenario":  scenario,
        "timestamp": _now_iso(),
        "desc":      desc,
        "passed":    passed,
        "extra":     extra if not passed else "",
    })


def build_run_log(exit_code: int) -> dict:
    finished_at = datetime.now(timezone.utc)

    scenarios: dict[str, dict] = {}
    for c in _CHECK_LOG:
        bucket = scenarios.setdefault(c["scenario"], {"checks": [], "calls": []})
        bucket["checks"].append(c)
    for c in _CALL_LOG:
        bucket = scenarios.setdefault(c["scenario"], {"checks": [], "calls": []})
        bucket["calls"].append(c)
    for name, bucket in scenarios.items():
        bucket["passed"] = sum(1 for x in bucket["checks"] if x["passed"])
        bucket["failed"] = sum(1 for x in bucket["checks"] if not x["passed"])

    total = len(_CHECK_LOG)
    passed = len(PASS)
    failed = len(FAIL)

    return {
        "meta": {
            "run_id":            _RUN_ID,
            "api_base":          API_BASE,
            "user_id":           USER_ID,
            "mode":              "stream (SSE)",
            "started_at":        _RUN_STARTED_AT.isoformat(timespec="milliseconds"),
            "finished_at":       finished_at.isoformat(timespec="milliseconds"),
            "duration_seconds":  round((finished_at - _RUN_STARTED_AT).total_seconds(), 3),
            "python_version":    sys.version.split()[0],
            "exit_code":         exit_code,
        },
        "summary": {
            "total_checks":  total,
            "passed":        passed,
            "failed":        failed,
            "pass_rate":     f"{(passed / total * 100):.1f}%" if total else "N/A",
            "scenarios_run": sorted(s for s in scenarios if s is not None),
            "failed_checks": list(FAIL),
        },
        "scenarios":  scenarios,
        "calls":      _CALL_LOG,
        "checks":     _CHECK_LOG,
    }


def save_run_log(log: dict) -> str | None:
    if not LOG_ENABLED:
        return None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"hitl_stream_test_log_{_RUN_ID}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        print(f"  ⚠️ 日志写入失败（不影响测试结果）：{e}")
        return None


# ══════════════════════════════════════════════════════
# SSE 解析：把 requests 的流式响应体解析成结构化事件列表
# ══════════════════════════════════════════════════════

def parse_sse(response: requests.Response) -> list[dict]:
    """
    把 SSE 响应体解析成 [{"event": str, "data": str}, ...]。

    SSE 帧格式（服务端 api.py 的写法）：
        data: <token>\n\n                          → event 缺省为 "message"
        event: interrupted\ndata: {...}\n\n         → 显式 event 类型

    按空行分帧；同一帧内可能有多行 "data:"（本项目目前没用到，但按 SSE
    规范支持，用 "\n".join 拼起来更稳妥）。
    """
    events: list[dict] = []
    event_type = "message"
    data_lines: list[str] = []

    def _flush():
        nonlocal event_type, data_lines
        if data_lines:
            events.append({"event": event_type, "data": "\n".join(data_lines)})
        event_type = "message"
        data_lines = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line
        if line == "":
            _flush()
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # 其它行（比如 SSE 的 "id:"/"retry:" 或注释行 ":..."）本项目不产生，忽略即可

    _flush()  # 服务端末尾没有额外空行时，兜底把最后一帧收进来
    return events


def summarize_stream(events: list[dict]) -> dict:
    """
    把原始事件列表归纳成测试断言常用的几个字段，对齐 api.py 文档里
    描述的事件格式：
      - data: <token>                       → 拼进 answer_tokens
      - data: [DONE:<tid>]                  → final_marker="DONE"
      - data: [WAITING_HUMAN:<tid>]         → final_marker="WAITING_HUMAN"
      - data: [ERROR] ...                   → final_marker="ERROR"
      - event: interrupted + data: {...}    → interrupted=解析后的dict
      - event: rejected    + data: {...}    → rejected=解析后的dict
    """
    tokens: list[str] = []
    final_marker = None
    final_thread_id = None
    error_message = None
    interrupted_payload = None
    rejected_payload = None

    for ev in events:
        etype, data = ev["event"], ev["data"]

        if etype == "interrupted":
            try:
                interrupted_payload = json.loads(data)
            except Exception:
                interrupted_payload = {"_raw": data}
            continue

        if etype == "rejected":
            try:
                rejected_payload = json.loads(data)
            except Exception:
                rejected_payload = {"_raw": data}
            continue

        # etype == "message"（普通 data: 行）
        if data.startswith("[DONE:") and data.endswith("]"):
            final_marker = "DONE"
            final_thread_id = data[len("[DONE:"):-1]
        elif data.startswith("[WAITING_HUMAN:") and data.endswith("]"):
            final_marker = "WAITING_HUMAN"
            final_thread_id = data[len("[WAITING_HUMAN:"):-1]
        elif data.startswith("[ERROR]"):
            final_marker = "ERROR"
            error_message = data[len("[ERROR]"):].strip()
        else:
            tokens.append(data)

    return {
        "events":            events,
        "answer_tokens":     "".join(tokens),
        "final_marker":      final_marker,
        "final_thread_id":   final_thread_id,
        "error_message":     error_message,
        "interrupted":       interrupted_payload,
        "rejected":          rejected_payload,
        "is_awaiting_human": final_marker == "WAITING_HUMAN" or interrupted_payload is not None,
    }


# ══════════════════════════════════════════════════════
# HTTP 封装（流式三件套 + 复用非流式 /state 做结果校验）
# ══════════════════════════════════════════════════════

def _do_stream_request(method: str, url: str, *, params=None, json_body=None) -> dict:
    t0 = time.perf_counter()
    if method == "GET":
        res = requests.get(url, params=params, stream=True, timeout=_REQUEST_TIMEOUT)
    else:
        res = requests.post(url, json=json_body, stream=True, timeout=_REQUEST_TIMEOUT)

    status_code = res.status_code
    try:
        events = parse_sse(res)
    finally:
        res.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    summary = summarize_stream(events)
    logged_request = params if method == "GET" else json_body
    _record_call(method, url, logged_request, status_code, summary, elapsed_ms)

    summary["status_code"] = status_code
    return summary


def chat_stream(question: str, thread_id: str) -> dict:
    """GET /chat/stream —— SSE 流式接口。"""
    url = f"{API_BASE}/chat/stream"
    params = {"question": question, "user_id": USER_ID, "thread_id": thread_id}
    return _do_stream_request("GET", url, params=params)


def resume_stream(thread_id: str, decisions: list[dict]) -> dict:
    """POST /session/{user_id}/{thread_id}/resume/stream —— SSE 流式接口。"""
    url = f"{API_BASE}/session/{USER_ID}/{thread_id}/resume/stream"
    return _do_stream_request("POST", url, json_body={"decisions": decisions})


def abort_stream(thread_id: str) -> dict:
    """POST /session/{user_id}/{thread_id}/abort/stream —— SSE 流式接口。"""
    url = f"{API_BASE}/session/{USER_ID}/{thread_id}/abort/stream"
    return _do_stream_request("POST", url, json_body=None)


def get_state(thread_id: str) -> dict:
    """GET /session/{user_id}/{thread_id}/state —— 非流式，用于事后核对 task_plan 真实状态。"""
    url = f"{API_BASE}/session/{USER_ID}/{thread_id}/state"
    t0 = time.perf_counter()
    res = requests.get(url, timeout=_REQUEST_TIMEOUT)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    body = _safe_json(res)
    _record_call("GET", url, None, res.status_code, body, elapsed_ms)
    return {"status_code": res.status_code, "body": body}


def _safe_json(res: requests.Response):
    try:
        return res.json()
    except Exception:
        return {"_raw_text": res.text}


def new_thread() -> str:
    return f"hitlstream_{uuid.uuid4().hex[:10]}"


def task_status(state_body: dict, task_id: int) -> str | None:
    for t in state_body.get("task_plan", []):
        if t.get("task_id") == task_id:
            return t.get("status")
    return None


def check(scenario: str, desc: str, condition: bool, extra: str = "") -> None:
    label = f"[{scenario}] {desc}"
    _record_check(scenario, desc, bool(condition), extra)
    if condition:
        print(f"  \u2705 {label}")
        PASS.append(label)
    else:
        print(f"  \u274c {label}  {extra}")
        FAIL.append(f"{label}  {extra}")


# ══════════════════════════════════════════════════════
# 场景1：自动重试对用户透明（流式）
# ══════════════════════════════════════════════════════
def test_scenario1_autoretry_stream():
    s = "场景1-自动重试(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    r = chat_stream("/hitl_test scenario1_autoretry", tid)
    check(s, "HTTP 200", r["status_code"] == 200, str(r))
    check(s, "未被中断（没有 interrupted 事件）", r["interrupted"] is None, str(r))
    check(s, "收尾标记为 [DONE:...]", r["final_marker"] == "DONE", str(r))
    check(s, "[DONE] 携带的 thread_id 正确", r["final_thread_id"] == tid, str(r))

    st = get_state(tid)["body"]
    check(s, "plan_status=completed", st.get("plan_status") == "completed", str(st))
    check(s, "任务#0最终状态为done", task_status(st, 0) == "done", str(st))
    check(s, "state接口 pending_gate_items 为空", len(st.get("pending_gate_items", [])) == 0)


# ══════════════════════════════════════════════════════
# 场景2：重试耗尽转人工 + 多轮 resume/stream
# ══════════════════════════════════════════════════════
def test_scenario2_retry_exhausted_stream():
    s = "场景2-重试耗尽(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    r = chat_stream("/hitl_test scenario2_retry_exhausted", tid)
    check(s, "收到 event: interrupted", r["interrupted"] is not None, str(r))
    check(s, "收尾标记为 [WAITING_HUMAN:...]", r["final_marker"] == "WAITING_HUMAN", str(r))
    if r["interrupted"]:
        check(s, "interrupted.plan_status=waiting_human",
              r["interrupted"].get("plan_status") == "waiting_human", str(r["interrupted"]))
        gate_items = r["interrupted"].get("pending_gate_items", [])
        check(s, "恰好1项待办事项", len(gate_items) == 1, str(gate_items))
        if gate_items:
            check(s, "待办事项reason=needs_human", gate_items[0].get("reason") == "needs_human")

    # 第一次 resume/stream：选择 retry，但测试任务永远失败 → 应该又产生新一批待办事项
    r2 = resume_stream(tid, [{"task_id": 0, "action": "retry", "patch": None}])
    check(s, "resume/stream后仍是[WAITING_HUMAN:...]（测试任务永远失败）",
          r2["final_marker"] == "WAITING_HUMAN", str(r2))
    check(s, "resume/stream后仍收到interrupted事件且有新一批待办事项",
          r2["interrupted"] is not None and len(r2["interrupted"].get("pending_gate_items", [])) == 1,
          str(r2))

    # 第二次 resume/stream：选择 skip，应该正常完成
    r3 = resume_stream(tid, [{"task_id": 0, "action": "skip",
                               "patch": {"manual_result": "人工确认已知故障，跳过"}}])
    check(s, "skip后收到[DONE:...]", r3["final_marker"] == "DONE", str(r3))
    check(s, "skip后没有新的interrupted事件", r3["interrupted"] is None, str(r3))

    st = get_state(tid)["body"]
    check(s, "skip后plan_status=completed", st.get("plan_status") == "completed", str(st))
    check(s, "任务#0最终状态为skipped", task_status(st, 0) == "skipped", str(st))


# ══════════════════════════════════════════════════════
# 场景3：级联阻塞 + 独立任务不受影响 + 修正后解除阻塞（流式）
# ══════════════════════════════════════════════════════
def test_scenario3_cascade_stream():
    s = "场景3-级联阻塞(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat_stream("/hitl_test scenario3_cascade", tid)
    st = get_state(tid)["body"]

    check(s, "任务#0(A)状态为needs_human", task_status(st, 0) == "needs_human", str(st))
    check(s, "任务#1(B,依赖A)状态为blocked", task_status(st, 1) == "blocked", str(st))
    check(s, "任务#2(C,独立)状态为done（不受A失败影响）", task_status(st, 2) == "done", str(st))

    # 人工修正A（patch里带FIXED关键字），走 resume/stream
    r = resume_stream(tid, [{"task_id": 0, "action": "edit_and_retry",
                              "patch": {"description": "已确认参数问题已修正 FIXED"}}])
    check(s, "修正后收到[DONE:...]", r["final_marker"] == "DONE", str(r))

    st2 = get_state(tid)["body"]
    check(s, "修正后plan_status=completed", st2.get("plan_status") == "completed", str(st2))
    check(s, "任务#0(A)修正后为done", task_status(st2, 0) == "done", str(st2))
    check(s, "任务#1(B)级联解除阻塞后自动执行为done", task_status(st2, 1) == "done", str(st2))
    check(s, "任务#2(C)保持done", task_status(st2, 2) == "done")


# ══════════════════════════════════════════════════════
# 场景4：高风险操作预批准（流式）
# ══════════════════════════════════════════════════════
def test_scenario4_high_risk_stream():
    s = "场景4-高风险审批(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat_stream("/hitl_test scenario4_high_risk", tid)
    st = get_state(tid)["body"]

    check(s, "任务#0(A,高风险)状态为pending_approval", task_status(st, 0) == "pending_approval", str(st))
    check(s, "任务#1(B,同层普通任务)已完成（不等待A审批）", task_status(st, 1) == "done", str(st))

    gate_items = st.get("pending_gate_items", [])
    check(s, "待办事项reason=pending_approval",
          bool(gate_items) and gate_items[0].get("reason") == "pending_approval")

    r = resume_stream(tid, [{"task_id": 0, "action": "approve", "patch": None}])
    check(s, "批准后收到[DONE:...]", r["final_marker"] == "DONE", str(r))
    st_final = get_state(tid)["body"]
    check(s, "批准后plan_status=completed", st_final.get("plan_status") == "completed", str(st_final))

    # 拒绝分支：另开一个会话验证 reject
    tid2 = new_thread()
    chat_stream("/hitl_test scenario4_high_risk", tid2)
    r2 = resume_stream(tid2, [{"task_id": 0, "action": "reject", "patch": None}])
    check(s, "拒绝分支：resume/stream后收到[DONE:...]", r2["final_marker"] == "DONE", str(r2))
    st2 = get_state(tid2)["body"]
    check(s, "拒绝分支：任务#0状态为skipped", task_status(st2, 0) == "skipped", str(st2))


# ══════════════════════════════════════════════════════
# 场景5：终止整个计划（两种流式方式）
# ══════════════════════════════════════════════════════
def test_scenario5_abort_stream():
    s = "场景5-终止计划(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")

    # 方式A：通过 resume/stream 提交 abort_all
    tid = new_thread()
    chat_stream("/hitl_test scenario5_abort", tid)
    r = resume_stream(tid, [{"task_id": 0, "action": "abort_all", "patch": None}])
    check(s, "[方式A] resume/stream(abort_all)后收到[DONE:...]", r["final_marker"] == "DONE", str(r))
    check(s, "[方式A] 没有产生新的interrupted事件（abort_all直接收尾）", r["interrupted"] is None, str(r))
    st = get_state(tid)["body"]
    check(s, "[方式A] plan_status=aborted", st.get("plan_status") == "aborted", str(st))
    check(s, "[方式A] 任务#1(B,依赖A)未被强行执行（非done）", task_status(st, 1) != "done", str(st))

    # 方式B：通过 /abort/stream 快捷接口
    tid2 = new_thread()
    chat_stream("/hitl_test scenario5_abort", tid2)
    r2 = abort_stream(tid2)
    check(s, "[方式B] /abort/stream后收到[DONE:...]", r2["final_marker"] == "DONE", str(r2))
    check(s, "[方式B] /abort/stream没有interrupted事件", r2["interrupted"] is None, str(r2))
    st2 = get_state(tid2)["body"]
    check(s, "[方式B] plan_status=aborted", st2.get("plan_status") == "aborted", str(st2))


# ══════════════════════════════════════════════════════
# 场景6：一批里混合多种待办事项（流式）
# ══════════════════════════════════════════════════════
def test_scenario6_batch_mixed_stream():
    s = "场景6-混合批量(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat_stream("/hitl_test scenario6_batch_mixed", tid)
    st = get_state(tid)["body"]

    gate_items = st.get("pending_gate_items", [])
    check(s, "恰好3项待办事项(A/B/C)", len(gate_items) == 3, str(gate_items))
    check(s, "任务#3(D,普通任务)已完成", task_status(st, 3) == "done", str(st))

    reasons = sorted(item.get("reason") for item in gate_items)
    check(s, "reason种类包含needs_human和pending_approval",
          "needs_human" in reasons and "pending_approval" in reasons, str(reasons))

    # 一次性通过 resume/stream 提交全部3项决策
    r = resume_stream(tid, [
        {"task_id": 0, "action": "skip", "patch": None},
        {"task_id": 1, "action": "skip", "patch": None},
        {"task_id": 2, "action": "approve", "patch": None},
    ])
    check(s, "全部提交后收到[DONE:...]", r["final_marker"] == "DONE", str(r))

    st2 = get_state(tid)["body"]
    check(s, "全部提交后plan_status=completed", st2.get("plan_status") == "completed", str(st2))


# ══════════════════════════════════════════════════════
# 场景8：并发保护（流式版）—— 会话冻结时新消息应被 event:rejected 拒绝
#
# ★ 注意跟非流式版的关键差异：/chat 是普通 JSON 接口，冻结时用 HTTP 409
#   拒绝；/chat/stream 是 SSE 接口，HTTP 状态码从连接建立那一刻就已经
#   是 200 了（后续才开始吐 event: rejected），所以这里不能再断言
#   status_code == 409，而是要断言 SSE 流里出现了 event: rejected。
# ══════════════════════════════════════════════════════
def test_scenario8_concurrent_rejection_stream():
    s = "场景8-并发保护(流式)"
    _enter_scenario(s)
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat_stream("/hitl_test scenario2_retry_exhausted", tid)  # 让会话卡在 waiting_human，不处理它

    r = chat_stream("你好，这是一条不相关的新消息（流式）", tid)  # 同一个 thread_id 上再发消息
    check(s, "HTTP状态码仍为200（SSE连接本身建立成功）", r["status_code"] == 200, str(r))
    check(s, "被冻结会话上发新消息应收到 event: rejected", r["rejected"] is not None, str(r))
    check(s, "rejected事件后直接结束，没有[DONE]/[WAITING_HUMAN]收尾标记",
          r["final_marker"] is None, str(r))
    if r["rejected"]:
        check(s, "rejected.plan_status=waiting_human",
              r["rejected"].get("plan_status") == "waiting_human", str(r["rejected"]))

    # 确认原来的待办事项没有被这次"误操作"污染/清空
    st = get_state(tid)["body"]
    check(s, "原有待办事项仍然是1项（未被新消息覆盖）", len(st.get("pending_gate_items", [])) == 1, str(st))
    check(s, "plan_status仍是waiting_human", st.get("plan_status") == "waiting_human")


# ══════════════════════════════════════════════════════

def main():
    print(f"HITL 流式（SSE）自动化测试开始，目标后端：{API_BASE}")
    try:
        requests.get(f"{API_BASE}/health", timeout=5)
    except Exception as e:
        print(f"❌ 无法连接后端 {API_BASE}：{e}\n请先确认 api.py 已启动。")
        sys.exit(1)

    for fn in (
        test_scenario1_autoretry_stream,
        test_scenario2_retry_exhausted_stream,
        test_scenario3_cascade_stream,
        test_scenario4_high_risk_stream,
        test_scenario5_abort_stream,
        test_scenario6_batch_mixed_stream,
        test_scenario8_concurrent_rejection_stream,
    ):
        try:
            fn()
        except Exception as e:
            _enter_scenario(fn.__name__)
            FAIL.append(f"[{fn.__name__}] 测试执行本身抛出异常：{e}")
            _record_check(fn.__name__, "测试执行本身未抛异常", False, str(e))
            print(f"  💥 [{fn.__name__}] 执行异常：{e}")

    print("\n" + "=" * 60)
    print(f"通过：{len(PASS)}    失败：{len(FAIL)}")
    exit_code = 0
    if FAIL:
        print("\n失败详情：")
        for f in FAIL:
            print(f"  - {f}")
        exit_code = 1
    else:
        print("🎉 全部通过")

    log = build_run_log(exit_code)
    log_path = save_run_log(log)
    if log_path:
        print(f"\n📄 详细日志已保存：{log_path}")
        print(f"   （本次运行共记录 {len(_CALL_LOG)} 次 HTTP 调用、{len(_CHECK_LOG)} 条断言）")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
