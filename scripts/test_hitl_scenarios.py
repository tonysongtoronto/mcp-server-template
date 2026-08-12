#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_hitl_scenarios.py

HITL（人工介入断点续传）功能自动化回归测试脚本
────────────────────────────────────────────────────────
直接调用 api.py 的真实 HTTP 接口，端到端跑一遍 HITL 断点续传机制的
场景1-6 + 场景8（并发保护），每一步都带断言，跑完打印汇总报告。

依赖的服务端配合（见 src/hitl_test_scenarios.py + src/langgraph_parallel_agent.py
里三处极小的旁路钩子）：
  发一句 "/hitl_test <scenario_name>" 会跳过真实 Planner LLM 规划和
  final_answer 的真实 LLM 汇总，直接用手工编排好的确定性任务计划跑完整条
  planner → parallel_executor → human_review_gate → final_answer 状态机。
  这样本脚本可以在没有可用 LLM API Key / 无法访问外部 LLM 服务的环境
  （比如 CI、隔离沙盒）里，依然对"调度逻辑本身"做完整的端到端验证。

依赖：pip install requests

用法：
    python scripts/test_hitl_scenarios.py
    API_BASE=http://localhost:8000 python scripts/test_hitl_scenarios.py

前置条件：
    - api.py 已经在跑（默认 http://localhost:8000，见 /health）
    - src/hitl_test_scenarios.py 存在（HITL 测试专用模块，独立文件）
    - langgraph_parallel_agent.py 里的三处 "★ HITL 测试专用钩子" 还在
      （没有被删掉），"/hitl_test <scenario>" 才能正常触发测试场景
────────────────────────────────────────────────────────
"""

import os
import sys
import uuid
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
USER_ID  = "hitl_test_user"

PASS, FAIL = [], []


# ── HTTP 封装 ────────────────────────────────────────────

def chat(question: str, thread_id: str) -> dict:
    """POST /chat —— 非流式 JSON 接口，直接拿完整 ChatResponse。"""
    res = requests.post(f"{API_BASE}/chat", json={
        "question": question, "user_id": USER_ID, "thread_id": thread_id,
    })
    return {"status_code": res.status_code, "body": _safe_json(res)}


def get_state(thread_id: str) -> dict:
    res = requests.get(f"{API_BASE}/session/{USER_ID}/{thread_id}/state")
    return {"status_code": res.status_code, "body": _safe_json(res)}


def resume(thread_id: str, decisions: list[dict]) -> dict:
    res = requests.post(
        f"{API_BASE}/session/{USER_ID}/{thread_id}/resume",
        json={"decisions": decisions},
    )
    return {"status_code": res.status_code, "body": _safe_json(res)}


def abort(thread_id: str) -> dict:
    res = requests.post(f"{API_BASE}/session/{USER_ID}/{thread_id}/abort")
    return {"status_code": res.status_code, "body": _safe_json(res)}


def _safe_json(res):
    try:
        return res.json()
    except Exception:
        return {"_raw_text": res.text}


def new_thread() -> str:
    return f"hitltest_{uuid.uuid4().hex[:10]}"


def task_status(state_body: dict, task_id: int) -> str | None:
    for t in state_body.get("task_plan", []):
        if t.get("task_id") == task_id:
            return t.get("status")
    return None


# ── 断言辅助（记录结果，不 raise，跑完全部用例再统一汇总）────

def check(scenario: str, desc: str, condition: bool, extra: str = ""):
    label = f"[{scenario}] {desc}"
    if condition:
        print(f"  \u2705 {label}")
        PASS.append(label)
    else:
        print(f"  \u274c {label}  {extra}")
        FAIL.append(f"{label}  {extra}")


# ══════════════════════════════════════════════════════
# 场景1：自动重试对用户透明
# ══════════════════════════════════════════════════════
def test_scenario1_autoretry():
    s = "场景1-自动重试"
    print(f"\n=== {s} ===")
    tid = new_thread()

    r = chat("/hitl_test scenario1_autoretry", tid)
    check(s, "HTTP 200", r["status_code"] == 200, str(r))
    body = r["body"]
    check(s, "未被中断（is_awaiting_human=False）", body.get("is_awaiting_human") is False, str(body))
    check(s, "plan_status=completed", body.get("plan_status") == "completed", str(body))
    check(s, "回答里包含'自动重试后成功执行'", "自动重试后成功执行" in body.get("answer", ""), body.get("answer"))

    st = get_state(tid)["body"]
    check(s, "任务#0最终状态为done", task_status(st, 0) == "done", str(st))
    check(s, "state接口 pending_gate_items 为空", len(st.get("pending_gate_items", [])) == 0)


# ══════════════════════════════════════════════════════
# 场景2：重试耗尽转人工 + 多轮 resume
# ══════════════════════════════════════════════════════
def test_scenario2_retry_exhausted():
    s = "场景2-重试耗尽"
    print(f"\n=== {s} ===")
    tid = new_thread()

    r = chat("/hitl_test scenario2_retry_exhausted", tid)
    body = r["body"]
    check(s, "被中断（is_awaiting_human=True）", body.get("is_awaiting_human") is True, str(body))
    check(s, "plan_status=waiting_human", body.get("plan_status") == "waiting_human")
    gate_items = body.get("pending_gate_items", [])
    check(s, "恰好1项待办事项", len(gate_items) == 1, str(gate_items))
    if gate_items:
        check(s, "待办事项reason=needs_human", gate_items[0].get("reason") == "needs_human")

    # 第一次 resume：选择 retry，但测试任务永远失败 → 应该又产生新一批待办事项
    r2 = resume(tid, [{"task_id": 0, "action": "retry", "patch": None}])
    body2 = r2["body"]
    check(s, "resume后仍是waiting_human（因为测试任务永远失败）", body2.get("is_awaiting_human") is True, str(body2))
    check(s, "resume后仍有新一批待办事项", len(body2.get("pending_gate_items", [])) == 1)

    # 第二次 resume：选择 skip，应该正常完成
    r3 = resume(tid, [{"task_id": 0, "action": "skip", "patch": {"manual_result": "人工确认已知故障，跳过"}}])
    body3 = r3["body"]
    check(s, "skip后plan_status=completed", body3.get("plan_status") == "completed", str(body3))
    check(s, "skip后不再等待人工", body3.get("is_awaiting_human") is False)

    st = get_state(tid)["body"]
    check(s, "任务#0最终状态为skipped", task_status(st, 0) == "skipped", str(st))


# ══════════════════════════════════════════════════════
# 场景3：级联阻塞 + 独立任务不受影响 + 修正后解除阻塞
# ══════════════════════════════════════════════════════
def test_scenario3_cascade():
    s = "场景3-级联阻塞"
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat("/hitl_test scenario3_cascade", tid)
    st = get_state(tid)["body"]

    check(s, "任务#0(A)状态为needs_human", task_status(st, 0) == "needs_human", str(st))
    check(s, "任务#1(B,依赖A)状态为blocked", task_status(st, 1) == "blocked", str(st))
    check(s, "任务#2(C,独立)状态为done（不受A失败影响）", task_status(st, 2) == "done", str(st))

    # 人工修正A（patch里带FIXED关键字）
    r = resume(tid, [{"task_id": 0, "action": "edit_and_retry",
                       "patch": {"description": "已确认参数问题已修正 FIXED"}}])
    body = r["body"]
    check(s, "修正后plan_status=completed", body.get("plan_status") == "completed", str(body))

    st2 = get_state(tid)["body"]
    check(s, "任务#0(A)修正后为done", task_status(st2, 0) == "done", str(st2))
    check(s, "任务#1(B)级联解除阻塞后自动执行为done", task_status(st2, 1) == "done", str(st2))
    check(s, "任务#2(C)保持done", task_status(st2, 2) == "done")


# ══════════════════════════════════════════════════════
# 场景4：高风险操作预批准
# ══════════════════════════════════════════════════════
def test_scenario4_high_risk():
    s = "场景4-高风险审批"
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat("/hitl_test scenario4_high_risk", tid)
    st = get_state(tid)["body"]

    check(s, "任务#0(A,高风险)状态为pending_approval", task_status(st, 0) == "pending_approval", str(st))
    check(s, "任务#1(B,同层普通任务)已完成（不等待A审批）", task_status(st, 1) == "done", str(st))

    gate_items = st.get("pending_gate_items", [])
    check(s, "待办事项reason=pending_approval", bool(gate_items) and gate_items[0].get("reason") == "pending_approval")

    r = resume(tid, [{"task_id": 0, "action": "approve", "patch": None}])
    check(s, "批准后plan_status=completed", r["body"].get("plan_status") == "completed", str(r["body"]))

    # 拒绝分支：另开一个会话验证 reject
    tid2 = new_thread()
    chat("/hitl_test scenario4_high_risk", tid2)
    r2 = resume(tid2, [{"task_id": 0, "action": "reject", "patch": None}])
    st2 = get_state(tid2)["body"]
    check(s, "拒绝分支：任务#0状态为skipped", task_status(st2, 0) == "skipped", str(st2))


# ══════════════════════════════════════════════════════
# 场景5：终止整个计划（两种方式）
# ══════════════════════════════════════════════════════
def test_scenario5_abort():
    s = "场景5-终止计划"
    print(f"\n=== {s} ===")

    # 方式A：通过 resume 提交 abort_all
    tid = new_thread()
    chat("/hitl_test scenario5_abort", tid)
    r = resume(tid, [{"task_id": 0, "action": "abort_all", "patch": None}])
    check(s, "[方式A] resume(abort_all)后plan_status=aborted", r["body"].get("plan_status") == "aborted", str(r["body"]))
    st = get_state(tid)["body"]
    check(s, "[方式A] 任务#1(B,依赖A)未被强行执行（非done）", task_status(st, 1) != "done", str(st))

    # 方式B：通过 /abort 快捷接口
    tid2 = new_thread()
    chat("/hitl_test scenario5_abort", tid2)
    r2 = abort(tid2)
    check(s, "[方式B] /abort 后plan_status=aborted", r2["body"].get("plan_status") == "aborted", str(r2["body"]))
    check(s, "[方式B] /abort 后不再等待人工", r2["body"].get("is_awaiting_human") is False)


# ══════════════════════════════════════════════════════
# 场景6：一批里混合多种待办事项
# ══════════════════════════════════════════════════════
def test_scenario6_batch_mixed():
    s = "场景6-混合批量"
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat("/hitl_test scenario6_batch_mixed", tid)
    st = get_state(tid)["body"]

    gate_items = st.get("pending_gate_items", [])
    check(s, "恰好3项待办事项(A/B/C)", len(gate_items) == 3, str(gate_items))
    check(s, "任务#3(D,普通任务)已完成", task_status(st, 3) == "done", str(st))

    reasons = sorted(item.get("reason") for item in gate_items)
    check(s, "reason种类包含needs_human和pending_approval",
          "needs_human" in reasons and "pending_approval" in reasons, str(reasons))

    # 一次性提交全部3项决策
    r = resume(tid, [
        {"task_id": 0, "action": "skip", "patch": None},
        {"task_id": 1, "action": "skip", "patch": None},
        {"task_id": 2, "action": "approve", "patch": None},
    ])
    check(s, "全部提交后plan_status=completed", r["body"].get("plan_status") == "completed", str(r["body"]))


# ══════════════════════════════════════════════════════
# 场景8：并发保护 —— 会话冻结时新消息应被 409 拒绝
# ══════════════════════════════════════════════════════
def test_scenario8_concurrent_rejection():
    s = "场景8-并发保护"
    print(f"\n=== {s} ===")
    tid = new_thread()

    chat("/hitl_test scenario2_retry_exhausted", tid)   # 让会话卡在 waiting_human，不处理它

    r = chat("你好，这是一条不相关的新消息", tid)   # 同一个 thread_id 上再发消息
    check(s, "被冻结会话上发新消息应返回 409", r["status_code"] == 409, str(r))

    # 确认原来的待办事项没有被这次"误操作"污染/清空
    st = get_state(tid)["body"]
    check(s, "原有待办事项仍然是1项（未被新消息覆盖）", len(st.get("pending_gate_items", [])) == 1, str(st))
    check(s, "plan_status仍是waiting_human", st.get("plan_status") == "waiting_human")


# ══════════════════════════════════════════════════════

def main():
    print(f"HITL 自动化测试开始，目标后端：{API_BASE}")
    try:
        requests.get(f"{API_BASE}/health", timeout=5)
    except Exception as e:
        print(f"❌ 无法连接后端 {API_BASE}：{e}\n请先确认 api.py 已启动。")
        sys.exit(1)

    for fn in (
        test_scenario1_autoretry,
        test_scenario2_retry_exhausted,
        test_scenario3_cascade,
        test_scenario4_high_risk,
        test_scenario5_abort,
        test_scenario6_batch_mixed,
        test_scenario8_concurrent_rejection,
    ):
        try:
            fn()
        except Exception as e:
            FAIL.append(f"[{fn.__name__}] 测试执行本身抛出异常：{e}")
            print(f"  💥 [{fn.__name__}] 执行异常：{e}")

    print("\n" + "=" * 60)
    print(f"通过：{len(PASS)}    失败：{len(FAIL)}")
    if FAIL:
        print("\n失败详情：")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("🎉 全部通过")
        sys.exit(0)


if __name__ == "__main__":
    main()


# uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

# 另开一个终端，跑 HITL 测试

# uv run python scripts/test_hitl_scenarios.py