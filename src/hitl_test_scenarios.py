"""
src/hitl_test_scenarios.py

★ HITL 测试专用模块（独立文件，与 langgraph_parallel_agent.py 解耦）
────────────────────────────────────────────────────────────────────
职责：
  给 HITL_测试方案.md 里描述的 8 个场景（本文件实现其中 6 个可自动化的：
  scenario1/2/3/4/5/6，scenario8 是在 scenario2 基础上叠加并发请求，
  复用 scenario2 即可，不需要单独定义）提供：

    1. build_scenario_task_plan(name) -> list[Task]
       —— 跳过真实 Planner LLM 调用，直接产出一份"手工编排"的 task_plan，
          任务的 agent 字段固定为 "test_agent"（除个别 direct 占位任务外）。

    2. run_test_agent(task, state) -> str
       —— 跳过真实 MCP 工具调用 / LLM 调用，根据任务上携带的 "_test_kind"
          字段（写在 Task 字典里的自定义键，TypedDict 允许运行时附加，
          不影响其余逻辑）确定性地返回成功文本，或抛出异常模拟失败
          （异常类型/文案经过挑选，能触发 langgraph_parallel_agent.py 里
          _classify_failure() 分类为 "retryable" 或 "permanent"）。

    3. build_final_answer(task_plan, plan_status) -> str
       —— 跳过 final_answer_node 真正调用 LLM 做自然语言汇总的那一步，
          直接用任务结果拼一份可读文本，避免测试跑在没有 LLM API Key /
          没有出网权限的环境（CI、沙盒）时被"最后一步生成总结"卡住。

  这样设计的好处：
    - 整条 planner → parallel_executor → human_review_gate → final_answer
      的状态机（拓扑分层、层内自动重试、级联阻塞、pending_approval 预批准、
      interrupt()/Command(resume=...) 断点续传、abort_all 等）全部走真实代码路径，
      被测的是"图的调度逻辑"本身，不是某个 mock。
    - 不消耗真实 LLM token，不要求网络能访问 LLM/MCP 服务，跑得快、可重复，
      适合放进 CI 做回归测试。

  与 langgraph_parallel_agent.py 的耦合点（★ 全部是极小的旁路判断，
  不修改任何已有的调度 / 状态机逻辑）：
    - planner_node：当用户消息以 "/hitl_test " 开头时，调用
      build_scenario_task_plan() 替代真实 LLM 规划。
    - parallel_executor_node._exec_one：当 task["agent"] == "test_agent" 时，
      调用 run_test_agent() 替代 run_agent_isolated()（真实 MCP 工具调用）。
    - final_answer_node：当当前这一轮的用户消息以 "/hitl_test " 开头时，
      调用 build_final_answer() 替代真实 LLM 生成的自然语言总结。
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════
# 0. 触发前缀 & 场景名解析
# ══════════════════════════════════════════════════════

COMMAND_PREFIX = "/hitl_test"
TEST_AGENT_NAME = "test_agent"


def is_test_command(text: str) -> bool:
    """判断这条用户消息是不是本模块要接管的 HITL 测试指令。"""
    return (text or "").strip().startswith(COMMAND_PREFIX)


def parse_scenario_name(text: str) -> str:
    """从 "/hitl_test scenario1_autoretry" 中提取 "scenario1_autoretry"。"""
    parts = (text or "").strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


# ══════════════════════════════════════════════════════
# 1. 失败模拟：错误文案要能命中/避开
#    langgraph_parallel_agent._classify_failure() 的可重试关键词表
# ══════════════════════════════════════════════════════
#
# _classify_failure() 源码摘要（见 langgraph_parallel_agent.py）：
#   isinstance(exc, (TimeoutError, ConnectionError, ...)) → "retryable"
#   否则看 str(exc).lower() 是否命中
#     ("timeout","connection","temporarily","rate limit","429",...)
#   都不命中 → "permanent"
#
# 所以：
#   - 想让任务"可自动重试" → 抛 TimeoutError 或消息里带 "timeout"
#   - 想让任务"不可重试，直接转人工" → 抛普通 ValueError，消息里避开这些关键词


class _RetryableTestError(TimeoutError):
    """模拟网络抖动等可自动重试的瞬时故障。"""


class _PermanentTestError(ValueError):
    """模拟参数错误等不可自动重试的永久故障。"""


# ══════════════════════════════════════════════════════
# 2. Task 构造辅助
# ══════════════════════════════════════════════════════

def _task(
    task_id: int,
    description: str,
    kind: str,
    *,
    depends_on: list[int] | None = None,
    high_risk: bool = False,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    构造一个符合 langgraph_parallel_agent.Task 结构的字典。

    额外携带的 "_test_kind" 字段不属于 Task 的正式字段，但 Python 的
    TypedDict 只在静态类型检查时约束字段集合，运行时就是普通 dict，
    可以安全地附加自定义键——run_test_agent() 靠这个键决定行为，
    其余所有调度代码完全不关心这个键的存在。
    """
    return {
        "task_id": task_id,
        "description": description,
        "agent": TEST_AGENT_NAME,
        "inputs": {},
        "depends_on": list(depends_on or []),
        "status": "pending",
        "result": "",
        "_resolved_description": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "last_error": "",
        "high_risk": high_risk,
        "_test_kind": kind,
    }


# ══════════════════════════════════════════════════════
# 3. 场景定义
# ══════════════════════════════════════════════════════

def _scenario1_autoretry() -> list[dict]:
    """场景1：任务先失败一次（可重试错误），第二次自动重试后成功，用户全程无感知。"""
    return [
        _task(0, "场景1：模拟一个偶发网络抖动、重试一次就能成功的任务",
              kind="fail_once_then_succeed", max_retries=2),
    ]


def _scenario2_retry_exhausted() -> list[dict]:
    """场景2：任务持续失败（可重试类型），重试到上限后转人工，支持多轮 resume。"""
    return [
        _task(0, "场景2：模拟一个持续失败、重试耗尽后需要人工介入的任务",
              kind="always_fail_retryable", max_retries=1),
    ]


def _scenario3_cascade() -> list[dict]:
    """场景3：A 失败阻塞下游 B，独立任务 C 不受影响；人工修正 A（patch 带 FIXED）后 B 自动解除阻塞。"""
    return [
        _task(0, "场景3-任务A：需要人工修正参数才能成功的任务",
              kind="fixable", max_retries=2),
        _task(1, "场景3-任务B：依赖任务A的结果", kind="succeed",
              depends_on=[0], max_retries=2),
        _task(2, "场景3-任务C：与A/B完全独立的任务", kind="succeed",
              max_retries=2),
    ]


def _scenario4_high_risk() -> list[dict]:
    """场景4：高风险任务A预批准挂起，同层普通任务B不等待A，照常执行。"""
    return [
        _task(0, "场景4-任务A：写入类高风险操作，需要人工批准",
              kind="succeed", high_risk=True),
        _task(1, "场景4-任务B：同层的普通只读任务，不受A审批影响",
              kind="succeed"),
    ]


def _scenario5_abort() -> list[dict]:
    """场景5：任务A永久失败转人工，任务B依赖A；人工选择终止整个计划。"""
    return [
        _task(0, "场景5-任务A：一个不可重试、必然失败的任务",
              kind="always_fail_permanent"),
        _task(1, "场景5-任务B：依赖任务A的结果", kind="succeed",
              depends_on=[0]),
    ]


def _scenario6_batch_mixed() -> list[dict]:
    """场景6：一批里混合 needs_human ×2 + pending_approval ×1，另有一个不受影响的普通任务D。"""
    return [
        _task(0, "场景6-任务A：不可重试的失败任务(1)", kind="always_fail_permanent"),
        _task(1, "场景6-任务B：不可重试的失败任务(2)", kind="always_fail_permanent"),
        _task(2, "场景6-任务C：高风险任务，需要审批", kind="succeed", high_risk=True),
        _task(3, "场景6-任务D：普通任务，不受A/B/C影响", kind="succeed"),
    ]


# 场景名 → 构造函数。scenario8（并发保护）复用 scenario2，测试脚本层面处理，不在此重复定义。
SCENARIOS: dict[str, Any] = {
    "scenario1_autoretry":        _scenario1_autoretry,
    "scenario2_retry_exhausted":  _scenario2_retry_exhausted,
    "scenario3_cascade":          _scenario3_cascade,
    "scenario4_high_risk":        _scenario4_high_risk,
    "scenario5_abort":            _scenario5_abort,
    "scenario6_batch_mixed":      _scenario6_batch_mixed,
}


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS.keys())


def build_scenario_task_plan(scenario_name: str) -> list[dict]:
    """
    根据场景名产出一份现成的 task_plan（不经过真实 Planner LLM）。
    未知场景名 → 抛 KeyError，调用方（planner_node）负责兜底成一条错误提示任务。
    """
    builder = SCENARIOS.get(scenario_name)
    if builder is None:
        raise KeyError(
            f"未知的 HITL 测试场景：'{scenario_name}'。"
            f"可用场景：{', '.join(list_scenarios())}"
        )
    # 每次都重新构造，避免多个会话共享同一份可变字典导致互相污染。
    return builder()


# ══════════════════════════════════════════════════════
# 4. test_agent 执行体（替代真实 MCP 工具调用 / run_agent_isolated）
# ══════════════════════════════════════════════════════

async def run_test_agent(task: dict, state: dict) -> str:
    """
    根据 task["_test_kind"] 确定性地返回结果或抛出异常。

    重要：不使用任何"模块级计数器"之类的外部可变状态来判断第几次执行，
    而是完全依赖 task 字典自身携带的 retry_count / description /
    _resolved_description——这些字段本身就是 LangGraph checkpoint 会持久化
    的内容，跨进程恢复、多轮 resume 时依然正确，不会因为测试脚本重启
    或多个会话并发而互相干扰。
    """
    kind = task.get("_test_kind", "succeed")
    task_id = task.get("task_id")

    if kind == "succeed":
        return f"[测试任务] task[{task_id}] 执行成功，结果=OK"

    if kind == "fail_once_then_succeed":
        # retry_count == 0 → 这是第一次尝试，模拟失败；
        # 之后（parallel_executor 会自动把 retry_count 加到 1 再重跑）→ 成功。
        if task.get("retry_count", 0) == 0:
            raise _RetryableTestError("模拟网络超时（timeout），预期会被自动重试")
        return "自动重试后成功执行（模拟第2次尝试成功，结果=OK）"

    if kind == "always_fail_retryable":
        raise _RetryableTestError("模拟持续性网络超时（timeout），预期重试耗尽后转人工")

    if kind == "always_fail_permanent":
        raise _PermanentTestError("模拟不可重试的参数错误，预期立即转人工，不做自动重试")

    if kind == "fixable":
        text = f"{task.get('description', '')} {task.get('_resolved_description', '')}"
        if "FIXED" in text:
            return "人工修正参数后执行成功，结果=OK"
        raise _PermanentTestError("模拟参数不合法的永久错误，需人工修正 description 后 edit_and_retry")

    # 未知 kind：保守地当作永久失败处理，方便发现测试场景定义里的笔误。
    raise _PermanentTestError(f"未知的测试任务类型 _test_kind={kind!r}，请检查场景定义")


# ══════════════════════════════════════════════════════
# 5. final_answer 汇总文本（替代真实 LLM 生成的自然语言总结）
# ══════════════════════════════════════════════════════

_STATUS_LABELS = {
    "done":              "✅已完成",
    "skipped":           "⏭️已跳过（人工决定）",
    "blocked":           "⛔未执行（依赖任务未就绪）",
    "needs_human":       "⏸️等待人工处理",
    "pending_approval":  "⏸️等待人工审批（高风险操作）",
    "pending":           "⏳未执行",
    "in_progress":       "⏳执行中",
    "failed":            "❌失败",
}


def build_final_answer(task_plan: list[dict], plan_status: str) -> str:
    """
    用任务结果拼一份可读的中文总结文本，格式与真实 final_answer_node
    的风格保持一致，方便测试脚本用子串匹配的方式做断言
    （比如场景1要求 answer 里出现"自动重试后成功执行"）。
    """
    lines: list[str] = [f"[HITL 测试模式] 本轮任务计划状态：{plan_status}"]
    for t in task_plan:
        label = _STATUS_LABELS.get(t.get("status", ""), f"❓{t.get('status')}")
        lines.append(
            f"  任务[{t.get('task_id')}]（{t.get('description', '')}）"
            f"[{label}]：{t.get('result', '')}"
        )
    if plan_status == "aborted":
        lines.append("用户在人工审核环节主动终止了本次任务计划。")
    return "\n".join(lines)
