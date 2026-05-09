"""
run_dataset_with_evaluate.py — 方法一：批量跑数据集 + 自动评分（aevaluate）

功能：
  - 使用 LangSmith aevaluate() 批量跑数据集
  - 挂载 4 个 evaluator：关键词匹配 / 语义正确性 / 工具调用验证 / 延迟评分
  - 将评估结果保存到 TestDataSetResult/<timestamp>_eval.json
  - 单独文件存放，每次运行生成一个新文件
  - 结果同时在 LangSmith UI → Experiments 里可视化查看

运行方式：
    uv run python TestDataSetScript/run_dataset_with_evaluate.py

结果位置：
    TestDataSetResult/<YYYYMMDD_HHMMSS>_eval.json
"""

import asyncio
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import aevaluate

load_dotenv()

# ── 路径设置 ──────────────────────────────────────────────────────────────────
THIS_DIR    = Path(__file__).parent          # TestDataSetScript/
PROJECT_DIR = THIS_DIR.parent               # 项目根目录
OUTPUT_DIR  = PROJECT_DIR / "TestDataSetResultWithEaluate"
SRC_DIR     = PROJECT_DIR / "src"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(SRC_DIR))

# ══════════════════════════════════════════════════════════════════════════════
# 配置区（按需修改）
# ══════════════════════════════════════════════════════════════════════════════
DATASET_NAME       = "backup"      # ← 手工改成你的数据集名称
EXPERIMENT_PREFIX  = "backup-eval" # ← LangSmith Experiments 里显示的前缀
MAX_CONCURRENCY    = 1             # 并发数，建议先用 1
TIMEOUT_SECONDS    = 120           # 单条超时（秒）
LATENCY_LIMIT_MS   = 60_000        # 超时阈值（毫秒），超过算失败
KEYWORD_THRESHOLD  = 0.6           # 关键词命中率阈值

# ══════════════════════════════════════════════════════════════════════════════
# Judge 模型（语义评估用）
# ══════════════════════════════════════════════════════════════════════════════
judge_llm = ChatOpenAI(
    model    = "deepseek-chat",
    api_key  = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com",
    temperature = 0,
)

# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def extract_question(inputs: dict) -> str:
    """
    兼容两种 input 格式：
      1. {"messages": [{"role": "human", "content": "..."}]}  ← dataset3 格式
      2. {"human": "..."}                                      ← backup 格式
    """
    for msg in inputs.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "human":
            return msg.get("content", "").strip()

    if "human" in inputs:
        return str(inputs["human"]).strip()

    if "content" in inputs:
        return str(inputs["content"]).strip()

    for v in inputs.values():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def extract_reference_answer(outputs: dict) -> str:
    """从 reference output 中提取期望答案文本"""
    if not outputs:
        return ""

    for msg in reversed(outputs.get("messages", [])):
        msg_type = msg.get("type", "") or msg.get("role", "")
        if msg_type in ("ai", "assistant"):
            return msg.get("content", "").strip()

    for key in ("output", "answer", "response", "ai", "assistant"):
        if key in outputs and isinstance(outputs[key], str):
            return outputs[key].strip()

    return ""


def extract_expected_agents(outputs: dict) -> list[str]:
    """从 reference output 的 task_plan 中取期望被调用的 agent 列表"""
    return [
        t.get("agent", "") for t in outputs.get("task_plan", [])
        if t.get("agent")
    ]


def strip_emoji(text: str) -> str:
    return re.sub(r'[^\w\s@.\-|:，。]', '', text, flags=re.UNICODE).strip()


def extract_keywords(reference: str) -> list[str]:
    """从期望答案的 markdown 表格中提取关键词（用户名、年龄、日期等）"""
    keywords = []
    for line in reference.split("\n"):
        line = strip_emoji(line.strip())
        if "|" in line and "@" in line:
            for part in line.split("|"):
                part = strip_emoji(part.strip())
                if not part or part.startswith("-") or "@" in part:
                    continue
                if len(part) < 20:
                    keywords.append(part)
    return keywords


def build_output_filename(prefix: str = "eval") -> Path:
    """生成带时间戳的输出文件名"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_{prefix}.json"


# ══════════════════════════════════════════════════════════════════════════════
# Target：调用 agent，返回回答 + 元数据
# 注意：target 在模块级别定义，agent_module 通过闭包注入
# ══════════════════════════════════════════════════════════════════════════════

# agent_module 由 main() 启动后注入
_agent_module = None


async def target(inputs: dict) -> dict:
    """aevaluate() 的 target 函数，调用 agent 并返回结构化结果"""
    if _agent_module is None:
        return {"output": "", "task_plan": [], "latency_ms": 0,
                "token_in": 0, "token_out": 0}

    question = extract_question(inputs)
    if not question:
        return {"output": "", "task_plan": [], "latency_ms": 0,
                "token_in": 0, "token_out": 0}

    print(f"\n🤖 [{time.strftime('%H:%M:%S')}] 处理: {question[:50]}")

    thread_id = f"eval_{abs(hash(question)) % 100_000}"
    config    = {"configurable": {"thread_id": thread_id}}

    t0    = time.monotonic()
    error = None

    try:
        result = await asyncio.wait_for(
            _agent_module.graph.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            ),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        error  = f"超时（>{TIMEOUT_SECONDS}s）"
        print(f"   ⚠️  {error}")
        return {"output": "", "task_plan": [], "latency_ms": TIMEOUT_SECONDS * 1000,
                "token_in": 0, "token_out": 0, "error": error}
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"   ❌ 错误: {error}")
        traceback.print_exc()
        return {"output": "", "task_plan": [], "latency_ms": 0,
                "token_in": 0, "token_out": 0, "error": error}

    latency_ms = int((time.monotonic() - t0) * 1000)
    task_plan  = result.get("task_plan", [])
    all_msgs   = result.get("messages", [])

    # 优先从 task_plan 拼完整结果
    task_results = [
        t.get("result", "") for t in task_plan
        if t.get("status") == "done" and t.get("result")
    ]
    if task_results:
        output = "\n\n".join(task_results)
        print(f"   📨 取 task_plan result，共 {len(task_results)} 段，总长 {len(output)} 字")
    else:
        ai_msgs = [
            m for m in all_msgs
            if hasattr(m, "content") and m.content
            and type(m).__name__ == "AIMessage"
        ]
        output = max(ai_msgs, key=lambda m: len(str(m.content))).content if ai_msgs else ""
        print(f"   📨 fallback AIMessage，最长 {len(output)} 字")

    token_in  = sum(
        len(str(m.content)) for m in all_msgs
        if hasattr(m, "content") and isinstance(m.content, str)
    ) // 4
    token_out = len(output) // 4

    print(f"   ⏱️  耗时 {latency_ms/1000:.1f}s  |  输出 {len(output)} 字  |  "
          f"任务数 {len(task_plan)}  |  agents: {[t.get('agent') for t in task_plan]}")

    return {
        "output":     output,
        "task_plan":  task_plan,
        "latency_ms": latency_ms,
        "token_in":   token_in,
        "token_out":  token_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator 1：关键词匹配（无需 LLM，速度快）
# ══════════════════════════════════════════════════════════════════════════════

async def keyword_judge(run, example) -> dict:
    reference = extract_reference_answer(example.outputs or {})
    actual    = (run.outputs or {}).get("output", "")

    if not reference:
        return {"key": "keyword_match", "score": 0, "comment": "无期望答案"}
    if not actual:
        return {"key": "keyword_match", "score": 0, "comment": "Agent无输出"}

    keywords = extract_keywords(reference)
    if not keywords:
        score = 1 if len(actual) > 50 else 0
        return {"key": "keyword_match", "score": score, "comment": "无关键词，按长度判断"}

    matched = sum(1 for kw in keywords if kw and kw in actual)
    total   = len(keywords)
    ratio   = matched / total if total > 0 else 0
    score   = 1 if ratio >= KEYWORD_THRESHOLD else 0

    return {
        "key":     "keyword_match",
        "score":   score,
        "comment": f"关键词命中 {matched}/{total} ({ratio:.0%})"
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator 2：语义正确性（DeepSeek LLM 评判）
# ══════════════════════════════════════════════════════════════════════════════

async def deepseek_judge(run, example) -> dict:
    question  = extract_question(example.inputs or {})
    reference = extract_reference_answer(example.outputs or {})
    actual    = (run.outputs or {}).get("output", "")

    if not reference:
        return {"key": "semantic_correctness", "score": 0, "comment": "无期望答案"}
    if not actual:
        return {"key": "semantic_correctness", "score": 0, "comment": "Agent无输出"}

    prompt = f"""请判断"实际答案"是否正确回答了用户问题，并与"期望答案"的核心内容一致。

用户问题：{question}
期望答案：{reference[:500]}
实际答案：{actual[:500]}

评估标准：数据是否基本吻合（数量、用户名等关键信息），不要求格式完全一样。

只返回 JSON：{{"correct": true或false, "reason": "一句话说明"}}"""

    try:
        resp   = await judge_llm.ainvoke([
            SystemMessage(content="你是评估专家，只返回 JSON，不要加任何 markdown 代码块。"),
            HumanMessage(content=prompt),
        ])
        raw    = resp.content.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        score  = 1 if result.get("correct") else 0
        return {
            "key":     "semantic_correctness",
            "score":   score,
            "comment": result.get("reason", "")
        }
    except Exception as e:
        return {"key": "semantic_correctness", "score": 0, "comment": f"解析失败: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator 3：工具调用验证
# ══════════════════════════════════════════════════════════════════════════════

async def tool_called_judge(run, example) -> dict:
    """验证 agent 是否真的调用了期望的工具，而不是凭空猜测"""
    expected_agents = extract_expected_agents(example.outputs or {})
    actual_plan     = (run.outputs or {}).get("task_plan", [])
    actual_agents   = [t.get("agent", "") for t in actual_plan]

    # 过滤掉 direct（直接回答不算工具调用）
    tool_agents = [a for a in expected_agents if a != "direct"]

    if not tool_agents:
        return {"key": "tool_called", "score": 1,
                "comment": "期望无工具调用（direct任务），跳过"}

    if not actual_plan:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"Agent未规划任何任务（task_plan为空），期望调用: {tool_agents}"
        }

    missing = [a for a in tool_agents if a not in actual_agents]
    if missing:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"缺少工具调用: {missing}，实际执行: {actual_agents}"
        }

    done_agents = [t.get("agent") for t in actual_plan if t.get("status") == "done"]
    not_done    = [a for a in tool_agents if a not in done_agents]
    if not_done:
        return {
            "key":     "tool_called",
            "score":   0,
            "comment": f"工具调用未完成(status≠done): {not_done}"
        }

    return {
        "key":     "tool_called",
        "score":   1,
        "comment": f"工具调用正常 ✓ agents={actual_agents}"
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator 4：延迟评分
# ══════════════════════════════════════════════════════════════════════════════

async def latency_judge(run, example) -> dict:
    """
    计分公式：
      latency <= 20s  → 1.0（优秀）
      latency <= 40s  → 0.7（良好）
      latency <= 60s  → 0.4（勉强）
      latency >  60s  → 0.0（超时）
    """
    latency_ms = (run.outputs or {}).get("latency_ms", 0)
    token_in   = (run.outputs or {}).get("token_in",   0)
    token_out  = (run.outputs or {}).get("token_out",  0)

    if latency_ms <= 20_000:
        score, level = 1.0, "优秀"
    elif latency_ms <= 40_000:
        score, level = 0.7, "良好"
    elif latency_ms <= LATENCY_LIMIT_MS:
        score, level = 0.4, "勉强"
    else:
        score, level = 0.0, "超时"

    return {
        "key":     "latency",
        "score":   score,
        "comment": f"{level} {latency_ms/1000:.1f}s  token≈in:{token_in} out:{token_out}"
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    global _agent_module

    # ── 动态 import agent ────────────────────────────────────────────────────
    try:
        import langgraph_parallel_agent as agent_module
        from langgraph_parallel_agent import (
            _start_mcp_sessions_stdio,
            _stop_mcp_sessions,
        )
        _agent_module = agent_module
    except ImportError as e:
        print(f"❌ 无法导入 agent 模块: {e}")
        print(f"   请确认 src/ 目录下存在 langgraph_parallel_agent.py")
        sys.exit(1)

    # ── 启动 MCP ────────────────────────────────────────────────────────────
    print("🚀 启动 MCP sessions（stdio 模式）...")
    await _start_mcp_sessions_stdio()
    print(f"✅ MCP 初始化完成，agents: {agent_module._registry.agents}\n")

    # ── 运行时间戳（用于文件名和 experiment 名称唯一性）────────────────────────
    run_ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name   = f"{EXPERIMENT_PREFIX}-{run_ts}"
    run_start_iso     = datetime.now().isoformat()

    # ── 评估结果容器 ──────────────────────────────────────────────────────────
    all_records: list[dict] = []
    scores: dict[str, list] = {
        "keyword_match":        [],
        "semantic_correctness": [],
        "tool_called":          [],
        "latency":              [],
    }

    try:
        print(f"📊 开始评估数据集 [{DATASET_NAME}]，实验名: {experiment_name}")
        print(f"   Evaluators: keyword_match / semantic_correctness / tool_called / latency\n")

        eval_results = await aevaluate(
            target,
            data            = DATASET_NAME,
            evaluators      = [
                keyword_judge,       # E1: 关键词匹配
                deepseek_judge,      # E2: 语义正确性
                tool_called_judge,   # E3: 工具调用验证
                latency_judge,       # E4: 延迟评分
            ],
            experiment_prefix = experiment_name,
            max_concurrency   = MAX_CONCURRENCY,
        )

        # ── 遍历结果 ─────────────────────────────────────────────────────────
        print("\n" + "═" * 70)
        async for r in eval_results:
            example  = r.get("example", {})
            run_obj  = r.get("run", {})
            evals    = r.get("evaluation_results", {}).get("results", [])

            question = extract_question(getattr(example, "inputs", {}) or {})
            ref_ans  = extract_reference_answer(getattr(example, "outputs", {}) or {})
            output   = (getattr(run_obj, "outputs", {}) or {}).get("output", "")
            task_plan= (getattr(run_obj, "outputs", {}) or {}).get("task_plan", [])

            # 构建 eval_map
            ev_map = {ev.key: ev for ev in evals}

            def get_ev(key):
                ev = ev_map.get(key)
                return (
                    (ev.score   if ev else None),
                    (ev.comment if ev else ""),
                )

            k_s, k_c = get_ev("keyword_match")
            s_s, s_c = get_ev("semantic_correctness")
            t_s, t_c = get_ev("tool_called")
            l_s, l_c = get_ev("latency")

            # 累积分数
            for key, val in [
                ("keyword_match",        k_s),
                ("semantic_correctness", s_s),
                ("tool_called",          t_s),
                ("latency",              l_s),
            ]:
                if isinstance(val, (int, float)):
                    scores[key].append(val)

            all_pass = all(
                v == 1 for v in [k_s, s_s, t_s]
                if isinstance(v, (int, float))
            )
            icon = "✅" if all_pass else "❌"

            # 序列化 task_plan
            serializable_plan = []
            for t in (task_plan or []):
                if isinstance(t, dict):
                    serializable_plan.append({
                        "agent":       t.get("agent", ""),
                        "description": t.get("description", ""),
                        "status":      t.get("status", ""),
                        "result":      str(t.get("result", ""))[:2000],
                        "task_id":     t.get("task_id", ""),
                    })

            record = {
                "example_id":        str(getattr(example, "id", "")),
                "question":          question,
                "reference_answer":  ref_ans,
                "output":            output,
                "task_plan":         serializable_plan,
                "all_pass":          all_pass,
                "scores": {
                    "keyword_match":        {"score": k_s, "comment": k_c},
                    "semantic_correctness": {"score": s_s, "comment": s_c},
                    "tool_called":          {"score": t_s, "comment": t_c},
                    "latency":              {"score": l_s, "comment": l_c},
                },
                "timestamp": datetime.now().isoformat(),
            }
            all_records.append(record)

            # 控制台打印
            print(f"{icon} {question[:50]}")
            print(f"   📝 回答:     {str(output)[:70]}...")
            print(f"   🔑 关键词:   [{k_s}] {k_c}")
            print(f"   🧠 语义:     [{s_s}] {s_c}")
            print(f"   🔧 工具调用: [{t_s}] {t_c}")
            print(f"   ⏱️  延迟:     [{l_s}] {l_c}")
            print("-" * 70)

    finally:
        await _stop_mcp_sessions()
        print("\n🛑 MCP sessions 已关闭")

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    print()
    score_summary = {}
    labels = {
        "keyword_match":        "🔑 关键词匹配",
        "semantic_correctness": "🧠 语义正确性",
        "tool_called":          "🔧 工具调用验证",
        "latency":              "⏱️  延迟评分",
    }
    for key, label in labels.items():
        vals = scores[key]
        if vals:
            avg    = sum(vals) / len(vals)
            passed = sum(1 for v in vals if v >= 1.0)
            print(f"{label}: {avg:.0%}  ({passed}/{len(vals)} 条通过)")
            score_summary[key] = {
                "avg":    round(avg, 4),
                "passed": passed,
                "total":  len(vals),
            }

    # ── 保存文件 ─────────────────────────────────────────────────────────────
    out_data = {
        "meta": {
            "dataset_name":      DATASET_NAME,
            "run_type":          "evaluate",          # 方法一标识
            "experiment_name":   experiment_name,
            "run_start":         run_start_iso,
            "run_end":           datetime.now().isoformat(),
            "total":             len(all_records),
            "all_pass_count":    sum(1 for r in all_records if r.get("all_pass")),
            "score_summary":     score_summary,
        },
        "results": all_records,
    }

    out_file = OUTPUT_DIR / f"{run_ts}_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 结果文件: {out_file}")
    print(f"\n🔗 LangSmith UI → Datasets → [{DATASET_NAME}] → Experiments")
    print(f"   实验名: {experiment_name}")
    print("═" * 70)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
    
    # uv run python TestDataSet/TestDataSetScript/run_dataset_with_evaluate.py