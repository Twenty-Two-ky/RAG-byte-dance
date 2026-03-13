"""
Phase 4 自测脚本 — ReAct Agent 手写推理循环

测试范围：
  Part 1: 模块导入 & 数据结构
  Part 2: _parse_action 正则解析（纯单元测试，无 API）
  Part 3: _build_system_prompt 工具描述生成
  Part 4: _execute_tool 工具调度
  Part 5: run() ReAct 循环集成测试（需 DASHSCOPE_API_KEY）
  Part 6: run_with_quality_loop() 质量迭代测试（需 DASHSCOPE_API_KEY）
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []


def check(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name} — {e}")
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────
# Part 1: 模块导入 & 数据结构
# ─────────────────────────────────────────────
print("\n=== Part 1: 模块导入 & 数据结构 ===")


def test_import_react_agent():
    from core.agent.react_agent import TestCaseReActAgent, AgentStep, AgentResult


def test_agent_step_creation():
    from core.agent.react_agent import AgentStep
    step = AgentStep(step_number=1, thought="分析需求", action="rag_search",
                     action_input={"query": "登录"}, observation="找到3条结果")
    assert step.step_number == 1
    assert step.action == "rag_search"


def test_agent_result_creation():
    from core.agent.react_agent import AgentResult, AgentStep
    result = AgentResult(
        final_answer="测试用例生成完毕",
        steps=[AgentStep(step_number=1, thought="开始")],
        iterations=1
    )
    assert result.final_answer == "测试用例生成完毕"
    assert result.quality_score is None


def test_import_prompts():
    from config.prompts import PromptManager
    assert hasattr(PromptManager, "REACT_SYSTEM_PROMPT")
    assert hasattr(PromptManager, "REACT_QUALITY_LOOP_PROMPT")
    assert "{tool_descriptions}" in PromptManager.REACT_SYSTEM_PROMPT
    assert "{max_steps}" in PromptManager.REACT_SYSTEM_PROMPT


check("core.agent.react_agent 模块导入", test_import_react_agent)
check("AgentStep 数据结构创建", test_agent_step_creation)
check("AgentResult 数据结构创建", test_agent_result_creation)
check("PromptManager 新增 ReAct 模板", test_import_prompts)


# ─────────────────────────────────────────────
# Part 2: _parse_action 正则解析（纯单元测试）
# ─────────────────────────────────────────────
print("\n=== Part 2: _parse_action 正则解析 ===")

dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")

# 用 mock key 创建 Agent 实例（仅测解析逻辑，不调 API）
from unittest.mock import patch

with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key-for-parsing"}):
    from core.agent.react_agent import TestCaseReActAgent
    # 用 mock 避免真正初始化 ChatTongyi
    try:
        mock_agent = TestCaseReActAgent.__new__(TestCaseReActAgent)
        mock_agent.api_key = "test"
        mock_agent.max_steps = 8
        mock_agent.target_score = 70
        mock_agent.max_retries = 2
        mock_agent.tool_map = {}
    except Exception:
        mock_agent = None


def test_parse_final_answer():
    if not mock_agent:
        raise Exception("mock_agent 创建失败")
    text = """Thought: 我已经有了足够的信息。
Final Answer: [{"id": "TC_001", "module": "登录"}]"""
    thought, action, inp = mock_agent._parse_action(text)
    assert action is None, f"应为 None（Final Answer），实际: {action}"
    assert "足够的信息" in thought


def test_parse_action_standard():
    if not mock_agent:
        raise Exception("mock_agent 创建失败")
    text = """Thought: 需要先搜索知识库
Action: rag_search
Action Input: {"query": "登录功能测试"}"""
    thought, action, inp = mock_agent._parse_action(text)
    assert action == "rag_search", f"期望 rag_search，实际: {action}"
    assert inp["query"] == "登录功能测试"
    assert "搜索知识库" in thought


def test_parse_action_with_chinese_colon():
    if not mock_agent:
        raise Exception("mock_agent 创建失败")
    text = """Thought：需要生成用例
Action：generate_test_cases
Action Input：{"prd_text": "登录需求"}"""
    thought, action, inp = mock_agent._parse_action(text)
    assert action == "generate_test_cases"
    assert inp["prd_text"] == "登录需求"


def test_parse_action_malformed():
    if not mock_agent:
        raise Exception("mock_agent 创建失败")
    text = "这是一段没有任何格式的文本，不包含 Action 或 Final Answer"
    thought, action, inp = mock_agent._parse_action(text)
    assert action == "__PARSE_FAILED__", f"应为 __PARSE_FAILED__，实际: {action}"


def test_parse_action_json_in_markdown():
    if not mock_agent:
        raise Exception("mock_agent 创建失败")
    text = """Thought: 搜索一下
Action: rag_search
Action Input: ```json
{"query": "支付流程"}
```"""
    thought, action, inp = mock_agent._parse_action(text)
    assert action == "rag_search"
    assert inp.get("query") == "支付流程"


check("解析 Final Answer", test_parse_final_answer)
check("解析标准 Action + Action Input", test_parse_action_standard)
check("解析中文冒号格式", test_parse_action_with_chinese_colon)
check("解析格式异常（降级处理）", test_parse_action_malformed)
check("解析 Markdown 代码块中的 JSON", test_parse_action_json_in_markdown)


# ─────────────────────────────────────────────
# Part 3: _build_system_prompt 工具描述
# ─────────────────────────────────────────────
print("\n=== Part 3: _build_system_prompt 工具描述 ===")

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("_build_system_prompt 包含所有工具", None, "跳过"))
    results.append(("_build_system_prompt 格式正确", None, "跳过"))
else:
    agent = TestCaseReActAgent(api_key=dashscope_api_key, max_steps=3)

    def test_system_prompt_contains_tools():
        prompt = agent._build_system_prompt()
        for name in ["rag_search", "add_knowledge", "parse_file",
                      "generate_test_cases", "evaluate_test_quality"]:
            assert name in prompt, f"System Prompt 缺少工具: {name}"

    def test_system_prompt_format():
        prompt = agent._build_system_prompt()
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Final Answer:" in prompt

    check("_build_system_prompt 包含所有工具", test_system_prompt_contains_tools)
    check("_build_system_prompt 格式正确", test_system_prompt_format)


# ─────────────────────────────────────────────
# Part 4: _execute_tool 工具调度
# ─────────────────────────────────────────────
print("\n=== Part 4: _execute_tool 工具调度 ===")

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("_execute_tool 已知工具", None, "跳过"))
    results.append(("_execute_tool 未知工具降级", None, "跳过"))
else:
    def test_execute_known_tool():
        result = agent._execute_tool("rag_search", {"query": "登录测试"})
        assert isinstance(result, str)
        assert "[工具不存在]" not in result

    def test_execute_unknown_tool():
        result = agent._execute_tool("nonexistent_tool", {})
        assert "[工具不存在]" in result
        assert "nonexistent_tool" in result

    check("_execute_tool 已知工具调用", test_execute_known_tool)
    check("_execute_tool 未知工具降级", test_execute_unknown_tool)


# ─────────────────────────────────────────────
# Part 5: run() ReAct 循环集成测试
# ─────────────────────────────────────────────
print("\n=== Part 5: run() ReAct 循环 ===")

SAMPLE_PRD = """
用户登录功能需求：
1. 支持手机号 + 验证码登录
2. 手机号格式：11位数字，1开头
3. 验证码：6位数字，有效期5分钟
4. 连续失败3次锁定账号10分钟
"""

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    for name in ["run() 基础调用", "run() 返回推理步骤", "run() max_steps 限制"]:
        results.append((name, None, "跳过"))
else:
    def test_run_basic():
        result = agent.run(f"请根据以下需求生成测试用例：{SAMPLE_PRD}")
        assert isinstance(result.final_answer, str)
        assert len(result.final_answer) > 0
        assert result.iterations > 0
        print(f"  → 推理轮数: {result.iterations}")
        print(f"  → 回答长度: {len(result.final_answer)} 字符")

    def test_run_has_steps():
        result = agent.run(f"请根据以下需求生成测试用例：{SAMPLE_PRD}")
        assert len(result.steps) > 0, "应至少有一个推理步骤"
        # 打印推理过程
        for step in result.steps:
            action_info = f"Action: {step.action}" if step.action else "Final Answer"
            print(f"  → Step {step.step_number}: {action_info}")

    def test_run_max_steps_limit():
        limited_agent = TestCaseReActAgent(api_key=dashscope_api_key, max_steps=2)
        result = limited_agent.run(f"生成测试用例：{SAMPLE_PRD}")
        assert result.iterations <= 2, f"应不超过 2 步，实际: {result.iterations}"

    check("run() 基础调用", test_run_basic)
    check("run() 返回推理步骤", test_run_has_steps)
    check("run() max_steps 限制", test_run_max_steps_limit)


# ─────────────────────────────────────────────
# Part 6: run_with_quality_loop() 质量迭代
# ─────────────────────────────────────────────
print("\n=== Part 6: run_with_quality_loop() 质量迭代 ===")

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("run_with_quality_loop 端到端", None, "跳过"))
else:
    def test_quality_loop():
        loop_agent = TestCaseReActAgent(
            api_key=dashscope_api_key,
            max_steps=5,
            target_score=60,  # 降低阈值以提高测试通过率
            max_retries=1
        )
        result = loop_agent.run_with_quality_loop(prd_text=SAMPLE_PRD)
        assert isinstance(result.final_answer, str)
        assert len(result.final_answer) > 0
        print(f"  → 最终回答长度: {len(result.final_answer)} 字符")
        if result.quality_score is not None:
            print(f"  → 质量分数: {result.quality_score}")
        if result.quality_report:
            print(f"  → 评估摘要: {result.quality_report.get('summary', 'N/A')}")

    check("run_with_quality_loop 端到端", test_quality_loop)


# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("测试汇总")
print("=" * 50)
passed  = sum(1 for _, ok, _ in results if ok is True)
failed  = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
print(f"通过: {passed}  失败: {failed}  跳过: {skipped}  共: {len(results)}")

if failed > 0:
    print("\n失败明细:")
    for name, ok, err in results:
        if ok is False:
            print(f"  {FAIL} {name}: {err}")

sys.exit(1 if failed > 0 else 0)
