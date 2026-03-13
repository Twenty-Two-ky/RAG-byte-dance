"""
Phase 3 自测脚本 — LangChain 工具库 + Evaluator 升级

测试范围：
  Part 1: 依赖包导入 & 模块结构
  Part 2: Tool 元数据验证（名称/描述/参数 schema）
  Part 3: AGENT_TOOLS 列表完整性
  Part 4: Evaluator 升级验证（DashScope Key）
  Part 5: rag_search 工具真实调用（需 DASHSCOPE_API_KEY）
  Part 6: generate_test_cases 工具真实调用（需 DASHSCOPE_API_KEY）
  Part 7: evaluate_test_quality 工具真实调用（需 DASHSCOPE_API_KEY）
"""

import os
import sys

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
# Part 1: 依赖包导入 & 模块结构
# ─────────────────────────────────────────────
print("\n=== Part 1: 依赖包导入 ===")


def test_import_tools_module():
    from core.agent import tools  # noqa


def test_import_agent_tools_list():
    from core.agent.tools import AGENT_TOOLS
    assert isinstance(AGENT_TOOLS, list)


def test_import_evaluator_new():
    from core.evaluator import Evaluator
    import inspect
    sig = inspect.signature(Evaluator.__init__)
    # 新版 __init__ 的 api_key 应有默认值 None
    assert sig.parameters["api_key"].default is None


def test_import_tool_functions():
    from core.agent.tools import (
        rag_search, add_knowledge, parse_file,
        generate_test_cases, evaluate_test_quality
    )


check("core.agent.tools 模块导入", test_import_tools_module)
check("AGENT_TOOLS 列表可导入", test_import_agent_tools_list)
check("Evaluator 新接口（api_key=None）", test_import_evaluator_new)
check("5 个 tool 函数可导入", test_import_tool_functions)


# ─────────────────────────────────────────────
# Part 2: Tool 元数据验证
# ─────────────────────────────────────────────
print("\n=== Part 2: Tool 元数据验证 ===")

from core.agent.tools import (
    rag_search, add_knowledge, parse_file,
    generate_test_cases, evaluate_test_quality
)

EXPECTED_TOOLS = {
    "rag_search": ["query"],
    "add_knowledge": ["content", "source_name"],
    "parse_file": ["file_path", "prompt"],
    "generate_test_cases": ["prd_text", "rag_context"],
    "evaluate_test_quality": ["prd_text", "test_cases_json"],
}


def make_tool_meta_test(t, expected_params):
    def _test():
        assert t.name, "工具缺少 name"
        assert t.description, "工具缺少 description"
        schema = t.args_schema.model_json_schema() if hasattr(t, "args_schema") and t.args_schema else {}
        props = schema.get("properties", {})
        for param in expected_params:
            assert param in props, f"参数 '{param}' 不在 schema 中: {list(props.keys())}"
    return _test


for tool_fn in [rag_search, add_knowledge, parse_file, generate_test_cases, evaluate_test_quality]:
    expected = EXPECTED_TOOLS[tool_fn.name]
    check(f"Tool '{tool_fn.name}' 元数据完整", make_tool_meta_test(tool_fn, expected))


# ─────────────────────────────────────────────
# Part 3: AGENT_TOOLS 列表完整性
# ─────────────────────────────────────────────
print("\n=== Part 3: AGENT_TOOLS 列表 ===")

from core.agent.tools import AGENT_TOOLS


def test_tools_count():
    assert len(AGENT_TOOLS) == 5, f"期望 5 个工具，实际 {len(AGENT_TOOLS)} 个"


def test_tools_names():
    names = {t.name for t in AGENT_TOOLS}
    expected = {"rag_search", "add_knowledge", "parse_file",
                "generate_test_cases", "evaluate_test_quality"}
    assert names == expected, f"工具名不匹配: {names}"


def test_tools_all_have_description():
    for t in AGENT_TOOLS:
        assert t.description, f"工具 '{t.name}' 缺少 description"


check("AGENT_TOOLS 共 5 个工具", test_tools_count)
check("AGENT_TOOLS 名称集合正确", test_tools_names)
check("所有工具均有 description", test_tools_all_have_description)


# ─────────────────────────────────────────────
# Part 4: Evaluator 升级验证
# ─────────────────────────────────────────────
print("\n=== Part 4: Evaluator 升级验证 ===")

dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")


def test_evaluator_no_gemini_import():
    import core.evaluator as ev_module
    import inspect
    src = inspect.getsource(ev_module)
    assert "google.generativeai" not in src, "evaluator.py 仍有 google.generativeai 引用"
    assert "TongyiChainManager" in src, "evaluator.py 未引用 TongyiChainManager"


def test_evaluator_init_with_key():
    if not dashscope_api_key:
        raise Exception("跳过（无 DASHSCOPE_API_KEY）")
    from core.evaluator import Evaluator
    ev = Evaluator(api_key=dashscope_api_key)
    assert ev.api_key == dashscope_api_key


def test_evaluator_no_key_raises():
    import unittest.mock
    with unittest.mock.patch.dict(os.environ, {}, clear=True):
        from core.evaluator import Evaluator
        try:
            Evaluator(api_key=None)
            assert False, "应抛出 ValueError"
        except ValueError:
            pass


check("evaluator.py 已移除 google.generativeai", test_evaluator_no_gemini_import)

if dashscope_api_key:
    check("Evaluator 用 DashScope Key 初始化", test_evaluator_init_with_key)
    check("Evaluator 无 Key 报 ValueError", test_evaluator_no_key_raises)
else:
    print(f"{SKIP} Evaluator 初始化（无 DASHSCOPE_API_KEY）")
    results.append(("Evaluator 用 DashScope Key 初始化", None, "跳过"))
    results.append(("Evaluator 无 Key 报 ValueError", None, "跳过"))


# ─────────────────────────────────────────────
# Part 5: rag_search 工具真实调用
# ─────────────────────────────────────────────
print("\n=== Part 5: rag_search 真实调用 ===")

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("rag_search 工具调用", None, "跳过"))
else:
    def test_rag_search_call():
        result = rag_search.invoke({"query": "登录功能测试规范"})
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"  → 返回长度: {len(result)} 字符")

    check("rag_search 工具调用（知识库可能为空）", test_rag_search_call)


# ─────────────────────────────────────────────
# Part 6: generate_test_cases 工具真实调用
# ─────────────────────────────────────────────
print("\n=== Part 6: generate_test_cases 真实调用 ===")

SAMPLE_PRD = """
用户登录功能需求：
1. 支持手机号 + 验证码登录
2. 手机号格式：11位数字，1开头
3. 验证码：6位数字，有效期5分钟
4. 连续失败3次锁定账号10分钟
"""

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("generate_test_cases 工具调用", None, "跳过"))
    results.append(("generate_test_cases 带 rag_context", None, "跳过"))
else:
    def test_generate_cases_basic():
        result = generate_test_cases.invoke({"prd_text": SAMPLE_PRD})
        assert isinstance(result, str) and len(result) > 0
        assert not result.startswith("[生成失败]"), f"生成失败: {result[:100]}"
        print(f"  → 返回长度: {len(result)} 字符")

    def test_generate_cases_with_context():
        result = generate_test_cases.invoke({
            "prd_text": SAMPLE_PRD,
            "rag_context": "规范：手机号验证码登录须覆盖边界值和异常流程。"
        })
        assert isinstance(result, str) and len(result) > 0

    check("generate_test_cases 基础调用", test_generate_cases_basic)
    check("generate_test_cases 带 rag_context", test_generate_cases_with_context)


# ─────────────────────────────────────────────
# Part 7: evaluate_test_quality 工具真实调用
# ─────────────────────────────────────────────
print("\n=== Part 7: evaluate_test_quality 真实调用 ===")

SAMPLE_CASES = '''[
  {"id":"TC_001","module":"登录","precondition":"未登录","step":"输入正确手机号和验证码","expected":"登录成功","priority":"P0","design_strategy":"正常流程"},
  {"id":"TC_002","module":"登录","precondition":"未登录","step":"输入错误格式手机号","expected":"提示手机号格式错误","priority":"P1","design_strategy":"边界值"}
]'''

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("evaluate_test_quality 工具调用", None, "跳过"))
    results.append(("evaluate_test_quality 返回 score 字段", None, "跳过"))
else:
    import json as _json

    def test_evaluate_call():
        result = evaluate_test_quality.invoke({
            "prd_text": SAMPLE_PRD,
            "test_cases_json": SAMPLE_CASES
        })
        assert isinstance(result, str) and len(result) > 0
        print(f"  → 返回长度: {len(result)} 字符")

    def test_evaluate_returns_score():
        result = evaluate_test_quality.invoke({
            "prd_text": SAMPLE_PRD,
            "test_cases_json": SAMPLE_CASES
        })
        report = _json.loads(result)
        assert "score" in report, f"报告缺少 score 字段: {list(report.keys())}"
        assert isinstance(report["score"], (int, float))
        print(f"  → 质量分数: {report['score']}")

    check("evaluate_test_quality 工具调用", test_evaluate_call)
    check("evaluate_test_quality 返回 score 字段", test_evaluate_returns_score)


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
