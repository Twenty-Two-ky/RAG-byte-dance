"""
Phase 2 自测脚本 — LLM Client 升级（通义千问 LCEL + DashScope Embedding）

测试范围：
  Part 1: 依赖包导入
  Part 2: TongyiChainManager 初始化 & 基础调用
  Part 3: generate() 单轮生成
  Part 4: chat() 多轮对话 + history 格式转换
  Part 5: DashScopeEmbeddingFunction 向量化
  Part 6: RAGEngine 新接口初始化
  Part 7: llm_client.py 函数可用性
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
# Part 1: 依赖包导入
# ─────────────────────────────────────────────
print("\n=== Part 1: 依赖包导入 ===")

def test_import_langchain_community():
    from langchain_community.chat_models.tongyi import ChatTongyi

def test_import_dashscope():
    import dashscope

def test_import_lc_chain():
    from core.lc_chain import TongyiChainManager, ClaudeChainManager  # 兼容别名

def test_import_llm_client():
    from core.llm_client import get_tongyi_chat_response, generate_summary, get_available_models, extract_json_from_text

def test_import_rag_engine_new():
    from core.rag_engine import RAGEngine, DashScopeEmbeddingFunction

check("langchain_community.ChatTongyi 导入", test_import_langchain_community)
check("dashscope 导入", test_import_dashscope)
check("core.lc_chain 导入", test_import_lc_chain)
check("core.llm_client 导入（新版）", test_import_llm_client)
check("core.rag_engine DashScopeEmbeddingFunction 导入", test_import_rag_engine_new)

# ─────────────────────────────────────────────
# Part 2: TongyiChainManager 初始化
# ─────────────────────────────────────────────
print("\n=== Part 2: TongyiChainManager 初始化 ===")

from core.lc_chain import TongyiChainManager

dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
chain_mgr = None

if not dashscope_api_key:
    print(f"{SKIP} 未找到 DASHSCOPE_API_KEY，跳过通义相关测试")
    results.append(("TongyiChainManager 初始化", None, "跳过"))
else:
    def test_chain_init():
        global chain_mgr
        chain_mgr = TongyiChainManager(api_key=dashscope_api_key)
        assert chain_mgr.llm is not None
        assert chain_mgr.simple_chain is not None
        assert chain_mgr.chat_chain is not None

    def test_chain_no_key_raises():
        import unittest.mock
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            try:
                TongyiChainManager(api_key=None)
                assert False, "应抛出 ValueError"
            except ValueError:
                pass

    check("TongyiChainManager 初始化", test_chain_init)
    check("TongyiChainManager 无 Key 报错", test_chain_no_key_raises)

# ─────────────────────────────────────────────
# Part 3: generate() 单轮生成
# ─────────────────────────────────────────────
print("\n=== Part 3: generate() 单轮生成 ===")

if not chain_mgr:
    print(f"{SKIP} 跳过（无 API Key）")
    results.append(("generate() 基础调用", None, "跳过"))
    results.append(("generate() 超长内容截断", None, "跳过"))
else:
    def test_generate_basic():
        result = chain_mgr.generate(
            content="用一句话解释什么是RAG",
            system_prompt="你是一个简洁的AI助手，只用一句话回答。"
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("[生成失败]"), f"生成失败: {result}"
        print(f"  → 返回: {result[:80]}")

    def test_generate_long_content():
        long_content = "测试内容。" * 2000
        result = chain_mgr.generate(content=long_content, system_prompt="总结一下")
        assert isinstance(result, str)

    check("generate() 基础调用", test_generate_basic)
    check("generate() 超长内容截断", test_generate_long_content)

# ─────────────────────────────────────────────
# Part 4: chat() 多轮对话
# ─────────────────────────────────────────────
print("\n=== Part 4: chat() 多轮对话 ===")

if not chain_mgr:
    print(f"{SKIP} 跳过（无 API Key）")
    for name in ["chat() 空历史", "chat() history格式转换", "chat() 多轮累积历史"]:
        results.append((name, None, "跳过"))
else:
    def test_chat_empty_history():
        response, history = chain_mgr.chat(
            user_input="你好，用一句话介绍你自己",
            history=[],
            system_prompt="你是一个简洁的助手。"
        )
        assert isinstance(response, str) and len(response) > 0
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"
        print(f"  → 回复: {response[:60]}")

    def test_chat_history_format():
        history = [
            {"role": "user",  "parts": ["你好"]},
            {"role": "model", "parts": ["你好！有什么可以帮你？"]}
        ]
        response, new_history = chain_mgr.chat(
            user_input="继续",
            history=history,
            system_prompt="你是一个助手。"
        )
        assert len(new_history) == 4
        assert new_history[2]["role"] == "user"
        assert new_history[3]["role"] == "model"

    def test_chat_appends_history():
        _, h1 = chain_mgr.chat("第一轮", [], "你是助手")
        _, h2 = chain_mgr.chat("第二轮", h1, "你是助手")
        assert len(h2) == 4

    check("chat() 空历史", test_chat_empty_history)
    check("chat() Gemini history格式转换", test_chat_history_format)
    check("chat() 多轮累积历史", test_chat_appends_history)

# ─────────────────────────────────────────────
# Part 5: DashScopeEmbeddingFunction
# ─────────────────────────────────────────────
print("\n=== Part 5: DashScope Embedding ===")

from core.rag_engine import DashScopeEmbeddingFunction

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    for name in ["DashScope Embedding 初始化", "embedding 单文本", "embedding 批量文本", "embedding 维度 1024"]:
        results.append((name, None, "跳过"))
else:
    embed_fn = DashScopeEmbeddingFunction(api_key=dashscope_api_key)

    def test_embed_init():
        assert embed_fn is not None

    def test_embed_single():
        vectors = embed_fn(["用户登录功能测试"])
        assert len(vectors) == 1
        assert len(vectors[0]) == DashScopeEmbeddingFunction.EMBEDDING_DIM
        assert any(v != 0.0 for v in vectors[0])

    def test_embed_batch():
        texts = ["登录功能", "支付模块", "用户注册"]
        vectors = embed_fn(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == DashScopeEmbeddingFunction.EMBEDDING_DIM

    def test_embed_dimension():
        vectors = embed_fn(["测试"])
        assert len(vectors[0]) == 1024

    check("DashScope Embedding 初始化", test_embed_init)
    check("embedding 单文本", test_embed_single)
    check("embedding 批量文本", test_embed_batch)
    check("embedding 维度 1024", test_embed_dimension)

# ─────────────────────────────────────────────
# Part 6: RAGEngine 新接口初始化
# ─────────────────────────────────────────────
print("\n=== Part 6: RAGEngine 新接口 ===")

from core.rag_engine import RAGEngine

if not dashscope_api_key:
    print(f"{SKIP} 跳过（无 DASHSCOPE_API_KEY）")
    results.append(("RAGEngine 新接口初始化", None, "跳过"))
    results.append(("RAGEngine 缺少 DashScope Key 报错", None, "跳过"))
else:
    def test_rag_engine_new_interface():
        engine = RAGEngine(dashscope_api_key=dashscope_api_key)
        assert isinstance(engine.embedding_fn, DashScopeEmbeddingFunction)
        assert engine.dashscope_api_key == dashscope_api_key

    def test_rag_engine_missing_key():
        import unittest.mock
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            try:
                RAGEngine(dashscope_api_key=None)
                assert False, "应该抛出 ValueError"
            except ValueError:
                pass

    check("RAGEngine 新接口初始化", test_rag_engine_new_interface)
    check("RAGEngine 缺少 DashScope Key 报错", test_rag_engine_missing_key)

# ─────────────────────────────────────────────
# Part 7: llm_client.py 函数可用性
# ─────────────────────────────────────────────
print("\n=== Part 7: llm_client 函数可用性 ===")

from core.llm_client import (
    get_available_models, extract_json_from_text,
    get_tongyi_chat_response, generate_summary,
    get_gemini_chat_response, get_claude_chat_response
)

def test_get_available_models():
    models = get_available_models()
    assert isinstance(models, list) and len(models) > 0
    assert any("qwen" in m.lower() for m in models)

def test_extract_json_valid():
    text = '```json\n[{"id": "TC_001", "module": "登录"}]\n```'
    result = extract_json_from_text(text)
    assert isinstance(result, list) and result[0]["id"] == "TC_001"

def test_extract_json_plain():
    result = extract_json_from_text('[{"id": "TC_002"}]')
    assert result[0]["id"] == "TC_002"

def test_extract_json_invalid():
    assert extract_json_from_text("普通文本，没有JSON") is None

def test_backward_compat_aliases():
    assert get_gemini_chat_response is get_tongyi_chat_response
    assert get_claude_chat_response is get_tongyi_chat_response

check("get_available_models 返回通义列表", test_get_available_models)
check("extract_json_from_text 代码块提取", test_extract_json_valid)
check("extract_json_from_text 纯 JSON 提取", test_extract_json_plain)
check("extract_json_from_text 无 JSON 返回 None", test_extract_json_invalid)
check("向后兼容别名正确", test_backward_compat_aliases)

if dashscope_api_key:
    def test_generate_summary_call():
        result = generate_summary(api_key=dashscope_api_key, content='[{"id":"TC_001","module":"登录"}]')
        assert isinstance(result, str) and len(result) > 0
        print(f"  → 摘要: {result}")

    check("generate_summary 可调用", test_generate_summary_call)
else:
    results.append(("generate_summary 可调用", None, "跳过"))

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
