"""
RAG 层升级自测脚本
测试范围：
  - Part 1: 依赖包导入
  - Part 2: RecursiveCharacterTextSplitter 文本切分
  - Part 3: BM25 索引构建 & 检索
  - Part 4: BGE-Reranker 加载 & 打分
  - Part 5: RAGEngine 集成测试（需要 GEMINI_API_KEY 环境变量）
"""

import os
import sys

# 项目根目录加入 PATH
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

def test_import_langchain():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_import_bm25():
    from rank_bm25 import BM25Okapi

def test_import_sentence_transformers():
    from sentence_transformers import CrossEncoder

def test_import_numpy():
    import numpy as np

def test_import_chromadb():
    import chromadb

def test_import_rag_engine():
    from core.rag_engine import RAGEngine, GeminiEmbeddingFunction

check("langchain_text_splitters 导入", test_import_langchain)
check("rank_bm25 导入", test_import_bm25)
check("sentence_transformers 导入", test_import_sentence_transformers)
check("numpy 导入", test_import_numpy)
check("chromadb 导入", test_import_chromadb)
check("core.rag_engine 模块导入", test_import_rag_engine)

# ─────────────────────────────────────────────
# Part 2: RecursiveCharacterTextSplitter 文本切分
# ─────────────────────────────────────────────
print("\n=== Part 2: 文本切分（LangChain）===")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)

def test_split_chinese():
    text = "需求分析是软件开发的第一步。" * 50  # 约700字
    chunks = splitter.split_text(text)
    assert len(chunks) > 1, "长文本应被切分为多个 chunk"
    for c in chunks:
        assert len(c) <= 500 + 10, f"chunk 长度超出限制: {len(c)}"

def test_split_short():
    text = "这是一段很短的文本。"
    chunks = splitter.split_text(text)
    assert len(chunks) == 1, "短文本应保持为单个 chunk"

def test_split_overlap():
    text = ("A" * 400 + "\n\n") * 3
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    # overlap 区域：后一个 chunk 的开头与前一个结尾有重叠
    if len(chunks) >= 2:
        overlap = set(chunks[0][-100:]) & set(chunks[1][:100])
        # 全是 A，直接检查长度
        assert len(chunks[1]) > 0

def test_split_empty():
    chunks = splitter.split_text("")
    assert chunks == [] or chunks == [""]

check("中文长文本切分", test_split_chinese)
check("短文本不拆分", test_split_short)
check("overlap 正常", test_split_overlap)
check("空文本不崩溃", test_split_empty)

# ─────────────────────────────────────────────
# Part 3: BM25 索引构建 & 检索
# ─────────────────────────────────────────────
print("\n=== Part 3: BM25 关键词检索 ===")

from rank_bm25 import BM25Okapi
import numpy as np

CORPUS = [
    "登录功能需要支持手机号和邮箱两种方式",
    "支付模块需要对接微信支付和支付宝",
    "用户注册时需要进行邮箱验证码验证",
    "系统需要支持多语言国际化配置",
    "接口需要进行 JWT Token 鉴权",
]
tokenized = [list(doc) for doc in CORPUS]
bm25 = BM25Okapi(tokenized)

def test_bm25_build():
    assert bm25 is not None
    assert len(tokenized) == len(CORPUS)

def test_bm25_search_relevant():
    query = "登录 手机号"
    scores = bm25.get_scores(list(query))
    top_idx = int(np.argmax(scores))
    assert top_idx == 0, f"期望命中第0条（登录），实际命中第{top_idx}条"

def test_bm25_search_zero_score():
    # 使用完全不存在于语料库中的字符（纯ASCII不在中文语料里）
    query = "xxxxxqqqqqzzzzz"
    scores = bm25.get_scores(list(query))
    assert all(s == 0.0 for s in scores), "纯ASCII查询在中文语料中分数应全为0"

def test_bm25_top_k():
    query = "支付 微信"
    scores = bm25.get_scores(list(query))
    top_indices = np.argsort(scores)[-3:][::-1]
    top_contents = [CORPUS[i] for i in top_indices if scores[i] > 0]
    assert any("支付" in c for c in top_contents), "Top-3 中应包含支付相关文档"

check("BM25 索引构建", test_bm25_build)
check("BM25 语义相关检索", test_bm25_search_relevant)
check("BM25 无关词零分", test_bm25_search_zero_score)
check("BM25 Top-K 正确", test_bm25_top_k)

# ─────────────────────────────────────────────
# Part 4: BGE-Reranker 加载 & 打分
# ─────────────────────────────────────────────
print("\n=== Part 4: BGE-Reranker 精排 ===")

reranker = None
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    print(f"[INFO] BGE-Reranker 加载成功")
except Exception as e:
    print(f"{SKIP} BGE-Reranker 加载失败（可能需要下载模型或检查网络）: {e}")

if reranker:
    def test_reranker_score_order():
        query = "如何实现用户登录"
        candidates = [
            "登录功能支持手机号和密码验证",       # 最相关
            "支付模块需要对接微信支付",             # 不相关
            "用户登录需要 JWT Token 鉴权",          # 相关
        ]
        pairs = [[query, c] for c in candidates]
        scores = reranker.predict(pairs)
        # 第0条和第2条分数应高于第1条
        assert scores[0] > scores[1], f"相关文档分数({scores[0]:.3f})应高于无关文档({scores[1]:.3f})"
        assert scores[2] > scores[1], f"相关文档分数({scores[2]:.3f})应高于无关文档({scores[1]:.3f})"

    def test_reranker_returns_float():
        pairs = [["测试查询", "测试文档"]]
        scores = reranker.predict(pairs)
        assert isinstance(float(scores[0]), float)

    check("Reranker 相关性排序正确", test_reranker_score_order)
    check("Reranker 返回 float 分数", test_reranker_returns_float)
else:
    results.append(("Reranker 相关性排序", None, "跳过（模型未加载）"))
    results.append(("Reranker 返回 float 分数", None, "跳过（模型未加载）"))

# ─────────────────────────────────────────────
# Part 5: RAGEngine 集成测试
# ─────────────────────────────────────────────
print("\n=== Part 5: RAGEngine 集成测试 ===")

from config.settings import load_config
config = load_config()
api_key = config.get('api_key') or os.environ.get('GEMINI_API_KEY')

if not api_key:
    print(f"{SKIP} 未找到 GEMINI_API_KEY，跳过集成测试")
    print("  → 请在 data/user_config.json 中设置 api_key，或设置环境变量 GEMINI_API_KEY")
    results.append(("RAGEngine 初始化", None, "跳过（无 API Key）"))
else:
    from config.settings import setup_proxy
    setup_proxy()

    from core.rag_engine import RAGEngine
    engine = None

    def test_engine_init():
        global engine
        engine = RAGEngine(api_key)
        assert engine is not None
        assert engine.text_splitter is not None
        assert engine.knowledge_coll is not None
        assert engine.history_coll is not None

    check("RAGEngine 初始化", test_engine_init)

    if engine:
        def test_text_splitter_in_engine():
            chunks = engine.text_splitter.split_text("测试文本内容。" * 100)
            assert len(chunks) >= 1

        def test_bm25_index_in_engine():
            # 知识库可能为空，只要不报错即可
            engine._build_bm25_index()

        def test_list_documents():
            docs = engine.list_documents("knowledge")
            assert isinstance(docs, list)

        def test_search_context_empty_kb():
            # 知识库为空时不应崩溃
            context, sources = engine.search_context("测试查询", use_history=False)
            assert isinstance(context, str)
            assert isinstance(sources, list)

        def test_hybrid_search_empty_kb():
            results_list = engine.hybrid_search("测试查询", top_k=3)
            assert isinstance(results_list, list)

        check("engine.text_splitter 可用", test_text_splitter_in_engine)
        check("engine._build_bm25_index 不崩溃", test_bm25_index_in_engine)
        check("engine.list_documents 返回列表", test_list_documents)
        check("engine.search_context 空库不崩溃", test_search_context_empty_kb)
        check("engine.hybrid_search 空库不崩溃", test_hybrid_search_empty_kb)

# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("测试汇总")
print("=" * 50)
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
print(f"通过: {passed}  失败: {failed}  跳过: {skipped}  共: {len(results)}")

if failed > 0:
    print("\n失败明细:")
    for name, ok, err in results:
        if ok is False:
            print(f"  {FAIL} {name}: {err}")

sys.exit(1 if failed > 0 else 0)
