# 变更记录 002 — LLM 层升级：Claude LCEL + DashScope Embedding

- **日期**：2026-03-13
- **阶段**：Phase 2
- **涉及文件**：`core/lc_chain.py`（新建）、`core/rag_engine.py`、`core/llm_client.py`、`config/settings.py`、`requirements.txt`

---

## 一、变更背景

原始 LLM 调用层使用 `google-generativeai` SDK 直接调用 Gemini API：
1. 调用方式原始，无链式组合能力（无法复用、无法扩展）
2. Embedding 依赖 Gemini `text-embedding-004`，与 LLM 强绑定同一供应商
3. 用户现有 Claude API Key，希望统一使用 Claude 作为主模型
4. `llm_client.py` 中混入了 `streamlit` 依赖（`@st.cache_data`），破坏后端纯洁性

---

## 二、具体变更

### 2.1 新建 `core/lc_chain.py`（核心新文件）

**用途**：封装所有 Claude 调用，使用 LangChain LCEL 管道语法。

**包含内容**：

#### `ClaudeChainManager` 类

```python
# LCEL 管道：Prompt → LLM → OutputParser
simple_chain = ChatPromptTemplate.from_messages([...]) | llm | StrOutputParser()
chat_chain   = ChatPromptTemplate.from_messages([..., MessagesPlaceholder(...)]) | llm | StrOutputParser()
```

| 方法 | 用途 |
|------|------|
| `generate(content, system_prompt)` | 单轮生成（摘要、文档处理） |
| `chat(user_input, history, system_prompt)` | 多轮对话，history 兼容 Gemini 格式 |
| `parse_file(file_bytes, media_type, prompt)` | Claude Vision 解析图片/PDF |

**History 格式转换**：
```python
# Gemini 格式（输入/输出，向后兼容）
{"role": "user",  "parts": ["用户输入"]}
{"role": "model", "parts": ["模型回复"]}

# LangChain 格式（内部使用）
HumanMessage(content="用户输入")
AIMessage(content="模型回复")
```

---

### 2.2 修改 `core/rag_engine.py`

#### 删除 GeminiEmbeddingFunction，新增 DashScopeEmbeddingFunction

```python
# 之前
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input):
        result = genai.embed_content(model="models/text-embedding-004", ...)
        return result['embedding']

# 之后
class DashScopeEmbeddingFunction(chromadb.EmbeddingFunction):
    EMBEDDING_DIM = 1024  # text-embedding-v4 默认维度

    def __call__(self, input):
        result = TextEmbedding.call(model="text-embedding-v4", input=input, ...)
        return [item['embedding'] for item in result.output['embeddings']]
```

#### RAGEngine `__init__` 签名变更

```python
# 之前
def __init__(self, api_key):
    self.embedding_fn = GeminiEmbeddingFunction(api_key)

# 之后
def __init__(self, claude_api_key=None, dashscope_api_key=None):
    self.claude_api_key    = claude_api_key    or os.environ.get("ANTHROPIC_API_KEY")
    self.dashscope_api_key = dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")
    self.embedding_fn = DashScopeEmbeddingFunction(self.dashscope_api_key)
```

#### parse_file_content 改用 Claude Vision

```python
# 之前：Gemini GenerativeModel + PIL.Image
genai.GenerativeModel(model_name).generate_content([prompt, img])

# 之后：ClaudeChainManager.parse_file()
chain_mgr = ClaudeChainManager(api_key=self.claude_api_key)
chain_mgr.parse_file(file_bytes, media_type, prompt)
```

#### 删除 imports

```python
# 删除
import google.generativeai as genai
from PIL import Image
```

---

### 2.3 重写 `core/llm_client.py`

| 旧函数 | 新函数 | 变化说明 |
|--------|--------|---------|
| `get_gemini_chat_response()` | `get_claude_chat_response()` | 内部改用 ClaudeChainManager.chat() |
| `generate_summary()` | `generate_summary()` | 内部改用 ClaudeChainManager.generate() |
| `get_available_models()` | `get_available_models()` | 返回预设 Claude 模型列表，不再调用 API |
| `extract_json_from_text()` | 无变化 | 纯工具函数 |

**删除**：
- `import streamlit as st`
- `@st.cache_data(ttl=3600)` 装饰器
- `import google.generativeai as genai`

**新增**：
- `get_gemini_chat_response = get_claude_chat_response`（向后兼容别名）

---

### 2.4 修改 `config/settings.py`

新增从环境变量加载两个 key：
```python
config['claude_api_key']    = os.environ.get("ANTHROPIC_API_KEY")
config['dashscope_api_key'] = os.environ.get("DASHSCOPE_API_KEY")
```

---

### 2.5 修改 `requirements.txt`

```diff
- google-generativeai==0.8.5
+ langchain-anthropic>=1.3.0
+ langchain-core>=1.2.0
+ anthropic>=0.84.0
+ dashscope>=1.20.0
  langchain-text-splitters>=1.1.0   # 版本升级（兼容 langchain-core 1.x）
```

---

## 三、使用的模型

| 用途 | 模型 | 供应商 |
|------|------|--------|
| 文本生成 / 对话 | `global.anthropic.claude-sonnet-4-6` | Anthropic |
| 文件解析（Vision） | `global.anthropic.claude-sonnet-4-6` | Anthropic |
| 文本向量化 | `text-embedding-v4` | 阿里云 DashScope |
| 重排序精排 | `BAAI/bge-reranker-base` | 本地（HuggingFace） |

---

## 四、环境变量配置

| 变量名 | 用途 | 必填 |
|--------|------|------|
| `ANTHROPIC_API_KEY` | Claude 调用 | 是（文件解析、对话） |
| `DASHSCOPE_API_KEY` | Embedding 向量化 | 是（RAG 检索必需） |

---

## 五、影响范围

| 模块 | 是否受影响 | 说明 |
|------|-----------|------|
| `core/lc_chain.py` | 是（新建） | Phase 2 核心文件 |
| `core/rag_engine.py` | 是 | embedding 替换，接口签名变更 |
| `core/llm_client.py` | 是 | 全面重写，保留向后兼容别名 |
| `config/settings.py` | 是 | 新增两个 key 加载 |
| `requirements.txt` | 是 | 移除 google-generativeai，新增 4 个包 |
| `ui/main.py` | 需适配 | RAGEngine 初始化参数变了（Phase 7 处理） |

---

## 六、测试结果

```
通过: 17  失败: 0  跳过: 6（Claude API Key 未配置）  共: 23
```

跳过的 6 项为 Claude 对话/生成相关测试，需配置 `ANTHROPIC_API_KEY` 后运行。
