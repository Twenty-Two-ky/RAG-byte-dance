# 变更记录 002 — LLM 层升级：通义千问 LCEL + DashScope 统一替代 Gemini

- **日期**：2026-03-13
- **阶段**：Phase 2
- **涉及文件**：`core/lc_chain.py`（新建）、`core/rag_engine.py`、`core/llm_client.py`、`config/settings.py`、`requirements.txt`

---

## 一、变更背景

原始 LLM 调用层使用 `google-generativeai` SDK 直接调用 Gemini API，存在以下问题：
1. 调用方式原始，无链式组合能力（无法复用、无法扩展）
2. `llm_client.py` 混入了 `streamlit` 依赖（`@st.cache_data`），破坏后端纯洁性
3. LLM（Gemini）和 Embedding（Gemini）绑定同一供应商，切换成本高

**升级目标**：
- LLM 切换为**通义千问 qwen3-max**（阿里云 DashScope）
- Embedding 继续使用 **text-embedding-v4**（DashScope）
- **统一使用同一个 DashScope API Key**，零额外配置
- 引入 **LangChain LCEL** 管道语法，提升代码可读性和可扩展性

---

## 二、具体变更

### 2.1 新建 `core/lc_chain.py`（核心新文件）

**核心类：`TongyiChainManager`**

使用 LCEL `|` 管道语法组装两条链：

```python
# Chain 1: 单轮生成（摘要、文档解析）
simple_chain = (
    ChatPromptTemplate.from_messages([("system", "{system_prompt}"), ("human", "{content}")])
    | ChatTongyi(model="qwen3-max", ...)
    | StrOutputParser()
)

# Chain 2: 多轮对话（带历史记录）
chat_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
        ("human", "{input}")
    ])
    | ChatTongyi(model="qwen3-max", ...)
    | StrOutputParser()
)
```

| 方法 | 用途 |
|------|------|
| `generate(content, system_prompt)` | 单轮生成，无历史 |
| `chat(user_input, history, system_prompt)` | 多轮对话，Gemini 格式 history 兼容 |
| `parse_file(file_bytes, media_type, prompt)` | 图片/PDF 解析（qwen-vl-max Vision） |

**History 格式双向转换**（向后兼容 ui/main.py）：
```
Gemini 格式（外部）  ←→  LangChain Message（内部）
{"role": "user",  "parts": [...]}  ←→  HumanMessage(content=...)
{"role": "model", "parts": [...]}  ←→  AIMessage(content=...)
```

**向后兼容别名**：
```python
ClaudeChainManager = TongyiChainManager  # 保留旧名，防止其他模块引用报错
```

---

### 2.2 修改 `core/rag_engine.py`

#### RAGEngine `__init__` 简化

```python
# 之前（需要 Gemini API Key）
def __init__(self, api_key):
    self.embedding_fn = GeminiEmbeddingFunction(api_key)

# 之后（只需 DashScope Key，LLM 和 Embedding 共用）
def __init__(self, claude_api_key=None, dashscope_api_key=None):
    self.dashscope_api_key = dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")
    self.embedding_fn = DashScopeEmbeddingFunction(self.dashscope_api_key)
```

#### parse_file_content 改用通义 Vision

```python
# 之前：Gemini Vision
genai.GenerativeModel(model_name).generate_content([prompt, img])

# 之后：TongyiChainManager.parse_file()（qwen-vl-max）
chain_mgr = TongyiChainManager(api_key=self.dashscope_api_key)
chain_mgr.parse_file(file_bytes, media_type, prompt)
```

---

### 2.3 重写 `core/llm_client.py`

| 旧函数 | 新函数 | 变化 |
|--------|--------|------|
| `get_gemini_chat_response()` | `get_tongyi_chat_response()` | 内部改用 TongyiChainManager.chat() |
| `generate_summary()` | 同名 | 内部改用 TongyiChainManager.generate() |
| `get_available_models()` | 同名 | 返回通义模型列表，不调用 API |
| `extract_json_from_text()` | 无变化 | 纯工具函数 |

**删除**：
- `import streamlit as st` + `@st.cache_data`
- `import google.generativeai as genai`

**新增向后兼容别名**：
```python
get_gemini_chat_response = get_tongyi_chat_response
get_claude_chat_response = get_tongyi_chat_response
```

---

### 2.4 修改 `config/settings.py`

新增加载逻辑：
```python
anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
if anthropic_key:
    config['claude_api_key'] = anthropic_key
if os.environ.get("DASHSCOPE_API_KEY"):
    config['dashscope_api_key'] = os.environ.get("DASHSCOPE_API_KEY")
```

---

### 2.5 修改 `requirements.txt`

```diff
- google-generativeai==0.8.5
- anthropic>=0.84.0
- langchain-anthropic>=1.3.0
+ langchain-community>=0.4.0   # 包含 ChatTongyi
  langchain-core>=1.2.0
  dashscope>=1.20.0             # LLM + Embedding 共用
```

---

## 三、使用的模型

| 用途 | 模型 | 供应商 | API Key |
|------|------|--------|---------|
| 文本生成 / 对话 | `qwen3-max` | 阿里云 DashScope | DASHSCOPE_API_KEY |
| 图片/PDF 解析 | `qwen-vl-max` | 阿里云 DashScope | DASHSCOPE_API_KEY |
| 文本向量化 | `text-embedding-v4` | 阿里云 DashScope | DASHSCOPE_API_KEY |
| 重排序精排 | `BAAI/bge-reranker-base` | 本地（HuggingFace） | 无需 |

---

## 四、环境变量配置

| 变量名 | 用途 | 必填 |
|--------|------|------|
| `DASHSCOPE_API_KEY` | 所有 DashScope 调用 | 是 |

---

## 五、影响范围

| 模块 | 是否受影响 | 说明 |
|------|-----------|------|
| `core/lc_chain.py` | 是（新建） | Phase 2 核心文件 |
| `core/rag_engine.py` | 是 | embedding + 文件解析替换 |
| `core/llm_client.py` | 是 | 全面重写，保留向后兼容别名 |
| `config/settings.py` | 是 | 新增 key 加载 |
| `requirements.txt` | 是 | 移除 google-generativeai/anthropic |
| `ui/main.py` | 需适配 | RAGEngine 初始化参数变了（Phase 7 处理） |

---

## 六、测试结果

```
通过: 24  失败: 0  跳过: 0  共: 24
```

所有测试全部通过，包括真实 API 调用验证：
- `generate()` 成功返回 RAG 的一句话解释
- `chat()` 成功完成多轮对话并正确累积历史
- `generate_summary()` 成功生成摘要："登录模块测试用例 TC_001"
- DashScope Embedding 返回 1024 维真实向量

---

## 七、测试过程中遇到的问题

### 问题 1：langchain-text-splitters 与 langchain-core 版本冲突

**现象**：安装 Phase 2 依赖时，pip 依赖解析报冲突：
```
langchain-text-splitters 0.3.0 requires langchain-core<0.4,>=0.3
langchain-community requires langchain-core>=1.0
```
安装完成后运行 Phase 1 测试出现 ImportError 或运行时异常。

**原因**：`langchain-community>=0.4.0` 拉入了 `langchain-core>=1.2`，与已锁定的 `langchain-text-splitters==0.3.0` 的上限约束（`langchain-core<0.4`）冲突，pip 降级 langchain-core 后导致其他包连锁失效。

**解决过程**：
1. 确认 `langchain-text-splitters 1.x` 已适配 `langchain-core 1.x`
2. 将 `requirements.txt` 中的 `langchain-text-splitters==0.3.0` 改为 `langchain-text-splitters>=1.1.0`
3. 重新安装依赖，版本冲突消除，Phase 1 与 Phase 2 测试均正常通过

---

### 问题 2：ANTHROPIC_AUTH_TOKEN 鉴权失败（401）

**现象**：最初计划使用 Claude 作为 LLM，配置 `ANTHROPIC_AUTH_TOKEN` 后调用报错：
```
anthropic.AuthenticationError: 401 {"error":{"type":"authentication_error","message":"invalid x-api-key"}}
```

**原因分析**：
- `ANTHROPIC_AUTH_TOKEN` 是 Bearer Token 格式，需放在 `Authorization: Bearer <token>` 请求头
- `langchain-anthropic` 的 `ChatAnthropic` 固定将 key 放在 `x-api-key` 请求头
- 两种鉴权方式不兼容，Anthropic 服务端拒绝请求

**尝试的修复方案**：

尝试在 `ChatAnthropic` 初始化后注入自定义客户端，绕过 header 限制：
```python
raw_client = anthropic.Anthropic(auth_token=token)
llm.client = raw_client.messages  # 试图覆盖内部客户端字段
```
结果：Pydantic v2 的 model 实例不允许初始化后对字段赋值，抛出 `ValidationError`，方案失败。

随后尝试子类化 `ChatAnthropic` 并设置 `model_config = ConfigDict(arbitrary_types_allowed=True)` 后再覆盖，依然被 Pydantic v2 保护机制拦截。

**最终解决方案**：

放弃使用 Claude，将 LLM 整体切换为**通义千问 qwen3-max**（阿里云 DashScope）：
- `DASHSCOPE_API_KEY` 同时用于 LLM（qwen3-max）和 Embedding（text-embedding-v4），一个 Key 覆盖所有调用
- 删除 `langchain-anthropic`、`anthropic` 依赖，改用 `langchain-community` 中的 `ChatTongyi`
- 切换后 24 项测试全部通过（24/24）
