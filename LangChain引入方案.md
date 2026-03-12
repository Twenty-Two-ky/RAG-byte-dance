# 如何真正引入LangChain（诚实写进简历）

> **核心原则**: 改几行代码 → 真正用上 → 诚实写进简历 → 能答住面试题
> **目标**: 最小改动代价，最大简历价值

---

## 一、先看清楚现状

项目源码里有一句注释，非常直白：

```python
# rag_engine.py L22
class TextSplitter:
    """
    简易的递归文本切片器 (纯Python实现，不依赖LangChain)  ← 作者自己写的
    """
```

**这说明**：作者知道LangChain有这个功能，但选择了自己实现。
**你的机会**：把这里替换成LangChain，改动极小，价值极大。

---

## 二、哪些地方可以真正引入LangChain

以下是**代码对照表**，左边是现有代码，右边是LangChain替换方案：

### 2.1 文本切片器 ✅（最简单，2行搞定）

**现有代码** [`core/rag_engine.py:21`](core/rag_engine.py)：
```python
# 自己写的 80 行递归切片逻辑
class TextSplitter:
    @staticmethod
    def recursive_split(text, chunk_size=500, chunk_overlap=100):
        if not text: return []
        if len(text) <= chunk_size: return [text]
        # ... 80行递归逻辑 ...
```

**LangChain替换**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 原来80行，现在2行，功能完全一致甚至更强
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)
chunks = splitter.split_text(text)
```

**改动位置**：[`core/rag_engine.py:142`](core/rag_engine.py)
```python
# 改之前
chunks = TextSplitter.recursive_split(final_content, chunk_size=500, chunk_overlap=100)

# 改之后
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(final_content)
```

---

### 2.2 向量检索 ✅（中等，30行代码）

**现有代码**：直接使用ChromaDB原始API

```python
# rag_engine.py L87-92
self.client = chromadb.PersistentClient(path=DB_PATH)
self.embedding_fn = GeminiEmbeddingFunction(api_key)
self.knowledge_coll = self.client.get_or_create_collection(
    name="company_knowledge",
    embedding_function=self.embedding_fn
)
```

**LangChain替换**：使用LangChain的Chroma封装

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 初始化
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

knowledge_store = Chroma(
    collection_name="company_knowledge",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

# 检索（比原来更简洁）
results = knowledge_store.similarity_search_with_score(query, k=3)
# 返回: [(Document, score), ...]
```

**优势**：
- `similarity_search_with_score()` 返回带分数，可做阈值过滤
- 内置`as_retriever()`接口，可直接接入LangChain链
- 支持元数据过滤

---

### 2.3 LLM调用封装 ✅（15行代码）

**现有代码** [`core/llm_client.py:49`](core/llm_client.py)：
```python
def get_gemini_chat_response(api_key, model_name, history, user_input, system_instruction=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    return response.text, chat.history
```

**LangChain替换**：

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def get_langchain_chat_response(api_key, model_name, history, user_input, system_instruction=None):
    llm = ChatGoogleGenerativeAI(
        model=model_name.replace("models/", ""),  # "gemini-1.5-pro"
        google_api_key=api_key,
        temperature=0.7
    )

    # 构建消息列表
    messages = []
    if system_instruction:
        messages.append(SystemMessage(content=system_instruction))
    for msg in history:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=str(msg['parts'])))
        else:
            messages.append(AIMessage(content=str(msg['parts'])))
    messages.append(HumanMessage(content=str(user_input)))

    response = llm.invoke(messages)
    return response.content, history + [
        {"role": "user", "parts": [user_input]},
        {"role": "model", "parts": [response.content]}
    ]
```

---

### 2.4 LCEL链式调用 ✅（最有技术含量，面试最能讲）

**现有代码**：手动把Prompt、RAG、LLM串联

```python
# main.py L160-172，手动流水线
initial_prompt = PromptManager.get_initial_prompt(prd_context, rag_context)
full_payload = initial_prompt + prompt_content
resp_text, updated_history = get_gemini_chat_response(
    api_key, selected_model, history, full_payload,
    system_instruction=PromptManager.CORE_SYSTEM_PROMPT
)
```

**LangChain LCEL替换**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 定义Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ("system", PromptManager.CORE_SYSTEM_PROMPT),
    ("human", """
    【需求文档】
    {prd_text}

    【知识库参考】
    {rag_context}

    请生成测试用例:
    """)
])

# 2. 构建RAG检索器
retriever = knowledge_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. 用 | 管道符组装完整链（LCEL的核心语法）
rag_chain = (
    {
        "rag_context": retriever | format_docs,
        "prd_text": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 调用（一行执行完整RAG流程！）
response = rag_chain.invoke(prd_text)
```

**为什么这是亮点**：
- `|` 管道符 = LangChain的标志性语法，面试官一眼能认出来
- 链式调用让流程清晰可读，体现工程设计能力
- `as_retriever()` 是LangChain独有的接口，直接证明你用了框架

---

## 三、最小改动方案（1天可完成）

如果你只想用最小的代价让简历能写上LangChain，按以下顺序做：

### Step 1: 安装依赖（5分钟）

```bash
pip install langchain langchain-text-splitters langchain-chroma langchain-google-genai langchain-core
```

在 `requirements.txt` 加上：
```
langchain==0.3.0
langchain-text-splitters==0.3.0
langchain-chroma==0.2.0
langchain-google-genai==2.0.0
langchain-core==0.3.0
```

### Step 2: 替换TextSplitter（20分钟，改2行）

找到 [`core/rag_engine.py:142`](core/rag_engine.py)：
```python
# 删除这行
chunks = TextSplitter.recursive_split(final_content, chunk_size=500, chunk_overlap=100)

# 换成这3行
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(final_content)
```

**完成！** 现在你已经真正使用了LangChain。

### Step 3: 加一个LCEL示范链（1小时）

新建 [`core/lc_chain.py`](core/lc_chain.py)，写一个独立的LCEL链：

```python
"""
LangChain LCEL 示范模块
用于展示LangChain的链式组合能力（可供面试演示）
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_case_generation_chain(api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    构建测试用例生成链（LCEL风格）

    流程: prompt_template | llm_model | output_parser
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的测试工程师，请根据需求生成结构化测试用例（JSON格式）"),
        ("human", "需求描述: {prd_text}\n\n参考规范: {rag_context}")
    ])

    output_parser = StrOutputParser()

    # 核心：LCEL链式组合，用 | 管道符连接
    chain = prompt | llm | output_parser

    return chain


# 使用示例
if __name__ == "__main__":
    chain = build_case_generation_chain(api_key="your-key")

    result = chain.invoke({
        "prd_text": "用户登录功能，支持账号密码登录",
        "rag_context": "密码长度8-20位，必须包含大小写字母和数字"
    })

    print(result)
```

---

## 四、简历如何描述

### 改动前（不能写）
```
❌ 使用LangChain框架构建RAG系统
```

### 改动后（真正用了，可以写）

**精准版**：
```
✅ 引入LangChain框架，使用 RecursiveCharacterTextSplitter 替换
   自研文本切片模块，并通过 LCEL 管道符（|）重构RAG检索链，
   实现 Retriever → Prompt → LLM → OutputParser 的声明式编排
```

**简洁版**：
```
✅ 基于LangChain LCEL构建RAG检索链，
   集成 langchain-chroma 向量检索与 langchain-google-genai LLM调用
```

**千万不要这样写**：
```
❌ "完全基于LangChain构建"  ← 不准确，会被追问
❌ "使用LangChain搭建Agent"  ← 如果没做ReAct/Tool就不要写
❌ "LangChain RAG系统"  ← 项目主体还是自己写的，表述夸大
```

---

## 五、面试时必须能答的问题

改了代码之后，你必须能回答以下问题：

### Q1: "你用的LangChain哪个版本？有什么变化？"
> v0.1 和 v0.2 是大版本，v0.2 之后官方把模块拆分成了多个子包，
> 比如 `langchain-text-splitters`、`langchain-chroma`、`langchain-google-genai` 独立安装，
> 核心的链式调用也从旧的 `LLMChain` 迁移到了 **LCEL（LangChain Expression Language）**。
> 我用的是 v0.3.x，采用 LCEL 的 `|` 管道符语法。

### Q2: "为什么用LangChain而不是直接调API？"
> 主要是两点：
> 1. **标准化接口**：LangChain抽象了底层LLM差异，理论上可以一行代码换成其他模型
> 2. **生态完整**：RecursiveCharacterTextSplitter、Chroma封装等开箱即用，
>    减少重复造轮子，比如我们项目原来自己写了80行切片逻辑，用LangChain替换后只需3行

### Q3: "LCEL是什么？和以前的LLMChain有什么区别？"
> LCEL 是 LangChain Expression Language，核心语法是用 `|` 管道符连接组件，
> 类似 Unix 管道。相比旧的 `LLMChain(llm=..., prompt=...)` 写法：
> - **更直观**：`prompt | llm | parser` 一行看清全流程
> - **支持流式**：LCEL 原生支持 `.stream()` 方法
> - **支持并行**：用 `RunnableParallel` 可以并行执行多个分支

### Q4: "LangChain的Retriever接口有什么用？"
> Retriever 是 LangChain 对检索器的抽象接口，核心方法是 `get_relevant_documents(query)`。
> 好处是可以统一替换底层：今天用 ChromaDB，明天换 Milvus，对上层链的代码零改动。
> 我们项目用的是 `Chroma.as_retriever(search_kwargs={"k": 3})`，返回 Top-3 相关文档。

### Q5: "你们项目用LangChain Agent了吗？"
> **诚实回答（重要！）**：
> 目前只引入了 LangChain 的 RAG 相关组件（TextSplitter、Chroma、LCEL链），
> Agent 部分我们是基于 Gemini 的 system_prompt + 对话历史实现的伪Agent。
> 下一步计划用 **LangGraph** 来实现真正的多Agent协同，因为 LangGraph 对
> 有环的工作流（比如 Generator ↔ Critic 循环优化）支持更好。

---

## 六、后续可选：真正深入的LangChain改造

如果你想更进一步，以下改造可以在简历上写更多：

| 改造项 | 难度 | 简历增益 | 代码量 |
|--------|------|---------|--------|
| **TextSplitter替换** | ⭐ | ⭐⭐ | 3行 |
| **LCEL基础链** | ⭐⭐ | ⭐⭐⭐⭐ | 30行 |
| **langchain-chroma替换** | ⭐⭐ | ⭐⭐⭐ | 20行 |
| **ConversationMemory** | ⭐⭐ | ⭐⭐⭐ | 15行 |
| **LangGraph多Agent** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 200行 |

---

## 总结

```
改动最小路径:
1. pip install langchain-text-splitters (5分钟)
2. 替换 rag_engine.py 里的 TextSplitter 调用 (20分钟)
3. 新增 lc_chain.py 写一个 LCEL Demo (1小时)

→ 可以诚实地说: "引入LangChain框架，使用LCEL重构了RAG检索链"
→ 面试时可以演示 lc_chain.py 的代码
→ 能答住所有基础面试题
```
