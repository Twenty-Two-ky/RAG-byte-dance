# Auto PRD Test Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![LLM](https://img.shields.io/badge/LLM-通义千问_qwen3--max-orange)
![RAG](https://img.shields.io/badge/RAG-ChromaDB+BM25+Reranker-purple)

基于 **RAG 混合检索 + 手写 ReAct Agent** 的智能测试用例生成系统。输入 PRD 需求文档，Agent 自主决策调用工具链，自动生成结构化测试用例并通过质量评估闭环迭代优化。

项目来源于字节跳动训练营结题项目，在原始版本基础上进行了 4 个阶段的架构升级，从单次 LLM 调用演进为具备 RAG 检索、LCEL 链式调用、LangChain Tool 工具库、ReAct 推理循环的完整 Agent 系统。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Auto PRD Test Agent                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PRD 需求文档                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────┐    Phase 1: RAG 混合检索引擎                 │
│  │   RAG Engine      │    向量检索(70%) + BM25(30%) + RRF 融合      │
│  │   rag_engine.py   │──► BGE-Reranker 精排 ──► LLM 上下文去噪     │
│  └────────┬─────────┘                                               │
│           │ 检索上下文                                               │
│           ▼                                                         │
│  ┌──────────────────┐    Phase 4: 手写 ReAct 推理循环              │
│  │   ReAct Agent     │    Thought ──► Action ──► Observation        │
│  │   react_agent.py  │         ◄── 循环直到 Final Answer            │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ├── rag_search         ──► 知识库检索                     │
│           ├── generate_test_cases ──► 用例生成     Phase 3: 工具库  │
│           ├── evaluate_test_quality ► 质量评估                      │
│           ├── add_knowledge      ──► 知识入库                       │
│           └── parse_file         ──► 文件解析                       │
│                                                                     │
│           │ 质量不达标？                                             │
│           ▼                                                         │
│  ┌──────────────────┐    质量迭代闭环                               │
│  │ 评估反馈 + 改进   │    score < target ──► 注入反馈 ──► 重新生成  │
│  └────────┬─────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│     结构化测试用例 (JSON)                                           │
│                                                                     │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│  Phase 2: LLM 层 (TongyiChainManager)                              │
│  ChatTongyi(qwen3-max) | ChatPromptTemplate | StrOutputParser       │
│  LCEL pipe 语法: simple_chain(单轮) + chat_chain(多轮)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心技术亮点

### 1. RAG 混合检索引擎 (Phase 1)

```
原始文档 ──► RecursiveCharacterTextSplitter 切片
         ──► ChromaDB 向量存储 (text-embedding-v4, 1024维)

查询 ──► 向量检索 Top-K (权重 0.7)
     ──► BM25 关键词检索 (权重 0.3)
     ──► RRF (Reciprocal Rank Fusion) 融合排序
     ──► BGE-Reranker 精排 (BAAI/bge-reranker-base)
     ──► LLM 上下文去噪 (RAG_FILTER_PROMPT)
```

- **混合检索**：向量语义 + BM25 关键词互补，解决纯向量检索漏召回问题
- **精排重排序**：Cross-Encoder 模型对候选文档重新打分，提升相关性
- **优雅降级**：Reranker 加载失败自动跳过，hybrid_search 异常回退纯向量检索

### 2. LCEL 链式调用 (Phase 2)

```python
# LangChain Expression Language — pipe 语法
simple_chain = ChatPromptTemplate | ChatTongyi(model="qwen3-max") | StrOutputParser()
chat_chain   = ChatPromptTemplate | ChatTongyi(model="qwen3-max") | StrOutputParser()
```

- **TongyiChainManager**：封装两条 LCEL 链，对外提供 `generate()` 和 `chat()` 接口
- **DashScope 统一**：LLM (qwen3-max) + Embedding (text-embedding-v4) + 评估，全部使用同一个 API Key
- **多模态支持**：qwen-vl-max 模型处理 UI 截图识别

### 3. LangChain Tool 工具库 (Phase 3)

| 工具名 | 功能 | 封装的底层能力 |
|--------|------|---------------|
| `rag_search` | 知识库检索 | RAGEngine.search_context() |
| `add_knowledge` | 知识入库 | RAGEngine.add_knowledge() |
| `parse_file` | 文件解析 | TongyiChainManager.parse_file() |
| `generate_test_cases` | 用例生成 | TongyiChainManager.generate() |
| `evaluate_test_quality` | 质量评估 | Evaluator.evaluate_cases() |

所有工具使用 `@tool` 装饰器定义，LangChain 自动生成 JSON Schema 供 Agent 推理时读取工具说明。

### 4. ReAct Agent 手写推理循环 (Phase 4)

**为什么手写而不用 AgentExecutor**：AgentExecutor 已被 LangChain 标记为 legacy，手写循环教育价值更高，可逐行解释 ReAct 原理。

```
循环（最多 max_steps 次）：
  1. LLM 生成推理文本
  2. 正则解析：Thought / Action / Action Input / Final Answer
  3. Final Answer? ──► 返回结果
  4. Action? ──► tool_map[name].invoke(input) ──► Observation
  5. Observation 追加到 scratchpad，继续下一轮
```

关键设计：
- **三层正则解析**：标准格式 → 清理 Markdown 代码块 → 兜底包装为 `{"input": raw_text}`
- **容错机制**：连续 3 次解析失败才终止，工具执行错误作为 Observation 反馈供 Agent 自我修正
- **tool_map 动态调度**：`{tool.name: tool for tool in AGENT_TOOLS}`，增删工具无需改 Agent 代码
- **质量迭代闭环**：`run_with_quality_loop()` 自动执行 生成 → 评估 → 注入反馈 → 重新生成

---

## 项目结构

```
├── config/
│   ├── prompts.py            # Prompt 模板管理（含 ReAct System Prompt）
│   └── settings.py           # 全局配置（代理、路径）
│
├── core/
│   ├── rag_engine.py         # RAG 引擎：混合检索 + Reranker + ChromaDB
│   ├── lc_chain.py           # TongyiChainManager：LCEL 双链封装
│   ├── llm_client.py         # LLM 工具函数（JSON 提取等）
│   ├── evaluator.py          # 测试用例质量评估器
│   └── agent/
│       ├── tools.py           # 5 个 LangChain Tool 定义 + AGENT_TOOLS 列表
│       └── react_agent.py     # ReAct Agent：手写推理循环 + 质量迭代
│
├── ui/
│   ├── main.py               # Streamlit 主界面
│   ├── sidebar.py            # 侧边栏配置
│   └── components.py         # UI 组件（表格渲染、导出）
│
├── tests/
│   ├── test_rag_engine.py    # RAG 引擎测试
│   ├── test_lc_chain.py      # LCEL 链测试
│   ├── test_tools.py         # 工具库测试
│   └── test_react_agent.py   # ReAct Agent 测试
│
├── data/
│   ├── vector_db/            # ChromaDB 向量数据库
│   └── raw_files/            # 上传的原始文档
│
├── 变更/                      # 每次升级的变更日志
├── requirements.txt
└── readme.md
```

---

## 技术栈

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| LLM | 通义千问 qwen3-max (DashScope API) | 阿里云大模型，替代原 Google Gemini |
| Embedding | text-embedding-v4 (1024维) | DashScope 向量化服务 |
| RAG 框架 | LangChain + ChromaDB | LCEL 链式调用 + 本地向量数据库 |
| 关键词检索 | rank-bm25 | BM25 算法，与向量检索互补 |
| 精排模型 | BAAI/bge-reranker-base | Cross-Encoder 重排序 |
| Agent | 手写 ReAct 循环 + LangChain Tool | 无额外框架依赖 |
| 前端 | Streamlit | 数据应用快速原型 |

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

DashScope API Key 从 [阿里云百炼平台](https://bailian.console.aliyun.com/) 获取，LLM + Embedding + 评估全部使用同一个 Key。

### 3. 启动应用

```bash
streamlit run ui/main.py
```

浏览器访问 http://localhost:8501 即可使用。

### 4. 代码调用（无需 UI）

```python
from core.agent.react_agent import TestCaseReActAgent

agent = TestCaseReActAgent(max_steps=8, target_score=70)

# 基础调用
result = agent.run("请根据以下PRD生成测试用例：用户登录功能...")
print(result.final_answer)

# 带质量迭代
result = agent.run_with_quality_loop(prd_text="用户登录功能需求...")
print(f"质量分数: {result.quality_score}")
```

---

## 测试

```bash
# 运行全部测试（需要 DASHSCOPE_API_KEY 环境变量）
python tests/test_rag_engine.py
python tests/test_lc_chain.py
python tests/test_tools.py
python tests/test_react_agent.py
```

| 测试文件 | 覆盖范围 | 结果 |
|---------|---------|------|
| test_rag_engine.py | RAG 混合检索、Reranker、ChromaDB | 16 通过 / 1 跳过 |
| test_lc_chain.py | LCEL 链、ChatTongyi、多轮对话 | 24 通过 |
| test_tools.py | 5 个 LangChain Tool 调用 | 20 通过 |
| test_react_agent.py | ReAct 循环、正则解析、质量迭代 | 17 通过 |

---

## 变更日志

| 阶段 | 文档 | 核心内容 |
|------|------|---------|
| Phase 1 | [001_RAG层升级_混合检索+重排序.md](变更/001_RAG层升级_混合检索+重排序.md) | 向量+BM25 混合检索、RRF 融合、BGE-Reranker |
| Phase 2 | [002_LLM层升级_通义千问LCEL+DashScope替代Gemini.md](变更/002_LLM层升级_通义千问LCEL+DashScope替代Gemini.md) | ChatTongyi LCEL 链、DashScope Embedding |
| Phase 3 | [003_工具库_LangChain_Tools封装.md](变更/003_工具库_LangChain_Tools封装.md) | 5 个 @tool 封装、Evaluator 升级 |
| Phase 4 | [004_ReAct_Agent_手写推理循环.md](变更/004_ReAct_Agent_手写推理循环.md) | 手写 ReAct 循环、三层解析、质量迭代闭环 |

---

## 致谢

本项目原始版本来源于字节跳动训练营结题项目，在此基础上进行了完整的架构升级。
