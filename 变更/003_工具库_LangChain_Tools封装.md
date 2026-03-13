# 变更记录 003 — 工具库：LangChain Tools 封装

- **日期**：2026-03-13
- **阶段**：Phase 3
- **涉及文件**：`core/agent/__init__.py`（新建）、`core/agent/tools.py`（新建）、`core/evaluator.py`（升级）

---

## 一、变更背景

Phase 2 完成了 LLM 层的 LCEL 化改造，但所有能力（RAG 检索、文件解析、用例生成、质量评估）仍以"硬编码调用"方式串联，无法被 Agent 动态选择和组合。

**Phase 3 目标**：
- 将核心业务能力封装为 LangChain `Tool` 对象
- 为 Phase 4 的 ReAct Agent 提供标准化的工具集接口
- 顺带将 `core/evaluator.py` 中残留的 `google.generativeai` 替换为 `TongyiChainManager`

---

## 二、具体变更

### 2.1 新建 `core/agent/__init__.py`

标识 `core/agent/` 为 Python 包，后续 Phase 4~6 的 `react_agent.py`、`memory.py`、`graph.py` 均在此目录下新增。

---

### 2.2 新建 `core/agent/tools.py`（核心新文件）

封装 5 个 LangChain Tool，统一使用 `DASHSCOPE_API_KEY` 环境变量，无需额外配置。

| 工具名 | 封装的现有能力 | 关键参数 |
|--------|--------------|---------|
| `rag_search` | `RAGEngine.search_context()` | `query: str` |
| `add_knowledge` | `RAGEngine.add_knowledge()` | `content: str`, `source_name: str` |
| `parse_file` | `TongyiChainManager.parse_file()` | `file_path: str`, `prompt: str` |
| `generate_test_cases` | `TongyiChainManager.generate()` | `prd_text: str`, `rag_context: str` |
| `evaluate_test_quality` | `Evaluator.evaluate_cases()` | `prd_text: str`, `test_cases_json: str` |

**所有工具均用 `@tool` 装饰器定义**，LangChain 自动生成 JSON Schema（供 Agent 推理时读取工具说明）：

```python
from langchain_core.tools import tool

@tool
def rag_search(query: str) -> str:
    """在测试知识库中检索与查询最相关的内容。..."""
    context, _ = _rag_engine().search_context(query)
    return context if context else "知识库中未找到相关内容。"
```

**工具注册列表**（Phase 4 直接使用）：
```python
AGENT_TOOLS = [
    rag_search, add_knowledge, parse_file,
    generate_test_cases, evaluate_test_quality
]
```

---

### 2.3 升级 `core/evaluator.py`

| 项目 | 旧版 | 新版 |
|------|------|------|
| LLM 依赖 | `google.generativeai` (Gemini) | `TongyiChainManager` (通义千问) |
| `__init__` 参数 | `api_key`（Gemini Key） | `api_key`（DashScope Key，支持从环境变量读取） |
| `evaluate_cases` 签名 | `(self, model_name, prd_text, ...)` | `(self, prd_text, ..., model_name=None)` |
| model_name | 决定实际调用的模型 | 废弃参数，保留以向后兼容 |

---

## 三、文件结构变化

```
core/
  agent/                ← 新增目录
    __init__.py         ← 新增
    tools.py            ← 新增（核心文件）
  evaluator.py          ← 升级（移除 Gemini）
  lc_chain.py           ← 不变
  llm_client.py         ← 不变
  rag_engine.py         ← 不变
```

---

## 四、影响范围

| 模块 | 是否受影响 | 说明 |
|------|-----------|------|
| `core/agent/tools.py` | 是（新建） | Phase 3 核心文件 |
| `core/agent/__init__.py` | 是（新建） | 包标识文件 |
| `core/evaluator.py` | 是 | 移除 google.generativeai，改用通义千问 |
| `core/rag_engine.py` | 否 | 直接复用，接口不变 |
| `core/lc_chain.py` | 否 | 直接复用 |
| `core/llm_client.py` | 否 | 直接复用 |
| `ui/main.py` | 否 | 工具层对 UI 透明 |
| `requirements.txt` | 否 | langchain-core 已包含 Tool 支持 |

---

## 五、测试结果

```
通过: 20  失败: 0  跳过: 0  共: 20
```

所有测试全部通过，包括真实 API 调用验证：
- `rag_search` 成功调用（知识库为空时返回兜底提示）
- `generate_test_cases` 成功生成 2300+ 字符的测试用例 JSON
- `evaluate_test_quality` 成功返回评估报告，质量分数: 60

---

## 六、测试过程中遇到的问题

### 问题 1：`rag_search` 调用了不存在的 `top_k` 参数

**现象**：运行测试时终端输出：
```
rag_search 失败: RAGEngine.search_context() got an unexpected keyword argument 'top_k'
```
测试因工具的异常兜底逻辑（返回固定字符串）而"假通过"。

**原因**：`tools.py` 中写了 `engine.search_context(query, top_k=5)`，但 `search_context` 方法签名为 `(self, query, use_history=True, use_knowledge=True)`，不接受 `top_k`。

**解决过程**：
1. 用 `grep` 定位 `search_context` 实际方法签名（line 391）
2. 将调用改为 `engine.search_context(query)`（不传 `top_k`）
3. 重新运行测试，错误消除

---

### 问题 2：`search_context` 返回元组，工具直接当字符串使用

**现象**：修复问题 1 后，`rag_search` 测试从"假通过"变为真正的 `[FAIL]`，断言报空字符串错误。

**原因**：`search_context` 返回 `(context_str, sources_list)` 元组，而工具代码直接用了 `result = engine.search_context(query)`，对元组做真值判断永远为 `True`（即使内容为空），最终返回的是元组对象而非字符串。

**解决过程**：
1. 读取 `search_context` 方法末尾（line 431）确认返回类型为元组
2. 将工具代码改为解包：`context, _ = engine.search_context(query)`
3. 对 `context` 字符串做空值判断，知识库为空时返回兜底提示
4. 重新运行，20/20 全部通过
