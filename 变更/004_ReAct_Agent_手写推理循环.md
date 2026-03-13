# 变更记录 004 — ReAct Agent：手写推理循环

- **日期**：2026-03-13
- **阶段**：Phase 4
- **涉及文件**：`core/agent/react_agent.py`（新建）、`config/prompts.py`（修改）

---

## 一、变更背景

Phase 1~3 完成了 RAG 混合检索、LCEL 链、5 个 LangChain Tool 的封装，但核心调用逻辑仍是硬编码的：
```
用户输入 → 固定调用 LLM → 返回结果
```

这不是真正的 Agent。Phase 4 引入 ReAct（Reasoning + Acting）模式：
```
用户输入 → Thought（推理）→ Action（调用工具）→ Observation（观察结果）→ 循环 → Final Answer
```

**方案选择：手写 ReAct 循环（方案 D）**

| 对比方案 | 选择理由 |
|---------|---------|
| AgentExecutor | 已被 LangChain 标记为 legacy，黑盒不利于面试讲解 |
| LangGraph | 需额外依赖，ChatTongyi 的 bind_tools 兼容性不确定 |
| **手写循环** | 教育价值最高，面试可逐行解释，零新增依赖 |

---

## 二、具体变更

### 2.1 修改 `config/prompts.py`

新增两个 Prompt 模板：

**REACT_SYSTEM_PROMPT** — ReAct 推理格式
- 定义 `Thought / Action / Action Input / Observation / Final Answer` 五个关键词
- 占位符 `{tool_descriptions}` 运行时从 AGENT_TOOLS 自动生成
- 占位符 `{max_steps}` 控制最大推理步数
- 5 条严格规则：单次单工具、不编造 Observation 等

**REACT_QUALITY_LOOP_PROMPT** — 质量迭代反馈
- 注入评估报告的 coverage_gap、logic_issues、suggestions
- 指导 LLM 针对性改进测试用例

---

### 2.2 新建 `core/agent/react_agent.py`（核心文件）

#### 数据结构

```python
@dataclass
class AgentStep:
    step_number: int      # 步骤序号
    thought: str          # 推理内容
    action: str | None    # 工具名（None = Final Answer）
    action_input: dict    # 工具参数
    observation: str      # 工具返回结果

@dataclass
class AgentResult:
    final_answer: str           # 最终回答
    steps: list[AgentStep]      # 完整推理过程
    iterations: int             # 实际推理轮数
    quality_score: int | None   # 质量分数（质量迭代时有值）
    quality_report: dict | None # 完整评估报告
```

#### TestCaseReActAgent 类

| 方法 | 用途 |
|------|------|
| `__init__(api_key, max_steps, target_score, max_retries)` | 初始化 LLM + 工具映射表 |
| `run(user_request) → AgentResult` | 核心 ReAct 循环 |
| `run_with_quality_loop(prd_text, rag_context) → AgentResult` | 带质量迭代的端到端流程 |
| `_build_system_prompt()` | 自动生成含工具描述的 System Prompt |
| `_parse_action(llm_output)` | 三层正则解析 LLM 输出 |
| `_execute_tool(name, input)` | 通过 tool_map 动态调度工具 |
| `_format_step(step)` | 格式化推理记录为 scratchpad 文本 |

#### ReAct 循环流程（`run` 方法）

```
1. _build_system_prompt() 组装含工具描述的 Prompt
2. 循环（最多 max_steps 次）：
   a. chain_mgr.generate(scratchpad + user_request)
   b. _parse_action(output) 解析
   c. Final Answer? → 返回结果
   d. Action? → _execute_tool() → Observation 追加到 scratchpad
   e. 解析失败? → 容错提示，连续 3 次则终止
3. 超出步数 → 返回当前最佳结果
```

#### 正则解析策略（`_parse_action`）

```python
# 三层解析优先级
1. Final Answer: → 返回 (thought, None, None)
2. Action: + Action Input: → 返回 (thought, name, dict)
   - JSON 解析失败 → 清理 markdown 代码块重试
   - 仍失败 → 兜底包装为 {"input": raw_text}
3. 都不匹配 → 返回 (__PARSE_FAILED__, None)
```

#### 质量迭代闭环（`run_with_quality_loop`）

```
初始 PRD → run() 生成用例 → evaluate_test_quality 评估
                 ↓
         score >= target_score? → 返回
                 ↓ (否)
         注入 REACT_QUALITY_LOOP_PROMPT → run() 重新生成
                 ↓
         最多 max_retries 次 → 返回最佳结果
```

---

## 三、影响范围

| 模块 | 是否受影响 | 说明 |
|------|-----------|------|
| `core/agent/react_agent.py` | 是（新建） | Phase 4 核心文件 |
| `config/prompts.py` | 是 | 新增 2 个 Prompt 模板 |
| `core/agent/tools.py` | 否 | 直接复用 AGENT_TOOLS |
| `core/lc_chain.py` | 否 | 通过 generate() 调用 |
| `core/evaluator.py` | 否 | 通过 evaluate_test_quality 工具间接调用 |
| `requirements.txt` | 否 | 零新增依赖 |
| `ui/main.py` | 否 | Agent 层对 UI 透明 |

---

## 四、使用示例

```python
from core.agent.react_agent import TestCaseReActAgent

agent = TestCaseReActAgent(max_steps=8, target_score=70)

# 方式 1: 基础 ReAct 循环
result = agent.run("请根据以下PRD生成测试用例：用户登录功能...")
print(result.final_answer)
for step in result.steps:
    print(f"Step {step.step_number}: {step.action or 'Final Answer'}")

# 方式 2: 带质量迭代
result = agent.run_with_quality_loop(prd_text="用户登录功能...")
print(f"质量分数: {result.quality_score}")
```

---

## 五、面试讲解要点

1. **ReAct 是什么**：Reasoning + Acting，LLM 交替"思考"和"行动"，每步行动的结果作为下一步思考的输入
2. **为什么手写**：AgentExecutor 是黑盒且已 deprecated；手写让我理解了 Prompt 工程 + 正则解析 + 工具调度的完整链路
3. **核心难点**：LLM 输出格式不稳定，需要三层正则 + 容错机制；Prompt 越长越容易格式漂移
4. **质量闭环**：不只是"生成"，而是"生成-评估-改进"的自动迭代，模拟了真实 QA Review 流程
5. **工具层解耦**：用 `tool_map` 字典动态调度，增删工具不需要改 Agent 代码

---

## 六、测试结果

```
通过: 17  失败: 0  跳过: 0  共: 17
```

所有测试全部通过，包括：
- 5 项正则解析单元测试（标准格式、中文冒号、Markdown 代码块、异常格式）
- 2 项 System Prompt 生成验证
- 2 项工具调度验证（已知工具、未知工具降级）
- 3 项 `run()` 集成测试：Agent 自主执行了 `rag_search → generate_test_cases → evaluate_test_quality` 调用链
- 1 项 `run_with_quality_loop()` 端到端测试：质量分数 92/100

---

## 七、测试过程中遇到的问题

### 问题 1：Agent 推理中工具参数格式不匹配

**现象**：测试日志中出现：
```
工具 'generate_test_cases' 执行失败: 1 validation error for generate_test_cases
prd_text: Field required [type=missing]
```

**原因**：LLM 有时将 Action Input 生成为嵌套 JSON 或将多个参数合并为单个 `"input"` 字段，导致 Pydantic schema 校验失败（缺少必填字段 `prd_text`）。

**解决过程**：
1. 在 `_parse_action` 中增加 JSON 解析的兜底逻辑：清理 markdown 代码块、包装为 `{"input": raw_text}`
2. 在 `_execute_tool` 中用 try/except 捕获 Pydantic ValidationError，返回错误信息作为 Observation
3. Agent 收到错误 Observation 后自动调整参数格式重试（ReAct 循环的自愈能力）
4. 容错机制确保连续 3 次解析失败才终止，而不是单次失败就崩溃

**关键启发**：这正是 ReAct 模式的优势 — Agent 能从工具执行的错误反馈中学习并自我修正。
