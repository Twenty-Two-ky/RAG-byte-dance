"""
core/agent/react_agent.py — Phase 4: 手写 ReAct 推理循环

核心机制（面试讲解要点）：
1. 将 ReAct 格式 Prompt 发送给 LLM（qwen3-max）
2. 用正则解析 LLM 输出中的 Thought / Action / Action Input / Final Answer
3. 通过 LangChain Tool.invoke() 执行工具，获取 Observation
4. 将 Observation 拼接回 Prompt，继续下一轮推理
5. 直到 LLM 输出 Final Answer 或达到最大步数

为什么手写而不用 AgentExecutor：
- AgentExecutor 已被 LangChain 标记为 legacy
- 手写可以逐行解释 ReAct 的 Thought-Action-Observation 循环
- 更利于面试讲解和对 Agent 原理的理解
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field

from core.lc_chain import TongyiChainManager
from core.agent.tools import AGENT_TOOLS
from core.llm_client import extract_json_from_text
from config.prompts import PromptManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """单步推理记录"""
    step_number: int
    thought: str
    action: str = None          # None 表示 Final Answer
    action_input: dict = None
    observation: str = None


@dataclass
class AgentResult:
    """Agent 执行结果"""
    final_answer: str
    steps: list = field(default_factory=list)
    iterations: int = 0
    quality_score: int = None
    quality_report: dict = None


# ─────────────────────────────────────────────────────────
# 正则模式（解析 LLM 输出）
# ─────────────────────────────────────────────────────────

# 匹配 Final Answer
RE_FINAL_ANSWER = re.compile(
    r"Final\s*Answer\s*[:：]\s*(.+)",
    re.DOTALL
)

# 匹配 Action + Action Input
RE_ACTION = re.compile(
    r"Action\s*[:：]\s*(.+?)\s*\n\s*Action\s*Input\s*[:：]\s*(.+)",
    re.DOTALL
)

# 匹配 Thought
RE_THOUGHT = re.compile(
    r"Thought\s*[:：]\s*(.+?)(?=\nAction|\nFinal\s*Answer|$)",
    re.DOTALL
)


# ─────────────────────────────────────────────────────────
# ReAct Agent
# ─────────────────────────────────────────────────────────

class TestCaseReActAgent:
    """
    手写 ReAct 推理循环 Agent。

    使用 AGENT_TOOLS 中的 5 个 LangChain Tool，
    通过 Thought → Action → Observation 循环自主完成测试用例生成任务。
    """

    def __init__(self, api_key: str = None, max_steps: int = 8,
                 target_score: int = 70, max_retries: int = 2):
        """
        Args:
            api_key:      DashScope API Key
            max_steps:    单次 run 最大推理步数
            target_score: 质量迭代的目标分数（0-100）
            max_retries:  质量不达标时的最大重试次数
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要 DASHSCOPE_API_KEY")

        self.max_steps = max_steps
        self.target_score = target_score
        self.max_retries = max_retries

        # LLM 调用层
        self.chain_mgr = TongyiChainManager(api_key=self.api_key)

        # 工具名 → 工具对象 的映射（动态调度的关键）
        self.tool_map = {tool.name: tool for tool in AGENT_TOOLS}

        logger.info(f"ReAct Agent 初始化完成，工具: {list(self.tool_map.keys())}")

    # ─── 公共接口 ───

    def run(self, user_request: str) -> AgentResult:
        """
        执行一次完整的 ReAct 推理循环。

        Args:
            user_request: 用户任务描述

        Returns:
            AgentResult 包含最终回答和推理过程
        """
        system_prompt = self._build_system_prompt()
        steps = []
        scratchpad = ""
        parse_failures = 0  # 连续解析失败计数

        for i in range(1, self.max_steps + 1):
            # 组装当前轮的输入：用户请求 + 历史 scratchpad
            content = f"用户任务：{user_request}\n\n{scratchpad}".strip()

            # 调用 LLM
            llm_output = self.chain_mgr.generate(
                content=content,
                system_prompt=system_prompt
            )
            logger.debug(f"Step {i} LLM 输出:\n{llm_output[:300]}")

            # 解析 LLM 输出
            thought, action_name, action_input = self._parse_action(llm_output)

            if action_name is None:
                # Final Answer — 推理结束
                final_text = self._extract_final_answer(llm_output)
                steps.append(AgentStep(
                    step_number=i, thought=thought,
                    action=None, action_input=None, observation=None
                ))
                return AgentResult(
                    final_answer=final_text,
                    steps=steps,
                    iterations=i
                )

            if action_name == "__PARSE_FAILED__":
                # 解析失败，容错处理
                parse_failures += 1
                if parse_failures >= 3:
                    logger.warning("连续 3 次解析失败，强制终止")
                    return AgentResult(
                        final_answer=llm_output,
                        steps=steps,
                        iterations=i
                    )
                # 提示 LLM 重新格式化
                scratchpad += (
                    f"\n\nThought: {thought}\n"
                    f"Observation: 格式错误，请严格按照 "
                    f"Thought/Action/Action Input 或 Final Answer 格式回答。\n"
                )
                continue

            # 重置连续失败计数
            parse_failures = 0

            # 执行工具
            observation = self._execute_tool(action_name, action_input)

            # 记录步骤
            step = AgentStep(
                step_number=i,
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation[:2000]  # 截断过长的观察结果
            )
            steps.append(step)

            # 将本轮结果追加到 scratchpad
            scratchpad += self._format_step(step)

        # 超出最大步数，返回已有信息
        logger.warning(f"达到最大步数 {self.max_steps}，强制返回")
        return AgentResult(
            final_answer=f"[达到最大推理步数 {self.max_steps}]\n{scratchpad}",
            steps=steps,
            iterations=self.max_steps
        )

    def run_with_quality_loop(self, prd_text: str,
                               rag_context: str = "") -> AgentResult:
        """
        带质量迭代的端到端流程：
        1. 用 ReAct 循环生成测试用例
        2. 自动评估质量
        3. 分数不达标 → 注入评估建议 → 重新生成
        4. 重复直到达标或达到 max_retries

        Args:
            prd_text:    PRD 需求文本
            rag_context: 可选的 RAG 检索上下文

        Returns:
            AgentResult，含 quality_score 和 quality_report
        """
        # 构建初始任务
        task = f"请根据以下 PRD 需求文档生成完整的测试用例（JSON 格式）。\n\n{prd_text}"
        if rag_context:
            task += f"\n\n参考规范：\n{rag_context}"

        best_result = None

        for attempt in range(1, self.max_retries + 2):  # +1 是初始生成
            logger.info(f"质量迭代第 {attempt} 轮")

            result = self.run(task)

            # 从 final_answer 中提取 JSON 测试用例
            cases_json = extract_json_from_text(result.final_answer)

            if not cases_json:
                logger.warning(f"第 {attempt} 轮未提取到 JSON 用例")
                best_result = result
                break

            # 评估质量
            cases_str = json.dumps(cases_json, ensure_ascii=False)
            eval_tool = self.tool_map.get("evaluate_test_quality")

            if eval_tool:
                eval_result = eval_tool.invoke({
                    "prd_text": prd_text,
                    "test_cases_json": cases_str
                })
                try:
                    report = json.loads(eval_result)
                except (json.JSONDecodeError, TypeError):
                    report = {"score": 0, "summary": eval_result}
            else:
                report = {"score": 100, "summary": "评估工具不可用，跳过"}

            score = report.get("score", 0)
            result.quality_score = score
            result.quality_report = report
            best_result = result

            logger.info(f"第 {attempt} 轮质量分数: {score}/{self.target_score}")

            if score >= self.target_score:
                logger.info("质量达标，结束迭代")
                break

            if attempt > self.max_retries:
                logger.info("达到最大重试次数，返回最佳结果")
                break

            # 注入改进指令，开始下一轮
            feedback = PromptManager.REACT_QUALITY_LOOP_PROMPT.format(
                score=score,
                target_score=self.target_score,
                coverage_gap=report.get("coverage_gap", []),
                logic_issues=report.get("logic_issues", []),
                suggestions=report.get("suggestions", [])
            )
            task = f"{feedback}\n\n原始 PRD：\n{prd_text}"

        return best_result

    # ─── 内部方法 ───

    def _build_system_prompt(self) -> str:
        """组装 ReAct System Prompt，自动填充工具描述。"""
        tool_desc_lines = []
        for t in AGENT_TOOLS:
            # 提取工具参数 schema
            if hasattr(t, "args_schema") and t.args_schema:
                schema = t.args_schema.model_json_schema()
                props = schema.get("properties", {})
                params = ", ".join(
                    f"{k}: {v.get('type', 'any')}"
                    for k, v in props.items()
                )
            else:
                params = "无参数"

            tool_desc_lines.append(
                f"- {t.name}({params}): {t.description}"
            )

        tool_descriptions = "\n".join(tool_desc_lines)

        return PromptManager.REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            max_steps=self.max_steps
        )

    def _parse_action(self, llm_output: str):
        """
        解析 LLM 输出，提取 Thought / Action / Action Input 或 Final Answer。

        Returns:
            (thought, action_name, action_input) — 正常工具调用
            (thought, None, None) — Final Answer
            (thought, "__PARSE_FAILED__", None) — 解析失败
        """
        # 提取 Thought
        thought_match = RE_THOUGHT.search(llm_output)
        thought = thought_match.group(1).strip() if thought_match else llm_output[:200]

        # 检查 Final Answer
        if RE_FINAL_ANSWER.search(llm_output):
            return thought, None, None

        # 匹配 Action + Action Input
        action_match = RE_ACTION.search(llm_output)
        if action_match:
            action_name = action_match.group(1).strip()
            raw_input = action_match.group(2).strip()

            # 尝试解析 JSON 参数
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError:
                # 清理 markdown 代码块后重试
                cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw_input).strip()
                try:
                    action_input = json.loads(cleaned)
                except json.JSONDecodeError:
                    # 兜底：把原始文本作为第一个参数的值
                    action_input = {"input": raw_input}

            return thought, action_name, action_input

        # 两者都匹配不到 → 解析失败
        return thought, "__PARSE_FAILED__", None

    def _extract_final_answer(self, llm_output: str) -> str:
        """从 LLM 输出中提取 Final Answer 后的内容。"""
        match = RE_FINAL_ANSWER.search(llm_output)
        if match:
            return match.group(1).strip()
        return llm_output

    def _execute_tool(self, action_name: str, action_input: dict) -> str:
        """通过名称查找 Tool 并执行。"""
        tool = self.tool_map.get(action_name)
        if not tool:
            available = list(self.tool_map.keys())
            return f"[工具不存在] '{action_name}' 不在可用工具列表中: {available}"

        try:
            result = tool.invoke(action_input)
            return str(result)
        except Exception as e:
            logger.error(f"工具 '{action_name}' 执行失败: {e}")
            return f"[工具执行失败] {action_name}: {str(e)}"

    def _format_step(self, step: AgentStep) -> str:
        """将单步推理记录格式化为 scratchpad 文本。"""
        text = f"\nThought: {step.thought}\n"
        if step.action:
            input_str = json.dumps(step.action_input, ensure_ascii=False)
            text += f"Action: {step.action}\nAction Input: {input_str}\n"
            text += f"Observation: {step.observation}\n"
        return text
