"""
core/evaluator.py — Phase 3 升级版

将原 Gemini SDK 调用替换为 TongyiChainManager（通义千问），
与 LLM 层保持一致，统一使用 DASHSCOPE_API_KEY。

变更对照:
  __init__(api_key)          — api_key 现在是 DashScope Key
  evaluate_cases(model_name, ...) — model_name 参数废弃保留，内部改用通义千问
"""

import json
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.prompts import PromptManager
from core.llm_client import extract_json_from_text
from core.lc_chain import TongyiChainManager

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("Evaluator 需要 DashScope API Key：传入 api_key 或设置 DASHSCOPE_API_KEY 环境变量")

    def evaluate_cases(self, prd_text, current_cases, rag_context=None,
                       golden_cases_content=None, model_name=None):
        """
        执行测试用例评估。

        Args:
            prd_text:              原始需求文本
            current_cases:         当前 AI 生成的测试用例（List/Dict 或 JSON String）
            rag_context:           RAG 检索到的规范上下文（可选）
            golden_cases_content:  人工上传的标准用例内容（可选）
            model_name:            已废弃，保留以向后兼容

        Returns:
            dict: 包含分数、建议等信息的结构化报告
        """
        try:
            chain_mgr = TongyiChainManager(api_key=self.api_key)

            prompt_text = PromptManager.get_evaluation_prompt(
                prd_text,
                current_cases,
                rag_text=rag_context or "",
                golden_cases_text=golden_cases_content or ""
            )

            raw_response = chain_mgr.generate(
                content=prompt_text,
                system_prompt=PromptManager.EVALUATOR_SYSTEM_PROMPT
            )

            report_json = extract_json_from_text(raw_response)

            if not report_json:
                return {
                    "score": 0,
                    "summary": "AI 未能生成有效的 JSON 格式报告，请重试。",
                    "coverage_gap": [],
                    "logic_issues": [],
                    "duplicates": [],
                    "suggestions": [f"原始响应: {raw_response[:200]}..."]
                }

            return report_json

        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            return {
                "score": 0,
                "summary": f"评估服务发生错误: {str(e)}",
                "coverage_gap": [],
                "logic_issues": [],
                "duplicates": [],
                "suggestions": ["请检查 DASHSCOPE_API_KEY 是否配置正确"]
            }