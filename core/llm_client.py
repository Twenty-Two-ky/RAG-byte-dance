"""
core/llm_client.py — Phase 2 升级版

将原 Gemini SDK 调用全部替换为通义千问（阿里云 DashScope）。
通过 TongyiChainManager（LCEL 链）实现，LLM 与 Embedding 共用同一 DashScope Key。

变更对照:
  get_gemini_chat_response()  → get_tongyi_chat_response()
  generate_summary()          → 内部改用 TongyiChainManager.generate()
  get_available_models()      → 返回通义模型列表
  extract_json_from_text()    → 无变化（纯工具函数）
"""

import json
import re
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.prompts import PromptManager
from core.lc_chain import TongyiChainManager

logger = logging.getLogger(__name__)

# 可用的通义模型列表
AVAILABLE_MODELS = [
    "qwen3-max",
    "qwen3-plus",
    "qwen3-turbo",
    "qwen-max",
    "qwen-plus",
]


def get_available_models(api_key=None):
    """
    返回可用的通义模型列表。
    api_key 参数保留以向后兼容，不再使用。
    """
    return AVAILABLE_MODELS


def extract_json_from_text(text):
    """从 AI 回复中提取 JSON 代码块（无变化）"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        longest_match = max(matches, key=len)
        try:
            return json.loads(longest_match.strip())
        except json.JSONDecodeError:
            pass

    return None


def get_tongyi_chat_response(api_key, model_name, history, user_input, system_instruction=None):
    """
    支持上下文的通义千问对话接口。

    替代原 get_gemini_chat_response()，接口签名兼容。
    model_name 参数保留但不使用，实际模型由 TONGYI_MODEL 环境变量或默认值控制。

    Args:
        api_key:           DashScope API Key
        model_name:        已废弃（保留以兼容旧调用）
        history:           历史对话列表（Gemini 格式）
        user_input:        本轮用户输入
        system_instruction: 系统指令

    Returns:
        (response_text, updated_history)
    """
    try:
        chain_mgr = TongyiChainManager(api_key=api_key)
        return chain_mgr.chat(
            user_input=user_input,
            history=history,
            system_prompt=system_instruction or PromptManager.CORE_SYSTEM_PROMPT
        )
    except Exception as e:
        error_msg = f"模型调用出错: {str(e)}"
        logger.error(error_msg)
        return error_msg, history


def generate_summary(api_key, content, model_name=None):
    """
    生成内容摘要。

    Args:
        api_key:    DashScope API Key
        content:    待摘要的内容（JSON 或文本）
        model_name: 已废弃（保留以向后兼容）

    Returns:
        摘要字符串
    """
    try:
        chain_mgr = TongyiChainManager(api_key=api_key)
        return chain_mgr.generate(
            content=str(content)[:8000],
            system_prompt=PromptManager.SUMMARY_PROMPT
        ).strip()
    except Exception as e:
        logger.error(f"摘要生成失败: {e}")
        return "未命名业务文档"


# 向后兼容别名
get_gemini_chat_response = get_tongyi_chat_response
get_claude_chat_response = get_tongyi_chat_response
