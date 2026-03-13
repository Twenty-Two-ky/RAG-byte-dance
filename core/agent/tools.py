"""
core/agent/tools.py — Phase 3: LangChain 工具库

将核心业务能力封装为 LangChain Tool 对象，供 Phase 4 的 ReAct Agent 按需调用。

工具列表:
  rag_search            — 知识库混合检索
  add_knowledge         — 向知识库添加文档
  parse_file            — 解析本地图片/PDF（qwen-vl-max Vision）
  generate_test_cases   — 根据 PRD 生成测试用例
  evaluate_test_quality — 评估测试用例质量

所有工具均从 DASHSCOPE_API_KEY 环境变量读取 API Key，无需额外配置。
"""

import os
import json
import mimetypes
import logging

from langchain_core.tools import tool

from core.rag_engine import RAGEngine
from core.lc_chain import TongyiChainManager
from core.evaluator import Evaluator
from config.prompts import PromptManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 私有工厂函数（避免每个 tool 重复读取环境变量）
# ─────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，无法调用工具")
    return key


def _rag_engine() -> RAGEngine:
    return RAGEngine(dashscope_api_key=_api_key())


def _chain_manager() -> TongyiChainManager:
    return TongyiChainManager(api_key=_api_key())


# ─────────────────────────────────────────────────────────
# Tool 1: RAG 混合检索
# ─────────────────────────────────────────────────────────

@tool
def rag_search(query: str) -> str:
    """
    在测试知识库中检索与查询最相关的内容。
    返回历史测试规范、用例模板、业务规则等参考信息，用于指导测试用例生成。
    """
    try:
        engine = _rag_engine()
        context, _ = engine.search_context(query)
        return context if context else "知识库中未找到相关内容。"
    except Exception as e:
        logger.error(f"rag_search 失败: {e}")
        return f"[检索失败] {str(e)}"


# ─────────────────────────────────────────────────────────
# Tool 2: 添加知识到知识库
# ─────────────────────────────────────────────────────────

@tool
def add_knowledge(content: str, source_name: str = "agent_upload") -> str:
    """
    将新的测试规范、技术文档或业务规则添加到知识库中，供后续检索使用。
    content: 要添加的文档内容（纯文本）。
    source_name: 文档来源标识，默认 'agent_upload'。
    """
    try:
        engine = _rag_engine()
        engine.add_knowledge(content, source_name)
        return f"已成功将文档 '{source_name}' 添加到知识库。"
    except Exception as e:
        logger.error(f"add_knowledge 失败: {e}")
        return f"[添加失败] {str(e)}"


# ─────────────────────────────────────────────────────────
# Tool 3: 解析本地文件（图片 / PDF）
# ─────────────────────────────────────────────────────────

@tool
def parse_file(file_path: str, prompt: str = "请提取并整理文件中的所有关键信息") -> str:
    """
    解析本地图片（jpg/png 等）或 PDF 文件，提取其中的文字、布局和业务信息。
    file_path: 本地文件的绝对路径。
    prompt:    解析指令，默认提取所有关键信息。
    """
    try:
        if not os.path.exists(file_path):
            return f"[文件不存在] {file_path}"

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        media_type, _ = mimetypes.guess_type(file_path)
        if media_type is None:
            # 根据扩展名做兜底判断
            ext = os.path.splitext(file_path)[1].lower()
            media_type = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(ext, "application/octet-stream")

        chain_mgr = _chain_manager()
        return chain_mgr.parse_file(file_bytes, media_type, prompt)
    except Exception as e:
        logger.error(f"parse_file 失败: {e}")
        return f"[解析失败] {str(e)}"


# ─────────────────────────────────────────────────────────
# Tool 4: 生成测试用例
# ─────────────────────────────────────────────────────────

@tool
def generate_test_cases(prd_text: str, rag_context: str = "") -> str:
    """
    根据产品需求文档（PRD）和可选的知识库上下文，生成 JSON 格式的测试用例列表。
    prd_text:    需求文档的文本内容。
    rag_context: 可选，知识库检索到的参考规范（由 rag_search 提供）。
    """
    try:
        chain_mgr = _chain_manager()

        if rag_context:
            content = (
                f"【参考规范（知识库检索结果）】:\n{rag_context}\n\n"
                f"【PRD 需求文档】:\n{prd_text}"
            )
        else:
            content = f"【PRD 需求文档】:\n{prd_text}"

        return chain_mgr.generate(
            content=content,
            system_prompt=PromptManager.CORE_SYSTEM_PROMPT
        )
    except Exception as e:
        logger.error(f"generate_test_cases 失败: {e}")
        return f"[生成失败] {str(e)}"


# ─────────────────────────────────────────────────────────
# Tool 5: 评估测试用例质量
# ─────────────────────────────────────────────────────────

@tool
def evaluate_test_quality(prd_text: str, test_cases_json: str) -> str:
    """
    评估 AI 生成的测试用例质量，返回包含分数（0-100）、覆盖度分析、
    逻辑问题和改进建议的 JSON 格式报告。
    prd_text:        原始需求文档文本。
    test_cases_json: 待评估的测试用例，JSON 字符串格式。
    """
    try:
        evaluator = Evaluator(api_key=_api_key())
        report = evaluator.evaluate_cases(
            prd_text=prd_text,
            current_cases=test_cases_json,
        )
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"evaluate_test_quality 失败: {e}")
        return json.dumps({
            "score": 0,
            "summary": f"评估失败: {str(e)}",
            "coverage_gap": [], "logic_issues": [],
            "duplicates": [], "suggestions": []
        }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────
# 工具注册列表（供 Phase 4 Agent 直接使用）
# ─────────────────────────────────────────────────────────

AGENT_TOOLS = [
    rag_search,
    add_knowledge,
    parse_file,
    generate_test_cases,
    evaluate_test_quality,
]
