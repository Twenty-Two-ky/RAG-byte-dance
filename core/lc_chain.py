"""
core/lc_chain.py — Phase 2: LLM 调用层升级

用 LangChain LCEL 管道语法替代原始 google-generativeai SDK 调用。
底层模型: 通义千问（阿里云 DashScope），与 Embedding 共用同一 API Key。

主要提供:
  - TongyiChainManager: 封装两条 LCEL 链
      - simple_chain: 单轮生成（摘要、文档解析指令）
      - chat_chain:   多轮对话（带历史记录）
  - parse_file(): 利用通义千问 Vision 解析图片/PDF
"""

import os
import base64
import logging

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 默认模型（可通过环境变量覆盖）
TONGYI_MODEL = os.environ.get("TONGYI_MODEL", "qwen3-max")
# 视觉解析用的多模态模型
TONGYI_VL_MODEL = os.environ.get("TONGYI_VL_MODEL", "qwen-vl-max")


class TongyiChainManager:
    """
    通义千问 LCEL 链管理器。

    使用 LangChain LCEL（| 管道语法）组装 Prompt → LLM → OutputParser 链。
    LLM 和 Embedding 共用同一个 DashScope API Key，无需多套 Key 管理。
    """

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: 阿里云 DashScope API Key。
                     为空时自动读取 DASHSCOPE_API_KEY 环境变量。
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要 DashScope API Key：传入 api_key 参数或设置 DASHSCOPE_API_KEY 环境变量")

        # 核心 LLM 实例（所有 Chain 共用）
        self.llm = ChatTongyi(
            model=TONGYI_MODEL,
            dashscope_api_key=self.api_key,
            max_tokens=8192,
        )

        # === Chain 1: 单轮生成链 ===
        # 用途: 摘要生成、文档质检、单次内容生成
        # LCEL 流: ChatPromptTemplate | LLM | StrOutputParser
        self.simple_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("human", "{content}")
            ])
            | self.llm
            | StrOutputParser()
        )

        # === Chain 2: 多轮对话链 ===
        # 用途: 带上下文的测试用例生成对话
        # LCEL 流: ChatPromptTemplate(含 MessagesPlaceholder) | LLM | StrOutputParser
        self.chat_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"TongyiChainManager 初始化完成，模型: {TONGYI_MODEL}")

    def generate(self, content: str, system_prompt: str) -> str:
        """
        单轮生成，无历史记录。用于摘要生成、文档解析结果处理等。

        Args:
            content:       输入内容（超过 8000 字符自动截断）
            system_prompt: 系统指令

        Returns:
            模型生成的文本
        """
        try:
            return self.simple_chain.invoke({
                "system_prompt": system_prompt,
                "content": content[:8000]
            })
        except Exception as e:
            logger.error(f"通义生成失败: {e}")
            return f"[生成失败] {str(e)}"

    def chat(self, user_input: str, history: list, system_prompt: str = "") -> tuple:
        """
        多轮对话，支持历史记录。

        history 格式兼容原 Gemini 格式（向后兼容 ui/main.py）：
            [{"role": "user",  "parts": ["..."]},
             {"role": "model", "parts": ["..."]}]

        Args:
            user_input:    本轮用户输入
            history:       历史对话列表（Gemini 格式）
            system_prompt: 系统指令

        Returns:
            (response_text, updated_history)
            updated_history 保持 Gemini 格式，追加本轮对话
        """
        # 将 Gemini 格式 history 转为 LangChain Message 对象
        lc_history = []
        for msg in history:
            role = msg.get("role", "")
            parts = msg.get("parts", [])
            content = parts[0] if isinstance(parts, list) and parts else msg.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role in ("model", "assistant"):
                lc_history.append(AIMessage(content=content))

        try:
            response = self.chat_chain.invoke({
                "system_prompt": system_prompt or "你是一个专业的AI助手。",
                "history": lc_history,
                "input": user_input
            })
        except Exception as e:
            logger.error(f"通义对话失败: {e}")
            response = f"[对话失败] {str(e)}"

        # 追加本轮对话，保持 Gemini 兼容格式
        updated_history = list(history) + [
            {"role": "user",  "parts": [user_input]},
            {"role": "model", "parts": [response]}
        ]
        return response, updated_history

    def parse_file(self, file_bytes: bytes, media_type: str, prompt: str) -> str:
        """
        使用通义千问 Vision（qwen-vl-max）解析图片或 PDF 内容。

        替代原 Gemini 多模态解析。
        图片通过 base64 编码传入；PDF 先提取文本再送入通义处理。

        Args:
            file_bytes:  文件原始字节
            media_type:  MIME 类型，如 "image/jpeg"、"application/pdf"
            prompt:      解析指令

        Returns:
            解析后的纯文本内容
        """
        try:
            if "pdf" in media_type:
                return self._parse_pdf(file_bytes, prompt)
            elif "image" in media_type:
                return self._parse_image(file_bytes, media_type, prompt)
            else:
                return file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"通义文件解析失败: {e}")
            return f"[解析失败] {str(e)}"

    def _parse_image(self, file_bytes: bytes, media_type: str, prompt: str) -> str:
        """使用 qwen-vl-max 解析图片"""
        import dashscope
        from dashscope import MultiModalConversation

        encoded = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{media_type};base64,{encoded}"

        messages = [{
            "role": "user",
            "content": [
                {"image": data_url},
                {"text": prompt}
            ]
        }]

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=TONGYI_VL_MODEL,
            messages=messages
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            raise RuntimeError(f"通义 Vision API 错误: {response.code} - {response.message}")

    def _parse_pdf(self, file_bytes: bytes, prompt: str) -> str:
        """PDF 提取文本后送通义处理"""
        try:
            import io
            # 尝试用 pdfplumber 提取文本
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return self.generate(
                    content=f"以下是PDF内容：\n{text[:6000]}",
                    system_prompt=prompt
                )
        except ImportError:
            pass
        # fallback: 直接把 PDF 字节作为文本解析
        return self.generate(
            content=file_bytes.decode("utf-8", errors="ignore")[:6000],
            system_prompt=prompt
        )


# 向后兼容别名（旧代码可能引用 ClaudeChainManager）
ClaudeChainManager = TongyiChainManager
