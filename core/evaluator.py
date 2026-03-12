import google.generativeai as genai
import json
import sys
import os

# 路径适配
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.prompts import PromptManager
from core.llm_client import extract_json_from_text

class Evaluator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Evaluator 需要 API Key")
        self.api_key = api_key
        
    def evaluate_cases(self, model_name, prd_text, current_cases, rag_context=None, golden_cases_content=None):
        """
        执行测试用例评估
        
        Args:
            model_name: 使用的模型 (建议使用 Pro 版本以获得更好的逻辑推理能力)
            prd_text: 原始需求文本
            current_cases: 当前 AI 生成的测试用例 (List/Dict 或 JSON String)
            rag_context: RAG 检索到的规范上下文
            golden_cases_content: (可选) 人工上传的标准用例内容
            
        Returns:
            dict: 包含分数、建议等信息的结构化报告
        """
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name, 
                system_instruction=PromptManager.EVALUATOR_SYSTEM_PROMPT
            )
            
            # 构建 Prompt
            prompt_text = PromptManager.get_evaluation_prompt(
                prd_text, 
                current_cases, 
                rag_text=rag_context, 
                golden_cases_text=golden_cases_content
            )
            
            # 调用模型
            # 评估任务通常不需要流式传输，一次性生成即可
            response = model.generate_content(prompt_text)
            
            # 解析结果
            report_json = extract_json_from_text(response.text)
            
            if not report_json:
                # 兜底返回
                return {
                    "score": 0,
                    "summary": "AI 未能生成有效的 JSON 格式报告，请重试。",
                    "coverage_gap": [],
                    "logic_issues": [],
                    "duplicates": [],
                    "suggestions": [f"原始响应: {response.text[:200]}..."]
                }
                
            return report_json
            
        except Exception as e:
            print(f"评估过程出错: {e}")
            return {
                "score": 0,
                "summary": f"评估服务发生错误: {str(e)}",
                "coverage_gap": [],
                "logic_issues": [],
                "duplicates": [],
                "suggestions": ["请检查网络连接或 API Key 配额"]
            }