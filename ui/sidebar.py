import os
import streamlit as st
from config.settings import load_config, save_config
from core.llm_client import get_available_models

def render_sidebar():
    """渲染侧边栏并返回配置"""
    config = load_config()

    with st.sidebar:
        st.header("配置中心")

        # 1. API Key 输入（DashScope）
        api_key = st.text_input(
            "DashScope API Key",
            value=config.get('dashscope_api_key', ''),
            type="password",
            help="从阿里云百炼平台获取: https://bailian.console.aliyun.com/"
        )

        # 保存按钮
        if st.button("保存配置"):
            save_config({'dashscope_api_key': api_key})
            st.success("配置已保存")

        # 设置环境变量（后端模块从 env 读取）
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key

        st.divider()

        # 2. 模型选择
        selected_model = "qwen3-max"
        if api_key:
            available_models = get_available_models(api_key)
            if available_models:
                selected_model = st.selectbox("选择模型", available_models, index=0)
        else:
            st.info("请输入 API Key 以开始使用")

        # 3. 清空按钮
        st.divider()
        if st.button("清空工作台", type="secondary"):
            for key in ['res_data', 'messages', 'gemini_history', 'prd_context',
                        'rag_context', 'processed_files', 'eval_report']:
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
            st.rerun()

    return api_key, selected_model
