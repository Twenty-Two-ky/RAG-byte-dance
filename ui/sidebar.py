import streamlit as st
from config.settings import load_config, save_config
from core.llm_client import get_available_models

def render_sidebar():
    """渲染侧边栏并返回配置"""
    config = load_config()
    
    with st.sidebar:
        st.header("🔑 配置中心")
        
        # 1. API Key 输入
        api_key = st.text_input("Gemini API Key", value=config.get('api_key', ''), type="password")
        
        # 保存按钮
        if st.button("💾 保存配置"):
            save_config({'api_key': api_key})
            st.success("配置已保存")

        st.divider()

        # 2. 动态模型选择
        selected_model = "models/gemini-1.5-flash" # 默认兜底
        if api_key:
            with st.spinner("正在联网获取模型列表..."):
                available_models = get_available_models(api_key)
            
            if available_models:
                # 智能默认选中 Pro，因为效果更好
                default_idx = 0
                for i, name in enumerate(available_models):
                    if "gemini-1.5-pro" in name and "latest" in name:
                        default_idx = i
                        break
                    elif "gemini-1.5-pro" in name: # 次优选择
                        default_idx = i
                
                selected_model = st.selectbox("🤖 选择模型", available_models, index=default_idx)
            else:
                st.warning("无法获取模型列表，将使用默认 Flash 模型")
                selected_model = "models/gemini-1.5-flash"
        else:
            st.info("请输入 API Key 以解锁高级模型选择")
            
        # 3. 清空按钮
        st.divider()
        if st.button("🗑️ 清空工作台", type="secondary"):
            # 清除 session_state
            for key in ['res_df', 'res_data', 'prd_cache']:
                if key in st.session_state:
                    st.session_state[key] = None
            st.rerun()
            
    return api_key, selected_model