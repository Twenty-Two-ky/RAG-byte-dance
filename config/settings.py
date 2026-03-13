import os
import json


# --- 1. 代理配置 ---
def setup_proxy():
    """
    设置网络代理。
    DashScope 是阿里云国内服务，通常不需要代理。
    仅当用户主动通过环境变量设置了代理时才使用。
    """
    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        print(f"检测到系统代理设置，保持使用。")
    else:
        print("未设置代理，DashScope API 将直连（国内网络无需代理）。")


# --- 2. 配置文件路径 ---
# 获取当前脚本的上级目录的绝对路径 (即项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_FILE = os.path.join(DATA_DIR, 'user_config.json')

# 确保 data 目录存在
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def load_config():
    config = {}

    # 1. 尝试从本地 JSON 加载
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"读取配置文件失败: {e}")

    # 2. 环境变量优先（Docker/云部署时通过环境变量注入）
    if os.environ.get("DASHSCOPE_API_KEY"):
        config['dashscope_api_key'] = os.environ.get("DASHSCOPE_API_KEY")

    # 3. 向后兼容：旧 api_key 字段自动映射为 dashscope_api_key
    if not config.get('dashscope_api_key') and config.get('api_key'):
        config['dashscope_api_key'] = config['api_key']

    return config


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except Exception as e:
        print(f"保存配置失败: {e}")