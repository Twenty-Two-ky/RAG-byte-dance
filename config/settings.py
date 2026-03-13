import os
import json


# --- 1. 智能代理配置 ---
def setup_proxy():
    """
    设置网络代理。
    优先级：系统/Docker环境变量 > 代码硬编码默认值
    """
    # 如果系统环境（包括 Docker Compose 传进来的）已经设置了代理，就不要覆盖它！
    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        print(f"检测到系统代理设置，跳过本地配置。")
        return

    # 只有在本地直接运行且没有环境变量时，才使用这个默认值
    # 你可以保留这个方便自己本地调试，但提交到 GitHub 也没关系，因为别人运行时如果不配环境，只是连不上网，不会报错
    default_proxy = 'http://127.0.0.1:7897'
    os.environ['HTTP_PROXY'] = default_proxy
    os.environ['HTTPS_PROXY'] = default_proxy
    print(f"使用本地默认代理: {default_proxy}")


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

    # 2. 从环境变量加载各 API Key（Docker/云部署时通过环境变量注入）
    anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        config['claude_api_key'] = anthropic_key
    if os.environ.get("DASHSCOPE_API_KEY"):
        config['dashscope_api_key'] = os.environ.get("DASHSCOPE_API_KEY")

    # 向后兼容：旧版 GEMINI_API_KEY 映射（保留以防旧配置文件）
    if os.environ.get("GEMINI_API_KEY"):
        config.setdefault('api_key', os.environ.get("GEMINI_API_KEY"))

    return config


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    except Exception as e:
        print(f"保存配置失败: {e}")