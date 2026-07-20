"""统一模型配置：从 .env 读取 DeepSeek 凭证，构造 ChatOpenAI 实例。"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """返回指向 DeepSeek 的 ChatOpenAI 实例。

    max_retries=0：ToolExecutor 统一管理重试，避免 ChatOpenAI 内部重试与
    ToolExecutor 外部重试叠加（llm_classification retries=1 + ChatOpenAI retries=2 → 6次请求）。
    request_timeout=65：单次 HTTP 请求硬超时，覆盖 ReAct astream_events 内
    每个独立 LLM 请求，不对整个 ReAct 流套 LLM Policy timeout。
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        streaming=streaming,
        max_retries=0,
        request_timeout=65,
    )
