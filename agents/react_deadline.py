"""共享 ReAct Agent Deadline 包装 — 防止 ReAct 多轮循环无限运行。

外部取消（CancelledError）与 Agent Deadline 超时（TimeoutError）严格分离：
- TimeoutError → 追加提示后返回部分结果，不向上传播
- CancelledError → 关闭 generator 后重新抛出（让 Graph/Task 层处理取消）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_PARTIAL_SUFFIX = "\n\n[回答生成超时，以上为部分结果，请重试或简化问题]"
_FULL_TIMEOUT = "抱歉，当前请求处理时间较长，请稍后重试或尝试简化您的问题。"


async def run_react_with_deadline(
    agent,
    input_dict: dict,
    *,
    deadline: float,
    config: dict | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """在 deadline 秒内运行 agent.astream_events()，返回拼接后的文本。

    Args:
        agent:       已配置好的 LangGraph ReAct agent（有 astream_events 方法）
        input_dict:  传给 astream_events 的输入（{"messages": [...]}）
        deadline:    Agent 总执行时限（秒）
        config:      额外 config dict（如 {"recursion_limit": 30}），可为 None
        on_token:    每个 on_chat_model_stream token 的实时回调，可为 None
    """
    parts: list[str] = []
    agen = None  # astream_events() 返回的 async generator 引用，用于 aclose()

    async def _consume() -> None:
        nonlocal agen
        kwargs: dict = {}
        if config:
            kwargs["config"] = config
        agen = agent.astream_events(input_dict, version="v2", **kwargs)
        async for event in agen:
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    parts.append(chunk.content)
                    if on_token is not None:
                        on_token(chunk.content)

    try:
        await asyncio.wait_for(_consume(), timeout=deadline)

    except asyncio.TimeoutError:
        # asyncio.wait_for 已内部取消并等待 _consume 完成，generator 通常已关闭。
        # 显式 aclose() 作为双重保险，忽略异常。
        await _safe_aclose(agen)
        logger.warning(
            "[react_deadline] 超时 deadline=%.0fs parts_count=%d", deadline, len(parts)
        )
        if parts:
            parts.append(_PARTIAL_SUFFIX)
        else:
            parts.append(_FULL_TIMEOUT)

    except asyncio.CancelledError:
        await _safe_aclose(agen)
        raise  # 外部取消必须传播，不降级为超时提示

    except Exception:
        await _safe_aclose(agen)
        raise

    return "".join(parts)


async def _safe_aclose(agen) -> None:
    """尽力关闭 async generator，忽略所有异常。"""
    if agen is None:
        return
    try:
        await agen.aclose()
    except Exception:
        pass
