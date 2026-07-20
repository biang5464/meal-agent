"""Chat Agent：兜底闲聊，直接 LLM 回复，不调用任何工具。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from config import get_llm


def _build_system_prompt(user_brief: dict) -> str:
    likes = "、".join(user_brief.get("likes", [])) or "无"
    dislikes = "、".join(user_brief.get("dislikes", [])) or "无"
    period = user_brief.get("period_status", "")
    unread = user_brief.get("unread_reminders", [])

    lines = [
        "你是友好的个人饮食生活助手，可以回答饮食、健康、食谱等相关问题，也可以闲聊。",
        "",
        f"【用户背景】喜欢：{likes} | 忌口：{dislikes}",
    ]
    if period:
        lines.append(f"健康状态：{period}")
    if unread:
        reminder_texts = "；".join(r["message"] for r in unread[:3])
        lines.append(f"未读提醒：{reminder_texts}")

    return "\n".join(lines)


async def chat_agent(state: AgentState, queue: asyncio.Queue = None) -> AgentState:
    user_brief = state.get("user_brief", {})
    llm = get_llm(streaming=True)

    messages = [SystemMessage(content=_build_system_prompt(user_brief))]

    for msg in state.get("chat_history", []):
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=state["user_input"]))

    chunks: list[str] = []

    async def _collect():
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                chunks.append(chunk.content)
                if queue is not None:
                    queue.put_nowait(chunk.content)

    try:
        await asyncio.wait_for(_collect(), timeout=60)
    except asyncio.TimeoutError:
        if chunks:
            timeout_msg = "\n\n（回复超时，以上为部分内容）"
        else:
            timeout_msg = "（回复超时，请稍后重试）"
        chunks.append(timeout_msg)
        if queue is not None:
            queue.put_nowait(timeout_msg)

    return {**state, "result": "".join(chunks)}
