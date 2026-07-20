"""营养建议 Agent：RAG 检索营养知识库 + DeepSeek 流式生成回答。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from agents.context_consumer import (
    FOOD_SAFETY_CONSTRAINTS,
    NUTRITION_CONSTRAINTS,
    build_human_message,
    get_agent_context,
    rag_query,
)
from agents.state import AgentState
from config import get_llm
from tools.nutrition import search_food_safety_result, search_nutrition_result


def _build_query(user_input: str, user_brief: dict) -> str:
    """将用户输入与健康档案拼接为检索 query，提升召回准确度。"""
    restrictions = user_brief.get("active_restrictions", [])
    period = user_brief.get("period_status", "")

    query = user_input
    if restrictions:
        query += " " + " ".join(restrictions)
    if period:
        query += " " + period
    return query


def _build_prompt(user_brief: dict, chunks: list[dict], rag_status: str = "ok") -> str:
    restrictions = "、".join(user_brief.get("active_restrictions", [])) or "无"
    period = user_brief.get("period_status", "") or "无"
    dislikes = "、".join(user_brief.get("dislikes", [])) or "无"

    if rag_status == "fault":
        knowledge = "营养知识库暂时不可用，以下为一般性信息，不替代专业医疗或营养师建议。"
    elif chunks:
        knowledge = "\n\n".join(f"[来源：{c['source']}]\n{c['content']}" for c in chunks)
    else:
        knowledge = "（未检索到直接相关的知识库资料，以下为一般性饮食建议，请酌情参考。）"

    return f"""\
你是一位专业营养顾问，请根据以下参考知识和用户的健康情况，给出具体、实用的饮食建议。

【参考知识】
{knowledge}

【用户健康档案】
饮食限制：{restrictions}
健康状态：{period}
忌口：{dislikes}
"""


async def nutrition_agent(state: AgentState, queue: asyncio.Queue = None) -> AgentState:
    user_input = state["user_input"]
    user_brief = state.get("user_brief", {})

    context = get_agent_context(state, "health", NUTRITION_CONSTRAINTS)
    context_q = rag_query(user_input, context)
    query = _build_query(context_q, user_brief)
    rag_result = await search_nutrition_result(query, n_results=3)

    if not rag_result.ok:
        chunks = []
        rag_status = "fault"
    else:
        chunks = rag_result.data or []
        rag_status = "ok"

    prompt = _build_prompt(user_brief, chunks, rag_status)

    llm = get_llm(streaming=True)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=build_human_message(user_input, context)),
    ]

    async def _collect() -> str:
        full_response = ""
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                if queue is not None:
                    queue.put_nowait(chunk.content)
        return full_response

    try:
        full_response = await asyncio.wait_for(_collect(), timeout=60)
    except asyncio.TimeoutError:
        full_response = "营养知识查询超时，请稍后重试。"
        if queue is not None:
            queue.put_nowait(full_response)

    return {**state, "result": full_response}


async def food_safety_agent(state: AgentState, queue: asyncio.Queue = None) -> AgentState:
    user_input = state["user_input"]
    user_brief = state.get("user_brief", {})

    context = get_agent_context(state, "health", FOOD_SAFETY_CONSTRAINTS)
    rag_result = await search_food_safety_result(rag_query(user_input, context), n_results=3)

    if not rag_result.ok:
        full_response = (
            "食品安全知识库暂时不可用。"
            "对于用药、过敏或中毒风险，请咨询医生、药师或当地毒物信息中心。"
        )
        if queue is not None:
            queue.put_nowait(full_response)
        return {**state, "result": full_response}

    chunks = rag_result.data or []

    if chunks:
        knowledge = "\n\n".join(f"[来源：{c['source']}]\n{c['content']}" for c in chunks)
        sources = "、".join({c["source"] for c in chunks})
        source_line = f"\n\n回答末尾请注明：参考来源：{sources}。"
    else:
        knowledge = "（未检索到可核验的知识库资料，请谨慎作答，不要虚构来源。）"
        source_line = ""

    prompt = f"""\
你是一位食品安全专家，请根据以下参考知识，给出准确、实用的食品安全建议。回答应具体、准确，说明原因和正确做法。{source_line}

【参考知识】
{knowledge}
"""

    llm = get_llm(streaming=True)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=build_human_message(user_input, context)),
    ]

    async def _collect() -> str:
        full_response = ""
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_response += chunk.content
                if queue is not None:
                    queue.put_nowait(chunk.content)
        return full_response

    try:
        full_response = await asyncio.wait_for(_collect(), timeout=60)
    except asyncio.TimeoutError:
        full_response = "食品安全查询超时，请稍后重试。"
        if queue is not None:
            queue.put_nowait(full_response)

    return {**state, "result": full_response}
