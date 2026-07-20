"""LangGraph StateGraph：连接所有 Agent 节点，导出 compiled_graph。

拓扑结构：
    memory → supervisor → (条件路由) → meal / price / update / chat
    update → (条件路由) → meal（needs_meal=True）或 END
    meal / price / chat → END
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)
from langgraph.graph import END, StateGraph

from agents.chat_agent import chat_agent
from agents.clarify_agent import run_clarify_agent
from agents.context_manager import commit_context
from agents.electronics_price_agent import run_electronics_price_agent
from agents.meal_agent import meal_agent
from agents.memory_agent import memory_agent
from agents.nutrition_agent import food_safety_agent, nutrition_agent
from agents.price_agent import price_agent
from agents.routing import should_clarify
from agents.state import AgentState
from agents.supervisor_agent import supervisor_agent
from agents.update_agent import update_agent
from core.cache import append_chat_message_result, get_chat_history_result


# ── 路由函数 ──────────────────────────────────────────────────────────────────

def _route_intent(state: AgentState) -> str:
    """supervisor 执行后，根据 intent 决定下一个节点。"""
    return {
        "MEAL": "meal",
        "PRICE": "price",
        "MEMORY_UPDATE": "update",
        "UPDATE_AND_MEAL": "update",
        "HEALTH_CYCLE": "chat",   # 暂时兜底到 chat，后续可接入专属节点
        "CHAT": "chat",
        "NUTRITION": "nutrition",
        "FOOD_SAFETY": "food_safety",
    }.get(state.get("intent", "CHAT"), "chat")


def _route_after_update(state: AgentState) -> str:
    """update 执行后，needs_meal=True 时继续到 meal，否则结束。"""
    return "meal" if state.get("needs_meal", False) else END


# ── queue-aware 节点包装：从 config["configurable"]["queue"] 取 queue ────────

async def _meal_node(state: AgentState, config: RunnableConfig) -> AgentState:
    queue = (config.get("configurable") or {}).get("queue")
    return await meal_agent(state, queue=queue)


async def _chat_node(state: AgentState, config: RunnableConfig) -> AgentState:
    queue = (config.get("configurable") or {}).get("queue")
    return await chat_agent(state, queue=queue)


async def _nutrition_node(state: AgentState, config: RunnableConfig) -> AgentState:
    queue = (config.get("configurable") or {}).get("queue")
    return await nutrition_agent(state, queue=queue)


async def _food_safety_node(state: AgentState, config: RunnableConfig) -> AgentState:
    queue = (config.get("configurable") or {}).get("queue")
    return await food_safety_agent(state, queue=queue)


async def _context_commit_node(state: AgentState, config: RunnableConfig) -> AgentState:
    from core.cache import _client as _redis_client
    return await commit_context(state, _redis_client)


# ── 图构建 ────────────────────────────────────────────────────────────────────

_graph = StateGraph(AgentState)

_graph.add_node("memory", memory_agent)
_graph.add_node("supervisor", supervisor_agent)
_graph.add_node("context_commit", _context_commit_node)
_graph.add_node("clarify", run_clarify_agent)
_graph.add_node("meal", _meal_node)
_graph.add_node("price", price_agent)
_graph.add_node("electronics_price", run_electronics_price_agent)
_graph.add_node("update", update_agent)
_graph.add_node("chat", _chat_node)
_graph.add_node("nutrition", _nutrition_node)
_graph.add_node("food_safety", _food_safety_node)

_graph.set_entry_point("memory")
_graph.add_edge("memory", "supervisor")
_graph.add_edge("supervisor", "context_commit")

_graph.add_conditional_edges(
    "context_commit",
    should_clarify,
    {
        "clarify": "clarify",
        "meal": "meal",
        "price": "price",
        "electronics_price": "electronics_price",
        "memory_update": "update",
        "update_and_meal": "update",
        "health_cycle": "chat",
        "chat": "chat",
        "nutrition": "nutrition",
        "food_safety": "food_safety",
    },
)

_graph.add_edge("clarify", END)
_graph.add_edge("meal", END)
_graph.add_edge("price", END)
_graph.add_edge("electronics_price", END)
_graph.add_edge("chat", END)
_graph.add_edge("nutrition", END)
_graph.add_edge("food_safety", END)

_graph.add_conditional_edges(
    "update",
    _route_after_update,
    {
        "meal": "meal",
        END: END,
    },
)

compiled_graph = _graph.compile()


# ── 入口包装函数 ──────────────────────────────────────────────────────────────

async def run(user_id: str, user_input: str) -> str:
    """执行完整图流程：读取历史 → 运行 graph → 写回历史。

    所有外部调用方（main.py 等）应通过此函数而非直接调用 compiled_graph。
    """
    chat_result = await get_chat_history_result(user_id)
    if not chat_result.ok:
        logger.warning("[graph] 对话历史读取失败: %s", chat_result.error)
    history = chat_result.data or []

    final_state = await compiled_graph.ainvoke({
        "user_id": user_id,
        "user_input": user_input,
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
        "chat_history": history,
        "conversation_memory": "",
        "confidence": "HIGH",
        "missing_slots": [],
        "top2": None,
        "last_turn": {},
        "session_context": {},
        "context_candidate": {},
        "resolved_entity": "",
        "resolved_input": "",
        "turn_excluded": [],
        "turn_constraints": {},
        "context_status": "active",
    })

    result = final_state.get("result", "")
    if result:
        for _role, _content in [("user", user_input), ("assistant", result)]:
            _msg_r = await append_chat_message_result(user_id, _role, _content)
            if not _msg_r.ok:
                logger.warning("[graph] 对话历史追加失败 role=%s: %s", _role, _msg_r.error)
    return result


async def run_with_queue(user_id: str, user_input: str, queue: asyncio.Queue) -> str:
    """执行完整图流程，将 token 实时推入 queue，最后放入 [DONE] 哨兵。"""
    chat_result = await get_chat_history_result(user_id)
    if not chat_result.ok:
        logger.warning("[graph] 对话历史读取失败: %s", chat_result.error)
    history = chat_result.data or []

    final_state = await compiled_graph.ainvoke(
        {
            "user_id": user_id,
            "user_input": user_input,
            "user_brief": {},
            "intent": "",
            "result": "",
            "messages": [],
            "needs_meal": False,
            "chat_history": history,
            "conversation_memory": "",
            "confidence": "HIGH",
            "missing_slots": [],
            "top2": None,
            "last_turn": {},
            "session_context": {},
            "context_candidate": {},
            "resolved_entity": "",
            "resolved_input": "",
            "turn_excluded": [],
            "turn_constraints": {},
            "context_status": "active",
        },
        config={"configurable": {"queue": queue}},
    )

    result = final_state.get("result", "")
    if result:
        for _role, _content in [("user", user_input), ("assistant", result)]:
            _msg_r = await append_chat_message_result(user_id, _role, _content)
            if not _msg_r.ok:
                logger.warning("[graph] 对话历史追加失败 role=%s: %s", _role, _msg_r.error)

    # 非流式 agent 的结果需要在这里统一写入 queue
    # 流式 agent（MEAL/CHAT/NUTRITION/FOOD_SAFETY）在图执行期间已逐 token 写入
    _STREAMING_INTENTS = {"MEAL", "CHAT", "NUTRITION", "FOOD_SAFETY"}
    intent = final_state.get("intent", "CHAT")
    if result and intent not in _STREAMING_INTENTS:
        queue.put_nowait(result)

    # 异步存储对话摘要，不阻塞主流程；异常通过 done callback 记录
    from tools.memory import save_conversation_memory
    _task = asyncio.create_task(save_conversation_memory(
        user_id=final_state.get("user_id", "unknown"),
        user_input=final_state.get("user_input", ""),
        result=final_state.get("result", ""),
        intent=final_state.get("intent", "CHAT"),
    ))
    _task.add_done_callback(_log_background_task_result)

    queue.put_nowait("[DONE]")
    return result


def _log_background_task_result(task: asyncio.Task) -> None:
    """后台任务 done callback：记录异常，忽略 CancelledError，不重新抛出。"""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logger.error("[graph] save_conversation_memory 后台任务异常", exc_info=exc)
