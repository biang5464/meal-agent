"""Memory Agent：纯数据聚合，不调用 LLM。

读取 SQLite 用户画像 + 健康周期 + 提醒，组装 user_brief 注入 AgentState。
"""

from __future__ import annotations

import logging

from agents.state import AgentState
from core.cache import get_profile_cache_result, set_profile_cache_result, _client as _redis_client
from core.user_service import (
    get_preference_history_result,
    get_unread_reminders_readonly_result,
    get_user_profile_result,
    predict_next_period_result,
)

logger = logging.getLogger(__name__)


async def memory_agent(state: AgentState) -> AgentState:
    user_id = state["user_id"]

    # ── 基础画像（Redis 缓存 → SQLite 降级） ──────────────────
    cache_result = await get_profile_cache_result(user_id)
    if not cache_result.ok:
        logger.warning("[memory_agent] Redis 画像缓存故障: %s", cache_result.error)
    profile = cache_result.data  # None on miss or Redis error

    if profile is None:
        profile_result = await get_user_profile_result(user_id)
        if not profile_result.ok:
            logger.warning("[memory_agent] SQLite 用户画像读取失败: %s", profile_result.error)
        profile = profile_result.data or {}
        if profile:
            cache_write = await set_profile_cache_result(user_id, profile)
            if not cache_write.ok:
                logger.warning("[memory_agent] Redis 画像缓存写入失败: %s", cache_write.error)

    # ── 活跃饮食限制（未过期） ─────────────────────────────────
    pref_result = await get_preference_history_result(user_id)
    if not pref_result.ok:
        logger.warning("[memory_agent] 饮食偏好读取失败: %s", pref_result.error)
    active_restrictions = [r["value"] for r in (pref_result.data or [])]

    # ── 未读提醒（只读取，不标记已读） ────────────────────────
    rem_result = await get_unread_reminders_readonly_result(user_id)
    if not rem_result.ok:
        logger.warning("[memory_agent] 未读提醒读取失败: %s", rem_result.error)
    unread_reminders = rem_result.data or []

    # ── 经期状态（无 LLM，纯计算） ────────────────────────────
    period_result = await predict_next_period_result(user_id)
    if not period_result.ok:
        logger.warning("[memory_agent] 健康周期读取失败: %s", period_result.error)
    period_info = period_result.data or {}
    period_status = ""
    if period_info:
        phase = period_info.get("phase", "normal")
        days_until = period_info.get("days_until_next", 0)
        period_status = {
            "in_period": "经期中",
            "pre_period": f"经期前 {days_until} 天",
            "post_period": "经期后恢复期",
            "normal": "",
        }.get(phase, "")

    user_brief: dict = {
        "likes": profile.get("likes", []),
        "dislikes": profile.get("dislikes", []),
        "active_restrictions": active_restrictions,
        "recent_meals": profile.get("meal_history", [])[-5:],
        "unread_reminders": unread_reminders,
        "period_status": period_status,
        "budget_level": profile.get("budget") or "",
        "category_preferences": profile.get("category_preferences", {}),
    }

    # 检索相关历史摘要（失败时降级为空字符串，不阻塞主流程）
    from tools.memory import search_conversation_memory_result
    mem_result = await search_conversation_memory_result(
        user_id=state["user_id"],
        query=state["user_input"],
        n_results=3,
    )

    conversation_memory = ""
    if mem_result.ok:
        memories = mem_result.data or []
        if memories:
            lines = []
            for m in memories:
                tag = "【长期】" if m["memory_type"] == "permanent" else "【近期】"
                lines.append(f"{tag}{m['date']} {m['summary']}")
            conversation_memory = "\n".join(lines)

    # 提取上一轮对话（chat_history 最后一对 user+assistant 消息）
    last_turn: dict = {}
    chat_history = state.get("chat_history", [])
    if chat_history:
        last_user = next((m for m in reversed(chat_history) if m.get("role") == "user"), None)
        last_asst = next((m for m in reversed(chat_history) if m.get("role") == "assistant"), None)
        if last_user:
            last_turn = {
                "user_input": last_user.get("content", ""),
                "intent": "",  # Redis 不存储 intent，留空
                "result_preview": (last_asst.get("content", "") if last_asst else "")[:100],
            }

    from agents.context_manager import resolve_context_candidate
    merged_state = {**state, "user_brief": user_brief, "conversation_memory": conversation_memory, "last_turn": last_turn}
    return await resolve_context_candidate(merged_state, _redis_client)
