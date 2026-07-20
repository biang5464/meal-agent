"""Phase 9C tests: commit_context 持久化策略、Domain 校验、降级行为与 excluded/constraints 合并。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.session_context import ContextSlot, TopicCache, MAX_CONTEXTS


# ── 公用 helpers ──────────────────────────────────────────────────────────────

def _make_slot(ctx_id, entity, domain="electronics", excluded=None, constraints=None):
    return ContextSlot(
        ctx_id=ctx_id,
        domain=domain,
        entity=entity,
        topic_turns=1,
        constraints=constraints or {},
        excluded=excluded or [],
        last_intent="ELECTRONICS_PRICE",
        last_active="2026-07-01T00:00:00",
    )


def _make_commit_state(
    intent="ELECTRONICS_PRICE",
    candidate=None,
    confidence="HIGH",
    missing_slots=None,
    turn_excluded=None,
    turn_constraints=None,
):
    if candidate is None:
        candidate = {
            "ctx_id": "",
            "match_type": "new",
            "entity": "MacBook Air M3",
            "domain": "electronics",
            "existing_excluded": [],
            "existing_constraints": {},
        }
    return {
        "user_id": "test_user",
        "user_input": "测试",
        "intent": intent,
        "confidence": confidence,
        "missing_slots": missing_slots or [],
        "context_candidate": candidate,
        "turn_excluded": turn_excluded or [],
        "turn_constraints": turn_constraints or {},
        "session_context": {},
        "context_status": "active",
        "resolved_entity": candidate.get("entity", ""),
        "resolved_input": "",
        "user_brief": {},
        "result": "",
        "messages": [],
        "needs_meal": False,
        "chat_history": [],
        "conversation_memory": "",
        "top2": None,
        "last_turn": {},
    }


def _cache_ok(cache):
    return MagicMock(ok=True, data=cache, error=None)


def _cache_fail():
    from core.tool_protocol import tool_failure, ToolErrorCode
    return tool_failure(ToolErrorCode.NETWORK, "Redis down", tool_name="load_topic_cache")


def _save_ok():
    return MagicMock(ok=True, error=None)


def _save_fail():
    return MagicMock(ok=False, error=MagicMock(message="Redis down"))


# ── 1. 协议迁移 ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_uses_load_topic_cache_result():
    """commit_context 应使用 load_topic_cache_result（ToolResult 协议），不调用旧版 load_topic_cache。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state()  # entity="MacBook Air M3"，有意义

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))) as mock_new, \
         patch("agents.context_manager.load_topic_cache",
               new=AsyncMock()) as mock_old, \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        await commit_context(state, MagicMock())

    mock_new.assert_called_once()
    mock_old.assert_not_called()


# ── 2. Pronoun Domain 校验 ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_pronoun_same_domain_reuses():
    """pronoun + candidate_domain == current_domain → 复用已有 Slot，ctx_id 保持。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["ctx_id"] == "ctx_macbook"
    assert result["session_context"]["persistence_status"] == "saved"


@pytest.mark.asyncio
async def test_commit_pronoun_cross_domain_no_reuse():
    """pronoun + 历史 entity + 跨 domain → 历史 entity 丢弃，跳过持久化，旧 Slot 不变。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",  # domain=food，与 electronics 不匹配
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "pronoun",  # 历史继承来源
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    # pronoun 历史 entity 跨 domain → skipped，save 未调用
    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["context_persisted"] is False
    assert result["session_context"]["entity"] == ""  # 历史 entity 已丢弃
    assert result["session_context"]["domain"] == "food"  # 当前 intent domain
    assert slot.entity == "MacBook Air M3"  # 旧 slot 未修改


@pytest.mark.asyncio
async def test_commit_pronoun_cross_domain_does_not_modify_old_slot():
    """pronoun 跨 domain → 旧 Slot 的 entity 和 excluded 不得被修改。"""
    from agents.context_manager import commit_context

    original_entity = "MacBook Air M3"
    original_excluded = ["保护壳"]
    slot = _make_slot("ctx_macbook", original_entity, excluded=list(original_excluded))
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "pronoun",  # 历史继承来源
            "existing_excluded": list(original_excluded), "existing_constraints": {},
        },
        turn_excluded=["二手"],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        await commit_context(state, MagicMock())

    assert slot.entity == original_entity
    assert slot.excluded == original_excluded


@pytest.mark.asyncio
async def test_commit_candidate_empty_domain_no_reuse():
    """candidate domain 为空 → 不得复用，因为无法校验 domain 一致性。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "",  # domain 为空
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["ctx_id"] != "ctx_macbook"


@pytest.mark.asyncio
async def test_commit_entity_match_cross_domain_no_reuse():
    """entity_match 跨 domain → 不复用旧 Slot，旧 Slot 不被修改。"""
    from agents.context_manager import commit_context

    original_excluded = ["保护壳"]
    slot = _make_slot("ctx_macbook", "MacBook Air M3", excluded=list(original_excluded))
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",  # food domain
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "番茄炒蛋", "domain": "electronics",
            "existing_excluded": list(original_excluded), "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        await commit_context(state, MagicMock())

    assert slot.excluded == original_excluded
    assert slot.entity == "MacBook Air M3"


@pytest.mark.asyncio
async def test_commit_embedding_match_cross_domain_no_reuse():
    """embedding_match 跨 domain → 不复用旧 Slot。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "embedding_match",
            "entity": "沙拉", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["ctx_id"] != "ctx_macbook"


# ── 3. CHAT/MEMORY_UPDATE 非 Context 意图 ────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_chat_pronoun_no_save():
    """CHAT + pronoun → 非 Context 意图，一律跳过持久化，save 未调用。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="CHAT",
        candidate={
            "ctx_id": "ctx_test", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["context_persisted"] is False


@pytest.mark.asyncio
async def test_commit_memory_update_pronoun_no_save():
    """MEMORY_UPDATE + pronoun → 非 Context 意图，一律跳过持久化。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="MEMORY_UPDATE",
        candidate={
            "ctx_id": "ctx_test", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["context_persisted"] is False


# ── 4. LOW 置信度 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_low_no_missing_slots_pending_skipped():
    """LOW + missing_slots=[] → context_status='pending'，不持久化（不调用 save）。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        confidence="LOW",
        missing_slots=[],
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["context_status"] == "pending"
    assert result["session_context"]["context_persisted"] is False
    assert result["session_context"]["persistence_status"] == "skipped"


@pytest.mark.asyncio
async def test_commit_low_with_missing_slots_pending_skipped():
    """LOW + missing_slots 非空 → 同样不持久化。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        confidence="LOW",
        missing_slots=["model"],
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "苹果手机",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["context_status"] == "pending"
    assert result["session_context"]["persistence_status"] == "skipped"


# ── 5. MEDIUM/pending 策略 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_medium_missing_meaningful_saves_pending():
    """MEDIUM + missing_slots + entity 非空 → 保存 pending Slot，persistence_status='saved'。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        confidence="MEDIUM",
        missing_slots=["model"],
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "苹果手机",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["context_status"] == "pending"
    assert result["session_context"]["persistence_status"] == "saved"
    assert result["session_context"]["context_persisted"] is True


@pytest.mark.asyncio
async def test_commit_medium_missing_empty_context_skipped():
    """MEDIUM + missing_slots + 空 entity + 无排除无约束 → 跳过持久化（无意义 Context）。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        confidence="MEDIUM",
        missing_slots=["model"],
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["context_status"] == "pending"
    assert result["session_context"]["persistence_status"] == "skipped"


# ── 6. Redis 读写降级 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_redis_read_degraded_returns_read_degraded():
    """Redis 读取失败 → persistence_status='read_degraded'，不调用 save。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics", "existing_excluded": ["保护壳"],
            "existing_constraints": {"color": "blue"},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_fail())), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "read_degraded"
    assert result["session_context"]["context_persisted"] is False
    # 降级时应使用 candidate 数据
    assert result["session_context"]["entity"] == "iPhone 16"


@pytest.mark.asyncio
async def test_commit_redis_write_degraded_returns_write_degraded():
    """Redis 保存失败 → persistence_status='write_degraded'，仍返回完整 session_context。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state()  # entity="MacBook Air M3"，有意义

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_fail())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["persistence_status"] == "write_degraded"
    assert result["session_context"]["context_persisted"] is False
    assert "session_context" in result


@pytest.mark.asyncio
async def test_commit_save_success_returns_saved_and_persisted():
    """保存成功 → persistence_status='saved'，context_persisted=True。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state()

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["persistence_status"] == "saved"
    assert result["session_context"]["context_persisted"] is True


@pytest.mark.asyncio
async def test_commit_strategy_skip_returns_skipped_and_not_persisted():
    """策略跳过（CHAT）→ persistence_status='skipped'，context_persisted=False。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(intent="CHAT")

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["context_persisted"] is False


# ── 7. Excluded 稳定去重 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_excluded_stable_dedup_order():
    """复用 Slot 时 excluded 稳定去重：首次出现顺序保持，重复项只出现一次。"""
    from agents.context_manager import commit_context

    original_excluded = ["保护壳", "二手", "翻新"]
    slot = _make_slot("ctx_macbook", "MacBook Air M3", excluded=list(original_excluded))
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": list(original_excluded), "existing_constraints": {},
        },
        turn_excluded=["保护壳", "分期"],  # "保护壳" 重复
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    excluded = result["session_context"]["excluded"]
    assert excluded.count("保护壳") == 1
    assert "二手" in excluded
    assert "翻新" in excluded
    assert "分期" in excluded
    # 首次出现顺序：保护壳 早于 分期
    assert excluded.index("保护壳") < excluded.index("分期")


@pytest.mark.asyncio
async def test_commit_excluded_empty_history_no_error():
    """历史 excluded=[] + turn_excluded=[] → 结果为 []，不出错。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state(
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
        turn_excluded=[],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["excluded"] == []


@pytest.mark.asyncio
async def test_commit_excluded_history_not_mutated():
    """commit_context 不应原地修改 ContextSlot.excluded 的原始引用（防止副作用）。"""
    from agents.context_manager import commit_context

    original = ["保护壳"]
    slot = _make_slot("ctx_macbook", "MacBook Air M3", excluded=original)
    cache = TopicCache.empty()
    cache.add(slot)
    # 保存 slot.excluded 的 id，验证不是同一对象（_stable_unique 返回新列表）
    original_id = id(slot.excluded)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": ["保护壳"], "existing_constraints": {},
        },
        turn_excluded=["二手"],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        await commit_context(state, MagicMock())

    # slot.excluded 已被更新为新列表，不是原始对象
    assert id(slot.excluded) != original_id


# ── 8. Constraints 合并 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_constraints_current_turn_overwrites_history():
    """复用 Slot 时，turn_constraints 中的同名 key 覆盖历史值，不同名 key 保留。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3",
                      constraints={"color": "silver", "storage": "256GB"})
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {"color": "silver", "storage": "256GB"},
        },
        turn_constraints={"storage": "512GB"},
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    constraints = result["session_context"]["constraints"]
    assert constraints["color"] == "silver"    # 历史保留
    assert constraints["storage"] == "512GB"   # 当前轮覆盖


@pytest.mark.asyncio
async def test_commit_constraints_empty_turn_preserves_history():
    """empty turn_constraints 不应清空历史 constraints。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3",
                      constraints={"color": "silver"})
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "existing_excluded": [], "existing_constraints": {"color": "silver"},
        },
        turn_constraints={},
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    assert result["session_context"]["constraints"]["color"] == "silver"


# ── 9. Ghost slot / LRU 防护 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_ghost_slot_no_lru_eviction():
    """match_type='new' + entity='' + 无排除无约束 → 早期 ghost 检测，不读 Redis，不调 save。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "",
            "domain": "", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock()) as mock_load, \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_load.assert_not_called()
    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"


@pytest.mark.asyncio
async def test_commit_ghost_does_not_add_slot_to_full_cache():
    """完全空 Context 不新增 Slot，也不触发 LRU 淘汰，cache slot 数量保持不变。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    for i in range(MAX_CONTEXTS):
        cache.add(_make_slot(f"ctx_{i}", f"product_{i}"))
    assert len(cache.contexts) == MAX_CONTEXTS

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        await commit_context(state, MagicMock())

    # 由于早期 ghost 检测，根本不加载 cache，slot 数量不变
    assert len(cache.contexts) == MAX_CONTEXTS


# ── 10. Redis read degraded 降级内容 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_commit_read_degraded_excluded_merged():
    """Redis 读取失败时，降级 session_context 应合并 existing_excluded 和 turn_excluded（稳定去重）。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_test", "match_type": "entity_match",
            "entity": "iPhone 16", "domain": "electronics",
            "existing_excluded": ["保护壳", "二手"],
            "existing_constraints": {"color": "blue"},
        },
        turn_excluded=["保护壳", "钢化膜"],  # 保护壳 重复
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_fail())), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    excluded = result["session_context"]["excluded"]
    assert excluded.count("保护壳") == 1
    assert "二手" in excluded
    assert "钢化膜" in excluded
    constraints = result["session_context"]["constraints"]
    assert constraints["color"] == "blue"


# ── 11. graph._context_commit_node 使用 core.cache._client ───────────────────

@pytest.mark.asyncio
async def test_context_commit_node_uses_core_cache_client():
    """_context_commit_node 应传入 core.cache._client 单例给 commit_context，不再创建新连接。"""
    import agents.graph as graph_module
    import core.cache as cache_module

    sentinel = MagicMock(name="core_cache_singleton")
    captured = {}

    async def _capture(state, redis_client):
        captured["client"] = redis_client
        return {**state, "session_context": {}, "context_status": "active"}

    with patch.object(cache_module, "_client", sentinel), \
         patch("agents.graph.commit_context", side_effect=_capture):
        config = {"configurable": {}}
        minimal_state = {k: "" for k in [
            "user_id", "user_input", "intent", "result", "resolved_entity",
            "resolved_input", "context_status", "confidence",
        ]}
        minimal_state.update({
            "user_brief": {}, "messages": [], "needs_meal": False,
            "chat_history": [], "conversation_memory": "",
            "missing_slots": [], "top2": None, "last_turn": {},
            "session_context": {}, "context_candidate": {},
            "turn_excluded": [], "turn_constraints": {},
        })
        await graph_module._context_commit_node(minimal_state, config)

    assert captured.get("client") is sentinel


# ── 12. entity_source 过滤 — 历史继承 vs 当前轮明确 ───────────────────────────

@pytest.mark.asyncio
async def test_commit_negation_carry_cross_domain_entity_discarded():
    """negation_carry 属于历史继承来源，跨 domain 时 entity 丢弃，返回 skipped。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "entity_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "negation_carry",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["entity"] == ""


@pytest.mark.asyncio
async def test_commit_slot_carry_cross_domain_entity_discarded():
    """slot_carry 属于历史继承来源，跨 domain 时 entity 丢弃，返回 skipped。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "embedding_match",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "slot_carry",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["entity"] == ""


@pytest.mark.asyncio
async def test_commit_llm_fallback_cross_domain_entity_discarded():
    """llm_fallback 属于历史继承来源，跨 domain 时 entity 丢弃，返回 skipped。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "llm_fallback",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["entity"] == ""


@pytest.mark.asyncio
async def test_commit_llm_entity_source_cross_domain_creates_new_slot():
    """entity_source='llm' 为当前轮明确提取，跨 domain 时允许新建 Slot，entity 保留。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",  # food domain
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "番茄炒蛋", "domain": "electronics",
            "entity_source": "llm",  # 当前轮明确提取，允许跨 domain
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_called_once()
    assert result["session_context"]["persistence_status"] == "saved"
    assert result["session_context"]["entity"] == "番茄炒蛋"
    assert result["session_context"]["domain"] == "food"
    assert result["session_context"]["ctx_id"] != "ctx_macbook"


@pytest.mark.asyncio
async def test_commit_new_slot_does_not_inherit_existing_excluded():
    """新建 Slot 只含 turn_excluded，不携带旧 candidate.existing_excluded。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics",
            "existing_excluded": ["保护壳", "二手"],
            "existing_constraints": {},
        },
        turn_excluded=["分期"],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    excluded = result["session_context"]["excluded"]
    assert "保护壳" not in excluded
    assert "二手" not in excluded
    assert "分期" in excluded


@pytest.mark.asyncio
async def test_commit_new_slot_does_not_inherit_existing_constraints():
    """新建 Slot 只含 turn_constraints，不携带旧 candidate.existing_constraints。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics",
            "existing_excluded": [],
            "existing_constraints": {"color": "silver", "storage": "256GB"},
        },
        turn_constraints={"color": "black"},
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    constraints = result["session_context"]["constraints"]
    assert constraints.get("storage") is None
    assert constraints["color"] == "black"


@pytest.mark.asyncio
async def test_commit_skipped_domain_uses_current_intent_domain():
    """skipped session_context 的 domain 字段必须使用当前 Supervisor domain，不使用 candidate.domain。"""
    from agents.context_manager import commit_context

    state = _make_commit_state(
        intent="MEAL",  # current domain = "food"
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "",
            "domain": "electronics",  # 不应出现在 skipped session_context
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["domain"] == "food"


@pytest.mark.asyncio
async def test_commit_new_slot_turn_excluded_stable_dedup():
    """新建 Slot 的 turn_excluded 含重复项时，excluded 稳定去重，保持首次出现顺序。"""
    from agents.context_manager import commit_context

    cache = TopicCache.empty()
    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "", "match_type": "new", "entity": "iPhone 16",
            "domain": "electronics", "existing_excluded": [], "existing_constraints": {},
        },
        turn_excluded=["分期", "二手", "分期", "保护壳", "二手"],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock(return_value=_save_ok())):
        result = await commit_context(state, MagicMock())

    excluded = result["session_context"]["excluded"]
    assert excluded.count("分期") == 1
    assert excluded.count("二手") == 1
    assert excluded.count("保护壳") == 1
    assert excluded.index("分期") < excluded.index("二手") < excluded.index("保护壳")


@pytest.mark.asyncio
async def test_commit_empty_domain_historical_entity_source_skipped():
    """candidate_domain='' + entity_source 属于历史继承来源 → 同跨 domain 规则，skipped。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="ELECTRONICS_PRICE",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "",
            "entity_source": "pronoun",
            "existing_excluded": [], "existing_constraints": {},
        },
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["entity"] == ""


@pytest.mark.asyncio
async def test_commit_historical_carry_with_turn_excluded_skipped():
    """历史继承来源跨 domain，即使有 turn_excluded 也一律 skipped，旧 Slot 不变，turn_excluded 保留在本轮 sc。"""
    from agents.context_manager import commit_context

    slot = _make_slot("ctx_macbook", "MacBook Air M3", domain="electronics",
                      excluded=["保护壳"])
    cache = TopicCache.empty()
    cache.add(slot)

    state = _make_commit_state(
        intent="MEAL",
        candidate={
            "ctx_id": "ctx_macbook", "match_type": "pronoun",
            "entity": "MacBook Air M3", "domain": "electronics",
            "entity_source": "slot_carry",
            "existing_excluded": ["保护壳"], "existing_constraints": {},
        },
        turn_excluded=["二手"],
    )

    with patch("agents.context_manager.load_topic_cache_result",
               new=AsyncMock(return_value=_cache_ok(cache))), \
         patch("agents.context_manager.save_topic_cache_result",
               new=AsyncMock()) as mock_save:
        result = await commit_context(state, MagicMock())

    mock_save.assert_not_called()
    assert result["session_context"]["persistence_status"] == "skipped"
    assert result["session_context"]["entity"] == ""
    assert "二手" in result["session_context"]["excluded"]
    assert slot.excluded == ["保护壳"]
