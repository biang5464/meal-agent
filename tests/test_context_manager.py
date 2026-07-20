import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.session_context import (
    SessionContext, get_domain, extract_excluded_by_rule,
    needs_llm_extraction, TopicCache, ContextSlot, MAX_CONTEXTS,
    ENTITY_SIMILARITY_THRESHOLD
)
from agents.context_manager import is_same_topic, _build_resolved_input, commit_context

# get_domain 测试
def test_get_domain_electronics():
    assert get_domain("ELECTRONICS_PRICE") == "electronics"

def test_get_domain_food():
    assert get_domain("MEAL") == "food"
    assert get_domain("PRICE") == "food"

def test_get_domain_health():
    assert get_domain("NUTRITION") == "health"
    assert get_domain("FOOD_SAFETY") == "health"

# extract_excluded_by_rule 测试
def test_extract_excluded_basic():
    result = extract_excluded_by_rule("不要保护壳")
    assert "保护壳" in result

def test_extract_excluded_multiple():
    result = extract_excluded_by_rule("不要保护壳，也不要二手")
    assert len(result) >= 1

def test_extract_excluded_no_negation():
    result = extract_excluded_by_rule("MacBook Air M3价格")
    assert result == []

# needs_llm_extraction 测试
def test_needs_llm_when_rule_failed():
    assert needs_llm_extraction("我更偏向不是那种特别重的", []) == True

def test_no_llm_when_rule_succeeded():
    assert needs_llm_extraction("不要保护壳", ["保护壳"]) == False

def test_no_llm_when_no_negation():
    assert needs_llm_extraction("MacBook Air M3", []) == False

# is_same_topic 测试
def test_same_topic_by_entity():
    assert is_same_topic("MacBook Air M3", "MacBook Air", "ELECTRONICS_PRICE", "ELECTRONICS_PRICE") == True

def test_same_topic_by_domain():
    assert is_same_topic("", "", "MEAL", "PRICE") == True  # 同属 food domain

def test_different_topic():
    assert is_same_topic("iPhone", "番茄炒蛋", "ELECTRONICS_PRICE", "MEAL") == False

# SessionContext 测试
def test_session_context_to_dict():
    ctx = SessionContext(domain="electronics", entity="MacBook Air M3")
    d = ctx.to_dict()
    assert d["domain"] == "electronics"
    assert d["entity"] == "MacBook Air M3"

def test_session_context_from_dict():
    d = {"domain": "food", "entity": "番茄炒蛋", "topic_turns": 2,
         "constraints": {}, "excluded": [], "last_intent": "MEAL",
         "last_updated": "", "turn_count": 2}
    ctx = SessionContext.from_dict(d)
    assert ctx.domain == "food"
    assert ctx.topic_turns == 2


# ── TopicCache 测试 ──────────────────────────────────────────────────────────

def make_slot(ctx_id, entity, domain="electronics", excluded=None):
    return ContextSlot(
        ctx_id=ctx_id,
        domain=domain,
        entity=entity,
        topic_turns=1,
        constraints={},
        excluded=excluded or [],
        last_intent="ELECTRONICS_PRICE",
        last_active="2026-07-05T00:00:00",
    )


def test_topic_cache_add_and_get():
    cache = TopicCache.empty()
    slot = make_slot("ctx_macbook", "MacBook Air M3")
    cache.add(slot)
    assert cache.current_context_id == "ctx_macbook"
    assert cache.get_current().entity == "MacBook Air M3"


def test_topic_cache_lru_eviction():
    """超过5个时淘汰最久未访问的"""
    cache = TopicCache.empty()
    for i in range(MAX_CONTEXTS + 1):
        cache.add(make_slot(f"ctx_{i}", f"product_{i}"))
    assert len(cache.contexts) == MAX_CONTEXTS
    assert "ctx_0" not in cache.contexts  # 最早的被淘汰


def test_topic_cache_access_updates_lru():
    """访问后移到末尾"""
    cache = TopicCache.empty()
    cache.add(make_slot("ctx_a", "MacBook"))
    cache.add(make_slot("ctx_b", "iPhone"))
    cache.access("ctx_a")  # 访问 ctx_a，移到末尾
    keys = list(cache.contexts.keys())
    assert keys[-1] == "ctx_a"  # ctx_a 在末尾（最近访问）


def test_find_by_entity_exact_match():
    """精确匹配找到已有 context"""
    cache = TopicCache.empty()
    cache.add(make_slot("ctx_macbook", "MacBook Air M3"))
    found = cache.find_by_entity("MacBook Air")
    assert found is not None
    assert found.ctx_id == "ctx_macbook"


def test_find_by_entity_not_found():
    """找不到时返回 None"""
    cache = TopicCache.empty()
    cache.add(make_slot("ctx_macbook", "MacBook Air M3"))
    found = cache.find_by_entity("iPhone 16")
    assert found is None


def test_topic_cache_serialization():
    """序列化和反序列化"""
    cache = TopicCache.empty()
    cache.add(make_slot("ctx_macbook", "MacBook Air M3", excluded=["保护壳"]))
    d = cache.to_dict()
    restored = TopicCache.from_dict(d)
    assert restored.current_context_id == "ctx_macbook"
    assert restored.contexts["ctx_macbook"].excluded == ["保护壳"]


def test_multiple_electronics_contexts():
    """同 domain 多个产品互不覆盖"""
    cache = TopicCache.empty()
    cache.add(make_slot("ctx_macbook", "MacBook Air M3", excluded=["保护壳"]))
    cache.add(make_slot("ctx_iphone", "iPhone 17 Pro"))
    assert len(cache.contexts) == 2
    assert cache.contexts["ctx_macbook"].excluded == ["保护壳"]
    assert cache.contexts["ctx_iphone"].entity == "iPhone 17 Pro"


def test_switch_back_restores_context():
    """切回已有 context 时数据完整"""
    cache = TopicCache.empty()
    macbook_slot = make_slot("ctx_macbook", "MacBook Air M3", excluded=["保护壳"])
    cache.add(macbook_slot)
    cache.add(make_slot("ctx_iphone", "iPhone 17 Pro"))
    # 切回 macbook
    cache.access("ctx_macbook")
    current = cache.get_current()
    assert current.entity == "MacBook Air M3"
    assert "保护壳" in current.excluded


# ── _build_resolved_input 测试 ────────────────────────────────────────────────

def test_resolved_input_pronoun_replacement():
    candidate = {"match_type": "pronoun", "entity": "MacBook Air M3"}
    result = _build_resolved_input("这个多少钱", candidate)
    assert "MacBook Air M3" in result


def test_resolved_input_no_change_when_no_entity():
    candidate = {"match_type": "new", "entity": ""}
    result = _build_resolved_input("今天吃什么", candidate)
    assert result == "今天吃什么"


def test_resolved_input_entity_match_appends():
    candidate = {"match_type": "entity_match", "entity": "iPhone 16"}
    result = _build_resolved_input("多少钱", candidate)
    assert "iPhone 16" in result


# ── commit_context 测试 ───────────────────────────────────────────────────────

def _make_commit_state(intent, candidate):
    return {
        "user_id": "test_user",
        "user_input": "测试",
        "intent": intent,
        "confidence": "HIGH",
        "missing_slots": [],
        "context_candidate": candidate,
        "turn_excluded": [],
        "turn_constraints": {},
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


@pytest.mark.asyncio
async def test_commit_reuses_pronoun_context():
    """pronoun + domain 一致 → 复用已有 Slot，ctx_id 保持"""
    cache = TopicCache.empty()
    slot = ContextSlot(
        ctx_id="ctx_test",
        domain="electronics",
        entity="MacBook Air M3",
        topic_turns=1,
        constraints={},
        excluded=[],
        last_intent="ELECTRONICS_PRICE",
        last_active="",
    )
    cache.add(slot)

    mock_redis = MagicMock()
    _cache_ok = MagicMock(ok=True, data=cache, error=None)
    _save_ok = MagicMock(ok=True, error=None)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=_cache_ok)):
        with patch("agents.context_manager.save_topic_cache_result", new=AsyncMock(return_value=_save_ok)):
            state = _make_commit_state("ELECTRONICS_PRICE", {
                "ctx_id": "ctx_test",
                "match_type": "pronoun",
                "entity": "MacBook Air M3",
                "domain": "electronics",  # 与 ELECTRONICS_PRICE domain 一致
                "existing_excluded": [],
                "existing_constraints": {},
                "turn_excluded": [],
                "turn_constraints": {},
            })
            result = await commit_context(state, mock_redis)
            # pronoun + domain 一致 → 复用
            assert result["session_context"]["ctx_id"] == "ctx_test"
            assert result["session_context"]["persistence_status"] == "saved"


@pytest.mark.asyncio
async def test_commit_creates_new_when_domain_mismatch():
    """候选 domain electronics，实际 intent MEAL(food)：domain 不匹配，新建 context"""
    cache = TopicCache.empty()
    slot = ContextSlot(
        ctx_id="ctx_electronics",
        domain="electronics",
        entity="MacBook Air M3",
        topic_turns=1,
        constraints={},
        excluded=[],
        last_intent="ELECTRONICS_PRICE",
        last_active="",
    )
    cache.add(slot)

    mock_redis = MagicMock()
    _cache_ok = MagicMock(ok=True, data=cache, error=None)
    _save_ok = MagicMock(ok=True, error=None)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=_cache_ok)):
        with patch("agents.context_manager.save_topic_cache_result", new=AsyncMock(return_value=_save_ok)):
            state = _make_commit_state("MEAL", {
                "ctx_id": "ctx_electronics",
                "match_type": "entity_match",
                "entity": "番茄炒蛋",
                "domain": "electronics",  # 候选 domain=electronics，MEAL→food，不匹配
                "existing_excluded": [],
                "existing_constraints": {},
                "turn_excluded": [],
                "turn_constraints": {},
            })
            result = await commit_context(state, mock_redis)
            # domain 不匹配 → 新建（不复用 ctx_electronics）
            assert result["session_context"]["ctx_id"] != "ctx_electronics"


@pytest.mark.asyncio
async def test_commit_pending_when_missing_slots():
    """MEDIUM + missing_slots 非空 → context_status 应为 pending，仍持久化（entity 非空）"""
    cache = TopicCache.empty()
    mock_redis = MagicMock()
    _cache_ok = MagicMock(ok=True, data=cache, error=None)
    _save_ok = MagicMock(ok=True, error=None)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=_cache_ok)):
        with patch("agents.context_manager.save_topic_cache_result", new=AsyncMock(return_value=_save_ok)):
            state = _make_commit_state("ELECTRONICS_PRICE", {
                "ctx_id": "",
                "match_type": "new",
                "entity": "苹果手机",
                "domain": "electronics",
                "existing_excluded": [],
                "existing_constraints": {},
                "turn_excluded": [],
                "turn_constraints": {},
            })
            state["confidence"] = "MEDIUM"
            state["missing_slots"] = ["model"]
            result = await commit_context(state, mock_redis)
            assert result["context_status"] == "pending"
            assert result["session_context"]["context_status"] == "pending"
