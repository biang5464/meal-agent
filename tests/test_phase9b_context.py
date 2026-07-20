"""Phase 9B tests: Entity Recovery 只读语义、防御性拷贝、Supervisor Context Hint。"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.session_context import ContextSlot, TopicCache


# ── 公用 state 工厂 ───────────────────────────────────────────────────────────

def _make_state(user_input: str = "MacBook Air价格") -> dict:
    return {
        "user_id": "u1",
        "user_input": user_input,
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
        "chat_history": [],
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
    }


def _make_slot(entity: str, excluded: list | None = None, constraints: dict | None = None) -> ContextSlot:
    return ContextSlot(
        ctx_id="ctx_test",
        domain="electronics",
        entity=entity,
        topic_turns=1,
        constraints=constraints or {},
        excluded=excluded or [],
        last_intent="ELECTRONICS_PRICE",
        last_active="",
    )


# ── 1. resolve_context_candidate 只读语义 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_candidate_does_not_write_redis():
    """resolve_context_candidate 不应调用 save_topic_cache_result（只读操作）。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air")), \
         patch("agents.context_manager.save_topic_cache_result", new=AsyncMock()) as mock_save:
        await resolve_context_candidate(_make_state(), MagicMock())

    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_candidate_reads_topic_cache_with_tool_result():
    """resolve_context_candidate 应使用 load_topic_cache_result（ToolResult 协议）。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)) as mock_load, \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        await resolve_context_candidate(_make_state(), MagicMock())

    mock_load.assert_called_once()


# ── 2. 防御性拷贝：existing_excluded ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_existing_excluded_is_defensive_copy():
    """candidate["existing_excluded"] 修改后不影响原始 slot.excluded。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3", excluded=["保护壳"])
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        # "它" 触发 pronoun match → matched_slot = slot
        result = await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    candidate = result["context_candidate"]
    assert candidate["existing_excluded"] == ["保护壳"]

    # 修改候选副本，原始 slot 应不受影响
    candidate["existing_excluded"].append("皮质")
    assert slot.excluded == ["保护壳"], "existing_excluded 应为防御性拷贝"


@pytest.mark.asyncio
async def test_existing_excluded_not_same_reference():
    """candidate["existing_excluded"] 应是新列表对象，不与 slot.excluded 共享引用。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3", excluded=["保护壳"])
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    assert result["context_candidate"]["existing_excluded"] is not slot.excluded


# ── 3. 防御性拷贝：existing_constraints ──────────────────────────────────────

@pytest.mark.asyncio
async def test_existing_constraints_is_defensive_copy():
    """candidate["existing_constraints"] 修改后不影响原始 slot.constraints。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3", constraints={"color": "silver"})
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    candidate = result["context_candidate"]
    assert candidate["existing_constraints"] == {"color": "silver"}

    # 修改候选副本，原始 slot 应不受影响
    candidate["existing_constraints"]["size"] = "13inch"
    assert slot.constraints == {"color": "silver"}, "existing_constraints 应为防御性拷贝"


@pytest.mark.asyncio
async def test_existing_constraints_not_same_reference():
    """candidate["existing_constraints"] 应是新 dict 对象，不与 slot.constraints 共享引用。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3", constraints={"color": "silver"})
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    assert result["context_candidate"]["existing_constraints"] is not slot.constraints


# ── 4. _build_candidate_hint 格式验证 ────────────────────────────────────────

def test_build_candidate_hint_empty_candidate():
    from agents.supervisor_agent import _build_candidate_hint
    assert _build_candidate_hint({}) == ""


def test_build_candidate_hint_no_entity_no_excluded():
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {"entity": "", "match_type": "new", "turn_excluded": [], "existing_excluded": []}
    assert _build_candidate_hint(candidate) == ""


def test_build_candidate_hint_entity_only():
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {"entity": "iPhone 16", "match_type": "entity_match", "turn_excluded": [], "existing_excluded": []}
    hint = _build_candidate_hint(candidate)
    assert "iPhone 16" in hint
    assert "候选" in hint
    # match_type 是内部技术字段，不应暴露给 LLM
    assert "entity_match" not in hint


def test_build_candidate_hint_with_entity_and_excluded():
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "MacBook Air M3",
        "match_type": "pronoun",
        "turn_excluded": ["保护壳"],
        "existing_excluded": ["皮质"],
    }
    hint = _build_candidate_hint(candidate)
    assert "MacBook Air M3" in hint
    assert "保护壳" in hint
    assert "皮质" in hint
    assert "候选" in hint


def test_build_candidate_hint_excluded_only():
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {"entity": "", "match_type": "new", "turn_excluded": ["重"], "existing_excluded": []}
    hint = _build_candidate_hint(candidate)
    assert "重" in hint


def test_build_candidate_hint_not_impersonate_session_label():
    """候选 hint 不应使用【当前会话上下文】标签（那是 commit 后的 session_context 标签）。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "MacBook Air M3",
        "match_type": "pronoun",
        "turn_excluded": [],
        "existing_excluded": [],
    }
    hint = _build_candidate_hint(candidate)
    assert "【当前会话上下文】" not in hint


# ── 5. Supervisor 使用 candidate_hint ────────────────────────────────────────

def _make_supervisor_state(entity: str, existing_excluded: list | None = None, turn_excluded: list | None = None) -> dict:
    state = _make_state("它多少钱")
    state["context_candidate"] = {
        "entity": entity,
        "match_type": "pronoun",
        "turn_excluded": turn_excluded or [],
        "existing_excluded": existing_excluded or [],
    }
    state["resolved_entity"] = entity
    return state


@pytest.mark.asyncio
async def test_supervisor_candidate_entity_in_prompt():
    """supervisor_agent 的 prompt 中应包含 candidate 的实体名。"""
    from core.tool_protocol import ToolResult
    from agents.supervisor_agent import supervisor_agent

    class _FakeResp:
        content = "INTENT: ELECTRONICS_PRICE\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"

    captured: list = []

    async def _fake_execute(fn, messages, **kwargs):
        captured.extend(messages)
        return ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)

    with patch("tools.tool_executor.tool_executor.execute", side_effect=_fake_execute):
        await supervisor_agent(_make_supervisor_state("MacBook Air M3"))

    assert captured, "execute 应被调用一次"
    full_prompt = captured[0].content
    assert "MacBook Air M3" in full_prompt


@pytest.mark.asyncio
async def test_supervisor_existing_excluded_in_prompt():
    """supervisor_agent 的 prompt 中应包含 existing_excluded 信息。"""
    from core.tool_protocol import ToolResult
    from agents.supervisor_agent import supervisor_agent

    class _FakeResp:
        content = "INTENT: ELECTRONICS_PRICE\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"

    captured: list = []

    async def _fake_execute(fn, messages, **kwargs):
        captured.extend(messages)
        return ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)

    with patch("tools.tool_executor.tool_executor.execute", side_effect=_fake_execute):
        await supervisor_agent(_make_supervisor_state("MacBook Air M3", existing_excluded=["保护壳"]))

    full_prompt = captured[0].content
    assert "保护壳" in full_prompt


@pytest.mark.asyncio
async def test_supervisor_turn_excluded_in_prompt():
    """supervisor_agent 的 prompt 中应包含本轮 turn_excluded。"""
    from core.tool_protocol import ToolResult
    from agents.supervisor_agent import supervisor_agent

    class _FakeResp:
        content = "INTENT: MEAL\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"

    captured: list = []

    async def _fake_execute(fn, messages, **kwargs):
        captured.extend(messages)
        return ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)

    state = _make_supervisor_state("", turn_excluded=["重"])
    with patch("tools.tool_executor.tool_executor.execute", side_effect=_fake_execute):
        await supervisor_agent(state)

    full_prompt = captured[0].content
    assert "重" in full_prompt


@pytest.mark.asyncio
async def test_supervisor_candidate_hint_not_impersonate_session_context():
    """candidate hint 不应触发【当前会话上下文】标签（session_context 为空时该标签不应出现）。"""
    from core.tool_protocol import ToolResult
    from agents.supervisor_agent import supervisor_agent

    class _FakeResp:
        content = "INTENT: CHAT\nCONFIDENCE: LOW\nMISSING_SLOTS: NONE\nTOP2: NONE"

    captured: list = []

    async def _fake_execute(fn, messages, **kwargs):
        captured.extend(messages)
        return ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)

    state = _make_supervisor_state("MacBook Air M3")
    # session_context 为空（正常图流程中 commit 前的状态）
    state["session_context"] = {}

    with patch("tools.tool_executor.tool_executor.execute", side_effect=_fake_execute):
        await supervisor_agent(state)

    full_prompt = captured[0].content
    assert "【当前会话上下文】" not in full_prompt
    assert "候选" in full_prompt


@pytest.mark.asyncio
async def test_supervisor_empty_candidate_no_hint():
    """context_candidate 为空时 prompt 中不出现"候选上下文"块。"""
    from core.tool_protocol import ToolResult
    from agents.supervisor_agent import supervisor_agent

    class _FakeResp:
        content = "INTENT: MEAL\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"

    captured: list = []

    async def _fake_execute(fn, messages, **kwargs):
        captured.extend(messages)
        return ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)

    state = _make_state("今天吃什么")
    state["context_candidate"] = {}

    with patch("tools.tool_executor.tool_executor.execute", side_effect=_fake_execute):
        await supervisor_agent(state)

    full_prompt = captured[0].content
    assert "候选上下文" not in full_prompt


# ── 6. entity_source 元数据 ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_entity_source_pronoun():
    """代词引用 → entity_source = 'pronoun'。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    assert result["context_candidate"]["entity_source"] == "pronoun"


@pytest.mark.asyncio
async def test_entity_source_llm_new_entity():
    """LLM 提取到与 last_entity 不同的新实体 → entity_source = 'llm'。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("iPhone 16")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state("MacBook Air M3价格"), MagicMock())

    assert result["context_candidate"]["entity_source"] == "llm"


@pytest.mark.asyncio
async def test_entity_source_none_when_no_entity():
    """LLM 未提取到实体（返回空）→ entity_source = 'none'。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        result = await resolve_context_candidate(_make_state("今天吃什么"), MagicMock())

    assert result["context_candidate"]["entity_source"] == "none"


@pytest.mark.asyncio
async def test_entity_source_llm_fallback():
    """LLM 超时或失败，返回 last_entity → entity_source = 'llm_fallback'。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    # extract_entity 返回 last_entity 值（模拟 LLM 失败 fallback）
    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        # user_input 不含代词，且返回值等于 last_entity → llm_fallback
        result = await resolve_context_candidate(_make_state("多少钱"), MagicMock())

    assert result["context_candidate"]["entity_source"] == "llm_fallback"


# ── 7. cache_read_status 元数据 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_read_status_hit():
    """TopicCache 中有 context → cache_read_status = 'hit'。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        result = await resolve_context_candidate(_make_state(), MagicMock())

    assert result["context_candidate"]["cache_read_status"] == "hit"


@pytest.mark.asyncio
async def test_cache_read_status_miss():
    """TopicCache 为空（key 不存在）→ cache_read_status = 'miss'。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        result = await resolve_context_candidate(_make_state(), MagicMock())

    assert result["context_candidate"]["cache_read_status"] == "miss"


@pytest.mark.asyncio
async def test_cache_read_status_degraded():
    """Redis 故障 → cache_read_status = 'degraded'。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=False, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        result = await resolve_context_candidate(_make_state(), MagicMock())

    assert result["context_candidate"]["cache_read_status"] == "degraded"


# ── 8. existing_constraints 在 Hint 中 ───────────────────────────────────────

def test_build_candidate_hint_with_existing_constraints():
    """existing_constraints 非空时应出现在 hint 中。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "MacBook Air M3",
        "match_type": "entity_match",
        "turn_excluded": [],
        "existing_excluded": [],
        "existing_constraints": {"color": "silver"},
    }
    hint = _build_candidate_hint(candidate)
    assert "silver" in hint
    assert "历史约束" in hint


def test_build_candidate_hint_constraints_only():
    """entity/excluded 均空但 constraints 非空时 hint 也不为空。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "",
        "match_type": "new",
        "turn_excluded": [],
        "existing_excluded": [],
        "existing_constraints": {"size": "13inch"},
    }
    hint = _build_candidate_hint(candidate)
    assert "13inch" in hint


def test_build_candidate_hint_empty_constraints_skipped():
    """existing_constraints 为空 dict 时不添加约束行。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "iPhone 16",
        "match_type": "entity_match",
        "turn_excluded": [],
        "existing_excluded": [],
        "existing_constraints": {},
    }
    hint = _build_candidate_hint(candidate)
    assert "历史约束" not in hint


# ── 9. 安全边界 ───────────────────────────────────────────────────────────────

def test_sanitize_str_truncates_long_string():
    """_sanitize_str 超过 max_len 时截断。"""
    from agents.supervisor_agent import _sanitize_str
    long_str = "A" * 100
    result = _sanitize_str(long_str, max_len=50)
    assert len(result) == 50


def test_sanitize_str_removes_control_chars():
    """_sanitize_str 去除 \\n、\\r、\\0 等控制字符。"""
    from agents.supervisor_agent import _sanitize_str
    result = _sanitize_str("hello\nworld\r\x00")
    assert "\n" not in result
    assert "\r" not in result
    assert "\x00" not in result
    assert "hello" in result
    assert "world" in result


def test_sanitize_list_limits_items():
    """_sanitize_list 最多返回 max_items 条。"""
    from agents.supervisor_agent import _sanitize_list
    items = [f"item_{i}" for i in range(20)]
    result = _sanitize_list(items, max_items=5)
    assert len(result) == 5


def test_sanitize_list_filters_non_string():
    """_sanitize_list 过滤掉非 str 类型。"""
    from agents.supervisor_agent import _sanitize_list
    items = ["保护壳", 123, None, "皮质"]
    result = _sanitize_list(items)
    assert result == ["保护壳", "皮质"]


def test_build_candidate_hint_entity_truncated():
    """超长 entity 应被截断，不会破坏 hint 结构。"""
    from agents.supervisor_agent import _build_candidate_hint
    long_entity = "A" * 200
    candidate = {"entity": long_entity, "match_type": "entity_match", "turn_excluded": [], "existing_excluded": []}
    hint = _build_candidate_hint(candidate)
    assert "候选实体" in hint
    # hint 中出现的实体最多 50 字符
    import re
    m = re.search(r"候选实体：(.+)", hint)
    assert m and len(m.group(1)) <= 50


def test_build_candidate_hint_excluded_list_limit():
    """excluded 超过 5 条时 hint 只展示前 5 条。"""
    from agents.supervisor_agent import _build_candidate_hint
    many_excluded = [f"条件{i}" for i in range(10)]
    candidate = {
        "entity": "",
        "match_type": "new",
        "turn_excluded": many_excluded,
        "existing_excluded": [],
    }
    hint = _build_candidate_hint(candidate)
    # 最多展示 5 条，条件5~9 不应出现
    for i in range(5, 10):
        assert f"条件{i}" not in hint


def test_build_candidate_hint_control_chars_in_excluded():
    """excluded 中包含控制字符时应被过滤，不在 prompt 中额外换行。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "",
        "match_type": "new",
        "turn_excluded": ["保护壳\n注入攻击"],
        "existing_excluded": [],
    }
    hint = _build_candidate_hint(candidate)
    # hint 本身以 \n 分隔结构行，属正常；验证排除项中的 \n 已被去除
    # 即"本轮排除："那一行只有 1 行（不因 \n 而多出额外行）
    excluded_lines = [l for l in hint.split("\n") if "本轮排除" in l]
    assert len(excluded_lines) == 1, "排除项中的控制字符应被过滤，不应额外换行"
    assert "保护壳" in excluded_lines[0]


# ── 10. TopicCache / ContextSlot / user_input 不变验证 ───────────────────────

@pytest.mark.asyncio
async def test_resolve_candidate_does_not_modify_topic_cache():
    """resolve_context_candidate 执行后，TopicCache 中 context 数量不变。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    original_context_count = len(cache.contexts)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    assert len(cache.contexts) == original_context_count


@pytest.mark.asyncio
async def test_resolve_candidate_does_not_modify_slot_attributes():
    """resolve_context_candidate 执行后，ContextSlot 的 excluded/constraints 不变。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3", excluded=["保护壳"], constraints={"color": "silver"})
    cache = TopicCache.empty()
    cache.add(slot)
    original_excluded = list(slot.excluded)
    original_constraints = dict(slot.constraints)
    cache_result = MagicMock(ok=True, data=cache)

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        await resolve_context_candidate(_make_state("它多少钱"), MagicMock())

    assert slot.excluded == original_excluded
    assert slot.constraints == original_constraints


@pytest.mark.asyncio
async def test_resolve_candidate_does_not_modify_user_input():
    """resolve_context_candidate 不应修改原始 user_input（原始输入保持不变）。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)
    original_input = "MacBook Air M3价格"

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(_make_state(original_input), MagicMock())

    assert result["user_input"] == original_input


# ── 11. 否定词不再 carry last_entity ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_negation_does_not_skip_llm():
    """含否定词的输入不应跳过 LLM 提取，应正常调用 extract_entity。"""
    from agents.context_manager import resolve_context_candidate

    cache = TopicCache.empty()
    cache_result = MagicMock(ok=True, data=cache)
    llm_called = []

    async def _fake_extract(user_input, last_entity=""):
        llm_called.append(user_input)
        return "MacBook Air M3"

    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", side_effect=_fake_extract):
        await resolve_context_candidate(_make_state("不要保护壳，MacBook Air M3的价格"), MagicMock())

    # extract_entity 应被调用（否定词不再短路）
    assert len(llm_called) == 1


@pytest.mark.asyncio
async def test_negation_with_new_entity_extracts_correctly():
    """'不要X，查Y'类输入应能提取到新实体 Y，而不是返回旧实体。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("iPhone 16")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    # 模拟 LLM 从含否定词的输入中提取到新实体
    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(
            _make_state("不要保护壳，MacBook Air M3的价格"), MagicMock()
        )

    # 应提取到 MacBook Air M3，而非旧实体 iPhone 16
    assert result["context_candidate"]["entity"] == "MacBook Air M3"
    assert result["context_candidate"]["entity_source"] == "llm"


# ── 12. Hint 中不暴露英文技术字段 ────────────────────────────────────────────

def test_hint_does_not_expose_match_type_to_llm():
    """_build_candidate_hint 不应在 prompt 中出现 match_type 英文值。"""
    from agents.supervisor_agent import _build_candidate_hint
    for match_type in ("pronoun", "entity_match", "embedding_match", "new"):
        candidate = {
            "entity": "iPhone 16",
            "match_type": match_type,
            "turn_excluded": [],
            "existing_excluded": [],
        }
        hint = _build_candidate_hint(candidate)
        assert match_type not in hint, f"match_type='{match_type}' 不应出现在 hint 中"


# ── 13. Final Fix Batch: 否定语义精确来源 ────────────────────────────────────

@pytest.mark.asyncio
async def test_negation_only_entity_source_is_negation_carry():
    """纯否定输入（无新实体信号）的 entity_source 应为 negation_carry。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    state = _make_state("不要保护壳")
    # extract_entity 内部短路，返回 last_entity；这里 patch 以防 LLM 调用
    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(state, MagicMock())

    assert result["context_candidate"]["entity_source"] == "negation_carry"


@pytest.mark.asyncio
async def test_negation_only_preserves_last_entity():
    """纯否定输入应保留 last_entity 作为 entity，不丢失已知主题。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    state = _make_state("不要保护壳")
    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="MacBook Air M3")):
        result = await resolve_context_candidate(state, MagicMock())

    assert result["context_candidate"]["entity"] == "MacBook Air M3"


@pytest.mark.asyncio
async def test_entity_source_slot_carry_when_llm_returns_empty():
    """LLM 返回空字符串时，若 matched_slot 有 entity，来源应为 slot_carry。"""
    from agents.context_manager import resolve_context_candidate

    slot = _make_slot("MacBook Air M3")
    cache = TopicCache.empty()
    cache.add(slot)
    cache_result = MagicMock(ok=True, data=cache)

    # LLM 返回空 → slot_carry
    with patch("agents.context_manager.load_topic_cache_result", new=AsyncMock(return_value=cache_result)), \
         patch("agents.context_manager.extract_entity", new=AsyncMock(return_value="")):
        result = await resolve_context_candidate(_make_state("多少钱"), MagicMock())

    assert result["context_candidate"]["entity"] == "MacBook Air M3"
    assert result["context_candidate"]["entity_source"] == "slot_carry"


# ── 14. Final Fix Batch: Candidate Hint 健壮性 ──────────────────────────────

def test_build_candidate_hint_non_dict_returns_empty():
    """candidate 非 dict 类型时，_build_candidate_hint 应安全返回空串。"""
    from agents.supervisor_agent import _build_candidate_hint
    assert _build_candidate_hint(["bad"]) == ""
    assert _build_candidate_hint("string") == ""
    assert _build_candidate_hint(42) == ""


def test_build_candidate_hint_with_none_excluded():
    """turn_excluded 为 None 时，_build_candidate_hint 不应抛异常。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "iPhone 16",
        "turn_excluded": None,
        "existing_excluded": None,
    }
    hint = _build_candidate_hint(candidate)
    # 不抛异常即可；entity 存在则 hint 非空
    assert "iPhone 16" in hint


def test_sanitize_list_with_none_returns_empty():
    """_sanitize_list(None) 应返回空列表，不抛异常。"""
    from agents.supervisor_agent import _sanitize_list
    assert _sanitize_list(None) == []


def test_sanitize_list_with_non_iterable_returns_empty():
    """_sanitize_list 收到不可迭代对象时应返回空列表，不抛异常。"""
    from agents.supervisor_agent import _sanitize_list
    assert _sanitize_list(123) == []
    assert _sanitize_list(3.14) == []


def test_constraints_with_numeric_value():
    """existing_constraints 中数字值（int/float）应被包含在 hint 中，不静默丢弃。"""
    from agents.supervisor_agent import _build_candidate_hint
    candidate = {
        "entity": "",
        "turn_excluded": [],
        "existing_excluded": [],
        "existing_constraints": {"budget": 100, "min_score": 4.5},
    }
    hint = _build_candidate_hint(candidate)
    assert "budget=100" in hint
    assert "min_score=4.5" in hint
