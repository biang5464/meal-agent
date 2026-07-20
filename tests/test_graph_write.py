"""
tests/test_graph_write.py
Phase 6B：graph.py 写入路径和 TopicCache 保存 ToolResult 协议测试。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import redis as redis_lib


def _ok(tool_name="test", data=None):
    from core.tool_protocol import tool_success
    return tool_success(data, tool_name=tool_name)


def _fail(tool_name="test"):
    from core.tool_protocol import tool_failure, ToolErrorCode
    return tool_failure(ToolErrorCode.NETWORK, "down", tool_name=tool_name)


# ── append_chat_message_result 在 graph 路径中 ────────────────────────────────

@pytest.mark.asyncio
async def test_append_chat_message_result_called_in_graph_run():
    """graph.run() 应调用 append_chat_message_result，不再直接调用 append_chat_message。"""
    from agents.graph import run

    with (
        patch("agents.graph.get_chat_history_result",
              new=AsyncMock(return_value=_ok("get_chat_history", []))),
        patch("agents.graph.compiled_graph") as mock_graph,
        patch("agents.graph.append_chat_message_result",
              new=AsyncMock(return_value=_ok("append_chat_message"))) as mock_append,
    ):
        mock_graph.ainvoke = AsyncMock(return_value={"result": "你好！", "intent": "CHAT"})
        await run("u1", "你好")

    assert mock_append.call_count == 2, "应追加 user 和 assistant 两条消息"
    roles = [call.args[1] for call in mock_append.call_args_list]
    assert "user" in roles and "assistant" in roles


@pytest.mark.asyncio
async def test_append_chat_message_result_failure_does_not_raise():
    """append_chat_message_result 失败时，graph.run() 应记录警告但不抛异常。"""
    from agents.graph import run

    with (
        patch("agents.graph.get_chat_history_result",
              new=AsyncMock(return_value=_ok("get_chat_history", []))),
        patch("agents.graph.compiled_graph") as mock_graph,
        patch("agents.graph.append_chat_message_result",
              new=AsyncMock(return_value=_fail("append_chat_message"))),
    ):
        mock_graph.ainvoke = AsyncMock(return_value={"result": "回复", "intent": "CHAT"})
        result = await run("u1", "你好")

    assert result == "回复"  # 主流程正常返回


# ── save_topic_cache_result in commit_context ─────────────────────────────────

@pytest.mark.asyncio
async def test_save_topic_cache_result_ok():
    """TopicCache 保存成功 → commit_context 正常返回 session_context，persistence_status='saved'。"""
    from agents.context_manager import commit_context
    from agents.session_context import TopicCache

    state = {
        "user_id": "u1", "user_input": "今天吃什么", "intent": "MEAL",
        "confidence": "HIGH", "missing_slots": [],
        "context_candidate": {"ctx_id": "", "match_type": "new", "entity": "番茄炒蛋",
                               "domain": "", "existing_excluded": [], "existing_constraints": {},
                               "turn_excluded": [], "turn_constraints": {}},
        "turn_excluded": [], "turn_constraints": {},
    }

    with (
        patch("agents.context_manager.load_topic_cache_result",
              new=AsyncMock(return_value=MagicMock(ok=True, data=TopicCache.empty()))),
        patch("agents.context_manager.save_topic_cache_result",
              new=AsyncMock(return_value=_ok("save_topic_cache"))),
    ):
        result = await commit_context(state, MagicMock())

    assert "session_context" in result
    assert result["session_context"]["persistence_status"] == "saved"


@pytest.mark.asyncio
async def test_save_topic_cache_result_failure_does_not_block():
    """TopicCache 保存失败 → commit_context 记录 warning，仍返回 session_context，persistence_status='write_degraded'。"""
    from agents.context_manager import commit_context
    from agents.session_context import TopicCache

    state = {
        "user_id": "u1", "user_input": "今天吃什么", "intent": "MEAL",
        "confidence": "HIGH", "missing_slots": [],
        "context_candidate": {"ctx_id": "", "match_type": "new", "entity": "番茄炒蛋",
                               "domain": "", "existing_excluded": [], "existing_constraints": {},
                               "turn_excluded": [], "turn_constraints": {}},
        "turn_excluded": [], "turn_constraints": {},
    }

    with (
        patch("agents.context_manager.load_topic_cache_result",
              new=AsyncMock(return_value=MagicMock(ok=True, data=TopicCache.empty()))),
        patch("agents.context_manager.save_topic_cache_result",
              new=AsyncMock(return_value=_fail("save_topic_cache"))),
    ):
        result = await commit_context(state, MagicMock())

    assert "session_context" in result
    assert result["session_context"]["persistence_status"] == "write_degraded"


# ── save_topic_cache_result TTL & key ────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_topic_cache_result_ttl_and_key():
    """save_topic_cache_result 使用 TTL=7200，Key=topic_cache:{user_id}。"""
    from agents.context_manager import save_topic_cache_result, REDIS_SESSION_TTL
    from agents.session_context import TopicCache

    assert REDIS_SESSION_TTL == 7200, f"REDIS_SESSION_TTL 应为 7200，实际 {REDIS_SESSION_TTL}"

    with patch("core.cache._client") as mock_client:
        mock_client.setex = MagicMock()
        result = await save_topic_cache_result("test_user", TopicCache.empty())

    assert result.ok
    call_args = mock_client.setex.call_args
    key_arg = call_args.args[0] if call_args.args else call_args[0][0]
    ttl_arg = call_args.args[1] if len(call_args.args) > 1 else call_args[0][1]
    assert key_arg == "topic_cache:test_user", f"Key 错误: {key_arg}"
    assert ttl_arg == 7200, f"TTL 错误: {ttl_arg}"
