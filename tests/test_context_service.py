"""
tests/test_context_service.py
Phase 6A: load_topic_cache_result 单元测试

全部 mock tool_executor.execute，不依赖真实 Redis。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.session_context import ContextSlot, TopicCache
from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult


def _make_topic_cache() -> TopicCache:
    cache = TopicCache.empty()
    cache.add(ContextSlot(
        ctx_id="ctx_test",
        domain="electronics",
        entity="MacBook Air M3",
        topic_turns=2,
        constraints={},
        excluded=["保护壳"],
        last_intent="ELECTRONICS_PRICE",
        last_active="2026-07-01T00:00:00",
    ))
    return cache


def _ok(data):
    return ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(tool_name="load_topic_cache", elapsed_ms=20, attempts=1, source="topic_cache"),
    )


def _fail(data=None, fallback_used=False):
    return ToolResult(
        ok=False,
        data=data,
        error=ToolError(code=ToolErrorCode.NETWORK, message="Redis down", retryable=True),
        meta=ToolMeta(tool_name="load_topic_cache", elapsed_ms=300, attempts=2, fallback_used=fallback_used),
    )


@pytest.mark.asyncio
async def test_topic_cache_hit():
    """Redis 命中 → ok=True, data=TopicCache 包含上下文数据。"""
    cache = _make_topic_cache()
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(cache))
        from agents.context_manager import load_topic_cache_result
        result = await load_topic_cache_result("u1", MagicMock())
    assert result.ok is True
    assert isinstance(result.data, TopicCache)
    assert "ctx_test" in result.data.contexts


@pytest.mark.asyncio
async def test_topic_cache_miss():
    """Redis miss（key 不存在）→ ok=True, data=TopicCache.empty()。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(TopicCache.empty()))
        from agents.context_manager import load_topic_cache_result
        result = await load_topic_cache_result("u_new", MagicMock())
    assert result.ok is True
    assert isinstance(result.data, TopicCache)
    assert len(result.data.contexts) == 0


@pytest.mark.asyncio
async def test_topic_cache_redis_failure():
    """Redis 故障 → ok=False, data=TopicCache.empty(), fallback_used=True。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail(data=TopicCache.empty(), fallback_used=True))
        from agents.context_manager import load_topic_cache_result
        result = await load_topic_cache_result("u1", MagicMock())
    assert result.ok is False
    assert isinstance(result.data, TopicCache)
    assert result.meta.fallback_used is True
    assert result.error.code == ToolErrorCode.NETWORK


@pytest.mark.asyncio
async def test_topic_cache_data_always_usable():
    """无论成功还是 Redis 故障，result.data 必须是可用的 TopicCache（不为 None）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail(data=TopicCache.empty(), fallback_used=True))
        from agents.context_manager import load_topic_cache_result
        result = await load_topic_cache_result("u1", MagicMock())
    # 调用方可以安全地访问 result.data.contexts、result.data.get_current() 等
    assert result.data is not None
    assert result.data.get_current() is None  # empty cache


@pytest.mark.asyncio
async def test_topic_cache_uses_redis_read_policy():
    """load_topic_cache_result 应使用 redis_read 策略。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(TopicCache.empty()))
        from agents.context_manager import load_topic_cache_result
        await load_topic_cache_result("u1", MagicMock())
    call_kwargs = m.execute.call_args.kwargs
    assert call_kwargs.get("policy_name") == "redis_read"
