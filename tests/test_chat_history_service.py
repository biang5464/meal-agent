"""
tests/test_chat_history_service.py
Phase 6A: get_chat_history_result 单元测试

全部 mock tool_executor.execute，不依赖真实 Redis 或 MySQL。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult

_HISTORY = [
    {"role": "user", "content": "你好", "timestamp": "2026-07-01T00:00:00"},
    {"role": "assistant", "content": "你好！", "timestamp": "2026-07-01T00:00:01"},
]


def _ok(data, source="chat_cache"):
    return ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(tool_name="get_chat_history", elapsed_ms=30, attempts=1, source=source),
    )


def _fail(data=None, fallback_used=False):
    return ToolResult(
        ok=False,
        data=data,
        error=ToolError(code=ToolErrorCode.NETWORK, message="Redis down", retryable=True),
        meta=ToolMeta(tool_name="get_chat_history", elapsed_ms=300, attempts=2, fallback_used=fallback_used),
    )


@pytest.mark.asyncio
async def test_chat_history_found():
    """正常读取历史 → ok=True, data=[messages]。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(_HISTORY))
        from core.cache import get_chat_history_result
        result = await get_chat_history_result("u1")
    assert result.ok is True
    assert len(result.data) == 2
    assert result.data[0]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_history_empty_new_user():
    """新用户无历史 → ok=True, data=[]（成功无数据，区别于 Redis 故障）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok([]))
        from core.cache import get_chat_history_result
        result = await get_chat_history_result("new_user")
    assert result.ok is True
    assert result.data == []


@pytest.mark.asyncio
async def test_chat_history_redis_failure():
    """Redis 连接失败 → ok=False, data=[], fallback_used=True。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail(data=[], fallback_used=True))
        from core.cache import get_chat_history_result
        result = await get_chat_history_result("u1")
    assert result.ok is False
    assert result.data == []
    assert result.meta.fallback_used is True
    assert result.error.code == ToolErrorCode.NETWORK


@pytest.mark.asyncio
async def test_chat_history_uses_database_read_policy():
    """get_chat_history_result 应使用 database_read 策略（容纳 SQLite fallback 路径）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(_HISTORY))
        from core.cache import get_chat_history_result
        await get_chat_history_result("u1")
    call_kwargs = m.execute.call_args.kwargs
    assert call_kwargs.get("policy_name") == "database_read"


@pytest.mark.asyncio
async def test_chat_history_meta_preserved():
    """成功路径 meta 透传。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=ToolResult(
            ok=True,
            data=_HISTORY,
            meta=ToolMeta(tool_name="get_chat_history", elapsed_ms=150, attempts=1, source="chat_cache"),
        ))
        from core.cache import get_chat_history_result
        result = await get_chat_history_result("u1")
    assert result.meta.elapsed_ms == 150
    assert result.meta.source == "chat_cache"
