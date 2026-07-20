"""
tests/test_profile_service.py
Phase 6A: get_profile_cache_result / get_user_profile_result 单元测试

全部 mock tool_executor.execute，不依赖真实 Redis 或 SQLite。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult


def _ok(data=None, source="redis"):
    return ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(tool_name="t", elapsed_ms=10, attempts=1, source=source),
    )


def _fail(code: ToolErrorCode, data=None, fallback_used=False):
    return ToolResult(
        ok=False,
        data=data,
        error=ToolError(code=code, message="mock error", retryable=True),
        meta=ToolMeta(tool_name="t", elapsed_ms=300, attempts=2, fallback_used=fallback_used),
    )


# ── get_profile_cache_result ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_cache_hit():
    """Redis 命中 → ok=True, data=profile_dict, source="redis"。"""
    profile = {"user_id": "u1", "likes": ["辣"], "dislikes": []}
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(profile, source="redis"))
        from core.cache import get_profile_cache_result
        result = await get_profile_cache_result("u1")
    assert result.ok is True
    assert result.data == profile
    assert result.meta.source == "redis"


@pytest.mark.asyncio
async def test_profile_cache_miss():
    """Redis miss（key 不存在）→ ok=True, data=None, source="redis_miss"。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(None, source="redis"))
        from core.cache import get_profile_cache_result
        result = await get_profile_cache_result("u_new")
    assert result.ok is True
    assert result.data is None
    # get_profile_cache_result 将 source 从 "redis" 更新为 "redis_miss"
    assert result.meta.source == "redis_miss"


@pytest.mark.asyncio
async def test_profile_cache_redis_failure():
    """Redis 故障 → ok=False, data=None, fallback_used=True, error.code=NETWORK。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail(ToolErrorCode.NETWORK, data=None, fallback_used=True))
        from core.cache import get_profile_cache_result
        result = await get_profile_cache_result("u1")
    assert result.ok is False
    assert result.data is None
    assert result.meta.fallback_used is True
    assert result.error.code == ToolErrorCode.NETWORK


# ── get_user_profile_result ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_user_profile_found():
    """用户存在 → ok=True, data=dict。"""
    profile = {"user_id": "u1", "likes": ["辣"], "dislikes": [], "meal_history": []}
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(profile, source="sqlite"))
        from core.user_service import get_user_profile_result
        result = await get_user_profile_result("u1")
    assert result.ok is True
    assert result.data["user_id"] == "u1"


@pytest.mark.asyncio
async def test_user_profile_not_found():
    """新用户，SQLite 无记录 → ok=True, data=None（非错误）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(None, source="sqlite"))
        from core.user_service import get_user_profile_result
        result = await get_user_profile_result("new_user")
    assert result.ok is True
    assert result.data is None


@pytest.mark.asyncio
async def test_user_profile_sqlite_error():
    """SQLite 故障 → ok=False（不能伪装成空用户，必须区分故障和无数据）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail(ToolErrorCode.INTERNAL))
        from core.user_service import get_user_profile_result
        result = await get_user_profile_result("u1")
    assert result.ok is False
    assert result.error.code == ToolErrorCode.INTERNAL


@pytest.mark.asyncio
async def test_user_profile_meta_preserved():
    """成功路径 meta 透传。"""
    profile = {"user_id": "u1", "likes": []}
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=ToolResult(
            ok=True,
            data=profile,
            meta=ToolMeta(tool_name="get_user_profile", elapsed_ms=200, attempts=1, source="sqlite"),
        ))
        from core.user_service import get_user_profile_result
        result = await get_user_profile_result("u1")
    assert result.meta.elapsed_ms == 200
    assert result.meta.source == "sqlite"
