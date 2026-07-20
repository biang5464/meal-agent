"""
tests/test_user_service.py
Phase 6A: get_preference_history_result / get_unread_reminders_readonly_result /
          predict_next_period_result 单元测试

全部 mock tool_executor.execute，不依赖真实 SQLite。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult


def _ok(data, source="sqlite"):
    return ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(tool_name="t", elapsed_ms=50, attempts=1, source=source),
    )


def _fail(code: ToolErrorCode = ToolErrorCode.INTERNAL):
    return ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=code, message="mock db error", retryable=False),
        meta=ToolMeta(tool_name="t", elapsed_ms=3000, attempts=1),
    )


# ── get_preference_history_result ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preference_history_with_data():
    """有未过期限制记录 → ok=True, data=[...]。"""
    restrictions = [
        {"value": "坚果", "reason": "过敏", "weight": 1.0, "expires_at": None},
    ]
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(restrictions))
        from core.user_service import get_preference_history_result
        result = await get_preference_history_result("u1")
    assert result.ok is True
    assert len(result.data) == 1
    assert result.data[0]["value"] == "坚果"


@pytest.mark.asyncio
async def test_preference_history_empty():
    """无限制记录（新用户或已过期）→ ok=True, data=[]。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok([]))
        from core.user_service import get_preference_history_result
        result = await get_preference_history_result("u_new")
    assert result.ok is True
    assert result.data == []


@pytest.mark.asyncio
async def test_preference_history_sqlite_error():
    """SQLite 故障 → ok=False。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail())
        from core.user_service import get_preference_history_result
        result = await get_preference_history_result("u1")
    assert result.ok is False
    assert result.error.code == ToolErrorCode.INTERNAL


# ── get_unread_reminders_readonly_result ──────────────────────────────────────

@pytest.mark.asyncio
async def test_unread_reminders_with_data():
    """有未读提醒 → ok=True, data=[...]。"""
    reminders = [
        {"id": 1, "message": "记得补铁", "reminder_type": "period", "created_at": "2026-07-01T00:00:00"},
    ]
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(reminders))
        from core.user_service import get_unread_reminders_readonly_result
        result = await get_unread_reminders_readonly_result("u1")
    assert result.ok is True
    assert result.data[0]["id"] == 1


@pytest.mark.asyncio
async def test_unread_reminders_empty():
    """无未读提醒 → ok=True, data=[]。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok([]))
        from core.user_service import get_unread_reminders_readonly_result
        result = await get_unread_reminders_readonly_result("u1")
    assert result.ok is True
    assert result.data == []


@pytest.mark.asyncio
async def test_unread_reminders_sqlite_error():
    """SQLite 故障 → ok=False。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail())
        from core.user_service import get_unread_reminders_readonly_result
        result = await get_unread_reminders_readonly_result("u1")
    assert result.ok is False


@pytest.mark.asyncio
async def test_unread_reminders_impl_does_not_mark_as_read():
    """验证 impl 函数名称语义：只读，不含 is_read 写入。"""
    from core.user_service import _get_unread_reminders_readonly_impl
    import inspect
    src = inspect.getsource(_get_unread_reminders_readonly_impl)
    assert "is_read = True" not in src
    assert "commit()" not in src


# ── predict_next_period_result ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_next_period_with_cycle():
    """有健康周期记录 → ok=True, data=dict。"""
    period = {
        "user_id": "u1",
        "next_start": "2026-08-01",
        "days_until_next": 14,
        "in_period": False,
        "cycle_day": 14,
        "phase": "normal",
    }
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(period))
        from core.user_service import predict_next_period_result
        result = await predict_next_period_result("u1")
    assert result.ok is True
    assert result.data["phase"] == "normal"
    assert result.data["days_until_next"] == 14


@pytest.mark.asyncio
async def test_predict_next_period_no_cycle():
    """未设置健康周期 → ok=True, data=None（正常情况，非错误）。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_ok(None))
        from core.user_service import predict_next_period_result
        result = await predict_next_period_result("u1")
    assert result.ok is True
    assert result.data is None


@pytest.mark.asyncio
async def test_predict_next_period_sqlite_error():
    """SQLite 故障 → ok=False。"""
    with patch("tools.tool_executor.tool_executor") as m:
        m.execute = AsyncMock(return_value=_fail())
        from core.user_service import predict_next_period_result
        result = await predict_next_period_result("u1")
    assert result.ok is False
    assert result.error.code == ToolErrorCode.INTERNAL
