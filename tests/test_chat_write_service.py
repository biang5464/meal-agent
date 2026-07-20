"""
tests/test_chat_write_service.py
Phase 6B：cache.py 写入路径 ToolResult 协议测试。

测试 set_profile_cache_result 和 append_chat_message_result。
所有测试使用 Mock，不访问真实 Redis 或数据库。
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import redis as redis_lib


# ── set_profile_cache_result ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_profile_cache_result_ok():
    """Redis SET 成功 → ok=True，data={"cache_written": True}。"""
    from core.cache import set_profile_cache_result

    with patch("core.cache._set_profile_cache_impl",
               return_value={"cache_written": True}):
        result = await set_profile_cache_result("u1", {"user_id": "u1"})

    assert result.ok
    assert result.data == {"cache_written": True}


@pytest.mark.asyncio
async def test_set_profile_cache_result_redis_fail():
    """Redis ConnectionError → ok=False，fallback_data={"cache_written": False}。"""
    from core.cache import set_profile_cache_result

    with patch("core.cache._set_profile_cache_impl",
               side_effect=redis_lib.ConnectionError("down")):
        result = await set_profile_cache_result("u1", {"user_id": "u1"})

    assert not result.ok
    assert result.data == {"cache_written": False}


@pytest.mark.asyncio
async def test_set_profile_cache_result_no_retry():
    """redis_write 策略 retries=0：只尝试一次（attempts==1）。"""
    from core.cache import set_profile_cache_result

    call_count = 0

    def _fail(*_):
        nonlocal call_count
        call_count += 1
        raise redis_lib.ConnectionError("down")

    with patch("core.cache._set_profile_cache_impl", side_effect=_fail):
        result = await set_profile_cache_result("u1", {})

    assert not result.ok
    assert call_count == 1, f"redis_write 不应重试，实际调用 {call_count} 次"
    assert result.meta is not None and result.meta.attempts == 1


# ── append_chat_message_result ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_append_chat_message_result_ok():
    """追加消息成功 → ok=True。"""
    from core.cache import append_chat_message_result

    with patch("core.cache._append_chat_message_impl", return_value=None):
        result = await append_chat_message_result("u1", "user", "你好")

    assert result.ok


@pytest.mark.asyncio
async def test_append_chat_message_result_redis_fail():
    """追加失败（Redis/DB 异常）→ ok=False，不抛异常。"""
    from core.cache import append_chat_message_result

    with patch("core.cache._append_chat_message_impl",
               side_effect=redis_lib.ConnectionError("down")):
        result = await append_chat_message_result("u1", "user", "你好")

    assert not result.ok
    assert result.error is not None


@pytest.mark.asyncio
async def test_append_chat_message_no_retry():
    """database_write 策略 retries=0：只尝试一次。"""
    from core.cache import append_chat_message_result

    call_count = 0

    def _fail(*_):
        nonlocal call_count
        call_count += 1
        raise Exception("db error")

    with patch("core.cache._append_chat_message_impl", side_effect=_fail):
        result = await append_chat_message_result("u1", "user", "msg")

    assert not result.ok
    assert call_count == 1, f"database_write 不应重试，实际调用 {call_count} 次"
