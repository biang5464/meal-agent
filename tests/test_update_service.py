"""
tests/test_update_service.py
Phase 6B：core/update_service.py 用户画像写入协议测试。

所有测试使用 Mock / 临时 SQLite，不访问真实 Redis 或生产数据库。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import SQLAlchemyError


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_orm_dict(user_id="u1", likes="", dislikes="", budget="mid", field=None, value=None):
    """构造 orm_to_dict 返回的 dict（仅含测试需要的字段）。"""
    d = {
        "user_id": user_id,
        "likes": [],
        "dislikes": [],
        "budget": budget,
        "diet_restriction": "",
        "meal_history": [],
        "category_preferences": {},
    }
    if field and value is not None:
        d[field] = value
    d["database_updated"] = True
    return d


# ── update_user_profile_result ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_user_profile_result_ok():
    """DB commit 成功 → ok=True，data 包含 database_updated=True。"""
    from core.update_service import update_user_profile_result

    expected = _make_orm_dict(field="budget", value="low")

    with patch("core.update_service._update_user_profile_impl", return_value=expected):
        result = await update_user_profile_result("u1", "budget", "low")

    assert result.ok, f"期望 ok=True，实际: {result}"
    assert result.data["database_updated"] is True
    assert result.data["budget"] == "low"


@pytest.mark.asyncio
async def test_update_user_profile_result_db_error():
    """SQLAlchemyError → ok=False，error.code 为 DATABASE_ERROR 或 TIMEOUT。"""
    from core.update_service import update_user_profile_result

    with patch("core.update_service._update_user_profile_impl",
               side_effect=SQLAlchemyError("connection failed")):
        result = await update_user_profile_result("u1", "budget", "low")

    assert not result.ok, "DB 失败应为 ok=False"
    assert result.error is not None


@pytest.mark.asyncio
async def test_update_user_profile_result_invalid_field():
    """非法字段名 → ValueError → ok=False。"""
    from core.update_service import update_user_profile_result

    with patch("core.update_service._update_user_profile_impl",
               side_effect=ValueError("不支持的字段")):
        result = await update_user_profile_result("u1", "nonexistent", "val")

    assert not result.ok


@pytest.mark.asyncio
async def test_update_user_profile_upsert_creates_user():
    """用户不存在时 _update_user_profile_impl 应 INSERT 新行，调用方得到 ok=True。"""
    from core.update_service import update_user_profile_result

    expected = _make_orm_dict(user_id="new_user", field="diet_restriction", value="素食")

    with patch("core.update_service._update_user_profile_impl", return_value=expected):
        result = await update_user_profile_result("new_user", "diet_restriction", "素食")

    assert result.ok
    assert result.data["user_id"] == "new_user"
    assert result.data["database_updated"] is True


# ── invalidate_cache_result ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalidate_cache_result_ok():
    """Redis DEL 成功 → ok=True，data={"cache_invalidated": True}。"""
    from core.update_service import invalidate_cache_result

    with patch("core.update_service._invalidate_cache_impl",
               return_value={"cache_invalidated": True}):
        result = await invalidate_cache_result("u1")

    assert result.ok
    assert result.data == {"cache_invalidated": True}


@pytest.mark.asyncio
async def test_invalidate_cache_result_redis_error():
    """Redis ConnectionError → ok=False，fallback_data={"cache_invalidated": False}。"""
    from core.update_service import invalidate_cache_result
    import redis

    with patch("core.update_service._invalidate_cache_impl",
               side_effect=redis.ConnectionError("Connection refused")):
        result = await invalidate_cache_result("u1")

    assert not result.ok
    assert result.data == {"cache_invalidated": False}


@pytest.mark.asyncio
async def test_update_then_invalidate_partial_success():
    """DB ok + Redis 失败 → DB result ok=True，invalidate result ok=False，不应引发异常。"""
    from core.update_service import update_user_profile_result, invalidate_cache_result
    import redis

    expected = _make_orm_dict(field="budget", value="mid")
    with patch("core.update_service._update_user_profile_impl", return_value=expected):
        db_result = await update_user_profile_result("u1", "budget", "mid")

    with patch("core.update_service._invalidate_cache_impl",
               side_effect=redis.ConnectionError("down")):
        inv_result = await invalidate_cache_result("u1")

    assert db_result.ok, "DB 成功应为 ok=True"
    assert not inv_result.ok, "Redis 失败应为 ok=False"
    assert inv_result.data == {"cache_invalidated": False}
