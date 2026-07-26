"""
tests/test_daily_rec_service.py
Phase 6B：daily_recommendation_agent.py 写入协议测试。

测试：幂等命中、IntegrityError 返回 existing、DB 失败 raise、_enqueue_retry_result。
所有测试使用 Mock，不访问真实 Redis 或数据库。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_RECIPES = [
    {
        "name": "番茄炒蛋", "category": "蔬菜", "ingredients": [],
        "cuisine": "", "flavor": "", "budget_tier": "low", "steps": "",
    }
]
_RECIPES_MEAT = [
    {
        "name": "宫保鸡丁", "category": "荤菜", "ingredients": [],
        "cuisine": "", "flavor": "", "budget_tier": "mid", "steps": "",
    }
]


def _make_row(user_id="u1", dishes=None, meal_type="lunch"):
    row = MagicMock()
    row.id = 1
    row.user_id = user_id
    row.date = date(2026, 7, 18)
    row.meal_type = meal_type
    row.dishes = dishes or []
    row.reasoning = "{}"
    row.generated_by = "scheduled"
    row.source_status = "success"
    row.is_read = 0
    row.created_at = None
    return row


def _make_mock_session(first_return=None, commit_side_effect=None):
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock_q = MagicMock()
    mock.query.return_value = mock_q
    mock_q.filter.return_value = mock_q
    mock_q.first.return_value = first_return
    mock.add = MagicMock()
    mock.refresh = MagicMock()
    if commit_side_effect:
        mock.commit = MagicMock(side_effect=commit_side_effect)
    else:
        mock.commit = MagicMock()
    return mock


# ── generate_for_user 幂等命中 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_for_user_idempotent_returns_existing():
    """已存在相同 (user, date, meal_type) 记录时，直接返回 existing，不调用 LLM。"""
    from agents.daily_recommendation_agent import generate_for_user

    existing_row = _make_row(dishes=[{"name": "番茄炒蛋"}])
    mock_session = _make_mock_session(first_return=existing_row)

    with (
        patch("agents.daily_recommendation_agent._session", return_value=mock_session),
        patch("agents.daily_recommendation_agent._generate_reasoning") as mock_llm,
    ):
        result = await generate_for_user("u1", meal_type="lunch", date_str="2026-07-18")

    mock_llm.assert_not_called()
    assert result is not None
    assert result["user_id"] == "u1"
    assert result["dishes"] == [{"name": "番茄炒蛋"}]


@pytest.mark.asyncio
async def test_generate_for_user_integrity_error_returns_existing():
    """INSERT 引发 IntegrityError（并发冲突）时，重新查询并返回已有记录（不返回 None）。"""
    from agents.daily_recommendation_agent import generate_for_user
    from sqlalchemy.exc import IntegrityError
    from unittest.mock import AsyncMock as _AsyncMock

    existing_row = _make_row(dishes=[{"name": "番茄炒蛋"}])
    call_count = 0

    def _session_factory():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 第一次：幂等检查（锁外快速路径），用户不存在
            return _make_mock_session(first_return=None)
        elif call_count == 2:
            # 第二次：获得锁后幂等再确认，用户不存在
            return _make_mock_session(first_return=None)
        elif call_count == 3:
            # 第三次：写入尝试，commit 引发 IntegrityError
            return _make_mock_session(first_return=None, commit_side_effect=IntegrityError("", {}, None))
        else:
            # 第四次：IntegrityError 后重查，返回已有记录
            return _make_mock_session(first_return=existing_row)

    _lock_result = MagicMock()
    _lock_result.ok = True
    _lock_result.data = True

    with (
        patch("agents.daily_recommendation_agent._session", side_effect=_session_factory),
        patch("agents.daily_recommendation_agent._try_acquire_lock",
              new=_AsyncMock(return_value=_lock_result)),
        patch("agents.daily_recommendation_agent._release_daily_rec_lock",
              new=_AsyncMock()),
        patch("agents.daily_recommendation_agent._get_user_profile", return_value={}),
        patch("agents.daily_recommendation_agent._get_same_day_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent._get_recent_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent.get_all_recipes", return_value=_RECIPES),
        patch("agents.daily_recommendation_agent._generate_reasoning",
              new=AsyncMock(return_value="{}")),
    ):
        result = await generate_for_user("u1", meal_type="lunch", date_str="2026-07-18")

    assert result is not None, "IntegrityError 后应返回 existing，不返回 None"
    assert result["dishes"] == [{"name": "番茄炒蛋"}]


@pytest.mark.asyncio
async def test_generate_for_user_db_write_fail_raises():
    """非 IntegrityError DB 异常时，generate_for_user 应向上 raise，使 _enqueue_retry_result 生效。"""
    from agents.daily_recommendation_agent import generate_for_user
    from unittest.mock import AsyncMock as _AsyncMock

    mock_session = _make_mock_session(first_return=None, commit_side_effect=RuntimeError("disk full"))

    _lock_result = MagicMock()
    _lock_result.ok = True
    _lock_result.data = True

    with (
        patch("agents.daily_recommendation_agent._session", return_value=mock_session),
        patch("agents.daily_recommendation_agent._try_acquire_lock",
              new=_AsyncMock(return_value=_lock_result)),
        patch("agents.daily_recommendation_agent._release_daily_rec_lock",
              new=_AsyncMock()),
        patch("agents.daily_recommendation_agent._get_user_profile", return_value={}),
        patch("agents.daily_recommendation_agent._get_same_day_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent._get_recent_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent.get_all_recipes", return_value=_RECIPES_MEAT),
        patch("agents.daily_recommendation_agent._generate_reasoning",
              new=AsyncMock(return_value="{}")),
    ):
        with pytest.raises(RuntimeError, match="disk full"):
            await generate_for_user("u1", meal_type="lunch", date_str="2026-07-18")


# ── _enqueue_retry_result ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_retry_result_ok():
    """Redis RPUSH 成功 → ok=True。"""
    from agents.daily_recommendation_agent import _enqueue_retry_result

    with patch("core.cache._client") as mock_client:
        mock_client.rpush = MagicMock(return_value=1)
        result = await _enqueue_retry_result("u1", "lunch")

    assert result.ok


@pytest.mark.asyncio
async def test_enqueue_retry_result_redis_fail_ok_false():
    """Redis 入队失败 → ok=False，不抛异常，不覆盖原始错误。"""
    import redis as redis_lib
    from agents.daily_recommendation_agent import _enqueue_retry_result

    with patch("core.cache._client") as mock_client:
        mock_client.rpush = MagicMock(side_effect=redis_lib.ConnectionError("down"))
        result = await _enqueue_retry_result("u1", "dinner")

    assert not result.ok
    assert result.error is not None


@pytest.mark.asyncio
async def test_enqueue_retry_no_auto_retry():
    """redis_write 策略 retries=0：RPUSH 失败只尝试一次。"""
    import redis as redis_lib
    from agents.daily_recommendation_agent import _enqueue_retry_result

    call_count = 0

    def _fail(*_):
        nonlocal call_count
        call_count += 1
        raise redis_lib.ConnectionError("down")

    with patch("core.cache._client") as mock_client:
        mock_client.rpush = MagicMock(side_effect=_fail)
        result = await _enqueue_retry_result("u1", "dinner")

    assert not result.ok
    assert call_count == 1, f"不应重试，实际调用 {call_count} 次"


# ── Phase 11C-A：steps 规范化与快照 ───────────────────────────────────────────

def test_normalize_steps_string_wraps_to_single_element_list():
    """字符串步骤应被包装为单元素列表，首尾空白去除。"""
    from agents.daily_recommendation_agent import _normalize_recipe_steps
    result = _normalize_recipe_steps("  番茄切块，鸡蛋炒熟。  ")
    assert result == ["番茄切块，鸡蛋炒熟。"]


def test_normalize_steps_list_filters_blank_and_strips():
    """列表步骤应过滤空项、去除首尾空白，不修改原列表。"""
    from agents.daily_recommendation_agent import _normalize_recipe_steps
    original = ["步骤一", "  ", "", "  步骤三  ", 42]
    original_copy = list(original)
    result = _normalize_recipe_steps(original)
    assert result == ["步骤一", "步骤三"]
    assert original == original_copy, "原始列表不应被修改"


def test_normalize_steps_none_and_invalid_types_return_empty_list():
    """None 及非法类型应安全降级为空列表。"""
    from agents.daily_recommendation_agent import _normalize_recipe_steps
    assert _normalize_recipe_steps(None) == []
    assert _normalize_recipe_steps(123) == []
    assert _normalize_recipe_steps({"key": "val"}) == []
    assert _normalize_recipe_steps("") == []
    assert _normalize_recipe_steps("   ") == []


@pytest.mark.asyncio
async def test_dishes_snapshot_includes_steps_field():
    """generate_for_user 写入 DB 的 dishes 快照应包含 steps 字段（list[str]）。"""
    from agents.daily_recommendation_agent import generate_for_user
    from unittest.mock import AsyncMock as _AsyncMock

    recipe = {
        "name": "番茄炒蛋", "category": "半荤素",
        "ingredients": ["番茄", "鸡蛋"],
        "cuisine": "家常菜", "flavor": "鲜甜", "budget_tier": "low",
        "steps": "番茄切块翻炒，鸡蛋打散炒熟，合炒调味。",
    }

    captured: list = []

    def _track_add(obj):
        captured.append(obj)

    _lock_result = MagicMock()
    _lock_result.ok = True
    _lock_result.data = True

    mock_session = _make_mock_session(first_return=None)
    mock_session.add = MagicMock(side_effect=_track_add)

    with (
        patch("agents.daily_recommendation_agent._session", return_value=mock_session),
        patch("agents.daily_recommendation_agent._try_acquire_lock",
              new=_AsyncMock(return_value=_lock_result)),
        patch("agents.daily_recommendation_agent._release_daily_rec_lock", new=_AsyncMock()),
        patch("agents.daily_recommendation_agent._get_user_profile", return_value={}),
        patch("agents.daily_recommendation_agent._get_same_day_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent._get_recent_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent.get_all_recipes", return_value=[recipe]),
        patch("agents.daily_recommendation_agent._generate_reasoning",
              new=AsyncMock(return_value="{}")),
    ):
        await generate_for_user("u1", meal_type="lunch", date_str="2026-07-18")

    assert len(captured) == 1, "session.add は一度だけ呼ばれる"
    dishes = captured[0].dishes
    assert dishes, "dishes should not be empty"
    for dish in dishes:
        assert "steps" in dish, f"菜谱快照缺少 steps 字段: {dish}"
    assert isinstance(dishes[0]["steps"], list), "steps 应为 list[str]"
    assert dishes[0]["steps"] == ["番茄切块翻炒，鸡蛋打散炒熟，合炒调味。"]


@pytest.mark.asyncio
async def test_original_recipe_ingredients_not_mutated():
    """dishes_data 中的 ingredients 应为防御性拷贝，不修改原始 recipe dict。"""
    from agents.daily_recommendation_agent import generate_for_user
    from unittest.mock import AsyncMock as _AsyncMock

    original_ingredients = ["番茄", "鸡蛋"]
    recipe = {
        "name": "番茄炒蛋", "category": "半荤素",
        "ingredients": original_ingredients,
        "cuisine": "家常菜", "flavor": "鲜甜", "budget_tier": "low",
        "steps": "简单做法。",
    }

    captured: list = []

    def _track_add(obj):
        captured.append(obj)

    _lock_result = MagicMock()
    _lock_result.ok = True
    _lock_result.data = True

    mock_session = _make_mock_session(first_return=None)
    mock_session.add = MagicMock(side_effect=_track_add)

    with (
        patch("agents.daily_recommendation_agent._session", return_value=mock_session),
        patch("agents.daily_recommendation_agent._try_acquire_lock",
              new=_AsyncMock(return_value=_lock_result)),
        patch("agents.daily_recommendation_agent._release_daily_rec_lock", new=_AsyncMock()),
        patch("agents.daily_recommendation_agent._get_user_profile", return_value={}),
        patch("agents.daily_recommendation_agent._get_same_day_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent._get_recent_dishes", return_value=([], "empty")),
        patch("agents.daily_recommendation_agent.get_all_recipes", return_value=[recipe]),
        patch("agents.daily_recommendation_agent._generate_reasoning",
              new=AsyncMock(return_value="{}")),
    ):
        await generate_for_user("u1", meal_type="lunch", date_str="2026-07-18")

    assert len(captured) == 1
    dishes = captured[0].dishes
    # Verify the snapshot ingredients list is a different object from the original
    assert dishes[0]["ingredients"] is not original_ingredients, "应为防御性拷贝，不应是同一对象"
    assert dishes[0]["ingredients"] == original_ingredients
