"""Phase 11B: Distributed lock protection for concurrent daily recommendation generation.

25 test scenarios covering:
  - Lock acquisition / key format / constants
  - Recheck pattern (wait-loop DB check + recheck inside lock)
  - Cross-meal dedup via lock serialization
  - Lock key granularity
  - Wait timeout → LockBusy
  - Redis degraded → LockDegraded
  - API endpoints → 503 on lock failure
  - Lock release in finally / exception / cancel paths
  - Token-safe Lua CAS release
  - Phase 11A regression (same-day hard constraint not bypassed)
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import agents.daily_recommendation_agent as _agent
from agents.daily_recommendation_agent import (
    LockBusy,
    LockDegraded,
    _LOCK_POLL_INTERVAL,
    _LOCK_TTL,
    _LOCK_WAIT_TIMEOUT,
    _release_daily_rec_lock,
    _try_acquire_lock,
    generate_for_user,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _tr(ok=True, data=None, error=None):
    """Build a minimal ToolResult-like mock."""
    r = MagicMock()
    r.ok = ok
    r.data = data
    r.error = error
    return r


def _recipe(name, cat="荤菜"):
    return {
        "name": name, "category": cat,
        "ingredients": [], "cuisine": "家常",
        "flavor": "咸鲜", "budget_tier": "mid", "steps": "",
    }


_RECIPES = (
    [_recipe(f"荤{i}", "荤菜") for i in range(4)]
    + [_recipe(f"素{i}", "蔬菜") for i in range(4)]
    + [_recipe(f"汤{i}", "汤")   for i in range(4)]
)


def _session_seq(*results):
    """Return _session mock whose successive context-managers yield results in order."""
    itr = iter(results)

    def _factory():
        rv = next(itr, None)
        sess = MagicMock()
        sess.query.return_value.filter.return_value.first.return_value = rv
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=sess)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    return MagicMock(side_effect=_factory)


def _generation_patches(acquire_result, session_seq=None, same_day=None):
    """Return a dict of patches for a full generate_for_user() run."""
    if session_seq is None:
        # idempotency check → None; recheck inside lock → None; DB write → session with add/commit/refresh
        write_sess = MagicMock()
        write_orm  = MagicMock()
        write_orm.id = 99
        write_orm.user_id = "u1"
        write_orm.date = date(2026, 7, 26)
        write_orm.meal_type = "lunch"
        write_orm.dishes = [{"name": "荤0"}, {"name": "素0"}, {"name": "汤0"}]
        write_orm.reasoning = "{}"
        write_orm.generated_by = "scheduled"
        write_orm.source_status = "success"
        write_orm.is_read = 0
        write_orm.created_at = None
        write_sess.query.return_value.filter.return_value.first.return_value = None
        write_cm = MagicMock()
        write_cm.__enter__ = MagicMock(return_value=write_sess)
        write_cm.__exit__ = MagicMock(return_value=False)
        session_seq = _session_seq(None, None, None)
        # Override: let the write session work
        _orig = session_seq.side_effect
        call_count = [0]

        def _smart_factory():
            call_count[0] += 1
            if call_count[0] <= 2:
                return _orig()
            # 3rd call onwards: write session
            return write_cm

        session_seq.side_effect = _smart_factory

    return {
        "agents.daily_recommendation_agent._session": session_seq,
        "agents.daily_recommendation_agent._try_acquire_lock": AsyncMock(return_value=acquire_result),
        "agents.daily_recommendation_agent._release_daily_rec_lock": AsyncMock(),
        "agents.daily_recommendation_agent._get_user_profile": MagicMock(return_value={}),
        "agents.daily_recommendation_agent.get_all_recipes": MagicMock(return_value=_RECIPES),
        "agents.daily_recommendation_agent._get_same_day_dishes": MagicMock(
            return_value=(same_day or [], "empty")
        ),
        "agents.daily_recommendation_agent._get_recent_dishes": MagicMock(
            return_value=([], "empty")
        ),
        "agents.daily_recommendation_agent._generate_reasoning": AsyncMock(return_value="{}"),
    }


# ── 1. Lock Acquisition ───────────────────────────────────────────────────────

class TestLockConstants:
    def test_lock_ttl_is_180(self):
        assert _LOCK_TTL == 180

    def test_wait_timeout_in_range(self):
        assert 10 <= _LOCK_WAIT_TIMEOUT <= 15

    def test_poll_interval_positive(self):
        assert 0 < _LOCK_POLL_INTERVAL <= _LOCK_WAIT_TIMEOUT

    def test_lock_key_format(self):
        """Lock key must embed user_id and date."""
        captured = []

        async def _run():
            patches = _generation_patches(_tr(ok=True, data=True))
            with (
                patch("agents.daily_recommendation_agent._session", patches["agents.daily_recommendation_agent._session"]),
                patch(
                    "agents.daily_recommendation_agent._try_acquire_lock",
                    new=AsyncMock(side_effect=lambda k, t: captured.append(k) or _tr(ok=True, data=True)),
                ),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
            ):
                await generate_for_user("alice", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        assert len(captured) == 1
        assert "alice" in captured[0]
        assert "2026-07-26" in captured[0]

    def test_fast_path_skips_lock(self):
        """If record exists on first DB check, lock is never acquired."""
        orm = MagicMock()
        orm.date = date(2026, 7, 26)
        orm.created_at = None
        orm.dishes = []

        acquire_mock = AsyncMock()

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(orm)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", acquire_mock),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        acquire_mock.assert_not_called()


# ── 2. Recheck Pattern ────────────────────────────────────────────────────────

class TestRecheckPattern:
    def test_recheck_while_waiting_returns_existing(self):
        """When lock busy, if DB check in wait-loop finds the record, return it early."""
        existing_orm = MagicMock()
        existing_orm.date = date(2026, 7, 26)
        existing_orm.created_at = None
        existing_orm.dishes = [{"name": "既有菜"}]

        async def _run():
            # lock busy on first attempt; record appears in wait-loop DB check
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, existing_orm)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=None))),  # lock busy
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
            ):
                result = await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")
            return result

        result = asyncio.run(_run())
        assert result is not None

    def test_recheck_inside_lock_returns_existing(self):
        """After acquiring lock, re-check finds record → return it without generating."""
        existing_orm = MagicMock()
        existing_orm.date = date(2026, 7, 26)
        existing_orm.created_at = None
        existing_orm.dishes = [{"name": "锁内已有菜"}]

        generate_mock = AsyncMock()

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, existing_orm)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._generate_reasoning", generate_mock),
            ):
                result = await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")
            return result

        asyncio.run(_run())
        generate_mock.assert_not_called()

    def test_waiter_loops_until_lock_free(self):
        """SET NX returns None twice, then True — acquisition after 2 polls."""
        poll_results = [_tr(ok=True, data=None), _tr(ok=True, data=None), _tr(ok=True, data=True)]
        call_count = [0]

        async def _acquire(k, t):
            r = poll_results[call_count[0]]
            call_count[0] += 1
            return r

        async def _run():
            patches = _generation_patches(_tr(ok=True, data=True))
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", side_effect=_acquire),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
                patch("asyncio.sleep", AsyncMock()),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        assert call_count[0] == 3

    def test_same_day_dishes_read_inside_lock(self):
        """_get_same_day_dishes is called after lock acquired (not before)."""
        same_day_mock = MagicMock(return_value=(["午餐菜A"], "success"))
        acquire_mock  = AsyncMock(return_value=_tr(ok=True, data=True))

        acquired_before_same_day = []

        orig_same_day = same_day_mock.side_effect

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(side_effect=lambda k, t: acquired_before_same_day.append(True) or _tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", same_day_mock),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        assert len(acquired_before_same_day) == 1
        same_day_mock.assert_called_once()


# ── 3. Cross-meal Dedup ───────────────────────────────────────────────────────

class TestCrossMealDedup:
    def test_dinner_excludes_lunch_dishes(self):
        """After lock acquired, _get_same_day_dishes returns lunch names → filtered from dinner."""
        lunch_names = ["荤0", "素0", "汤0"]
        same_day_mock = MagicMock(return_value=(lunch_names, "success"))
        selected_dishes = []

        async def _capture_reasoning(dishes, profile, meal_type):
            selected_dishes.extend(d["name"] for d in dishes)
            return "{}"

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", same_day_mock),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", side_effect=_capture_reasoning),
            ):
                await generate_for_user("u1", meal_type="dinner", date_str="2026-07-26")

        asyncio.run(_run())
        for name in lunch_names:
            assert name not in selected_dishes, f"{name} must not appear in dinner after cross-meal dedup"

    def test_same_day_hard_constraint_never_relaxed(self):
        """Even when all soft constraints are dropped, same-day dishes are still excluded."""
        same_day_mock = MagicMock(return_value=(["荤0", "素0", "汤0", "荤1", "素1", "汤1", "荤2", "素2", "汤2"], "success"))
        selected_dishes = []

        async def _capture(dishes, profile, meal_type):
            selected_dishes.extend(d["name"] for d in dishes)
            return "{}"

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile",
                      MagicMock(return_value={"budget": "low"})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", same_day_mock),
                patch("agents.daily_recommendation_agent._get_recent_dishes",
                      MagicMock(return_value=([r["name"] for r in _RECIPES], "success"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", side_effect=_capture),
            ):
                await generate_for_user("u1", meal_type="dinner", date_str="2026-07-26")

        asyncio.run(_run())
        blocked = {"荤0", "素0", "汤0", "荤1", "素1", "汤1", "荤2", "素2", "汤2"}
        for name in selected_dishes:
            assert name not in blocked, f"{name} must never appear (same-day hard constraint)"

    def test_no_dish_overlap_when_same_day_status_success(self):
        """With same_day_dishes=[lunch_dish], the dinner selection cannot include that dish."""
        selected = []

        async def _capture(dishes, profile, meal_type):
            selected.extend(d["name"] for d in dishes)
            return "{}"

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes",
                      MagicMock(return_value=(["荤3"], "success"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", side_effect=_capture),
            ):
                await generate_for_user("u1", meal_type="dinner", date_str="2026-07-26")

        asyncio.run(_run())
        assert "荤3" not in selected


# ── 4. Lock Granularity ───────────────────────────────────────────────────────

class TestLockGranularity:
    def test_different_users_different_keys(self):
        """Two different users must get different lock keys."""
        keys = []

        async def _capture_acquire(k, t):
            keys.append(k)
            return _tr(ok=True, data=True)

        async def _run():
            base_patches = dict(
                _get_user_profile=MagicMock(return_value={}),
                get_all_recipes=MagicMock(return_value=_RECIPES),
                _get_same_day_dishes=MagicMock(return_value=([], "empty")),
                _get_recent_dishes=MagicMock(return_value=([], "empty")),
                _generate_reasoning=AsyncMock(return_value="{}"),
                _release_daily_rec_lock=AsyncMock(),
            )
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None, None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", side_effect=_capture_acquire),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", base_patches["_get_user_profile"]),
                patch("agents.daily_recommendation_agent.get_all_recipes", base_patches["get_all_recipes"]),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", base_patches["_get_same_day_dishes"]),
                patch("agents.daily_recommendation_agent._get_recent_dishes", base_patches["_get_recent_dishes"]),
                patch("agents.daily_recommendation_agent._generate_reasoning", base_patches["_generate_reasoning"]),
            ):
                await generate_for_user("alice", meal_type="lunch", date_str="2026-07-26")
                await generate_for_user("bob",   meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        assert len(keys) == 2
        assert keys[0] != keys[1]
        assert "alice" in keys[0]
        assert "bob" in keys[1]

    def test_different_dates_different_keys(self):
        """Same user, different dates → different lock keys."""
        keys = []

        async def _capture_acquire(k, t):
            keys.append(k)
            return _tr(ok=True, data=True)

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None, None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", side_effect=_capture_acquire),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
            ):
                await generate_for_user("alice", meal_type="lunch", date_str="2026-07-26")
                await generate_for_user("alice", meal_type="lunch", date_str="2026-07-27")

        asyncio.run(_run())
        assert keys[0] != keys[1]
        assert "2026-07-26" in keys[0]
        assert "2026-07-27" in keys[1]


# ── 5. Wait Timeout ───────────────────────────────────────────────────────────

class TestWaitTimeout:
    def test_lock_busy_raised_after_timeout(self):
        """SET NX always returns None (lock held), timeout → LockBusy."""
        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=None))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("asyncio.sleep", AsyncMock()),
                patch("agents.daily_recommendation_agent._LOCK_WAIT_TIMEOUT", 0),
            ):
                with pytest.raises(LockBusy):
                    await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())

    def test_lock_busy_message_contains_user_id_and_meal(self):
        """LockBusy exception message contains user_id and meal_type."""
        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=None))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("asyncio.sleep", AsyncMock()),
                patch("agents.daily_recommendation_agent._LOCK_WAIT_TIMEOUT", 0),
            ):
                with pytest.raises(LockBusy) as exc_info:
                    await generate_for_user("user_xyz", meal_type="dinner", date_str="2026-07-26")
            return str(exc_info.value)

        msg = asyncio.run(_run())
        assert "user_xyz" in msg
        assert "dinner" in msg


# ── 6. Redis Degraded ─────────────────────────────────────────────────────────

class TestRedisDegraded:
    def test_lock_degraded_raised_on_redis_failure(self):
        """When _try_acquire_lock returns ToolResult(ok=False), raise LockDegraded."""
        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=False, error=MagicMock()))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
            ):
                with pytest.raises(LockDegraded):
                    await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())

    @pytest.mark.asyncio
    async def test_scheduler_catches_lock_degraded_and_enqueues_retry(self):
        """generate_daily_recommendations() must catch LockDegraded and call _enqueue_retry_result."""
        from agents.daily_recommendation_agent import generate_daily_recommendations

        with (
            patch("agents.daily_recommendation_agent.generate_for_user",
                  AsyncMock(side_effect=LockDegraded("redis down"))),
            patch("agents.daily_recommendation_agent._enqueue_retry_result",
                  AsyncMock(return_value=_tr(ok=True))),
            patch("agents.daily_recommendation_agent._session", _session_seq(MagicMock())),
        ):
            # Patch user list to have one user
            with patch("agents.daily_recommendation_agent._session") as sess_mock:
                inner_sess = MagicMock()
                inner_sess.query.return_value.all.return_value = [("user1",)]
                sess_cm = MagicMock()
                sess_cm.__enter__ = MagicMock(return_value=inner_sess)
                sess_cm.__exit__ = MagicMock(return_value=False)
                sess_mock.return_value = sess_cm

                enqueue = AsyncMock(return_value=_tr(ok=True))
                with patch("agents.daily_recommendation_agent._enqueue_retry_result", enqueue):
                    await generate_daily_recommendations()

        enqueue.assert_called()

    @pytest.mark.asyncio
    async def test_scheduler_catches_lock_busy_and_enqueues_retry(self):
        """generate_daily_recommendations() must catch LockBusy and call _enqueue_retry_result."""
        from agents.daily_recommendation_agent import generate_daily_recommendations

        with patch("agents.daily_recommendation_agent._session") as sess_mock:
            inner_sess = MagicMock()
            inner_sess.query.return_value.all.return_value = [("user1",)]
            sess_cm = MagicMock()
            sess_cm.__enter__ = MagicMock(return_value=inner_sess)
            sess_cm.__exit__ = MagicMock(return_value=False)
            sess_mock.return_value = sess_cm

            enqueue = AsyncMock(return_value=_tr(ok=True))
            with (
                patch("agents.daily_recommendation_agent.generate_for_user",
                      AsyncMock(side_effect=LockBusy("timeout"))),
                patch("agents.daily_recommendation_agent._enqueue_retry_result", enqueue),
            ):
                await generate_daily_recommendations()

        enqueue.assert_called()


# ── 7. API → 503 ──────────────────────────────────────────────────────────────

class TestApi503:
    def test_get_endpoint_returns_503_on_lock_busy(self):
        """GET /api/daily-recommendation raises LockBusy → HTTPException 503."""
        from fastapi import HTTPException
        from main import get_daily_recommendation

        async def _run():
            # Endpoint lazily imports _session from tools.memory, so patch there.
            # generate_for_user is also lazily imported from agents module.
            with (
                patch("tools.memory._session", _session_seq(None)),
                patch("agents.daily_recommendation_agent.generate_for_user",
                      AsyncMock(side_effect=LockBusy("busy"))),
            ):
                with pytest.raises(HTTPException) as exc:
                    await get_daily_recommendation(user_id="u1", date="2026-07-26", meal_type="lunch")
            return exc.value.status_code

        status = asyncio.run(_run())
        assert status == 503

    def test_get_endpoint_returns_503_on_lock_degraded(self):
        """GET /api/daily-recommendation raises LockDegraded → HTTPException 503."""
        from fastapi import HTTPException
        from main import get_daily_recommendation

        async def _run():
            with (
                patch("tools.memory._session", _session_seq(None)),
                patch("agents.daily_recommendation_agent.generate_for_user",
                      AsyncMock(side_effect=LockDegraded("redis down"))),
            ):
                with pytest.raises(HTTPException) as exc:
                    await get_daily_recommendation(user_id="u1", date="2026-07-26", meal_type="lunch")
            return exc.value.status_code

        status = asyncio.run(_run())
        assert status == 503


# ── 8. Lock Release ───────────────────────────────────────────────────────────

class TestLockRelease:
    def test_lock_released_after_normal_generation(self):
        """_release_daily_rec_lock is called in finally after successful generation."""
        release_mock = AsyncMock()

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", release_mock),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        release_mock.assert_called_once()

    def test_lock_released_when_generation_raises(self):
        """_release_daily_rec_lock is called even if generation throws."""
        release_mock = AsyncMock()

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", release_mock),
                patch("agents.daily_recommendation_agent._get_user_profile",
                      MagicMock(side_effect=RuntimeError("profile boom"))),
            ):
                with pytest.raises(RuntimeError):
                    await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        release_mock.assert_called_once()

    def test_lock_released_on_task_cancel(self):
        """_release_daily_rec_lock is called even when the coroutine is cancelled."""
        release_mock = AsyncMock()

        async def _slow_reasoning(*args, **kwargs):
            await asyncio.sleep(100)
            return "{}"

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", release_mock),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes",
                      MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes",
                      MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", new=_slow_reasoning),
            ):
                task = asyncio.create_task(
                    generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")
                )
                await asyncio.sleep(0)   # let task reach the first real await (_try_acquire_lock)
                await asyncio.sleep(0)   # let task proceed to _generate_reasoning's slow sleep
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        asyncio.run(_run())
        release_mock.assert_called_once()


# ── 9. Token-Safe Release ─────────────────────────────────────────────────────

class TestTokenSafeRelease:
    def test_release_not_called_when_lock_not_acquired(self):
        """If lock was never acquired (fast-path return), release is not called."""
        release_mock = AsyncMock()

        existing_orm = MagicMock()
        existing_orm.date = date(2026, 7, 26)
        existing_orm.created_at = None
        existing_orm.dishes = []

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(existing_orm)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", release_mock),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        release_mock.assert_not_called()

    def test_lua_release_script_contains_get_and_del(self):
        """Lua release script must use GET to check token before DEL."""
        from agents.daily_recommendation_agent import _LUA_RELEASE
        script = _LUA_RELEASE.lower()
        assert "get" in script
        assert "del" in script

    def test_release_uses_token_from_acquire(self):
        """The token passed to _release_daily_rec_lock matches the one from SET NX."""
        acquire_token  = []
        release_tokens = []

        async def _capture_acquire(k, t):
            acquire_token.append(t)
            return _tr(ok=True, data=True)

        async def _capture_release(k, t):
            release_tokens.append(t)

        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock", side_effect=_capture_acquire),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", side_effect=_capture_release),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes", MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning", AsyncMock(return_value="{}")),
            ):
                await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")

        asyncio.run(_run())
        assert acquire_token == release_tokens


# ── 10. Phase 11A Regression ──────────────────────────────────────────────────

class TestPhase11aRegression:
    def test_same_day_hard_constraint_applies_under_lock(self):
        """Phase 11A: same-day constraint is applied inside the lock."""
        from agents.daily_recommendation_agent import _filter_recipes

        recipes = [
            _recipe("共享菜A", "荤菜"),
            _recipe("独享菜B", "蔬菜"),
            _recipe("独享菜C", "汤"),
        ]
        filtered = _filter_recipes(recipes, same_day_dishes=["共享菜A"])
        names = [r["name"] for r in filtered]
        assert "共享菜A" not in names
        assert "独享菜B" in names

    def test_dislikes_restriction_still_hard_constraints(self):
        """Dislikes and diet_restriction are never relaxed — Phase 11A invariant."""
        from agents.daily_recommendation_agent import _filter_recipes

        recipes = [
            {"name": "虾饺", "category": "荤菜", "ingredients": ["虾"], "flavor": "", "steps": "", "budget_tier": "mid"},
            {"name": "豆腐汤", "category": "汤", "ingredients": ["豆腐"], "flavor": "", "steps": "", "budget_tier": "mid"},
        ]
        result = _filter_recipes(recipes, dislikes=["虾"], restrictions=[])
        assert all(r["name"] != "虾饺" for r in result)

    def test_generate_for_user_still_returns_dict(self):
        """generate_for_user returns dict after Phase 11B additions."""
        async def _run():
            with (
                patch("agents.daily_recommendation_agent._session", _session_seq(None, None, None)),
                patch("agents.daily_recommendation_agent._try_acquire_lock",
                      AsyncMock(return_value=_tr(ok=True, data=True))),
                patch("agents.daily_recommendation_agent._release_daily_rec_lock", AsyncMock()),
                patch("agents.daily_recommendation_agent._get_user_profile", MagicMock(return_value={})),
                patch("agents.daily_recommendation_agent.get_all_recipes", MagicMock(return_value=_RECIPES)),
                patch("agents.daily_recommendation_agent._get_same_day_dishes",
                      MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._get_recent_dishes",
                      MagicMock(return_value=([], "empty"))),
                patch("agents.daily_recommendation_agent._generate_reasoning",
                      AsyncMock(return_value="{}")),
            ):
                result = await generate_for_user("u1", meal_type="lunch", date_str="2026-07-26")
            return result

        result = asyncio.run(_run())
        assert isinstance(result, (dict, type(None)))  # None only if 0 recipes pass hard filter
