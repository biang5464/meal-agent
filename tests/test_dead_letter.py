"""Phase 8 test: Dead-letter 持久化 — write 函数、幂等性、daily_rec 接线。"""

from __future__ import annotations

import asyncio
import importlib
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.dead_letter import (
    DeadLetterORM,
    _write_dead_letter_sync,
    init_dead_letter_db,
    write_dead_letter_result,
)


# ── fixture：每个测试独立的临时 DB ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_dl_globals():
    """保证每个测试前后 dead_letter 模块全局 _engine/_Session 都被重置。"""
    import tools.dead_letter as dl
    yield
    dl._engine = None
    dl._Session = None


@pytest.fixture
def dl_db(tmp_path):
    """初始化临时 Dead-letter DB，返回 db 路径。"""
    db_path = str(tmp_path / "dl.db")
    init_dead_letter_db(db_path)
    return db_path


# ── 1. 初始化建表 ───────────────────────────────────────────────────────────────

def test_init_creates_table(dl_db):
    """init_dead_letter_db 应成功创建 dead_letter_tasks 表。"""
    import tools.dead_letter as dl
    assert dl._Session is not None
    with dl._Session() as session:
        rows = session.query(DeadLetterORM).all()
        assert rows == []


# ── 2. 同步写入 ─────────────────────────────────────────────────────────────────

def test_write_inserts_row_with_correct_fields(dl_db):
    """_write_dead_letter_sync 应插入一行，所有字段匹配输入。"""
    import tools.dead_letter as dl

    _write_dead_letter_sync("user_001", "lunch", "2024-01-15", "TIMEOUT")

    with dl._Session() as session:
        row = session.query(DeadLetterORM).first()
    assert row is not None
    assert row.user_id == "user_001"
    assert row.meal_type == "lunch"
    assert row.target_date == "2024-01-15"
    assert row.error_code == "TIMEOUT"
    assert row.status == "pending"
    assert row.attempts == 1
    assert row.idempotency_key == "daily_recommendation:user_001:2024-01-15:lunch"


# ── 3. 幂等性 ───────────────────────────────────────────────────────────────────

def test_idempotency_increments_attempts_not_duplicates(dl_db):
    """相同 idempotency_key 再次写入应 attempts+1，不应插入重复行。"""
    import tools.dead_letter as dl

    _write_dead_letter_sync("user_002", "dinner", "2024-01-15", "NETWORK")
    _write_dead_letter_sync("user_002", "dinner", "2024-01-15", "TIMEOUT")

    with dl._Session() as session:
        rows = session.query(DeadLetterORM).all()

    assert len(rows) == 1
    assert rows[0].attempts == 2
    assert rows[0].error_code == "TIMEOUT"  # 最新的 error_code


# ── 4. write_dead_letter_result 返回 ok=True ──────────────────────────────────

@pytest.mark.asyncio
async def test_write_result_ok_on_success(dl_db):
    """write_dead_letter_result 写入成功时应返回 ok=True。"""
    result = await write_dead_letter_result("user_003", "lunch", "2024-01-16", "INTERNAL")
    assert result.ok is True


# ── 5. target_date 不被写入时间覆盖 ────────────────────────────────────────────

def test_target_date_not_overwritten_by_write_time(dl_db):
    """target_date 必须是原始业务日期，而不是写入时的当天日期。"""
    import tools.dead_letter as dl

    historical_date = "2024-01-01"
    _write_dead_letter_sync("user_004", "lunch", historical_date, "INTERNAL")

    with dl._Session() as session:
        row = session.query(DeadLetterORM).first()

    assert row.target_date == historical_date
    # 写入时间（created_at）是今天，与 target_date 无关
    assert row.target_date != date.today().isoformat()


# ── 6. 未初始化时返回 ok=False ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_result_fail_if_not_initialized():
    """在未调用 init_dead_letter_db 的情况下，write_dead_letter_result 应返回 ok=False。"""
    # reset_dl_globals fixture 保证本测试开始时 _Session is None
    result = await write_dead_letter_result("user_005", "lunch", "2024-01-17", "INTERNAL")
    assert result.ok is False


# ── 7. daily_rec 在 Redis 失败时写入 Dead-letter ──────────────────────────────

@pytest.mark.asyncio
async def test_daily_rec_writes_dead_letter_on_redis_failure(dl_db):
    """generate_daily_recommendations：Redis 入队失败时，应调用 write_dead_letter_result。"""
    from core.tool_protocol import ToolError, ToolErrorCode, ToolResult

    redis_fail = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=ToolErrorCode.NETWORK, message="connection refused", retryable=True),
        meta=None,
    )

    dl_calls: list[tuple] = []

    async def _fake_dl(user_id, meal_type, target_date, error_code="INTERNAL"):
        dl_calls.append((user_id, meal_type, target_date, error_code))
        return ToolResult(ok=True, data=None, error=None, meta=None)

    with patch(
        "agents.daily_recommendation_agent.generate_for_user",
        new_callable=AsyncMock,
        side_effect=RuntimeError("DB 写入失败"),
    ), patch(
        "agents.daily_recommendation_agent._enqueue_retry_result",
        new_callable=AsyncMock,
        return_value=redis_fail,
    ), patch(
        "agents.daily_recommendation_agent.write_dead_letter_result",
        new=_fake_dl,
    ), patch(
        "agents.daily_recommendation_agent._session"
    ) as mock_session:
        # 模拟返回 1 个用户
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)

        class _FakeRow:
            def __iter__(self):
                yield "user_999"

        ctx.query.return_value.all.return_value = [("user_999",)]
        mock_session.return_value = ctx

        from agents.daily_recommendation_agent import generate_daily_recommendations
        await generate_daily_recommendations()

    # 2 次（lunch + dinner）
    assert len(dl_calls) == 2
    # target_date 应为 today（与 generate_for_user 所用日期一致）
    today = date.today().isoformat()
    for user_id, meal_type, target_date, error_code in dl_calls:
        assert user_id == "user_999"
        assert target_date == today
        assert error_code == "NETWORK"


# ── 8. daily_rec 生成成功时不写 Dead-letter ───────────────────────────────────

@pytest.mark.asyncio
async def test_daily_rec_no_dead_letter_on_success():
    """generate_daily_recommendations：推荐生成成功时不应调用 write_dead_letter_result。"""
    dl_calls: list = []

    async def _fake_dl(*args, **kwargs):
        dl_calls.append(args)
        return MagicMock(ok=True)

    with patch(
        "agents.daily_recommendation_agent.generate_for_user",
        new_callable=AsyncMock,
        return_value={"id": 1},
    ), patch(
        "agents.daily_recommendation_agent.write_dead_letter_result",
        new=_fake_dl,
    ), patch(
        "agents.daily_recommendation_agent._session"
    ) as mock_session:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.query.return_value.all.return_value = [("user_888",)]
        mock_session.return_value = ctx

        from agents.daily_recommendation_agent import generate_daily_recommendations
        await generate_daily_recommendations()

    assert dl_calls == [], "生成成功时不应写入 Dead-letter"
