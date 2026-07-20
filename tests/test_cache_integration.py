"""
tests/test_cache_integration.py

验证 memory_agent + update_agent 的 Redis 缓存集成：
  1. 第一次读画像 → 缓存未命中，从 MySQL 读，写入 Redis
  2. 第二次读同一 user_id → 缓存命中，不查 MySQL
  3. update_agent 写入画像后 → Redis 缓存被删除
  4. Redis 不可用时 → 降级直接查 MySQL，不报错

注意：测试使用真实 MySQL（由 conftest.py 初始化），测试后清理。
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.memory_agent import memory_agent
from agents.state import AgentState
from core.cache import get_profile_cache, invalidate_profile_cache, _client

TEST_USER = "test_integration_user"


def _make_state(user_id: str = TEST_USER) -> AgentState:
    return AgentState(
        user_id=user_id,
        user_input="",
        user_brief={},
        intent="",
        result="",
        messages=[],
        needs_meal=False,
    )


@pytest.fixture(autouse=True)
def setup_db_and_cache(fake_redis_client, in_memory_mysql_session, monkeypatch):
    """FakeRedis + in-memory SQLite 隔离：不依赖真实 Redis 或 MySQL。
    预建 TEST_USER 画像，使 memory_agent 能找到数据并写入缓存。
    """
    from models.user import UserProfileORM
    monkeypatch.setattr("core.cache._client", fake_redis_client)
    with in_memory_mysql_session() as session:
        session.add(UserProfileORM(user_id=TEST_USER))
        session.commit()
    invalidate_profile_cache(TEST_USER)
    yield
    invalidate_profile_cache(TEST_USER)


# ── 测试 1：缓存未命中 → SQLite → Redis ──────────────────────────────────────

@pytest.mark.asyncio
async def test_first_call_populates_cache():
    """第一次读画像：缓存应为空，读完后 Redis 里有这个 key。"""
    # 确认 Redis 里还没有
    assert get_profile_cache(TEST_USER) is None, "测试前 Redis 应为空"

    state = _make_state()
    result = await memory_agent(state)

    cached = get_profile_cache(TEST_USER)
    print(f"\n[1] memory_agent 返回 user_brief: {list(result['user_brief'].keys())}")
    print(f"[1] Redis 缓存内容: {cached}")

    assert cached is not None, "读完后 Redis 应有画像缓存"
    assert cached["user_id"] == TEST_USER
    assert "user_brief" in result
    print("[1] PASS — 缓存未命中 → SQLite → Redis 写入成功")


# ── 测试 2：第二次读 → 缓存命中，SQLite 不被查询 ─────────────────────────────

@pytest.mark.asyncio
async def test_second_call_hits_cache():
    """第二次读同一 user_id：命中缓存，get_user_profile 不应再被调用。"""
    # 第一次：真实写入缓存
    await memory_agent(_make_state())
    assert get_profile_cache(TEST_USER) is not None, "第一次后缓存应有数据"

    # 第二次：mock get_user_profile_result，验证未被调用（缓存命中，跳过 DB 查询）
    mock_result_fn = AsyncMock()

    with patch("agents.memory_agent.get_user_profile_result", mock_result_fn):
        result = await memory_agent(_make_state())

    print(f"\n[2] get_user_profile_result 调用次数: {mock_result_fn.call_count}")
    mock_result_fn.assert_not_called()
    assert "user_brief" in result
    print("[2] PASS — 缓存命中，SQLite 未被查询")


# ── 测试 3：update_agent 写入后 → 缓存被删除 ─────────────────────────────────

@pytest.mark.asyncio
async def test_update_agent_invalidates_cache():
    """update_agent 执行后，Redis 缓存应被清除。"""
    from agents.update_agent import update_agent

    # 预置缓存
    from core.cache import set_profile_cache
    set_profile_cache(TEST_USER, {"user_id": TEST_USER, "likes": ["辣"]})
    assert get_profile_cache(TEST_USER) is not None, "写入后应有缓存"

    # Mock LLM + create_react_agent，跳过真实 LLM 调用
    async def _empty_stream(*args, **kwargs):
        return
        yield  # 空异步生成器

    mock_agent = MagicMock()
    mock_agent.astream_events = _empty_stream

    with patch("agents.update_agent.create_react_agent", return_value=mock_agent), \
         patch("agents.update_agent.get_llm", return_value=MagicMock()):
        await update_agent(_make_state())

    cached = get_profile_cache(TEST_USER)
    print(f"\n[3] update_agent 执行后 get_profile_cache → {cached!r}")

    assert cached is None, "update_agent 执行后缓存应被清除"
    print("[3] PASS — update_agent 完成后缓存已失效")


# ── 测试 4：Redis 不可用 → 降级 SQLite，不报错 ────────────────────────────────

@pytest.mark.asyncio
async def test_redis_down_falls_back_to_mysql():
    """Redis 故障时，memory_agent 应降级到 MySQL，正常返回结果。"""
    import redis

    state = _make_state()

    with patch("core.cache.get_profile_cache",
               side_effect=redis.ConnectionError("Connection refused")), \
         patch("core.cache.set_profile_cache",
               side_effect=redis.ConnectionError("Connection refused")):
        result = await memory_agent(state)

    print(f"\n[4] Redis 故障时 user_brief 字段: {list(result['user_brief'].keys())}")

    assert "user_brief" in result, "Redis 故障不应影响 user_brief"
    assert "likes" in result["user_brief"]
    assert "dislikes" in result["user_brief"]
    print("[4] PASS — Redis 不可用时降级 MySQL，流程不中断")
