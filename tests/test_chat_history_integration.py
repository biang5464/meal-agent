"""
tests/test_chat_history_integration.py

验证对话历史在 Redis + MySQL 之间的双写、降级和截断行为：
  1. 第一轮对话后，Redis 和 MySQL 都有记录
  2. 第二轮对话，graph.run() 将第一轮历史注入 state["chat_history"]
  3. Redis 清空后，get_chat_history 从 MySQL 补回历史
  4. 超过10轮（20条）后，Redis 自动裁剪；MySQL 保留全部

数据隔离：TEST_USER = "test_chat_hist_user"，每个测试前后清理。
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from core.cache import (
    CHAT_HISTORY_LIMIT,
    append_chat_message,
    clear_chat_history,
    get_chat_history,
    _client,
)

TEST_USER = "test_chat_hist_user"


@pytest.fixture(autouse=True)
def cleanup(fake_redis_client, in_memory_mysql_session, monkeypatch):
    """FakeRedis + in-memory SQLite 隔离：不依赖真实 Redis 或 MySQL。"""
    import sys
    _this = sys.modules[__name__]
    monkeypatch.setattr("core.cache._client", fake_redis_client)
    monkeypatch.setattr(_this, "_client", fake_redis_client)
    clear_chat_history(TEST_USER)
    yield
    clear_chat_history(TEST_USER)


# ── 测试 1：第一轮对话后，Redis 和 MySQL 都有记录 ─────────────────────────────

def test_first_round_writes_redis_and_mysql():
    """第一轮对话写入后，Redis 和 MySQL 各有2条记录。"""
    append_chat_message(TEST_USER, "user", "你好")
    append_chat_message(TEST_USER, "assistant", "你好！我是你的饮食助手。")

    # 验证 Redis
    history = get_chat_history(TEST_USER)
    print(f"\n[1] Redis history: {[(m['role'], m['content']) for m in history]}")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "你好"
    assert history[1]["role"] == "assistant"

    # 验证 MySQL
    from core.database import SessionLocal
    from models.chat_history import ChatHistoryORM
    with SessionLocal() as session:
        rows = (
            session.query(ChatHistoryORM)
            .filter_by(user_id=TEST_USER)
            .order_by(ChatHistoryORM.id)
            .all()
        )
    print(f"[1] MySQL rows: {[(r.role, r.content) for r in rows]}")
    assert len(rows) == 2
    assert rows[0].role == "user"
    assert rows[1].role == "assistant"
    print("[1] PASS — Redis 和 MySQL 各有2条记录")


# ── 测试 2：第二轮对话，graph.run() 注入第一轮历史到 state ───────────────────

@pytest.mark.asyncio
async def test_graph_injects_history_on_second_call():
    """run() 在第二轮调用时，将第一轮历史注入 state['chat_history']。"""
    # 模拟第一轮历史
    append_chat_message(TEST_USER, "user", "你好")
    append_chat_message(TEST_USER, "assistant", "你好！我是你的饮食助手。")

    # 第二轮：mock compiled_graph.ainvoke，捕获初始 state
    from agents.graph import run

    captured = {}

    async def mock_ainvoke(state, **kwargs):
        captured.update(state)
        return {**state, "result": "模拟：你刚才说了你好"}

    with patch("agents.graph.compiled_graph.ainvoke", new=mock_ainvoke):
        await run(TEST_USER, "我刚才说了什么？")

    chat_hist = captured.get("chat_history", [])
    print(f"\n[2] 注入的 chat_history ({len(chat_hist)} 条): "
          f"{[(m['role'], m['content']) for m in chat_hist]}")

    assert len(chat_hist) >= 2, "state['chat_history'] 应包含第一轮历史"
    assert chat_hist[0]["content"] == "你好"
    assert chat_hist[0]["role"] == "user"
    print("[2] PASS — 第二轮 state['chat_history'] 正确注入第一轮历史")


# ── 测试 3：Redis 清空后，从 MySQL 补回历史 ──────────────────────────────────

def test_mysql_fallback_when_redis_empty():
    """Redis key 过期/清空后，get_chat_history 从 MySQL 取回记录。"""
    append_chat_message(TEST_USER, "user", "我喜欢吃辣")
    append_chat_message(TEST_USER, "assistant", "好的，已记录你喜欢辣。")

    # 只清 Redis，保留 MySQL
    redis_key = f"chat_history:{TEST_USER}"
    _client.delete(redis_key)
    assert _client.llen(redis_key) == 0, "Redis 应已清空"

    # get_chat_history 应从 MySQL 补回并回填 Redis
    history = get_chat_history(TEST_USER)
    print(f"\n[3] MySQL 降级后 history: {[(m['role'], m['content']) for m in history]}")

    assert len(history) >= 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "我喜欢吃辣"

    # 确认已回填 Redis
    redis_len = _client.llen(redis_key)
    print(f"[3] Redis 回填后长度: {redis_len}")
    assert redis_len >= 2
    print("[3] PASS — Redis 清空后从 MySQL 降级并回填")


# ── 测试 4：超过10轮，Redis 裁剪到20条，MySQL 保留全部 ────────────────────────

def test_redis_trims_at_limit():
    """写入25条后，Redis 保留最新20条，MySQL 保留全部25条。"""
    total = CHAT_HISTORY_LIMIT * 2 + 5   # 25
    limit = CHAT_HISTORY_LIMIT * 2        # 20

    for i in range(total):
        role = "user" if i % 2 == 0 else "assistant"
        append_chat_message(TEST_USER, role, f"消息{i}")

    # Redis 应只有20条（最新的）
    history = get_chat_history(TEST_USER)
    print(f"\n[4] 写入{total}条后，Redis get_chat_history = {len(history)} 条")
    assert len(history) == limit, f"期望{limit}条，实际{len(history)}条"
    assert history[0]["content"] == f"消息{total - limit}"
    assert history[-1]["content"] == f"消息{total - 1}"

    # MySQL 应保留全部25条
    from core.database import SessionLocal
    from models.chat_history import ChatHistoryORM
    with SessionLocal() as session:
        mysql_count = (
            session.query(ChatHistoryORM)
            .filter_by(user_id=TEST_USER)
            .count()
        )
    print(f"[4] MySQL 记录数: {mysql_count}")
    assert mysql_count == total
    print(f"[4] PASS — Redis 裁剪到{limit}条，MySQL 保留全部{total}条")
