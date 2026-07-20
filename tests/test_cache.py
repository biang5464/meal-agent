"""
tests/test_cache.py

验证 core/cache.py 的用户画像缓存 + 对话历史功能。
测试数据隔离：所有测试使用 user_id="test_cache_user"，
测试结束后统一清理。

运行：.venv\\Scripts\\python.exe -m pytest tests/test_cache.py -v -s
"""

import pytest

from core.cache import (
    CHAT_HISTORY_LIMIT,
    PROFILE_TTL,
    append_chat_message,
    clear_chat_history,
    get_chat_history,
    get_profile_cache,
    invalidate_profile_cache,
    set_profile_cache,
    _client,
)

TEST_USER = "test_cache_user"


@pytest.fixture(autouse=True)
def cleanup(fake_redis_client, in_memory_mysql_session, monkeypatch):
    """每个测试前后各清理一次，保证隔离。使用 FakeRedis + in-memory SQLite 不依赖真实服务。"""
    import sys
    _this = sys.modules[__name__]
    monkeypatch.setattr("core.cache._client", fake_redis_client)
    monkeypatch.setattr(_this, "_client", fake_redis_client)
    invalidate_profile_cache(TEST_USER)
    clear_chat_history(TEST_USER)
    yield
    invalidate_profile_cache(TEST_USER)
    clear_chat_history(TEST_USER)


# ── 测试 1：set + get 画像 ────────────────────────────────────────────────────

def test_set_and_get_profile_cache():
    """测试 1：写入后能读回相同数据。"""
    profile = {"user_id": TEST_USER, "likes": ["辣", "川菜"], "budget": "mid"}
    set_profile_cache(TEST_USER, profile)

    result = get_profile_cache(TEST_USER)
    print(f"\n[1] get_profile_cache → {result}")

    assert result is not None
    assert result == profile
    print("[1] PASS — 写入与读取数据一致")


# ── 测试 2：查不存在的 key → None ─────────────────────────────────────────────

def test_get_profile_cache_miss():
    """测试 2：未写入时 get 返回 None。"""
    result = get_profile_cache("nonexistent_user_xyz_abc")
    print(f"\n[2] get_profile_cache(nonexistent) → {result!r}")

    assert result is None
    print("[2] PASS — 未命中返回 None")


# ── 测试 3：invalidate → get 返回 None ───────────────────────────────────────

def test_invalidate_profile_cache():
    """测试 3：写入后 invalidate，再 get 返回 None。"""
    set_profile_cache(TEST_USER, {"user_id": TEST_USER, "budget": "low"})
    assert get_profile_cache(TEST_USER) is not None

    invalidate_profile_cache(TEST_USER)
    result = get_profile_cache(TEST_USER)
    print(f"\n[3] get after invalidate → {result!r}")

    assert result is None
    print("[3] PASS — invalidate 后缓存已清除")


# ── 测试 4：append × 3 → 顺序正确 ────────────────────────────────────────────

def test_append_and_get_chat_history():
    """测试 4：追加3条消息，读回后顺序与追加顺序一致。"""
    append_chat_message(TEST_USER, "user", "我想吃川菜")
    append_chat_message(TEST_USER, "assistant", "好的，推荐麻婆豆腐")
    append_chat_message(TEST_USER, "user", "有没有素的？")

    history = get_chat_history(TEST_USER)
    print(f"\n[4] chat_history ({len(history)} 条):")
    for m in history:
        print(f"    [{m['role']}] {m['content']}")

    assert len(history) == 3
    assert history[0]["role"] == "user" and history[0]["content"] == "我想吃川菜"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user" and history[2]["content"] == "有没有素的？"
    assert all("timestamp" in m for m in history)
    print("[4] PASS — 3 条消息顺序正确")


# ── 测试 5：超出上限 → 自动裁剪 ──────────────────────────────────────────────

def test_chat_history_trim():
    """测试 5：写入超过 CHAT_HISTORY_LIMIT*2 条时，自动保留最新的上限条数。"""
    limit = CHAT_HISTORY_LIMIT * 2  # 20
    total = limit + 5               # 写入 25 条

    for i in range(total):
        role = "user" if i % 2 == 0 else "assistant"
        append_chat_message(TEST_USER, role, f"消息 {i}")

    history = get_chat_history(TEST_USER)
    print(f"\n[5] 写入 {total} 条，读回 {len(history)} 条（上限 {limit}）")
    print(f"    第一条内容: {history[0]['content']!r}")
    print(f"    最后一条内容: {history[-1]['content']!r}")

    assert len(history) == limit, f"期望 {limit} 条，实际 {len(history)} 条"
    # 应该保留最新的 limit 条（index 5 到 24）
    assert history[0]["content"] == f"消息 {total - limit}"
    assert history[-1]["content"] == f"消息 {total - 1}"
    print(f"[5] PASS — 超出上限后自动裁剪至 {limit} 条")


# ── 测试 6：clear → 返回空列表 ────────────────────────────────────────────────

def test_clear_chat_history():
    """测试 6：clear 后 get_chat_history 返回空列表。"""
    append_chat_message(TEST_USER, "user", "测试消息")
    assert len(get_chat_history(TEST_USER)) > 0

    clear_chat_history(TEST_USER)
    history = get_chat_history(TEST_USER)
    print(f"\n[6] clear 后 history = {history!r}")

    assert history == []
    print("[6] PASS — clear 后历史为空")


# ── 测试 7：TTL 验证 ──────────────────────────────────────────────────────────

def test_profile_cache_ttl():
    """测试 7：set_profile_cache 后，Redis key 的 TTL 在 (0, PROFILE_TTL] 范围内。"""
    set_profile_cache(TEST_USER, {"user_id": TEST_USER})

    key = f"user_profile:{TEST_USER}"
    ttl = _client.ttl(key)
    print(f"\n[7] TTL of {key!r} = {ttl}s（期望 0 < TTL <= {PROFILE_TTL}）")

    assert 0 < ttl <= PROFILE_TTL, f"TTL {ttl} 不在 (0, {PROFILE_TTL}] 范围内"
    print(f"[7] PASS — TTL={ttl}s 符合预期")
