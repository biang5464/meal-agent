import pytest
from core.tool_protocol import (
    ToolResult, ToolError, ToolMeta, ToolErrorCode,
    tool_success, tool_failure,
)
from core.tool_policy import ToolPolicy, TOOL_POLICIES, get_tool_policy


# ── tool_protocol 测试 ──────────────────────────────────────────

def test_tool_success_ok_true():
    result = tool_success(["item1"], tool_name="search")
    assert result.ok is True
    assert result.data == ["item1"]
    assert result.error is None

def test_tool_success_empty_list_is_success():
    """空列表 = 成功但无数据，不是失败"""
    result = tool_success([], tool_name="search")
    assert result.ok is True
    assert result.data == []
    assert result.error is None

def test_tool_success_none_data():
    result = tool_success(None, tool_name="search")
    assert result.ok is True
    assert result.data is None

def test_tool_success_meta_populated():
    result = tool_success(
        "data",
        tool_name="my_tool",
        elapsed_ms=42,
        attempts=2,
        source="woolworths",
        trace_id="abc123",
    )
    assert result.meta.tool_name == "my_tool"
    assert result.meta.elapsed_ms == 42
    assert result.meta.attempts == 2
    assert result.meta.fallback_used is False
    assert result.meta.source == "woolworths"
    assert result.meta.trace_id == "abc123"

def test_tool_failure_ok_false():
    result = tool_failure(
        ToolErrorCode.TIMEOUT,
        "timed out",
        tool_name="search",
    )
    assert result.ok is False
    assert result.error.code == ToolErrorCode.TIMEOUT
    assert result.error.message == "timed out"

def test_tool_failure_with_fallback_data():
    """失败时可以携带降级数据"""
    result = tool_failure(
        ToolErrorCode.TIMEOUT,
        "timed out",
        tool_name="search",
        fallback_data=[],
        fallback_used=True,
    )
    assert result.ok is False
    assert result.data == []
    assert result.meta.fallback_used is True

def test_tool_failure_retryable():
    result = tool_failure(
        ToolErrorCode.NETWORK,
        "connection error",
        tool_name="search",
        retryable=True,
    )
    assert result.error.retryable is True

def test_tool_failure_non_retryable():
    result = tool_failure(
        ToolErrorCode.INVALID_INPUT,
        "bad input",
        tool_name="search",
        retryable=False,
    )
    assert result.error.retryable is False

def test_tool_failure_meta_populated():
    result = tool_failure(
        ToolErrorCode.INTERNAL,
        "unexpected error",
        tool_name="my_tool",
        elapsed_ms=150,
        attempts=3,
    )
    assert result.meta.tool_name == "my_tool"
    assert result.meta.elapsed_ms == 150
    assert result.meta.attempts == 3

def test_success_vs_failure_distinction():
    """明确区分成功无数据和执行失败"""
    success = tool_success([], tool_name="search")
    failure = tool_failure(
        ToolErrorCode.TIMEOUT, "timed out",
        tool_name="search", fallback_data=[]
    )
    assert success.ok is True
    assert failure.ok is False
    assert success.data == failure.data  # 数据都是[]，但语义不同
    assert success.error is None
    assert failure.error is not None


# ── tool_policy 测试 ────────────────────────────────────────────

def test_all_policies_exist():
    expected = [
        "redis_read", "redis_write",
        "chroma_search",
        "database_read", "database_write",
        "external_http",
        "llm_classification", "llm_generation",
    ]
    for name in expected:
        assert name in TOOL_POLICIES, f"缺少策略: {name}"

def test_get_tool_policy_returns_correct():
    policy = get_tool_policy("external_http")
    assert policy.timeout == 8.0
    assert policy.retries == 2
    assert policy.idempotent is True

def test_get_tool_policy_unknown_raises():
    """未知策略必须抛出 ValueError，不能静默返回默认值"""
    with pytest.raises(ValueError, match="未知的 ToolPolicy"):
        get_tool_policy("nonexistent_policy")

def test_write_policies_not_idempotent():
    """写操作策略必须标记为非幂等"""
    assert TOOL_POLICIES["redis_write"].idempotent is False
    assert TOOL_POLICIES["database_write"].idempotent is False

def test_write_policies_no_retry():
    """写操作不重试"""
    assert TOOL_POLICIES["redis_write"].retries == 0
    assert TOOL_POLICIES["database_write"].retries == 0

def test_llm_generation_no_retry():
    """LLM 生成不重试（成本高）"""
    assert TOOL_POLICIES["llm_generation"].retries == 0

def test_external_http_retries():
    """外部 HTTP 允许重试"""
    assert TOOL_POLICIES["external_http"].retries == 2

def test_policy_frozen():
    """ToolPolicy 是不可变的"""
    policy = get_tool_policy("redis_read")
    with pytest.raises(Exception):
        policy.timeout = 999.0

def test_redis_timeout_fast():
    """Redis 超时必须很短"""
    assert TOOL_POLICIES["redis_read"].timeout <= 0.5
    assert TOOL_POLICIES["redis_write"].timeout <= 0.5
