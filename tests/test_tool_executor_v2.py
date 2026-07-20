"""
ToolExecutor.execute() 单元测试（第二阶段新增）

策略备忘（来自 core/tool_policy.py）：
    redis_read:       timeout=0.3, retries=1, backoff=0.5, idempotent=True
    chroma_search:    timeout=2.0, retries=1, backoff=0.5, idempotent=True
    external_http:    timeout=8.0, retries=2, backoff=0.5, idempotent=True
    redis_write:      timeout=0.3, retries=0,              idempotent=False
    database_write:   timeout=3.0, retries=0,              idempotent=False
"""
import asyncio
import pytest
from unittest.mock import patch

from core.tool_protocol import ToolErrorCode
from core.tool_policy import TOOL_POLICIES, ToolPolicy
from tools.tool_executor import ToolExecutor

executor = ToolExecutor()


# ── 成功路径 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_sync_success():
    """同步函数成功 → ok=True"""
    def sync_tool():
        return "sync_result"

    result = await executor.execute(sync_tool, policy_name="chroma_search", tool_name="sync")
    assert result.ok is True
    assert result.data == "sync_result"
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_async_success():
    """异步函数成功 → ok=True"""
    async def async_tool():
        return "async_result"

    result = await executor.execute(async_tool, policy_name="chroma_search", tool_name="async")
    assert result.ok is True
    assert result.data == "async_result"
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_empty_list_is_success():
    """成功返回空列表 → ok=True, data=[]（区分"成功无数据"与"执行失败"）"""
    async def empty_tool():
        return []

    result = await executor.execute(empty_tool, policy_name="chroma_search", tool_name="empty")
    assert result.ok is True
    assert result.data == []
    assert result.error is None


# ── 超时 ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_timeout_error_code():
    """超时 → ToolErrorCode.TIMEOUT"""
    async def always_slow():
        await asyncio.sleep(10)

    result = await executor.execute(
        always_slow,
        policy_name="redis_read",   # timeout=0.3, retries=1
        tool_name="slow",
    )
    assert result.ok is False
    assert result.error.code == ToolErrorCode.TIMEOUT
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_execute_timeout_triggers_retry():
    """超时后触发重试，第二次成功 → ok=True"""
    call_count = 0

    async def slow_then_fast():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(10)   # 第一次超时
        return "recovered"

    result = await executor.execute(
        slow_then_fast,
        policy_name="redis_read",   # timeout=0.3, retries=1
        tool_name="slow_then_fast",
    )
    assert result.ok is True
    assert result.data == "recovered"
    assert call_count == 2


# ── 网络错误 ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_network_error_code():
    """ConnectionError → ToolErrorCode.NETWORK，retryable=True"""
    async def always_net_fail():
        raise ConnectionError("network fail")

    result = await executor.execute(
        always_net_fail,
        policy_name="chroma_search",   # retries=1
        tool_name="net",
    )
    assert result.ok is False
    assert result.error.code == ToolErrorCode.NETWORK
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_execute_network_error_triggers_retry():
    """网络错误触发重试，第二次成功 → ok=True"""
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient")
        return "recovered"

    result = await executor.execute(
        flaky,
        policy_name="chroma_search",   # retries=1
        tool_name="flaky",
    )
    assert result.ok is True
    assert result.data == "recovered"
    assert call_count == 2


# ── 不可重试错误 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_value_error_no_retry():
    """ValueError → INVALID_INPUT，不重试"""
    call_count = 0

    async def bad_input():
        nonlocal call_count
        call_count += 1
        raise ValueError("invalid param")

    result = await executor.execute(
        bad_input,
        policy_name="external_http",   # retries=2，但 ValueError 不重试
        tool_name="bad",
    )
    assert result.ok is False
    assert result.error.code == ToolErrorCode.INVALID_INPUT
    assert result.error.retryable is False
    assert call_count == 1   # 严格只调用一次


@pytest.mark.asyncio
async def test_execute_unknown_exception_no_retry():
    """未知异常 → INTERNAL，不重试"""
    call_count = 0

    async def internal_error():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("something unexpected")

    result = await executor.execute(
        internal_error,
        policy_name="external_http",   # retries=2，但 RuntimeError 不重试
        tool_name="err",
    )
    assert result.ok is False
    assert result.error.code == ToolErrorCode.INTERNAL
    assert result.error.retryable is False
    assert call_count == 1


# ── Fallback ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_fallback_after_retries_exhausted():
    """所有重试失败后返回 fallback_data，meta.fallback_used=True"""
    async def always_fail():
        raise ConnectionError("network fail")

    result = await executor.execute(
        always_fail,
        policy_name="chroma_search",   # retries=1
        tool_name="fail",
        fallback_data=["cached_item"],
    )
    assert result.ok is False
    assert result.data == ["cached_item"]
    assert result.meta.fallback_used is True


@pytest.mark.asyncio
async def test_execute_no_fallback_data_is_none():
    """未提供 fallback_data 时，data=None，fallback_used=False"""
    async def always_fail():
        raise ConnectionError("fail")

    result = await executor.execute(
        always_fail,
        policy_name="chroma_search",
        tool_name="fail",
    )
    assert result.ok is False
    assert result.data is None
    assert result.meta.fallback_used is False


# ── 幂等性 ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_non_idempotent_no_retry():
    """非幂等策略（idempotent=False）：即使 policy.retries>0 也强制不重试"""
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("fail")

    # 临时注入 retries=2 但 idempotent=False 的策略
    fake_policy = ToolPolicy(timeout=5.0, retries=2, idempotent=False)
    with patch.dict(TOOL_POLICIES, {"_test_non_idempotent": fake_policy}):
        result = await executor.execute(
            always_fail,
            policy_name="_test_non_idempotent",
            tool_name="non_idempotent",
        )

    assert result.ok is False
    assert call_count == 1   # retries=2 被强制为 0，只调用一次


# ── Meta 字段 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_meta_attempts_counts_retries():
    """meta.attempts 正确记录实际调用次数（含重试）"""
    async def always_fail():
        raise ConnectionError("fail")

    result = await executor.execute(
        always_fail,
        policy_name="chroma_search",   # retries=1 → 共2次
        tool_name="retry_count",
    )
    assert result.meta.attempts == 2


@pytest.mark.asyncio
async def test_execute_meta_attempts_on_success():
    """第一次成功时 meta.attempts=1"""
    async def ok_tool():
        return 42

    result = await executor.execute(ok_tool, policy_name="chroma_search", tool_name="ok")
    assert result.meta.attempts == 1


@pytest.mark.asyncio
async def test_execute_meta_elapsed_ms_positive():
    """meta.elapsed_ms > 0（存在实际耗时）"""
    async def slightly_slow():
        await asyncio.sleep(0.01)
        return "ok"

    result = await executor.execute(
        slightly_slow,
        policy_name="chroma_search",
        tool_name="timed",
    )
    assert result.ok is True
    assert result.meta.elapsed_ms > 0


@pytest.mark.asyncio
async def test_execute_meta_tool_name_propagated():
    """meta.tool_name 与传入 tool_name 一致"""
    async def tool():
        return "x"

    result = await executor.execute(tool, policy_name="chroma_search", tool_name="my_search")
    assert result.meta.tool_name == "my_search"


@pytest.mark.asyncio
async def test_execute_source_and_trace_id_propagated():
    """source 和 trace_id 正确写入 meta"""
    async def tool():
        return "data"

    result = await executor.execute(
        tool,
        policy_name="chroma_search",
        tool_name="traced",
        source="woolworths",
        trace_id="req-abc-123",
    )
    assert result.ok is True
    assert result.meta.source == "woolworths"
    assert result.meta.trace_id == "req-abc-123"


@pytest.mark.asyncio
async def test_execute_unknown_policy_raises():
    """未知 policy_name 必须抛出 ValueError，不能静默返回"""
    async def tool():
        return "x"

    with pytest.raises(ValueError, match="未知的 ToolPolicy"):
        await executor.execute(tool, policy_name="nonexistent_policy", tool_name="t")


@pytest.mark.asyncio
async def test_execute_fallback_none_explicit():
    """调用方显式传 fallback_data=None 时：fallback_used=True，data=None"""
    async def always_fail():
        raise ConnectionError("fail")

    result = await executor.execute(
        always_fail,
        policy_name="chroma_search",
        tool_name="none_fallback",
        fallback_data=None,   # 显式传 None（与"未传"不同）
    )
    assert result.ok is False
    assert result.data is None
    assert result.meta.fallback_used is True  # 哨兵确保此处为 True
