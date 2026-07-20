import pytest
import asyncio
from tools.tool_executor import ToolExecutor

executor = ToolExecutor()


async def fast_tool():
    return "success"


async def slow_tool():
    await asyncio.sleep(10)
    return "too late"


async def failing_tool():
    raise ConnectionError("network error")


async def always_fail():
    raise Exception("business error")


@pytest.mark.asyncio
async def test_success():
    result = await executor.run(fast_tool, timeout=5, retries=0, fallback=None)
    assert result == "success"


@pytest.mark.asyncio
async def test_timeout_returns_fallback():
    result = await executor.run(slow_tool, timeout=0.1, retries=0, fallback=[])
    assert result == []


@pytest.mark.asyncio
async def test_retry_on_connection_error():
    call_count = 0

    async def flaky_tool():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("retry me")
        return "recovered"

    result = await executor.run(
        flaky_tool, timeout=5, retries=2, fallback="failed"
    )
    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_error_returns_fallback():
    """业务错误不重试，直接返回 fallback"""
    result = await executor.run(
        always_fail, timeout=5, retries=2, fallback="default"
    )
    assert result == "default"


@pytest.mark.asyncio
async def test_all_retries_exhausted():
    """所有重试失败后返回 fallback"""
    result = await executor.run(
        failing_tool, timeout=5, retries=2, fallback=[]
    )
    assert result == []


@pytest.mark.asyncio
async def test_exponential_backoff_timing():
    """验证指数退避不会太快"""
    import time
    start = time.time()
    await executor.run(failing_tool, timeout=5, retries=2, fallback=None)
    elapsed = time.time() - start
    # 0.5 + 1.0 = 1.5秒最少等待
    assert elapsed >= 1.4
