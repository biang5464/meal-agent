"""Phase 7 test: context_manager entity/exclusion extraction LLM timeout fallbacks."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agents.context_manager import extract_entity, extract_excluded_with_llm


@pytest.mark.asyncio
async def test_extract_entity_timeout_returns_last_entity():
    """LLM 超时时 extract_entity 应返回 last_entity（降级）。"""
    from core.tool_protocol import ToolResult, ToolError, ToolErrorCode

    timeout_result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=ToolErrorCode.TIMEOUT, message="超时", retryable=True),
        meta=None,
    )

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = timeout_result
        result = await extract_entity("帮我换一个", last_entity="MacBook Air")

    assert result == "MacBook Air"


@pytest.mark.asyncio
async def test_extract_entity_no_last_entity_timeout_returns_empty():
    """无 last_entity + LLM 超时 → 返回空字符串。"""
    from core.tool_protocol import ToolResult, ToolError, ToolErrorCode

    timeout_result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=ToolErrorCode.TIMEOUT, message="超时", retryable=True),
        meta=None,
    )

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = timeout_result
        result = await extract_entity("价格多少", last_entity="")

    assert result == ""


@pytest.mark.asyncio
async def test_extract_entity_pronoun_skips_llm():
    """代词引用时直接返回 last_entity，不调用 LLM。"""
    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        result = await extract_entity("它多少钱", last_entity="iPhone 16")

    mock_exec.assert_not_called()
    assert result == "iPhone 16"


@pytest.mark.asyncio
async def test_extract_entity_success():
    """LLM 成功时 extract_entity 返回提取结果。"""
    from core.tool_protocol import ToolResult

    class _FakeResp:
        content = "MacBook Air M3"

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)
        result = await extract_entity("MacBook Air M3价格")

    assert result == "MacBook Air M3"


@pytest.mark.asyncio
async def test_extract_excluded_timeout_returns_empty_list():
    """LLM 超时时 extract_excluded_with_llm 应返回空列表（降级）。"""
    from core.tool_protocol import ToolResult, ToolError, ToolErrorCode

    timeout_result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=ToolErrorCode.TIMEOUT, message="超时", retryable=True),
        meta=None,
    )

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = timeout_result
        result = await extract_excluded_with_llm("不要保护壳")

    assert result == []


@pytest.mark.asyncio
async def test_extract_excluded_success():
    """LLM 成功时 extract_excluded_with_llm 返回解析列表。"""
    from core.tool_protocol import ToolResult

    class _FakeResp:
        content = "保护壳,皮质"

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)
        result = await extract_excluded_with_llm("不要保护壳也不要皮质")

    assert result == ["保护壳", "皮质"]
