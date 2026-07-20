"""Phase 7 test: supervisor_agent LLM timeout → CHAT fallback."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agents.supervisor_agent import supervisor_agent, _SUPERVISOR_FALLBACK
from agents.state import AgentState


def _make_state(user_input: str = "你好") -> AgentState:
    return AgentState(
        user_id="u1",
        user_input=user_input,
        user_brief={},
        intent="",
        result="",
        messages=[],
        needs_meal=False,
    )


@pytest.mark.asyncio
async def test_supervisor_timeout_falls_back_to_chat():
    """LLM 超时时 supervisor_agent 应降级为 CHAT/LOW，不抛出异常。"""
    from core.tool_protocol import ToolResult, ToolError, ToolErrorCode

    timeout_result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(
            code=ToolErrorCode.TIMEOUT,
            message="超时>20.0s",
            retryable=True,
        ),
        meta=None,
    )

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = timeout_result
        result = await supervisor_agent(_make_state())

    assert result["intent"] == "CHAT"
    assert result["confidence"] == "LOW"
    assert result["missing_slots"] == []
    assert result["top2"] is None
    assert result["needs_meal"] is False


@pytest.mark.asyncio
async def test_supervisor_success_parses_intent():
    """LLM 成功时 supervisor_agent 应正确解析意图。"""
    from core.tool_protocol import ToolResult

    class _FakeResp:
        content = "INTENT: MEAL\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"

    success_result = ToolResult(
        ok=True,
        data=_FakeResp(),
        error=None,
        meta=None,
    )

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = success_result
        result = await supervisor_agent(_make_state("今天吃什么"))

    assert result["intent"] == "MEAL"
    assert result["confidence"] == "HIGH"
    assert result["missing_slots"] == []
    assert result["needs_meal"] is False


@pytest.mark.asyncio
async def test_supervisor_uses_llm_classification_policy():
    """supervisor_agent 必须使用 llm_classification policy。"""
    from core.tool_protocol import ToolResult

    class _FakeResp:
        content = "INTENT: CHAT\nCONFIDENCE: LOW\nMISSING_SLOTS: NONE\nTOP2: NONE"

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)
        await supervisor_agent(_make_state())

    _call_kwargs = mock_exec.call_args
    assert _call_kwargs.kwargs.get("policy_name") == "llm_classification" or \
           _call_kwargs.args[2] == "llm_classification" or \
           "llm_classification" in str(_call_kwargs)
