"""Phase 7 test: daily_recommendation_agent._generate_reasoning LLM timeout → fallback JSON."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_generate_reasoning_timeout_returns_fallback():
    """LLM 超时时 _generate_reasoning 应返回 fallback JSON，不抛出异常。"""
    from agents.daily_recommendation_agent import _generate_reasoning
    from core.tool_protocol import ToolResult, ToolError, ToolErrorCode

    timeout_result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=ToolErrorCode.TIMEOUT, message="超时", retryable=False),
        meta=None,
    )

    dish_names = [{"name": "红烧肉"}, {"name": "清炒时蔬"}]
    user_profile = {"likes": ["辣"], "dislikes": [], "budget": "mid"}

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = timeout_result
        raw = await _generate_reasoning(dish_names, user_profile, "lunch")

    data = json.loads(raw)
    assert "dishes" in data
    assert "summary" in data
    dish_list = data["dishes"]
    assert any(d["dish"] in ("红烧肉", "") for d in dish_list)
    assert len(dish_list) == 2


@pytest.mark.asyncio
async def test_generate_reasoning_success():
    """LLM 成功时 _generate_reasoning 应返回解析后的 JSON。"""
    from agents.daily_recommendation_agent import _generate_reasoning
    from core.tool_protocol import ToolResult

    payload = json.dumps({
        "dishes": [{"dish": "红烧肉", "reason": "蛋白质丰富"}],
        "summary": "均衡搭配。",
    }, ensure_ascii=False)

    class _FakeResp:
        content = payload

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)
        raw = await _generate_reasoning([{"name": "红烧肉"}], {}, "lunch")

    data = json.loads(raw)
    assert data["dishes"][0]["dish"] == "红烧肉"
    assert data["summary"] == "均衡搭配。"


@pytest.mark.asyncio
async def test_generate_reasoning_bad_json_returns_fallback():
    """LLM 返回非 JSON 内容时应降级为 fallback JSON，不抛出异常。"""
    from agents.daily_recommendation_agent import _generate_reasoning
    from core.tool_protocol import ToolResult

    class _FakeResp:
        content = "这不是JSON内容"

    with patch("tools.tool_executor.tool_executor.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = ToolResult(ok=True, data=_FakeResp(), error=None, meta=None)
        raw = await _generate_reasoning([{"name": "红烧肉"}], {}, "lunch")

    data = json.loads(raw)
    assert "dishes" in data
    assert "summary" in data
