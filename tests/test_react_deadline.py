"""Phase 8 test: ReAct Agent Deadline — run_react_with_deadline 核心行为。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.react_deadline import run_react_with_deadline
from agents.state import AgentState
from tools.timeout_config import TimeoutConfig


# ── 测试辅助 ────────────────────────────────────────────────────────────────────

class _FakeChunk:
    def __init__(self, content: str):
        self.content = content


def _event(token: str) -> dict:
    return {"event": "on_chat_model_stream", "data": {"chunk": _FakeChunk(token)}}


def _make_agent(tokens: list[str] | None = None, hang: bool = False, raise_exc=None):
    """返回一个带有 fake astream_events 的 mock agent。"""
    async def _astream(input_dict, version, config=None):
        if raise_exc is not None:
            raise raise_exc
        if tokens is not None:
            for tok in tokens:
                yield _event(tok)
        if hang:
            await asyncio.sleep(9999)

    agent = MagicMock()
    agent.astream_events = _astream
    return agent


# ── 1. 正常完成 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normal_completion_returns_all_tokens():
    """正常流式完成时，应返回所有 token 拼接后的字符串。"""
    agent = _make_agent(tokens=["你好，", "我是助手。"])
    result = await run_react_with_deadline(agent, {}, deadline=10.0)
    assert result == "你好，我是助手。"


# ── 2. 超时 + 部分内容 ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_with_partial_content_appends_suffix():
    """超时时如果有已收集的内容，应附加部分超时提示。"""
    # 第一个 token 快速产生，之后挂起
    collected: list[str] = []

    async def _slow_astream(input_dict, version, config=None):
        yield _event("partial_result")
        await asyncio.sleep(9999)

    agent = MagicMock()
    agent.astream_events = _slow_astream

    result = await run_react_with_deadline(agent, {}, deadline=0.05)
    assert "partial_result" in result
    assert "超时" in result


# ── 3. 超时 + 无内容 ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_with_no_content_returns_fallback():
    """超时时如果没有任何内容，应返回完整的降级提示消息。"""
    agent = _make_agent(hang=True)
    result = await run_react_with_deadline(agent, {}, deadline=0.05)
    assert "超时" in result or "处理时间" in result
    assert "partial" not in result.lower()


# ── 4. CancelledError 传播 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancelled_error_propagates():
    """外部取消时 CancelledError 必须向上传播，不得被转换为超时降级。"""
    agent = _make_agent(hang=True)

    task = asyncio.create_task(
        run_react_with_deadline(agent, {}, deadline=9999)
    )
    # 确保 task 已启动
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ── 5. on_token 回调 ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_token_callback_called_for_each_token():
    """on_token 回调应对每个 token 精确调用一次。"""
    tokens = ["A", "B", "C"]
    agent = _make_agent(tokens=tokens)
    received: list[str] = []

    await run_react_with_deadline(agent, {}, deadline=10.0, on_token=received.append)

    assert received == tokens


# ── 6. meal_agent 使用 MEAL_AGENT_DEADLINE ────────────────────────────────────

@pytest.mark.asyncio
async def test_meal_agent_uses_meal_agent_deadline():
    """meal_agent 调用 run_react_with_deadline 时必须传入 MEAL_AGENT_DEADLINE。"""
    from agents.meal_agent import meal_agent

    captured: list[float] = []

    async def _fake_helper(agent, input_dict, *, deadline, config=None, on_token=None):
        captured.append(deadline)
        return "ok"

    state = AgentState(
        user_id="u1", user_input="今天吃什么", user_brief={},
        intent="MEAL", result="", messages=[], needs_meal=False,
    )

    with patch("agents.meal_agent.run_react_with_deadline", new=_fake_helper), \
         patch("agents.meal_agent.get_llm", return_value=MagicMock()), \
         patch("agents.meal_agent.create_react_agent", return_value=MagicMock()):
        result = await meal_agent(state)

    assert captured == [TimeoutConfig.MEAL_AGENT_DEADLINE]
    assert result["result"] == "ok"


# ── 7. price_agent 使用 PRICE_AGENT_DEADLINE ──────────────────────────────────

@pytest.mark.asyncio
async def test_price_agent_uses_price_agent_deadline():
    """price_agent 调用 run_react_with_deadline 时必须传入 PRICE_AGENT_DEADLINE。"""
    from agents.price_agent import price_agent

    captured: list[float] = []

    async def _fake_helper(agent, input_dict, *, deadline, config=None, on_token=None):
        captured.append(deadline)
        return "ok"

    state = AgentState(
        user_id="u1", user_input="鸡胸肉多少钱", user_brief={},
        intent="PRICE", result="", messages=[], needs_meal=False,
    )

    with patch("agents.price_agent.run_react_with_deadline", new=_fake_helper), \
         patch("agents.price_agent.get_llm", return_value=MagicMock()), \
         patch("agents.price_agent.create_react_agent", return_value=MagicMock()):
        result = await price_agent(state)

    assert captured == [TimeoutConfig.PRICE_AGENT_DEADLINE]
    assert result["result"] == "ok"


# ── 8. update_agent 使用 UPDATE_AGENT_DEADLINE ────────────────────────────────

@pytest.mark.asyncio
async def test_update_agent_uses_update_agent_deadline():
    """update_agent 调用 run_react_with_deadline 时必须传入 UPDATE_AGENT_DEADLINE。"""
    from agents.update_agent import update_agent

    captured: list[float] = []

    async def _fake_helper(agent, input_dict, *, deadline, config=None, on_token=None):
        captured.append(deadline)
        return "ok"

    state = AgentState(
        user_id="u1", user_input="我不喜欢香菜", user_brief={},
        intent="UPDATE", result="", messages=[], needs_meal=False,
    )

    from core.tool_protocol import ToolResult
    fake_inv = ToolResult(ok=True, data=None, error=None, meta=None)

    with patch("agents.update_agent.run_react_with_deadline", new=_fake_helper), \
         patch("agents.update_agent.get_llm", return_value=MagicMock()), \
         patch("agents.update_agent.create_react_agent", return_value=MagicMock()), \
         patch("agents.update_agent.invalidate_cache_result", new_callable=AsyncMock) as m:
        m.return_value = fake_inv
        result = await update_agent(state)

    assert captured == [TimeoutConfig.UPDATE_AGENT_DEADLINE]
    assert result["result"] == "ok"


# ── 9. 非 TimeoutError 异常向外传播 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_unexpected_exception_propagates():
    """astream_events 抛出非超时异常时，应向外传播，不被吞掉。"""
    agent = _make_agent(raise_exc=ValueError("模型返回格式错误"))

    with pytest.raises(ValueError, match="模型返回格式错误"):
        await run_react_with_deadline(agent, {}, deadline=10.0)
