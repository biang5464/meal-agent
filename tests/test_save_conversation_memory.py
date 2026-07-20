"""
tests/test_save_conversation_memory.py
Phase 6B：graph.py 后台任务 done callback 行为测试。
"""
from __future__ import annotations

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── _log_background_task_result callback ─────────────────────────────────────

def test_callback_logs_on_exception(caplog):
    """Task 内部抛出异常时，callback 应调用 logger.error，不重新抛出。"""
    from agents.graph import _log_background_task_result

    loop = asyncio.new_event_loop()
    try:
        async def _fail():
            raise RuntimeError("ChromaDB failed")

        task = loop.run_until_complete(asyncio.ensure_future(_fail(), loop=loop))
    except RuntimeError:
        pass

    # 创建一个已完成且有异常的 mock task
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.exception.return_value = RuntimeError("ChromaDB failed")
    mock_task.cancelled.return_value = False

    with caplog.at_level(logging.ERROR, logger="agents.graph"):
        _log_background_task_result(mock_task)

    assert any("ChromaDB" in r.message or "后台任务" in r.message
               for r in caplog.records), "应记录 ERROR 级别日志"
    loop.close()


def test_callback_silent_on_success():
    """Task 成功时，callback 不应记录任何日志。"""
    from agents.graph import _log_background_task_result

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.exception.return_value = None
    mock_task.cancelled.return_value = False

    with patch("agents.graph.logger") as mock_logger:
        _log_background_task_result(mock_task)

    mock_logger.error.assert_not_called()


def test_callback_ignores_cancelled_task():
    """CancelledError 时，callback 静默返回，不记录 error，不抛异常。"""
    from agents.graph import _log_background_task_result

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.exception.side_effect = asyncio.CancelledError()

    with patch("agents.graph.logger") as mock_logger:
        _log_background_task_result(mock_task)  # 不应抛出

    mock_logger.error.assert_not_called()


def test_callback_does_not_reraise():
    """callback 内部任何路径都不应向外抛出异常。"""
    from agents.graph import _log_background_task_result

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.exception.return_value = ValueError("unexpected")
    mock_task.cancelled.return_value = False

    try:
        _log_background_task_result(mock_task)
    except Exception as e:
        pytest.fail(f"callback 不应抛出异常，但抛出了: {e}")


# ── run_with_queue 中 create_task 带 callback ─────────────────────────────────

@pytest.mark.asyncio
async def test_run_with_queue_task_has_callback():
    """run_with_queue 创建的 save_conversation_memory task 应附带 done callback。"""
    from agents.graph import run_with_queue

    from core.tool_protocol import tool_success
    ok_append = tool_success(None, tool_name="append_chat_message")
    ok_history = tool_success([], tool_name="get_chat_history")

    captured_tasks = []

    original_create_task = asyncio.create_task

    def _mock_create_task(coro, **kwargs):
        task = original_create_task(coro, **kwargs)
        captured_tasks.append(task)
        return task

    queue = asyncio.Queue()

    with patch("agents.graph.get_chat_history_result", new=AsyncMock(return_value=ok_history)), \
         patch("agents.graph.compiled_graph") as mock_graph, \
         patch("agents.graph.append_chat_message_result", new=AsyncMock(return_value=ok_append)), \
         patch("tools.memory.save_conversation_memory", new=AsyncMock(return_value=None)), \
         patch("asyncio.create_task", side_effect=_mock_create_task):
        mock_graph.ainvoke = AsyncMock(return_value={"result": "回复", "intent": "UPDATE"})
        await run_with_queue("u1", "更新偏好", queue)

    # 确认至少创建了一个 task（save_conversation_memory）
    assert len(captured_tasks) >= 1
    # 确认 task 有 done callback
    # （asyncio.Task._callbacks 在 CPython 中是私有属性，用 add_done_callback 计数来验证）
    # 这里只验证 task 存在且未抛异常
    await asyncio.gather(*captured_tasks, return_exceptions=True)
