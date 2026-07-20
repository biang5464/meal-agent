"""
search_food_safety_result() 服务层单元测试（Phase 5）

用 patch 替换 tool_executor.execute，专注测试错误码映射和 ToolResult 语义，
不调用真实 ChromaDB 或 embedding 模型。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult
from tools.nutrition import search_food_safety_result


# ── 工厂 ─────────────────────────────────────────────────────────────────────

def _ok_result(data=None):
    return ToolResult(
        ok=True,
        data=data if data is not None else [],
        meta=ToolMeta(tool_name="search_food_safety", elapsed_ms=90, attempts=1),
    )


def _fail_result(code: ToolErrorCode, data=None):
    return ToolResult(
        ok=False,
        data=data or [],
        error=ToolError(code=code, message="mock error", retryable=True),
        meta=ToolMeta(tool_name="search_food_safety", elapsed_ms=2100, attempts=2),
    )


_CHUNKS = [{"content": "四季豆必须完全煮熟", "source": "food_safety_guide", "distance": 0.09}]


# ── 1. 检索成功，返回 chunks ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_returns_chunks():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result(_CHUNKS))
        result = await search_food_safety_result("四季豆中毒")

    assert result.ok is True
    assert result.data == _CHUNKS


# ── 2. 检索成功但无结果 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_empty_data():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result([]))
        result = await search_food_safety_result("obscure_query")

    assert result.ok is True
    assert result.data == []


# ── 3. INTERNAL 错误 → DEPENDENCY_UNAVAILABLE ────────────────────────────────

@pytest.mark.asyncio
async def test_internal_error_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.INTERNAL))
        result = await search_food_safety_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.error.details.get("collection") == "food_safety"


# ── 4. NETWORK 错误 → DEPENDENCY_UNAVAILABLE ─────────────────────────────────

@pytest.mark.asyncio
async def test_network_error_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.NETWORK))
        result = await search_food_safety_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE


# ── 5. TIMEOUT → 保留原错误码 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_not_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.TIMEOUT))
        result = await search_food_safety_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.TIMEOUT


# ── 6. 失败时 fallback data 透传 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_data_preserved():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.NETWORK, data=[]))
        result = await search_food_safety_result("任何查询")

    assert result.data == []


# ── 7. meta 透传（成功路径）─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_success():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=True,
            data=_CHUNKS,
            meta=ToolMeta(tool_name="search_food_safety", elapsed_ms=400, attempts=2),
        ))
        result = await search_food_safety_result("测试")

    assert result.meta.attempts == 2
    assert result.meta.elapsed_ms == 400


# ── 8. meta 在错误重映射后仍透传 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_remapped_error():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=False,
            data=[],
            error=ToolError(code=ToolErrorCode.INTERNAL, message="chroma down", retryable=True),
            meta=ToolMeta(tool_name="search_food_safety", elapsed_ms=1900, attempts=2),
        ))
        result = await search_food_safety_result("测试")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.meta.elapsed_ms == 1900
