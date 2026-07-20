"""
search_nutrition_result() 服务层单元测试（Phase 5）

用 patch 替换 tool_executor.execute，专注测试错误码映射和 ToolResult 语义，
不调用真实 ChromaDB 或 embedding 模型。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult
from tools.nutrition import search_nutrition_result


# ── 工厂 ─────────────────────────────────────────────────────────────────────

def _ok_result(data=None):
    return ToolResult(
        ok=True,
        data=data if data is not None else [],
        meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=80, attempts=1),
    )


def _fail_result(code: ToolErrorCode, data=None):
    return ToolResult(
        ok=False,
        data=data or [],
        error=ToolError(code=code, message="mock error", retryable=True),
        meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=2100, attempts=2),
    )


_CHUNKS = [{"content": "低嘌呤饮食原则", "source": "gout_diet", "distance": 0.12}]


# ── 1. 检索成功，返回 chunks ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_returns_chunks():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result(_CHUNKS))
        result = await search_nutrition_result("痛风饮食")

    assert result.ok is True
    assert result.data == _CHUNKS


# ── 2. 检索成功但无结果（success-no-data）────────────────────────────────────

@pytest.mark.asyncio
async def test_success_empty_data():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result([]))
        result = await search_nutrition_result("obscure_query")

    assert result.ok is True
    assert result.data == []


# ── 3. INTERNAL 错误 → 重映射为 DEPENDENCY_UNAVAILABLE ─────────────────────

@pytest.mark.asyncio
async def test_internal_error_remapped_to_dependency_unavailable():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.INTERNAL))
        result = await search_nutrition_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.error.retryable is True
    assert result.error.details.get("collection") == "nutrition"


# ── 4. NETWORK 错误 → 重映射为 DEPENDENCY_UNAVAILABLE ──────────────────────

@pytest.mark.asyncio
async def test_network_error_remapped_to_dependency_unavailable():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.NETWORK))
        result = await search_nutrition_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE


# ── 5. TIMEOUT 错误 → 保留原错误码，不重映射 ─────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_not_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.TIMEOUT))
        result = await search_nutrition_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.TIMEOUT


# ── 6. fallback data 在失败时透传 ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_data_preserved_on_failure():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.NETWORK, data=[]))
        result = await search_nutrition_result("任何查询")

    assert result.ok is False
    assert result.data == []


# ── 7. meta 透传（attempts、elapsed_ms 不被丢弃）────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_success():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=True,
            data=_CHUNKS,
            meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=350, attempts=2),
        ))
        result = await search_nutrition_result("测试")

    assert result.meta.attempts == 2
    assert result.meta.elapsed_ms == 350


# ── 8. meta 在错误重映射后仍透传 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_remapped_error():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=False,
            data=[],
            error=ToolError(code=ToolErrorCode.INTERNAL, message="chroma down", retryable=True),
            meta=ToolMeta(tool_name="search_nutrition", elapsed_ms=2050, attempts=2),
        ))
        result = await search_nutrition_result("测试")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.meta.elapsed_ms == 2050
    assert result.meta.attempts == 2
