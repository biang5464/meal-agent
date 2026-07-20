"""
search_recipes_result() 服务层单元测试（Phase 5）

用 patch 替换 tool_executor.execute，专注测试错误码映射和 ToolResult 语义，
不调用真实 ChromaDB 或 embedding 模型。
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolError, ToolErrorCode, ToolMeta, ToolResult
from tools.recipe import search_recipes_result


# ── 工厂 ─────────────────────────────────────────────────────────────────────

def _ok_result(data=None):
    return ToolResult(
        ok=True,
        data=data if data is not None else [],
        meta=ToolMeta(tool_name="search_recipes", elapsed_ms=120, attempts=1),
    )


def _fail_result(code: ToolErrorCode, data=None):
    return ToolResult(
        ok=False,
        data=data or [],
        error=ToolError(code=code, message="mock error", retryable=True),
        meta=ToolMeta(tool_name="search_recipes", elapsed_ms=2100, attempts=2),
    )


_RECIPES = [
    {
        "name": "番茄炒蛋",
        "cuisine": "家常菜",
        "flavor": "鲜甜",
        "ingredients": ["番茄", "鸡蛋"],
        "steps": "...",
        "budget_tier": "low",
        "category": "半荤素",
        "relevance_score": 0.88,
    }
]


# ── 1. 检索成功，返回菜谱列表 ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_returns_recipes():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result(_RECIPES))
        result = await search_recipes_result("下饭的家常菜")

    assert result.ok is True
    assert result.data == _RECIPES


# ── 2. 检索成功但无匹配菜谱（success-no-data）───────────────────────────────

@pytest.mark.asyncio
async def test_success_empty_data():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result([]))
        result = await search_recipes_result("极其罕见的菜名")

    assert result.ok is True
    assert result.data == []


# ── 3. INTERNAL 错误 → DEPENDENCY_UNAVAILABLE ────────────────────────────────

@pytest.mark.asyncio
async def test_internal_error_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.INTERNAL))
        result = await search_recipes_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.error.retryable is True
    assert result.error.details.get("collection") == "recipes"


# ── 4. NETWORK 错误 → DEPENDENCY_UNAVAILABLE ─────────────────────────────────

@pytest.mark.asyncio
async def test_network_error_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.NETWORK))
        result = await search_recipes_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE


# ── 5. TIMEOUT → 保留原错误码，不重映射 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_not_remapped():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_fail_result(ToolErrorCode.TIMEOUT))
        result = await search_recipes_result("任何查询")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.TIMEOUT


# ── 6. user_likes / user_dislikes / budget_tier 传入 execute ─────────────────

@pytest.mark.asyncio
async def test_filters_forwarded_to_executor():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=_ok_result(_RECIPES))
        await search_recipes_result(
            "辣菜",
            user_likes=["辣"],
            user_dislikes=["花椒"],
            budget_tier="low",
            n_results=3,
        )

    call_kwargs = mock_exec.execute.call_args
    assert call_kwargs is not None
    # user_likes/dislikes/budget_tier 应作为 kwargs 传给 execute
    kwargs = call_kwargs.kwargs
    assert kwargs.get("user_likes") == ["辣"]
    assert kwargs.get("user_dislikes") == ["花椒"]
    assert kwargs.get("budget_tier") == "low"


# ── 7. meta 透传（成功路径）─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_success():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=True,
            data=_RECIPES,
            meta=ToolMeta(tool_name="search_recipes", elapsed_ms=500, attempts=2),
        ))
        result = await search_recipes_result("测试")

    assert result.meta.attempts == 2
    assert result.meta.elapsed_ms == 500


# ── 8. meta 在错误重映射后仍透传 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_preserved_on_remapped_error():
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(return_value=ToolResult(
            ok=False,
            data=[],
            error=ToolError(code=ToolErrorCode.INTERNAL, message="chroma down", retryable=True),
            meta=ToolMeta(tool_name="search_recipes", elapsed_ms=1800, attempts=2),
        ))
        result = await search_recipes_result("测试")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.meta.elapsed_ms == 1800
    assert result.meta.attempts == 2
