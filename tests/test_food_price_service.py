"""
get_food_prices_result() 服务层单元测试（第四阶段新增）

用 patch 替换 tool_executor.execute，
专注测试聚合逻辑与 ToolResult 语义，不产生真实 HTTP 流量。
"""
import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolResult, ToolMeta, ToolError, ToolErrorCode
from tools.price import get_food_prices_result


# ── 测试用 ToolResult 工厂 ─────────────────────────────────────────────────

def source_ok(platform: str, pr_dict: dict) -> ToolResult:
    return ToolResult(
        ok=True,
        data=pr_dict,
        meta=ToolMeta(tool_name=platform, elapsed_ms=200, attempts=1),
    )


def source_fail(platform: str) -> ToolResult:
    fallback = {
        "platform": platform,
        "price": None,
        "unit": "",
        "available": False,
        "url": None,
        "product_name": None,
    }
    return ToolResult(
        ok=False,
        data=fallback,
        error=ToolError(
            code=ToolErrorCode.NETWORK,
            message="connection refused",
            retryable=True,
        ),
        meta=ToolMeta(tool_name=platform, elapsed_ms=8000, attempts=3, fallback_used=True),
    )


def _avail(platform: str, price: float, unit: str = "kg") -> dict:
    return {
        "platform": platform,
        "price": price,
        "unit": unit,
        "available": True,
        "url": None,
        "product_name": f"{platform}_product",
    }


def _unavail(platform: str) -> dict:
    return {
        "platform": platform,
        "price": None,
        "unit": "",
        "available": False,
        "url": None,
        "product_name": None,
    }


# ── 两个源都成功 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_both_food_sources_succeed_ok_true():
    """两个源都成功 → ok=True，all_results 包含两个平台条目"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("woolworths", _avail("woolworths", 5.0)),
            source_ok("umall", _avail("umall", 4.5)),
        ])
        result = await get_food_prices_result("tomato")

    assert result.ok is True
    all_results = result.data["all_results"]
    assert len(all_results) == 2
    platforms = {r["platform"] for r in all_results}
    assert "woolworths" in platforms
    assert "umall" in platforms


@pytest.mark.asyncio
async def test_both_sources_succeed_all_ok_in_status():
    """两个源都成功 → sources 列表中两项均 ok=True，fallback_used=False"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("woolworths", _avail("woolworths", 5.0)),
            source_ok("umall", _avail("umall", 4.5)),
        ])
        result = await get_food_prices_result("tomato")

    assert result.meta.fallback_used is False
    assert all(s["ok"] for s in result.data["sources"])


# ── 一个源失败 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_source_fails_ok_true_partial():
    """一个源失败 → ok=True（仍有数据），fallback_used=True，失败源 ok=False"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("woolworths", _avail("woolworths", 5.0)),
            source_fail("umall"),
        ])
        result = await get_food_prices_result("tomato")

    assert result.ok is True
    assert result.meta.fallback_used is True
    sources = result.data["sources"]
    ww = next(s for s in sources if s["source"] == "woolworths")
    um = next(s for s in sources if s["source"] == "umall")
    assert ww["ok"] is True
    assert um["ok"] is False
    assert um["error_code"] == "NETWORK"


@pytest.mark.asyncio
async def test_one_source_fails_entry_still_exists():
    """一个源失败 → all_results 中仍有失败源的占位条目（available=False）"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("woolworths", _avail("woolworths", 5.0)),
            source_fail("umall"),
        ])
        result = await get_food_prices_result("tomato")

    all_results = result.data["all_results"]
    assert len(all_results) == 2
    umall_entry = next((r for r in all_results if r["platform"] == "umall"), None)
    assert umall_entry is not None
    assert umall_entry["available"] is False


# ── 所有源失败 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_sources_fail_ok_false():
    """所有源失败 → ok=False，error.code=DEPENDENCY_UNAVAILABLE，retryable=True"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_fail("woolworths"),
            source_fail("umall"),
        ])
        result = await get_food_prices_result("tomato")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_all_sources_fail_sources_status():
    """所有源失败 → sources 列表中所有项 ok=False"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_fail("woolworths"),
            source_fail("umall"),
        ])
        result = await get_food_prices_result("tomato")

    assert not any(s["ok"] for s in result.data["sources"])


# ── 源成功但无商品 ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_succeed_but_no_items():
    """源调用成功但无商品（available=False）→ ok=True（区分"成功无数据"与"执行失败"）"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("woolworths", _unavail("woolworths")),
            source_ok("umall", _unavail("umall")),
        ])
        result = await get_food_prices_result("xyz_product")

    assert result.ok is True
    assert all(not r["available"] for r in result.data["all_results"])
    assert all(s["ok"] for s in result.data["sources"])
    assert all(not s["found"] for s in result.data["sources"])


# ── meta 汇总 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_attempts_and_elapsed():
    """meta.attempts 为各源之和，elapsed_ms 为各源最大值"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            ToolResult(
                ok=True,
                data=_avail("woolworths", 5.0),
                meta=ToolMeta(tool_name="woolworths", elapsed_ms=300, attempts=2),
            ),
            ToolResult(
                ok=True,
                data=_avail("umall", 4.5),
                meta=ToolMeta(tool_name="umall", elapsed_ms=500, attempts=1),
            ),
        ])
        result = await get_food_prices_result("tomato")

    assert result.meta.attempts == 3       # 2 + 1
    assert result.meta.elapsed_ms == 500   # max(300, 500)
