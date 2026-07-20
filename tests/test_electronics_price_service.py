"""
get_electronics_prices_result() 服务层单元测试（第三阶段新增）

用 patch 替换 tool_executor.execute，
专注测试聚合逻辑与 ToolResult 语义，不产生真实 HTTP 流量。
"""
import pytest
from unittest.mock import AsyncMock, patch

from core.tool_protocol import ToolResult, ToolMeta, ToolError, ToolErrorCode
from tools.price import get_electronics_prices, get_electronics_prices_result


# ── 测试用 ToolResult 工厂 ─────────────────────────────────────────────────

def source_ok(platform: str, items: list[dict]) -> ToolResult:
    return ToolResult(
        ok=True,
        data=items,
        meta=ToolMeta(tool_name=platform, elapsed_ms=200, attempts=1),
    )


def source_fail(platform: str) -> ToolResult:
    return ToolResult(
        ok=False,
        data=[],
        error=ToolError(
            code=ToolErrorCode.NETWORK,
            message="connection refused",
            retryable=True,
        ),
        meta=ToolMeta(
            tool_name=platform,
            elapsed_ms=8000,
            attempts=3,
            fallback_used=True,
        ),
    )


# ── 两个源都成功 ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_both_sources_succeed_ok_true():
    """两个源都成功 → ok=True，items 按价格升序"""
    jb_items  = [{"platform": "jbhifi",   "name": "iPhone 16", "price": 1299.0, "url": ""}]
    gg_items  = [{"platform": "goodguys", "name": "iPhone 16", "price": 1279.0, "url": ""}]

    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi", jb_items),
            source_ok("goodguys", gg_items),
        ])
        result = await get_electronics_prices_result("iPhone 16")

    assert result.ok is True
    items = result.data["items"]
    assert len(items) == 2
    assert items[0]["price"] <= items[1]["price"]   # 按价格升序


@pytest.mark.asyncio
async def test_both_sources_succeed_all_sources_ok_in_status():
    """两个源都成功 → sources 列表中两项均 ok=True"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi",   [{"platform": "jbhifi",   "name": "X", "price": 10.0, "url": ""}]),
            source_ok("goodguys", [{"platform": "goodguys", "name": "X", "price": 9.0,  "url": ""}]),
        ])
        result = await get_electronics_prices_result("X")

    assert result.meta.fallback_used is False
    assert all(s["ok"] for s in result.data["sources"])


# ── 一个源失败 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_source_fails_ok_true_partial():
    """一个源失败 → ok=True（仍有数据），fallback_used=True"""
    jb_items = [{"platform": "jbhifi", "name": "iPhone 16", "price": 1299.0, "url": ""}]

    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi", jb_items),
            source_fail("goodguys"),
        ])
        result = await get_electronics_prices_result("iPhone 16")

    assert result.ok is True
    assert result.meta.fallback_used is True
    sources = result.data["sources"]
    jb = next(s for s in sources if s["source"] == "jbhifi")
    gg = next(s for s in sources if s["source"] == "goodguys")
    assert jb["ok"] is True
    assert gg["ok"] is False
    assert gg["error_code"] == "NETWORK"


@pytest.mark.asyncio
async def test_one_source_fails_items_from_remaining():
    """一个源失败 → items 仍包含另一个源的商品"""
    jb_items = [{"platform": "jbhifi", "name": "MacBook", "price": 1899.0, "url": ""}]

    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi", jb_items),
            source_fail("goodguys"),
        ])
        result = await get_electronics_prices_result("MacBook")

    items = result.data["items"]
    assert len(items) == 1
    assert items[0]["platform"] == "jbhifi"


# ── 所有源失败 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_sources_fail_ok_false():
    """所有源失败 → ok=False，error.code=DEPENDENCY_UNAVAILABLE"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_fail("jbhifi"),
            source_fail("goodguys"),
        ])
        result = await get_electronics_prices_result("iPhone 16")

    assert result.ok is False
    assert result.error.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_all_sources_fail_sources_status():
    """所有源失败 → sources 列表中所有项 ok=False"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_fail("jbhifi"),
            source_fail("goodguys"),
        ])
        result = await get_electronics_prices_result("X")

    assert not any(s["ok"] for s in result.data["sources"])


# ── 源成功但无商品 ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_succeed_but_no_items():
    """源调用成功但无商品 → ok=True，items=[]（区分"成功无数据"与"执行失败"）"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi",   []),
            source_ok("goodguys", []),
        ])
        result = await get_electronics_prices_result("产品xyz")

    assert result.ok is True
    assert result.data["items"] == []
    assert all(s["ok"] for s in result.data["sources"])


# ── 旧接口兼容 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_electronics_prices_compat_returns_list():
    """旧接口 get_electronics_prices() 仍返回 list[dict]"""
    items = [{"platform": "jbhifi", "name": "iPhone 16", "price": 1299.0, "url": ""}]

    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_ok("jbhifi",   items),
            source_ok("goodguys", []),
        ])
        result = await get_electronics_prices("iPhone 16")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["platform"] == "jbhifi"


@pytest.mark.asyncio
async def test_get_electronics_prices_compat_all_fail_empty_list():
    """旧接口：所有源失败 → 返回空列表（不抛异常）"""
    with patch("tools.tool_executor.tool_executor") as mock_exec:
        mock_exec.execute = AsyncMock(side_effect=[
            source_fail("jbhifi"),
            source_fail("goodguys"),
        ])
        result = await get_electronics_prices("X")

    assert isinstance(result, list)
    assert result == []
