import pytest
from unittest.mock import patch, AsyncMock
from agents.electronics_price_agent import run_electronics_price_agent
from core.tool_protocol import ToolResult, ToolMeta, ToolError, ToolErrorCode


# ── 测试数据工厂 ─────────────────────────────────────────────────────────────

def make_state(user_input="iPhone 16"):
    return {
        "user_input": user_input,
        "intent": "ELECTRONICS_PRICE",
        "confidence": "HIGH",
        "missing_slots": [],
        "top2": None,
        "result": "",
        "user_id": "test_user",
        "user_brief": {},
        "conversation_memory": "",
    }


def make_success_result(items, failed_sources=None):
    """构造 ok=True 的 ToolResult，可指定失败的 source 名称列表。"""
    failed_sources = failed_sources or []
    all_names = ["jbhifi", "goodguys"]
    sources = [
        {"source": n, "ok": n not in failed_sources,
         "found": n not in failed_sources and any(i["platform"] == n for i in items),
         "error_code": "NETWORK" if n in failed_sources else None}
        for n in all_names
    ]
    return ToolResult(
        ok=True,
        data={"items": items, "sources": sources},
        meta=ToolMeta(
            tool_name="get_electronics_prices",
            elapsed_ms=300,
            attempts=len(all_names),
            fallback_used=bool(failed_sources),
        ),
    )


def make_failure_result():
    """构造所有源失败的 ToolResult（ok=False）。"""
    return ToolResult(
        ok=False,
        data={"items": [], "sources": [
            {"source": "jbhifi", "ok": False, "found": False, "error_code": "NETWORK"},
            {"source": "goodguys", "ok": False, "found": False, "error_code": "NETWORK"},
        ]},
        error=ToolError(
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            message="所有电子产品价格源均不可用",
            retryable=True,
        ),
        meta=ToolMeta(tool_name="get_electronics_prices", elapsed_ms=8000, attempts=6),
    )


# ── 原有测试（迁移到 get_electronics_prices_result 接口）────────────────────

@pytest.mark.asyncio
async def test_electronics_price_returns_result():
    """正常查询返回格式化结果"""
    mock_items = [
        {"platform": "jbhifi",    "name": "iPhone 16 128GB", "price": 1299.0, "url": ""},
        {"platform": "goodguys",  "name": "iPhone 16 128GB", "price": 1279.0, "url": ""},
    ]
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(return_value=make_success_result(mock_items))):
        state = make_state("iPhone 16")
        result = await run_electronics_price_agent(state)
    assert "JB Hi-Fi" in result["result"]
    assert "The Good Guys" in result["result"]
    assert "1279" in result["result"] or "1299" in result["result"]


@pytest.mark.asyncio
async def test_electronics_price_empty_result():
    """无商品时返回友好提示"""
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(return_value=make_success_result([]))):
        state = make_state("不存在的产品xyz")
        result = await run_electronics_price_agent(state)
    assert "没有找到" in result["result"] or "暂时" in result["result"]


@pytest.mark.asyncio
async def test_electronics_price_exception_handled():
    """get_electronics_prices_result 抛出异常时不崩溃"""
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(side_effect=Exception("network error"))):
        state = make_state("iPhone 16")
        result = await run_electronics_price_agent(state)
    assert "错误" in result["result"] or "稍后" in result["result"]


# ── 新增测试：ToolResult 新行为 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_electronics_price_all_sources_down():
    """所有源失败（ok=False）→ 返回错误提示"""
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(return_value=make_failure_result())):
        state = make_state("iPhone 16")
        result = await run_electronics_price_agent(state)
    assert "错误" in result["result"] or "稍后" in result["result"]


@pytest.mark.asyncio
async def test_electronics_price_partial_failure_shows_warning():
    """一个源失败、一个成功 → ok=True，显示商品 + 不可用平台警告"""
    mock_items = [
        {"platform": "jbhifi", "name": "iPhone 16", "price": 1299.0, "url": ""},
    ]
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(return_value=make_success_result(mock_items, failed_sources=["goodguys"]))):
        state = make_state("iPhone 16")
        result = await run_electronics_price_agent(state)
    text = result["result"]
    assert "JB Hi-Fi" in text          # 可用平台的商品
    assert "The Good Guys" in text     # 不可用平台出现在警告中
    assert "⚠️" in text or "无法获取" in text


@pytest.mark.asyncio
async def test_electronics_price_result_contains_price():
    """返回内容包含价格（AUD 标记）"""
    mock_items = [
        {"platform": "jbhifi", "name": "MacBook Air M3", "price": 1899.0, "url": ""},
    ]
    with patch("agents.electronics_price_agent.get_electronics_prices_result",
               new=AsyncMock(return_value=make_success_result(mock_items))):
        state = make_state("MacBook Air M3")
        result = await run_electronics_price_agent(state)
    assert "1899" in result["result"]
    assert "AUD" in result["result"]


# ── 配置测试（不依赖接口，不需改动）────────────────────────────────────────

def test_electronics_price_in_valid_intents():
    """ELECTRONICS_PRICE 在白名单内"""
    from agents.supervisor_agent import VALID_INTENTS
    assert "ELECTRONICS_PRICE" in VALID_INTENTS


def test_electronics_price_in_slots():
    """ELECTRONICS_PRICE 槽位配置存在"""
    from agents.slots import REQUIRED_SLOTS, INTENT_DISPLAY
    assert "ELECTRONICS_PRICE" in REQUIRED_SLOTS
    assert "ELECTRONICS_PRICE" in INTENT_DISPLAY
