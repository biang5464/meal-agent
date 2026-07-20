# TODO: Coles API 返回 404，需要确认正确的 API endpoint
# 当前 Woolworths + Umall 正常，Coles 暂时以占位结果处理；ColesSource 待实现后接入

"""比价工具：查询 Woolworths / Umall 价格，供 Agent 调用。"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

from langchain_core.tools import tool

from tools.sources.goodguys_source import GoodGuysSource
from tools.sources.jbhifi_source import JBHiFiSource
from tools.sources.umall_source import UmallSource
from tools.sources.woolworths_source import WoolworthsSource
from tools.price_history import record_price

# ---------- 模块级平台实例 ----------

_ww_source = WoolworthsSource()
_umall_source = UmallSource()
_jbhifi_source = JBHiFiSource()
_goodguys_source = GoodGuysSource()


# ---------- 内部多平台查询辅助 ----------

async def get_food_prices_result(ingredient: str):
    """
    查询食材价格（Woolworths + Umall），通过 ToolExecutor.execute() 统一管理。
    返回 ToolResult：
    - ok=True：至少一个源调用成功（即使无商品）
    - ok=False / DEPENDENCY_UNAVAILABLE：所有源调用失败（网络/超时等依赖错误）
    - data.all_results：每个源一条记录（含 available 字段区分有无商品）
    - data.sources：每个源的调用状态
    """
    from core.tool_protocol import (
        ToolError,
        ToolErrorCode,
        ToolMeta,
        ToolResult,
    )
    from tools.tool_executor import tool_executor

    sources = [_ww_source, _umall_source]

    def _fetch(source) -> dict:
        fetch = getattr(source, "get_price_strict", source.get_price)
        pr = fetch(ingredient)
        return {
            "platform": pr.platform,
            "price": pr.price,
            "unit": pr.unit,
            "available": pr.available,
            "url": getattr(pr, "url", None),
            "product_name": getattr(pr, "product_name", None),
        }

    tasks = [
        tool_executor.execute(
            _fetch,
            source,
            policy_name="external_http",
            fallback_data={
                "platform": source.get_platform_name(),
                "price": None,
                "unit": "",
                "available": False,
                "url": None,
                "product_name": None,
            },
            tool_name=source.__class__.__name__,
            source=source.get_platform_name(),
        )
        for source in sources
    ]
    results = await asyncio.gather(*tasks)

    all_platform_results: list[dict] = []
    source_statuses: list[dict] = []
    successful_sources = 0
    total_attempts = 0
    elapsed_ms = 0

    for source, result in zip(sources, results):
        platform = source.get_platform_name()
        meta = result.meta
        total_attempts += meta.attempts if meta else 0
        elapsed_ms = max(elapsed_ms, meta.elapsed_ms if meta else 0)

        if result.ok:
            successful_sources += 1
            pr_dict = result.data or {
                "platform": platform, "price": None, "unit": "",
                "available": False, "url": None, "product_name": None,
            }
            all_platform_results.append(pr_dict)
            source_statuses.append({
                "source": platform,
                "ok": True,
                "found": pr_dict.get("available", False),
                "error_code": None,
            })
        else:
            fallback = result.data or {
                "platform": platform, "price": None, "unit": "",
                "available": False, "url": None, "product_name": None,
            }
            all_platform_results.append(fallback)
            error_code = result.error.code.value if result.error else "INTERNAL"
            source_statuses.append({
                "source": platform,
                "ok": False,
                "found": False,
                "error_code": error_code,
            })
            logger.warning("[price] %s 食材价格查询失败 code=%s", platform, error_code)

    data = {"all_results": all_platform_results, "sources": source_statuses}

    if successful_sources:
        return ToolResult(
            ok=True,
            data=data,
            meta=ToolMeta(
                tool_name="get_food_prices",
                elapsed_ms=elapsed_ms,
                attempts=total_attempts,
                fallback_used=successful_sources < len(sources),
                source="food",
            ),
        )

    return ToolResult(
        ok=False,
        data=data,
        error=ToolError(
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            message="所有食材价格源均不可用",
            retryable=True,
            details={"sources": source_statuses},
        ),
        meta=ToolMeta(
            tool_name="get_food_prices",
            elapsed_ms=elapsed_ms,
            attempts=total_attempts,
            fallback_used=True,
            source="food",
        ),
    )


def _get_price_multi(ingredient: str) -> dict:
    """
    同时查询 Woolworths 和 Umall（经 ToolExecutor 统一管理），
    以及 JBHiFi、The Good Guys（直接调用），Coles 保留 TODO 占位。
    返回统一多平台结果字典。
    """
    # 食材平台通过服务层（ToolExecutor.execute）查询
    food_tool_result = asyncio.run(get_food_prices_result(ingredient))
    food_data = food_tool_result.data or {}
    food_all_results: list[dict] = list(food_data.get("all_results", []))

    # 确保 woolworths / umall 占位条目始终存在（全部失败且无 fallback 时兜底）
    _present_platforms = {r["platform"] for r in food_all_results}
    for _platform in ("woolworths", "umall"):
        if _platform not in _present_platforms:
            food_all_results.append({
                "platform": _platform, "price": None, "unit": "",
                "available": False, "url": None, "product_name": None,
            })

    # 电子产品平台（保持直接调用）
    jbhifi_result = _jbhifi_source.get_price(ingredient)
    goodguys_result = _goodguys_source.get_price(ingredient)

    # TODO: ColesSource 实现后替换此占位
    coles_entry: dict = {
        "platform": "coles",
        "price": None,
        "unit": "",
        "available": False,
        "url": None,
        "product_name": None,
    }

    all_results = food_all_results + [
        {
            "platform": "jbhifi",
            "price": jbhifi_result.price,
            "unit": jbhifi_result.unit,
            "available": jbhifi_result.available,
            "url": jbhifi_result.url,
            "product_name": jbhifi_result.product_name,
        },
        {
            "platform": "goodguys",
            "price": goodguys_result.price,
            "unit": goodguys_result.unit,
            "available": goodguys_result.available,
            "url": goodguys_result.url,
            "product_name": goodguys_result.product_name,
        },
        coles_entry,
    ]

    available_results = [r for r in all_results if r["available"] and r["price"] is not None]
    units = set(r["unit"] for r in available_results)
    comparable = len(units) <= 1  # 0 个或 1 种单位都算可比

    if comparable and available_results:
        best = min(available_results, key=lambda r: r["price"])
        best_price: Optional[float] = best["price"]
        best_platform: Optional[str] = best["platform"]
        best_unit: str = best["unit"]
        note: Optional[str] = None
    else:
        best_price = None
        best_platform = None
        best_unit = ""
        note = "各平台单位不同，无法直接比价，以下为各平台价格供参考" if not comparable else None

    for r in all_results:
        if r["available"] and r["price"] is not None:
            record_price(
                query_term=ingredient,
                platform=r["platform"],
                price=r["price"],
                unit=r["unit"],
                product_name=r.get("product_name"),
                url=r.get("url"),
            )

    result: dict = {
        "item": ingredient,
        "comparable": comparable,
        "best_price": best_price,
        "best_platform": best_platform,
        "unit": best_unit,
        "all_results": all_results,
    }
    if note is not None:
        result["note"] = note
    return result


# ---------- LangChain Tools ----------

@tool
def get_ingredient_price(ingredient: str) -> dict:
    """
    同时查询 Coles 和 Woolworths 的食材价格，返回两家对比结果。

    Args:
        ingredient: 食材名称（英文效果最佳），如 "chicken breast"

    Returns:
        包含两家最低价、差价及推荐购买方的字典。
    """
    return _get_price_multi(ingredient)


@tool
def compare_recipe_cost(ingredients: List[str]) -> dict:
    """
    批量查询菜谱所需食材的最低价，估算总花费。

    Args:
        ingredients: 食材名称列表，如 ["chicken breast", "broccoli", "garlic"]

    Returns:
        包含每项食材最低价来源和总估算费用的字典。
    """
    breakdown = []
    total = 0.0
    unavailable = []

    for ingredient in ingredients:
        result = _get_price_multi(ingredient)

        if result["best_price"] is None:
            unavailable.append(ingredient)
            breakdown.append({
                "ingredient": ingredient,
                "price": None,
                "source": None,
                "note": "未找到价格",
            })
            continue

        total += result["best_price"]
        breakdown.append({
            "ingredient": ingredient,
            "price": result["best_price"],
            "unit": result["unit"],
            "source": result["best_platform"],
            "currency": "AUD",
        })

    return {
        "breakdown": breakdown,
        "total_estimate": round(total, 2),
        "currency": "AUD",
        "unavailable": unavailable,
        "note": (
            "总价为各食材最低单价之和，未含具体克重换算。"
            if breakdown
            else "所有食材均未查询到价格。"
        ),
    }


async def get_electronics_prices_result(query: str):
    """
    只查询电子产品平台（JB Hi-Fi、The Good Guys），
    跳过 Woolworths、Umall 等食材平台。
    每个平台独立执行并返回统一 ToolResult：
    - 至少一个平台调用成功：ok=True（即使成功但没有商品）
    - 所有平台调用失败：ok=False / DEPENDENCY_UNAVAILABLE
    - data.items 为排序后的商品，data.sources 保留每个平台状态
    """
    from core.tool_protocol import (
        ToolError,
        ToolErrorCode,
        ToolMeta,
        ToolResult,
    )
    from tools.tool_executor import tool_executor

    sources = [_jbhifi_source, _goodguys_source]

    def _fetch(source) -> list[dict]:
        # strict 接口会把依赖故障交给 ToolExecutor；旧 get_price 仍保持
        # “异常时返回 unavailable”行为，供其他旧调用兼容使用。
        fetch = getattr(source, "get_price_strict", source.get_price)
        result = fetch(query)
        if result.available and result.price is not None:
            return [{
                "platform": result.platform,
                "name": result.product_name or query,
                "price": result.price,
                "url": result.url,
            }]
        return []

    tasks = [
        tool_executor.execute(
            _fetch,
            source,
            policy_name="external_http",
            fallback_data=[],
            tool_name=source.__class__.__name__,
            source=source.get_platform_name(),
        )
        for source in sources
    ]
    results = await asyncio.gather(*tasks)

    all_results: list[dict] = []
    source_statuses: list[dict] = []
    successful_sources = 0
    total_attempts = 0
    elapsed_ms = 0

    for source, result in zip(sources, results):
        platform = source.get_platform_name()
        meta = result.meta
        total_attempts += meta.attempts if meta else 0
        elapsed_ms = max(elapsed_ms, meta.elapsed_ms if meta else 0)

        if result.ok:
            successful_sources += 1
            items = result.data or []
            all_results.extend(items)
            source_statuses.append({
                "source": platform,
                "ok": True,
                "found": bool(items),
                "error_code": None,
            })
        else:
            error_code = result.error.code.value if result.error else "INTERNAL"
            source_statuses.append({
                "source": platform,
                "ok": False,
                "found": False,
                "error_code": error_code,
            })
            logger.warning("[price] %s 查询失败 code=%s", platform, error_code)

    all_results.sort(key=lambda x: x.get("price", float("inf")))
    data = {"items": all_results, "sources": source_statuses}

    if successful_sources:
        return ToolResult(
            ok=True,
            data=data,
            meta=ToolMeta(
                tool_name="get_electronics_prices",
                elapsed_ms=elapsed_ms,
                attempts=total_attempts,
                fallback_used=successful_sources < len(sources),
                source="electronics",
            ),
        )

    return ToolResult(
        ok=False,
        data=data,
        error=ToolError(
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            message="所有电子产品价格源均不可用",
            retryable=True,
            details={"sources": source_statuses},
        ),
        meta=ToolMeta(
            tool_name="get_electronics_prices",
            elapsed_ms=elapsed_ms,
            attempts=total_attempts,
            fallback_used=True,
            source="electronics",
        ),
    )


async def get_electronics_prices(query: str) -> list[dict]:
    """旧接口兼容层：继续只返回商品列表。"""
    result = await get_electronics_prices_result(query)
    return (result.data or {}).get("items", [])


# 导出给 router.py 使用
PRICE_TOOLS = [get_ingredient_price, compare_recipe_cost]
