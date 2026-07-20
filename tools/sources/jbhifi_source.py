"""JBHiFiSource：从 JB Hi-Fi Shopify Predictive Search API 查询电子产品价格。"""

from __future__ import annotations

from datetime import datetime

import httpx

from tools.price_source import PriceResult, PriceSource
from tools.sources.translate_utils import _translate

_SEARCH_URL = (
    "https://www.jbhifi.com.au/search/suggest.json"
    "?q={q}&resources[type]=product&resources[limit]=10"
)
_BASE_URL = "https://www.jbhifi.com.au"
_TIMEOUT = 10

# 查询词（translate 之后的英文）→ 允许的 type 集合
# 不在映射表里的查询词不做过滤（兜底，返回所有 available 结果）
_ACCESSORY_BLACKLIST: list[str] = [
    "protector", "screen protector", "lens protector", "lens", "case", "cover",
    "charger", "cable", "adapter", "stand", "holder", "skin", "film",
    "tempered glass", "guard", "sleeve", "pouch", "bumper", "wrap",
    "armour", "armor", "bellroy", "zagg", "hardshell",
    "magic keyboard", "folio", "smart folio", "slim folio",
    "ear tip", "eartip",
    "贴膜", "保护膜", "手机壳", "充电器", "数据线",
]


def _is_accessory(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in _ACCESSORY_BLACKLIST)


_QUERY_TYPE_WHITELIST: dict[str, set[str]] = {
    # 手机
    "phone":            {"COMMUNICATIONS"},
    "iphone":           {"COMMUNICATIONS"},
    "samsung phone":    {"COMMUNICATIONS"},
    "huawei phone":     {"COMMUNICATIONS"},
    # 电脑
    "laptop":           {"COMPUTERS"},
    "macbook":          {"COMPUTERS"},
    "desktop computer": {"COMPUTERS"},
    # 平板
    "tablet":           {"COMPUTERS"},
    "ipad":             {"COMPUTERS"},
    # 耳机
    "headphones":           {"AUDIO"},
    "wireless headphones":  {"AUDIO"},
    "airpods":              {"AUDIO"},
    # 电视
    "tv":               {"VISUAL"},
    # 充电器/配件（本身就是配件，不过滤）
    "charger":  {"ACCESSORIES", "IT"},
    "cable":    {"ACCESSORIES", "IT"},
    "keyboard": {"IT", "ACCESSORIES"},
    "mouse":    {"IT", "ACCESSORIES"},
}


class JBHiFiSource(PriceSource):
    def get_platform_name(self) -> str:
        return "jbhifi"

    def get_price_strict(self, item_name: str) -> PriceResult:
        """查询价格；依赖故障会抛出异常，供统一 ToolExecutor 分类和重试。"""
        ts = datetime.now().isoformat()
        _unavailable = PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=None,
            unit="",
            currency="AUD",
            available=False,
            timestamp=ts,
        )

        query = _translate(item_name)

        url = _SEARCH_URL.format(q=query)
        try:
            resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        except httpx.RequestError as exc:
            raise ConnectionError(f"JB Hi-Fi 请求失败: {exc}") from exc

        if resp.status_code >= 500:
            raise ConnectionError(f"JB Hi-Fi 服务返回 {resp.status_code}")
        if resp.status_code != 200:
            return _unavailable

        try:
            data = resp.json()
            products = (
                data.get("resources", {})
                    .get("results", {})
                    .get("products", [])
            )
        except (ValueError, AttributeError) as exc:
            raise OSError("JB Hi-Fi 返回了无法解析的数据") from exc

        allowed_types = _QUERY_TYPE_WHITELIST.get(query.lower())

        # 仅当用户不是主动查询配件时才过滤配件标题
        # allowed_types 含 ACCESSORIES 说明用户本身就在查配件（如 charger/cable）
        query_is_accessory = allowed_types is not None and "ACCESSORIES" in allowed_types
        available_products = [
            p for p in products
            if p.get("available", False)
            and p.get("price")
            and (
                allowed_types is None
                or p.get("type", "").upper() in allowed_types
            )
            and (query_is_accessory or not _is_accessory(p.get("title", "")))
        ]
        if not available_products:
            return _unavailable

        best = min(available_products, key=lambda p: float(p["price"]))

        return PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=round(float(best["price"]), 2),
            unit="each",
            currency="AUD",
            available=True,
            timestamp=ts,
            url=_BASE_URL + best.get("url", ""),
            product_name=best.get("title", ""),
        )

    def get_price(self, item_name: str) -> PriceResult:
        """旧接口兼容：依赖故障仍降级为 unavailable。"""
        try:
            return self.get_price_strict(item_name)
        except Exception:
            return PriceResult(
                platform=self.get_platform_name(),
                item_name=item_name,
                price=None,
                unit="",
                currency="AUD",
                available=False,
                timestamp=datetime.now().isoformat(),
            )
