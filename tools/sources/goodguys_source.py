"""GoodGuysSource：从 The Good Guys Shopify Storefront GraphQL API 查询电子产品价格。"""

from __future__ import annotations

import os
from datetime import datetime

import httpx

from tools.price_source import PriceResult, PriceSource
from tools.sources.translate_utils import _translate

_STOREFRONT_URL = "https://tgg-prod.myshopify.com/api/2024-10/graphql.json"
_DEFAULT_TOKEN = "9a0c6d57d352710108867dede92258a0"
_BASE_URL = "https://www.thegoodguys.com.au"
_TIMEOUT = 10

# title 关键词黑名单：标题含这些词的结果视为配件，排除
# productType 字段在 TGG 恒为空，只能靠 title 过滤
_TITLE_BLACKLIST: list[str] = [
    "case", "cover", "screen protector", "lens protector", "lens", "protector",
    "charger", "cable", "adapter", "skin", "mount", "holder", "stand",
    "bag", "sleeve", "film", "tempered glass", "guard", "pouch", "bumper",
    "wrap", "keyboard", "magic keyboard", "folio",
    "armour", "armor", "zagg", "bellroy", "hardshell",
    "ear tip", "eartip",
    "贴膜", "保护膜", "手机壳", "充电器", "数据线",
]


def _is_blacklisted(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _TITLE_BLACKLIST)


def _build_query(q: str) -> str:
    escaped = q.replace('"', '\\"')
    return (
        f'{{ predictiveSearch(query: "{escaped}", limit: 10, types: PRODUCT) {{'
        f"  products {{"
        f"    title availableForSale handle"
        f"    priceRange {{ minVariantPrice {{ amount currencyCode }} }}"
        f"    variants(first: 1) {{ nodes {{ price {{ amount currencyCode }} availableForSale }} }}"
        f"  }}"
        f"}}"
        f"}}"
    )


def _variant_price(product: dict) -> float:
    """取 product 的最低可用价格（优先 variant，fallback priceRange）。"""
    nodes = product.get("variants", {}).get("nodes", [])
    if nodes:
        return float(nodes[0]["price"]["amount"])
    return float(product["priceRange"]["minVariantPrice"]["amount"])


class GoodGuysSource(PriceSource):
    def get_platform_name(self) -> str:
        return "goodguys"

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
        token = os.getenv("GOODGUYS_TOKEN", _DEFAULT_TOKEN)

        try:
            resp = httpx.post(
                _STOREFRONT_URL,
                json={"query": _build_query(query)},
                headers={
                    "X-Shopify-Storefront-Access-Token": token,
                    "Content-Type": "application/json",
                },
                timeout=_TIMEOUT,
            )
        except httpx.RequestError as exc:
            raise ConnectionError(f"The Good Guys 请求失败: {exc}") from exc

        if resp.status_code >= 500:
            raise ConnectionError(f"The Good Guys 服务返回 {resp.status_code}")
        if resp.status_code != 200:
            return _unavailable

        try:
            data = resp.json()
            products = (
                data.get("data", {})
                    .get("predictiveSearch", {})
                    .get("products", [])
            )
        except (ValueError, AttributeError) as exc:
            raise OSError("The Good Guys 返回了无法解析的数据") from exc

        candidates = [
            p for p in products
            if p.get("availableForSale", False)
            and not _is_blacklisted(p.get("title", ""))
        ]
        if not candidates:
            return _unavailable

        best = min(candidates, key=_variant_price)

        return PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=round(_variant_price(best), 2),
            unit="each",
            currency="AUD",
            available=True,
            timestamp=ts,
            url=f"{_BASE_URL}/products/{best.get('handle', '')}",
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
