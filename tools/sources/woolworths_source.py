"""WoolworthsSource：从 Woolworths API 查询价格的 PriceSource 实现。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from tools.price_source import PriceResult, PriceSource
from tools.sources.translate_utils import _translate
from tools.sources.unit_utils import _extract_unit_price

_TIMEOUT = 10
_WW_SEARCH = (
    "https://www.woolworths.com.au/apis/ui/Search/products"
    "?searchTerm={q}&pageNumber=1&pageSize=5"
)
_WW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://www.woolworths.com.au/",
    "Origin": "https://www.woolworths.com.au",
    "x-requested-with": "XMLHttpRequest",
}


def _parse_woolworths(data: dict) -> list[dict]:
    try:
        items = []
        for group in data.get("Products", []):
            for p in group.get("Products", []):
                price = p.get("Price")
                if price is None:
                    continue
                items.append({
                    "name": p.get("Name", "Unknown"),
                    "price": float(price),
                })
        return items
    except Exception:
        return []


def _unavailable(platform: str, item_name: str, ts: str, url: str) -> PriceResult:
    return PriceResult(
        platform=platform,
        item_name=item_name,
        price=None,
        unit="",
        currency="AUD",
        available=False,
        timestamp=ts,
        url=url,
    )


class WoolworthsSource(PriceSource):
    def get_platform_name(self) -> str:
        return "woolworths"

    def get_price_strict(self, item_name: str) -> PriceResult:
        """查询价格；依赖故障会抛出异常，供统一 ToolExecutor 分类和重试。"""
        query = _translate(item_name)
        ts = datetime.now().isoformat()
        url = _WW_SEARCH.format(q=query)

        try:
            resp = httpx.get(url, headers=_WW_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Woolworths 请求超时: {exc}") from exc
        except httpx.RequestError as exc:
            raise ConnectionError(f"Woolworths 请求失败: {exc}") from exc

        if resp.status_code >= 500:
            raise ConnectionError(f"Woolworths 服务返回 {resp.status_code}")
        if resp.status_code != 200:
            return _unavailable(self.get_platform_name(), item_name, ts, url)

        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            return _unavailable(self.get_platform_name(), item_name, ts, url)

        try:
            raw_json = resp.json()
        except ValueError as exc:
            raise OSError(f"Woolworths 返回了无效 JSON: {exc}") from exc

        items = _parse_woolworths(raw_json)

        if not items:
            return _unavailable(self.get_platform_name(), item_name, ts, url)

        # 对每件商品做克重换算，取换算后单价最低的
        best: Optional[tuple[float, str, str]] = None
        for item in items:
            per_kg, unit = _extract_unit_price(item["name"], item["price"])
            if best is None or per_kg < best[0]:
                best = (per_kg, unit, item["name"])

        if best is None:
            return _unavailable(self.get_platform_name(), item_name, ts, url)

        per_kg, unit, best_product_name = best
        return PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=per_kg,
            unit=unit,
            currency="AUD",
            available=True,
            timestamp=ts,
            url=url,
            product_name=best_product_name,
        )

    def get_price(self, item_name: str) -> PriceResult:
        """旧接口兼容：依赖故障仍降级为 unavailable。"""
        try:
            return self.get_price_strict(item_name)
        except Exception:
            ts = datetime.now().isoformat()
            url = _WW_SEARCH.format(q=_translate(item_name))
            return _unavailable(self.get_platform_name(), item_name, ts, url)
