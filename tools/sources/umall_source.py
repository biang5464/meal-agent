"""UmallSource：从 umall.com.au Predictive Search API 查询价格。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from tools.price_source import PriceResult, PriceSource
from tools.sources.translate_utils import _translate  # noqa: F401
from tools.sources.unit_utils import FRESH_KEYWORDS, _extract_unit_price, _is_fresh  # noqa: F401

_SEARCH_URL = (
    "https://www.umall.com.au/search/suggest.json"
    "?q={q}&resources[type]=product&resources[limit]=10"
)
_BASE_URL = "https://www.umall.com.au"
_TIMEOUT = 10


# ---------- PriceSource 实现 ----------

class UmallSource(PriceSource):
    def get_platform_name(self) -> str:
        return "umall"

    def get_price_strict(self, item_name: str) -> PriceResult:
        """查询价格；依赖故障会抛出异常，供统一 ToolExecutor 分类和重试。"""
        ts = datetime.now().isoformat()
        _unavail = PriceResult(
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
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"Umall 请求超时: {exc}") from exc
        except httpx.RequestError as exc:
            raise ConnectionError(f"Umall 请求失败: {exc}") from exc

        if resp.status_code >= 500:
            raise ConnectionError(f"Umall 服务返回 {resp.status_code}")
        if resp.status_code != 200:
            return _unavail

        try:
            data = resp.json()
        except ValueError as exc:
            raise OSError(f"Umall 返回了无效 JSON: {exc}") from exc

        try:
            products = (
                data.get("resources", {})
                    .get("results", {})
                    .get("products", [])
            )
        except AttributeError:
            return _unavail

        available_products = [p for p in products if p.get("available", False)]
        if not available_products:
            return _unavail

        best: Optional[tuple[float, str, str, dict]] = None
        for product in available_products:
            try:
                raw_price = float(product["price"])
            except (KeyError, ValueError, TypeError):
                continue

            title = product.get("title", "")
            per_kg, unit = _extract_unit_price(title, raw_price)
            product_url = product.get("url", "")

            if best is None or per_kg < best[0]:
                best = (per_kg, unit, product_url, product)

        if best is None:
            return _unavail

        per_kg, unit, product_url, best_product = best
        return PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=per_kg,
            unit=unit,
            currency="AUD",
            available=True,
            timestamp=ts,
            url=_BASE_URL + product_url if product_url else None,
            product_name=best_product.get("title", ""),
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
