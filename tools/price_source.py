"""PriceSource 抽象层：定义价格查询的统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class PriceResult:
    def __init__(
        self,
        platform: str,
        item_name: str,
        price: Optional[float],
        unit: str,
        currency: str,
        available: bool,
        timestamp: str,
        url: Optional[str] = None,
        product_name: Optional[str] = None,
    ):
        self.platform = platform
        self.item_name = item_name
        self.price = price
        self.unit = unit
        self.currency = currency
        self.available = available
        self.timestamp = timestamp
        self.url = url
        self.product_name = product_name

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "item_name": self.item_name,
            "price": self.price,
            "unit": self.unit,
            "currency": self.currency,
            "available": self.available,
            "timestamp": self.timestamp,
            "url": self.url,
            "product_name": self.product_name,
        }


class PriceSource(ABC):
    @abstractmethod
    def get_platform_name(self) -> str:
        pass

    @abstractmethod
    def get_price(self, item_name: str) -> PriceResult:
        pass

    def get_prices(self, item_names: list[str]) -> list[PriceResult]:
        return [self.get_price(name) for name in item_names]
