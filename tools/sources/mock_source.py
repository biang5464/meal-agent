"""MockSource：本地静态数据，用于测试和离线开发。"""

from __future__ import annotations

from datetime import datetime

from tools.price_source import PriceResult, PriceSource

MOCK_DATA: dict[str, tuple[float, str, str]] = {
    # 食材类 AUD/kg or AUD/dozen
    "番茄": (3.50, "kg", "AUD"),
    "鸡蛋": (4.00, "dozen", "AUD"),
    "鸡腿": (10.00, "kg", "AUD"),
    "猪肉": (14.00, "kg", "AUD"),
    # 电子产品类 AUD/piece
    "华为Mate60Pro": (1299.00, "piece", "AUD"),
    "AirPods Pro": (399.00, "piece", "AUD"),
    # 生活用品类 AUD/piece
    "洗发水": (12.00, "piece", "AUD"),
    "沐浴露": (10.00, "piece", "AUD"),
}


class MockSource(PriceSource):
    def get_platform_name(self) -> str:
        return "mock"

    def get_price(self, item_name: str) -> PriceResult:
        data = MOCK_DATA.get(item_name)
        if data:
            price, unit, currency = data
            return PriceResult(
                platform=self.get_platform_name(),
                item_name=item_name,
                price=price,
                unit=unit,
                currency=currency,
                available=True,
                timestamp=datetime.now().isoformat(),
            )
        return PriceResult(
            platform=self.get_platform_name(),
            item_name=item_name,
            price=None,
            unit="",
            currency="",
            available=False,
            timestamp=datetime.now().isoformat(),
        )
