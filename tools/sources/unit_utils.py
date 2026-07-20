"""unit_utils：跨平台共用的克重换算工具。"""

from __future__ import annotations

import re

# 生鲜食材关键词白名单（title 包含任意一个才换算 /kg）
FRESH_KEYWORDS = [
    "tomato", "potato", "onion", "garlic", "ginger", "capsicum",
    "carrot", "cucumber", "spinach", "cabbage", "broccoli",
    "green bean", "eggplant", "pork", "chicken", "beef", "lamb",
    "prawn", "fish", "salmon", "barramundi", "mince", "belly",
    "rib", "breast", "leg", "fillet", "egg", "tofu",
]


def _is_fresh(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in FRESH_KEYWORDS)


def _extract_unit_price(title: str, price: float) -> tuple[float, str]:
    """
    从 title 提取克重，换算 /kg 单价。返回 (换算后价格, 单位)。

    判断顺序：
      1. 匹配不到克重          → 原价, "pack"
      2. 匹配到克重但不在白名单 → 原价, "pack"
      3. 匹配到克重，换算>200  → 原价, "pack"
      4. 其余                  → 换算后单价, "kg"
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", title, re.IGNORECASE)
    if not match:
        return round(price, 2), "pack"

    if not _is_fresh(title):
        return round(price, 2), "pack"

    amount = float(match.group(1))
    unit_str = match.group(2).lower()
    grams = amount * 1000 if unit_str == "kg" else amount
    if grams <= 0:
        return round(price, 2), "pack"

    per_kg = price * (1000 / grams)
    if per_kg > 200:
        return round(price, 2), "pack"

    return round(per_kg, 2), "kg"
