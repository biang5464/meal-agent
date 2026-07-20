"""菜谱预算档位计算工具：根据食材价格计算 low/mid/high 档位。"""

from __future__ import annotations

# 价格阈值（AUD/kg）
_VEGGIE_LOW = 6.0
_VEGGIE_MID = 10.0
_MEAT_LOW = 12.0
_MEAT_MID = 22.0

_MEAT_KEYWORDS = [
    "猪肉", "鸡肉", "牛肉", "羊肉", "鸡腿", "鸡胸",
    "五花肉", "猪里脊", "猪绞肉", "牛绞肉", "虾", "鱼", "鲈鱼",
    "排骨", "salmon", "pork", "chicken", "beef", "lamb", "prawn", "fish",
]

_TIER_RANK: dict[str, int] = {"low": 0, "mid": 1, "high": 2}


def _is_meat(ingredient: str) -> bool:
    return any(kw in ingredient for kw in _MEAT_KEYWORDS)


def _price_to_tier(price: float, is_meat: bool) -> str:
    if is_meat:
        if price <= _MEAT_LOW:
            return "low"
        elif price <= _MEAT_MID:
            return "mid"
        else:
            return "high"
    else:
        if price <= _VEGGIE_LOW:
            return "low"
        elif price <= _VEGGIE_MID:
            return "mid"
        else:
            return "high"


def calc_recipe_budget_tier(ingredients: list[str]) -> str:
    """
    计算一道菜的 budget_tier（一票否决制：最贵食材决定整道菜档位）。
    查不到价格的食材跳过；全部查不到时返回 "unknown"。
    """
    from tools.price import get_ingredient_price

    worst_rank = -1
    worst_tier = "low"
    any_found = False

    for ingredient in ingredients:
        try:
            result = get_ingredient_price.invoke({"ingredient": ingredient})
        except Exception:
            continue

        best_price = result.get("best_price")
        if best_price is None:
            # comparable=False：取 all_results 里 kg 单位的最低价
            kg_prices = [
                r["price"] for r in result.get("all_results", [])
                if r.get("available") and r.get("unit") == "kg" and r.get("price")
            ]
            if not kg_prices:
                continue
            best_price = min(kg_prices)

        any_found = True
        tier = _price_to_tier(float(best_price), _is_meat(ingredient))
        rank = _TIER_RANK[tier]
        if rank > worst_rank:
            worst_rank = rank
            worst_tier = tier

    return worst_tier if any_found else "unknown"
