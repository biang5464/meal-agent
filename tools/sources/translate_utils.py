"""translate_utils：跨平台共用的中英商品名称映射。"""

from __future__ import annotations

# 食材类（原有，保持不变）
_FOOD_ZH_TO_EN: dict[str, str] = {
    "番茄": "tomato", "西红柿": "tomato", "土豆": "potato",
    "洋葱": "onion", "大蒜": "garlic", "姜": "ginger",
    "青椒": "capsicum", "胡萝卜": "carrot", "黄瓜": "cucumber",
    "菠菜": "spinach", "白菜": "cabbage", "西兰花": "broccoli",
    "豆腐": "tofu", "四季豆": "green bean", "茄子": "eggplant",
    "猪肉": "pork", "鸡肉": "chicken", "牛肉": "beef",
    "羊肉": "lamb", "猪排": "pork ribs", "鸡腿": "chicken leg",
    "鸡胸": "chicken breast", "五花肉": "pork belly",
    "猪绞肉": "pork mince", "牛绞肉": "beef mince",
    "虾": "prawn", "鱼": "fish", "鲈鱼": "barramundi",
    "三文鱼": "salmon", "螃蟹": "crab",
    "鸡蛋": "egg", "牛奶": "milk",
}

# 电子产品类（新增）
_ELECTRONICS_ZH_TO_EN: dict[str, str] = {
    "手机": "phone",
    "苹果手机": "iphone",
    "华为手机": "huawei phone",
    "三星手机": "samsung phone",
    "iPhone": "iphone",
    "电脑": "laptop",
    "笔记本": "laptop",
    "笔记本电脑": "laptop",
    "台式机": "desktop computer",
    "MacBook": "macbook",
    "平板": "tablet",
    "iPad": "ipad",
    "耳机": "headphones",
    "无线耳机": "wireless headphones",
    "AirPods": "airpods",
    "电视": "tv",
    "充电器": "charger",
    "数据线": "cable",
    "键盘": "keyboard",
    "鼠标": "mouse",
}

# 生活用品类（预留扩展）
_HOUSEHOLD_ZH_TO_EN: dict[str, str] = {
    "洗发水": "shampoo",
    "沐浴露": "body wash",
    "洗衣液": "laundry detergent",
    "纸巾": "tissue",
}

# 合并（后面的覆盖前面的）
ZH_TO_EN: dict[str, str] = {
    **_FOOD_ZH_TO_EN,
    **_ELECTRONICS_ZH_TO_EN,
    **_HOUSEHOLD_ZH_TO_EN,
}


def _translate(query: str) -> str:
    """中文查询词转英文；查不到则原样返回。"""
    return ZH_TO_EN.get(query.strip(), query)
