REQUIRED_SLOTS = {
    "MEAL": [],
    "PRICE": ["product"],
    "ELECTRONICS_PRICE": ["model"],
    "NUTRITION": ["topic"],
    "FOOD_SAFETY": ["ingredient"],
    "MEMORY_UPDATE": [],
    "UPDATE_AND_MEAL": [],
    "HEALTH_CYCLE": [],
    "CHAT": [],
}

PROFILE_FILLABLE_SLOTS = {
    "budget": "budget",
    "cuisine": "category_preferences",
    "health": "health_conditions",
    "brand": "category_preferences",
}

CLARIFY_TEMPLATES = {
    "missing_product": "请问您想查询什么商品的价格？\n（例如：苹果、猪肉、iPhone 16、耳机）",
    "missing_model": "请问您想查哪款具体型号？\n（例如：iPhone 16、iPhone 16 Pro、MacBook Air M3、Samsung S25）",
    "missing_ingredient": "请问您想了解哪种食材或药物的安全信息？（例如：头孢、菠菜、螃蟹）",
    "missing_topic": "请问您想了解哪方面的营养信息？（例如：糖尿病饮食、高血压食谱）",
    "ambiguous_category": "您是想查：\n1. {option_a}\n2. {option_b}\n请回复数字选择",
    "low_confidence": "您是想：\n1. {intent_a}\n2. {intent_b}\n请回复数字选择",
}

INTENT_DISPLAY = {
    "MEAL": "餐食推荐",
    "PRICE": "价格查询",
    "ELECTRONICS_PRICE": "电子产品价格查询",
    "NUTRITION": "营养饮食咨询",
    "FOOD_SAFETY": "食材安全查询",
    "MEMORY_UPDATE": "更新个人偏好",
    "UPDATE_AND_MEAL": "更新偏好并推荐",
    "HEALTH_CYCLE": "健康状态记录",
    "CHAT": "闲聊",
}
