import pytest
from unittest.mock import patch, MagicMock
from agents.clarify_agent import run_clarify_agent
from agents.routing import should_clarify

def make_state(**kwargs):
    defaults = {
        "user_input": "",
        "intent": "CHAT",
        "confidence": "HIGH",
        "missing_slots": [],
        "top2": None,
        "result": "",
        "user_id": "test_user",
        "user_brief": {},
        "conversation_memory": "",
    }
    defaults.update(kwargs)
    return defaults

# should_clarify 路由测试
def test_high_confidence_routes_directly():
    state = make_state(intent="MEAL", confidence="HIGH")
    assert should_clarify(state) == "meal"

def test_low_confidence_routes_to_clarify():
    state = make_state(intent="MEAL", confidence="LOW", top2="PRICE")
    assert should_clarify(state) == "clarify"

def test_medium_missing_required_slot_routes_to_clarify():
    state = make_state(intent="PRICE", confidence="MEDIUM", missing_slots=["product"])
    assert should_clarify(state) == "clarify"

def test_medium_profile_can_fill_slot_routes_directly():
    state = make_state(
        intent="MEAL",
        confidence="MEDIUM",
        missing_slots=["budget"],
        user_brief={"budget": "mid"}
    )
    assert should_clarify(state) == "meal"

# clarify_agent 输出测试
def test_clarify_low_confidence_shows_top2():
    state = make_state(
        confidence="LOW",
        intent="PRICE",
        top2="MEAL"
    )
    result = run_clarify_agent(state)
    assert "1." in result["result"]
    assert "2." in result["result"]

def test_clarify_missing_product():
    state = make_state(
        confidence="MEDIUM",
        intent="PRICE",
        missing_slots=["product"]
    )
    result = run_clarify_agent(state)
    assert "商品" in result["result"] or "价格" in result["result"]

def test_clarify_missing_ingredient():
    state = make_state(
        confidence="MEDIUM",
        intent="FOOD_SAFETY",
        missing_slots=["ingredient"]
    )
    result = run_clarify_agent(state)
    assert "食材" in result["result"] or "药物" in result["result"]

def test_clarify_fallback():
    state = make_state(
        confidence="MEDIUM",
        missing_slots=["unknown_slot"]
    )
    result = run_clarify_agent(state)
    assert len(result["result"]) > 0


def test_electronics_price_missing_model_routes_to_clarify():
    """苹果手机（有品牌无型号）→ 追问型号"""
    state = make_state(
        intent="ELECTRONICS_PRICE",
        confidence="MEDIUM",
        missing_slots=["model"]
    )
    assert should_clarify(state) == "clarify"


def test_electronics_price_with_model_routes_directly():
    """iPhone 16（有具体型号）→ 直接路由"""
    state = make_state(
        intent="ELECTRONICS_PRICE",
        confidence="HIGH",
        missing_slots=[]
    )
    assert should_clarify(state) == "electronics_price"


def test_clarify_missing_model_template():
    """追问型号时返回正确模板"""
    state = make_state(
        confidence="MEDIUM",
        intent="ELECTRONICS_PRICE",
        missing_slots=["model"]
    )
    result = run_clarify_agent(state)
    assert "型号" in result["result"]


def test_price_missing_product_routes_to_clarify():
    """完全没提商品 → 追问商品名"""
    state = make_state(
        intent="PRICE",
        confidence="MEDIUM",
        missing_slots=["product"]
    )
    assert should_clarify(state) == "clarify"
