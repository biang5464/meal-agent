"""Tests for daily recommendation filtering and selection logic (no DB/Redis/LLM calls)."""

import pytest

from agents.daily_recommendation_agent import (
    MAIN_DISH_CATEGORIES,
    VEGGIE_CATEGORIES,
    SOUP_CATEGORIES,
    _filter_recipes,
    _rule_fallback,
    _select_by_structure,
)

MOCK_RECIPES = [
    {
        "name": "红烧肉",
        "category": "肉类",
        "ingredients": ["五花肉", "生抽", "老抽"],
        "budget_tier": "high",
        "flavor": "咸甜",
        "cuisine": "家常菜",
        "steps": "五花肉焯水炖煮。",
    },
    {
        "name": "蒜蓉西兰花",
        "category": "蔬菜",
        "ingredients": ["西兰花", "大蒜"],
        "budget_tier": "low",
        "flavor": "清淡",
        "cuisine": "家常菜",
        "steps": "西兰花焯水翻炒。",
    },
    {
        "name": "冬瓜排骨汤",
        "category": "汤",
        "ingredients": ["排骨", "冬瓜"],
        "budget_tier": "mid",
        "flavor": "清淡",
        "cuisine": "家常菜",
        "steps": "排骨焯水熬汤。",
    },
    {
        "name": "清蒸鲈鱼",
        "category": "海鲜",
        "ingredients": ["鲈鱼", "姜", "葱"],
        "budget_tier": "high",
        "flavor": "清淡",
        "cuisine": "粤菜",
        "steps": "鱼蒸熟浇汁。",
    },
    {
        "name": "麻婆豆腐",
        "category": "豆腐",
        "ingredients": ["嫩豆腐", "猪肉末", "豆瓣酱"],
        "budget_tier": "low",
        "flavor": "麻辣",
        "cuisine": "川菜",
        "steps": "豆腐炖煮调味。",
    },
    {
        "name": "番茄炒蛋",
        "category": "半荤素",
        "ingredients": ["番茄", "鸡蛋"],
        "budget_tier": "low",
        "flavor": "鲜甜",
        "cuisine": "家常菜",
        "steps": "鸡蛋炒熟加番茄。",
    },
    {
        "name": "白灼虾",
        "category": "海鲜",
        "ingredients": ["鲜虾", "姜", "葱"],
        "budget_tier": "high",
        "flavor": "鲜甜",
        "cuisine": "粤菜",
        "steps": "虾汆水捞起蘸汁。",
    },
    {
        "name": "干煸四季豆",
        "category": "蔬菜",
        "ingredients": ["四季豆", "猪肉末", "干辣椒"],
        "budget_tier": "low",
        "flavor": "辣",
        "cuisine": "川菜",
        "steps": "四季豆干煸后翻炒。",
    },
]


# ── 过滤测试 ───────────────────────────────────────────────────────────────────

def test_filter_removes_disliked_ingredient():
    result = _filter_recipes(MOCK_RECIPES, dislikes=["五花肉"])
    names = [r["name"] for r in result]
    assert "红烧肉" not in names
    assert len(result) == len(MOCK_RECIPES) - 1


def test_filter_budget_soft():
    result = _filter_recipes(MOCK_RECIPES, budget="low")
    tiers = [r["budget_tier"] for r in result]
    assert all(t in ("low", "unknown") for t in tiers)
    assert not any(t == "high" for t in tiers)


def test_filter_recent_dedup():
    result = _filter_recipes(MOCK_RECIPES, recent_dishes=["红烧肉", "清蒸鲈鱼"])
    names = [r["name"] for r in result]
    assert "红烧肉" not in names
    assert "清蒸鲈鱼" not in names
    assert len(result) == len(MOCK_RECIPES) - 2


def test_filter_diet_restriction():
    # "辣" 关键词应过滤掉 flavor 含"辣"的菜（麻辣、辣）
    result = _filter_recipes(MOCK_RECIPES, restrictions=["辣"])
    names = [r["name"] for r in result]
    assert "麻婆豆腐" not in names   # flavor=麻辣
    assert "干煸四季豆" not in names  # flavor=辣
    # 宫保鸡丁不在 MOCK_RECIPES，其他菜不含"辣"
    assert "蒜蓉西兰花" in names


def test_filter_empty_when_all_restricted():
    all_ingredients = [
        "五花肉", "西兰花", "排骨", "鲈鱼",
        "嫩豆腐", "番茄", "鲜虾", "四季豆",
    ]
    result = _filter_recipes(MOCK_RECIPES, dislikes=all_ingredients)
    assert result == []


# ── 选菜测试 ───────────────────────────────────────────────────────────────────

def test_select_structure_main_veggie_soup():
    selected = _select_by_structure(MOCK_RECIPES)
    assert len(selected) == 3
    categories = [r.get("category") for r in selected]
    has_main  = any(c in MAIN_DISH_CATEGORIES for c in categories)
    has_veggie = any(c in VEGGIE_CATEGORIES   for c in categories)
    has_soup  = any(c in SOUP_CATEGORIES      for c in categories)
    assert has_main and has_veggie and has_soup


def test_select_fallback_when_missing_soup():
    no_soup = [r for r in MOCK_RECIPES if r["category"] != "汤"]
    selected = _select_by_structure(no_soup)
    assert len(selected) == 3


def test_select_no_duplicates():
    selected = _select_by_structure(MOCK_RECIPES)
    names = [r["name"] for r in selected]
    assert len(names) == len(set(names))


# ── 兜底测试 ───────────────────────────────────────────────────────────────────

def test_rule_fallback_returns_n():
    result = _rule_fallback(MOCK_RECIPES, n=3)
    assert len(result) == 3
    # 所有返回的菜必须来自输入
    all_names = {r["name"] for r in MOCK_RECIPES}
    assert all(r["name"] in all_names for r in result)
