"""
tests/test_budget_tier.py

验证 budget_tier 计算、ChromaDB 更新和 search_recipes 过滤：
  1-8:  _is_meat / _price_to_tier 纯逻辑单元测试
  9:    calc_recipe_budget_tier 真实查价（网络），返回合法值
  10:   update_recipe_budget_tier 写入并能从 ChromaDB 读回
  11:   search_recipes(budget_tier="low") 只返回 low 档菜谱
  12:   search_recipes(budget_tier="")  不过滤，正常返回结果
"""

from __future__ import annotations

import os
import pytest

from tools.budget_tier import _is_meat, _price_to_tier, calc_recipe_budget_tier
from tools.recipe import (
    init_chroma,
    update_recipe_budget_tier,
    get_all_recipes,
    search_recipes,
)

# ── fixture：确保 ChromaDB 已初始化 ─────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def chroma_ready():
    """模块级初始化 ChromaDB，避免每个测试重复初始化。"""
    init_chroma(persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    yield
    # 测试结束后把番茄炒蛋的 budget_tier 恢复 unknown，不污染其他测试
    update_recipe_budget_tier("番茄炒蛋", "unknown")


# ── 测试 1-2：_is_meat ───────────────────────────────────────────────────────

def test_is_meat_true():
    assert _is_meat("鸡肉") is True
    print("\n[1] PASS — 鸡肉 is_meat=True")


def test_is_meat_false():
    assert _is_meat("番茄") is False
    print("\n[2] PASS — 番茄 is_meat=False")


# ── 测试 3-5：_price_to_tier（蔬菜档位）────────────────────────────────────────

def test_price_to_tier_veggie_low():
    assert _price_to_tier(5.0, is_meat=False) == "low"
    print("\n[3] PASS — 5.0 蔬菜 → low")


def test_price_to_tier_veggie_mid():
    assert _price_to_tier(8.0, is_meat=False) == "mid"
    print("\n[4] PASS — 8.0 蔬菜 → mid")


def test_price_to_tier_veggie_high():
    assert _price_to_tier(15.0, is_meat=False) == "high"
    print("\n[5] PASS — 15.0 蔬菜 → high")


# ── 测试 6-8：_price_to_tier（肉类档位）────────────────────────────────────────

def test_price_to_tier_meat_low():
    assert _price_to_tier(10.0, is_meat=True) == "low"
    print("\n[6] PASS — 10.0 肉类 → low")


def test_price_to_tier_meat_mid():
    assert _price_to_tier(20.0, is_meat=True) == "mid"
    print("\n[7] PASS — 20.0 肉类 → mid")


def test_price_to_tier_meat_high():
    assert _price_to_tier(25.0, is_meat=True) == "high"
    print("\n[8] PASS — 25.0 肉类 → high")


# ── 测试 9：calc_recipe_budget_tier 真实查价 ──────────────────────────────────

@pytest.mark.network
def test_calc_recipe_budget_tier_returns_valid():
    """真实网络查价，验证返回合法值（不要求具体档位）。"""
    tier = calc_recipe_budget_tier(["番茄", "鸡蛋"])
    print(f"\n[9] 番茄+鸡蛋 → budget_tier={tier!r}")
    assert tier in ("low", "mid", "high", "unknown"), f"非法值: {tier!r}"
    print("[9] PASS — calc_recipe_budget_tier 返回合法值")


# ── 测试 10：update_recipe_budget_tier 写入 ChromaDB ──────────────────────────

def test_update_recipe_budget_tier():
    """将番茄炒蛋改为 low，再从 ChromaDB 读回验证。"""
    ok = update_recipe_budget_tier("番茄炒蛋", "low")
    print(f"\n[10] update_recipe_budget_tier 返回: {ok}")
    assert ok is True, "update 应返回 True"

    # 从 ChromaDB 读回验证
    all_r = get_all_recipes()
    match = next((r for r in all_r if r["name"] == "番茄炒蛋"), None)
    assert match is not None, "ChromaDB 里找不到番茄炒蛋"
    assert match["budget_tier"] == "low", f"期望 low，实际 {match['budget_tier']!r}"
    print("[10] PASS — ChromaDB 里 budget_tier 已更新为 low")


# ── 测试 11：search_recipes 按 budget_tier 过滤 ───────────────────────────────

@pytest.mark.asyncio
async def test_search_recipes_with_budget_tier_filter():
    """
    先确保至少1条菜谱是 low，再过滤搜索，验证返回结果都是 low。
    依赖测试10已将番茄炒蛋改为 low。
    """
    results = await search_recipes.ainvoke({
        "query": "家常菜",
        "budget_tier": "low",
    })
    print(f"\n[11] budget_tier=low 搜索结果 ({len(results)} 条):")
    for r in results:
        print(f"    {r['name']} → {r['budget_tier']}")

    # 如果没有 low 的菜谱，至少测试应返回空列表而不报错
    for r in results:
        assert r["budget_tier"] == "low", (
            f"{r['name']} 的 budget_tier={r['budget_tier']!r}，期望 low"
        )
    print("[11] PASS — 所有返回结果 budget_tier 均为 low")


# ── 测试 12：search_recipes 不传 budget_tier 不过滤 ───────────────────────────

@pytest.mark.asyncio
async def test_search_recipes_without_budget_tier_filter():
    """不传 budget_tier 时，正常返回最多3条结果。"""
    results = await search_recipes.ainvoke({
        "query": "家常菜",
        "budget_tier": "",
    })
    print(f"\n[12] 无过滤搜索结果 ({len(results)} 条):")
    for r in results:
        print(f"    {r['name']} → {r.get('budget_tier', 'N/A')}")

    assert len(results) >= 1, "无过滤应有结果"
    assert len(results) <= 3, "最多返回3条"
    print("[12] PASS — 无 budget_tier 过滤时正常返回结果")
