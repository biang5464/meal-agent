"""
tests/test_phase11a_daily_dedup.py

Phase 11A: 验证每日推荐午晚餐去重修复。
不调用 DeepSeek / Redis / MySQL / ChromaDB。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from agents.daily_recommendation_agent import (
    _filter_recipes,
    _get_recent_dishes,
    _get_same_day_dishes,
    _rule_fallback,
)

# ── 测试用菜谱数据 ────────────────────────────────────────────────────────────

RECIPES = [
    {"name": "红烧肉",    "category": "肉类",   "ingredients": ["五花肉"],        "budget_tier": "high", "flavor": "咸甜", "steps": "炖煮。"},
    {"name": "蒜蓉西兰花", "category": "蔬菜",   "ingredients": ["西兰花"],        "budget_tier": "low",  "flavor": "清淡", "steps": "翻炒。"},
    {"name": "冬瓜排骨汤", "category": "汤",     "ingredients": ["排骨"],          "budget_tier": "mid",  "flavor": "清淡", "steps": "熬汤。"},
    {"name": "清蒸鲈鱼",  "category": "海鲜",   "ingredients": ["鲈鱼"],          "budget_tier": "high", "flavor": "清淡", "steps": "蒸熟。"},
    {"name": "麻婆豆腐",  "category": "豆腐",   "ingredients": ["嫩豆腐"],        "budget_tier": "low",  "flavor": "麻辣", "steps": "炖煮。"},
    {"name": "番茄炒蛋",  "category": "半荤素",  "ingredients": ["番茄", "鸡蛋"],  "budget_tier": "low",  "flavor": "鲜甜", "steps": "翻炒。"},
    {"name": "白灼虾",    "category": "海鲜",   "ingredients": ["鲜虾"],          "budget_tier": "high", "flavor": "鲜甜", "steps": "汆水。"},
    {"name": "干煸四季豆", "category": "蔬菜",   "ingredients": ["四季豆"],        "budget_tier": "low",  "flavor": "辣",   "steps": "干煸。"},
]


def _names(recipes: list[dict]) -> list[str]:
    return [r["name"] for r in recipes]


# ── 1-3: _filter_recipes 新增 same_day_dishes 参数 ───────────────────────────

class TestFilterRecipesSameDayDishes:
    def test_same_day_dishes_hard_excluded(self):
        """same_day_dishes 中的菜必须被硬过滤掉，不受其他参数影响。"""
        result = _filter_recipes(RECIPES, same_day_dishes=["白灼虾", "番茄炒蛋"])
        names = _names(result)
        assert "白灼虾" not in names
        assert "番茄炒蛋" not in names
        assert len(result) == len(RECIPES) - 2

    def test_same_day_dishes_none_no_filter(self):
        """same_day_dishes=None 不过滤任何菜谱。"""
        result = _filter_recipes(RECIPES, same_day_dishes=None)
        assert len(result) == len(RECIPES)

    def test_same_day_dishes_empty_list_no_filter(self):
        """same_day_dishes=[] 不过滤任何菜谱。"""
        result = _filter_recipes(RECIPES, same_day_dishes=[])
        assert len(result) == len(RECIPES)


# ── 4-6: _get_recent_dishes 返回 (names, status) ─────────────────────────────

class TestGetRecentDishesStatus:
    def _mock_session(self, rows: list):
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = rows
        return mock_session

    def _make_row(self, dishes: list[dict]) -> MagicMock:
        row = MagicMock()
        row.dishes = dishes
        return row

    def test_returns_success_with_rows(self):
        """有历史记录时返回 (names, 'success')。"""
        row = self._make_row([{"name": "红烧肉"}, {"name": "蒜蓉西兰花"}])
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = self._mock_session([row])
            names, status = _get_recent_dishes("user1", days=7)
        assert status == "success"
        assert "红烧肉" in names
        assert "蒜蓉西兰花" in names

    def test_returns_empty_with_no_rows(self):
        """没有历史记录时返回 ([], 'empty')。"""
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = self._mock_session([])
            names, status = _get_recent_dishes("user1", days=7)
        assert status == "empty"
        assert names == []

    def test_returns_read_degraded_on_exception(self):
        """DB 异常时返回 ([], 'read_degraded')，不抛出异常。"""
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.side_effect = Exception("DB error")
            names, status = _get_recent_dishes("user1", days=7)
        assert status == "read_degraded"
        assert names == []


# ── 7-10: _get_same_day_dishes ────────────────────────────────────────────────

class TestGetSameDayDishes:
    def _mock_session_with_rows(self, rows: list) -> MagicMock:
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = rows
        return mock_session

    def _make_row(self, dishes: list[dict]) -> MagicMock:
        row = MagicMock()
        row.dishes = dishes
        return row

    def test_lunch_queries_dinner_dishes(self):
        """lunch 生成时，返回同天 dinner 的已推荐菜名。"""
        row = self._make_row([{"name": "白灼虾"}, {"name": "番茄炒蛋"}])
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = self._mock_session_with_rows([row])
            names, status = _get_same_day_dishes("user1", date.today(), "lunch")
        assert status == "success"
        assert "白灼虾" in names
        assert "番茄炒蛋" in names

    def test_dinner_queries_lunch_dishes(self):
        """dinner 生成时，返回同天 lunch 的已推荐菜名。"""
        row = self._make_row([{"name": "清蒸鲈鱼"}])
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = self._mock_session_with_rows([row])
            names, status = _get_same_day_dishes("user1", date.today(), "dinner")
        assert status == "success"
        assert "清蒸鲈鱼" in names

    def test_returns_empty_when_no_other_meal(self):
        """另一餐尚未生成时返回 ([], 'empty')。"""
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = self._mock_session_with_rows([])
            names, status = _get_same_day_dishes("user1", date.today(), "lunch")
        assert status == "empty"
        assert names == []

    def test_returns_read_degraded_on_exception(self):
        """DB 异常时返回 ([], 'read_degraded')，不抛出异常。"""
        with patch("agents.daily_recommendation_agent._session") as mock_ctx:
            mock_ctx.return_value.__enter__.side_effect = Exception("DB error")
            names, status = _get_same_day_dishes("user1", date.today(), "dinner")
        assert status == "read_degraded"
        assert names == []


# ── 11-14: 降级逻辑验证 ───────────────────────────────────────────────────────

class TestFallbackLevelLogic:
    """
    通过直接调用 _filter_recipes 验证 generate_for_user 的降级逻辑：
    - Level 0: hard_base + recent + budget
    - Level 1: hard_base + budget (放宽 7-day 历史)
    - Level 2: hard_base         (放宽预算 + 历史)
    same_day 约束在所有 level 中均保持。
    """

    def test_level0_all_constraints_applied(self):
        """Level 0：full constraints 后仍有 ≥3 道菜，且 same_day 与 recent 均被排除。"""
        same_day = ["白灼虾"]
        recent   = ["红烧肉"]
        budget   = "low"

        after_safety = _filter_recipes(RECIPES)
        hard_base    = _filter_recipes(after_safety, same_day_dishes=same_day)
        lv0          = _filter_recipes(hard_base, budget=budget, recent_dishes=recent)

        assert len(lv0) >= 3, f"lv0 只有 {len(lv0)} 道菜"
        assert all(r["name"] not in same_day for r in lv0), "same_day 菜出现在 lv0"
        assert all(r["name"] not in recent   for r in lv0), "recent 菜出现在 lv0"

    def test_level1_drops_history_recovers(self):
        """Level 1：去掉 7-day 历史使候选数量恢复，same_day 仍被排除。"""
        same_day = ["白灼虾"]
        # 把所有非 same_day 菜都加入 recent，让 lv0 不足
        recent = [r["name"] for r in RECIPES if r["name"] not in same_day]
        budget = ""

        after_safety = _filter_recipes(RECIPES)
        hard_base    = _filter_recipes(after_safety, same_day_dishes=same_day)
        lv0          = _filter_recipes(hard_base, budget=budget, recent_dishes=recent)
        lv1          = _filter_recipes(hard_base, budget=budget)

        assert len(lv0) < 3,           f"期望 lv0 < 3，实际 {len(lv0)}"
        assert len(lv1) >= len(lv0),   f"lv1 应 >= lv0，实际 lv1={len(lv1)} lv0={len(lv0)}"
        assert all(r["name"] not in same_day for r in lv1), "same_day 菜出现在 lv1"

    def test_level2_drops_budget_recovers(self):
        """Level 2：去掉预算约束后候选 ≥3，same_day 在所有层级均被排除。"""
        # same_day 排除高档菜以外的 high 菜，迫使 budget=high 时选项极少
        same_day = ["红烧肉", "清蒸鲈鱼"]   # 排除两道 high 菜
        recent   = []
        budget   = "high"

        after_safety = _filter_recipes(RECIPES)
        hard_base    = _filter_recipes(after_safety, same_day_dishes=same_day)
        lv0          = _filter_recipes(hard_base, budget=budget, recent_dishes=recent)
        lv1          = _filter_recipes(hard_base, budget=budget)
        lv2          = hard_base

        assert len(lv0) < 3, f"期望 lv0 < 3，实际 {len(lv0)}"
        assert len(lv1) < 3, f"期望 lv1 < 3，实际 {len(lv1)}"
        assert len(lv2) >= 3, f"期望 lv2 >= 3，实际 {len(lv2)}"

        for level_name, level in [("lv0", lv0), ("lv1", lv1), ("lv2", lv2)]:
            assert all(r["name"] not in same_day for r in level), \
                f"same_day 菜出现在 {level_name}"

    def test_same_day_never_bypassed_at_any_level(self):
        """同天去重约束在所有降级层级（lv0/lv1/lv2）中均严格执行。"""
        same_day = ["白灼虾", "番茄炒蛋", "冬瓜排骨汤"]

        after_safety = _filter_recipes(RECIPES)
        hard_base    = _filter_recipes(after_safety, same_day_dishes=same_day)
        lv0          = _filter_recipes(hard_base, budget="low",  recent_dishes=["红烧肉"])
        lv1          = _filter_recipes(hard_base, budget="low")
        lv2          = hard_base

        for level_name, level in [("lv0", lv0), ("lv1", lv1), ("lv2", lv2)]:
            for dish in same_day:
                assert dish not in _names(level), \
                    f"同天菜 '{dish}' 出现在 {level_name}，违反跨餐去重约束"


# ── 15-19: 审计补充场景 ───────────────────────────────────────────────────────

class TestAuditScenarios:
    """Phase 11A Final Audit 补充测试，覆盖五个未显式测试的场景。"""

    # 场景 3a：_rule_fallback 候选不足 n 道时，返回现有全部，不崩溃、不补位
    def test_rule_fallback_fewer_items_than_n_returns_all(self):
        """候选只有 2 道时，_rule_fallback(n=3) 返回 2 道，不尝试补到 3 道。"""
        pool = RECIPES[:2]
        result = _rule_fallback(pool, n=3)
        assert len(result) == 2
        assert all(r["name"] in _names(pool) for r in result)

    # 场景 3b：fallback level 3 路径 — 同天约束仍在，结果可 <3 道
    def test_fallback_level3_excludes_same_day_with_fewer_dishes(self):
        """hard_base 只剩 2 道时，同天约束仍不放宽，返回值为这 2 道（不补充同天菜）。"""
        # 把前 6 道菜列为同天约束，仅剩后 2 道（干煸四季豆、白灼虾）
        same_day = _names(RECIPES[:6])
        after_safety = _filter_recipes(RECIPES)
        hard_base    = _filter_recipes(after_safety, same_day_dishes=same_day)

        assert len(hard_base) == 2, f"期望 hard_base=2，实际 {len(hard_base)}"
        for r in hard_base:
            assert r["name"] not in same_day, \
                f"同天菜 '{r['name']}' 出现在 hard_base，违反约束"

    # 场景 5a：忌口食材在所有降级层级不放宽
    def test_dislikes_persist_through_all_fallback_levels(self):
        """忌口食材（五花肉→红烧肉）在 lv0/lv1/lv2 中均不出现。"""
        after_safety = _filter_recipes(RECIPES, dislikes=["五花肉"])
        hard_base    = _filter_recipes(after_safety, same_day_dishes=[])
        lv0 = _filter_recipes(hard_base, budget="high", recent_dishes=["清蒸鲈鱼"])
        lv1 = _filter_recipes(hard_base, budget="high")
        lv2 = hard_base

        for level_name, level in [("lv0", lv0), ("lv1", lv1), ("lv2", lv2)]:
            for r in level:
                assert "五花肉" not in r.get("ingredients", []), \
                    f"忌口食材五花肉（红烧肉）出现在 {level_name}"

    # 场景 5b：饮食限制关键词在所有降级层级不放宽
    def test_restrictions_persist_through_all_fallback_levels(self):
        """饮食限制'辣'（麻婆豆腐、干煸四季豆）在 lv0/lv1/lv2 中均不出现。"""
        after_safety = _filter_recipes(RECIPES, restrictions=["辣"])
        hard_base    = _filter_recipes(after_safety, same_day_dishes=[])
        lv0 = _filter_recipes(hard_base, budget="low", recent_dishes=["番茄炒蛋"])
        lv1 = _filter_recipes(hard_base, budget="low")
        lv2 = hard_base

        restricted = ["麻婆豆腐", "干煸四季豆"]
        for level_name, level in [("lv0", lv0), ("lv1", lv1), ("lv2", lv2)]:
            for name in restricted:
                assert name not in _names(level), \
                    f"饮食限制菜品 '{name}' 出现在 {level_name}"

    # 场景 7：Scheduler 对每个用户依次生成 lunch 再 dinner（保证 dinner 能看到 lunch 数据）
    @pytest.mark.asyncio
    async def test_scheduler_sequential_lunch_then_dinner(self):
        """generate_daily_recommendations 对每个用户按 lunch→dinner 顺序调用 generate_for_user。"""
        from agents.daily_recommendation_agent import generate_daily_recommendations

        call_log: list[tuple[str, str]] = []

        async def spy(uid, meal_type, date_str, generated_by="scheduled"):
            call_log.append((uid, meal_type))

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [("u1",), ("u2",)]

        with (
            patch("agents.daily_recommendation_agent._session") as mock_ctx,
            patch("agents.daily_recommendation_agent.generate_for_user", side_effect=spy),
        ):
            mock_ctx.return_value.__enter__.return_value = mock_session
            await generate_daily_recommendations()

        for uid in ("u1", "u2"):
            user_calls = [meal for u, meal in call_log if u == uid]
            assert user_calls == ["lunch", "dinner"], \
                f"user={uid} 期望 ['lunch','dinner']，实际 {user_calls}"
