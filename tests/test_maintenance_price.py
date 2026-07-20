"""
tests/test_maintenance_price.py

验证 get_tracked_terms 和 daily_price_update 的行为：
  1. price_history 为空时 get_tracked_terms 返回 []
  2. 插入3条不同 query_term 后 get_tracked_terms 返回3个词
  3. get_tracked_terms(days=7) 只返回7天内的，30天前的不返回
  4. price_history 为空时 daily_price_update 直接跳过，不报错
  5. 有追踪词时 daily_price_update 执行后 price_history 有新记录（mock 网络请求）
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from tools.price_history import get_tracked_terms, record_price, get_price_history

TERM_A = "maint_test_apple"
TERM_B = "maint_test_banana"
TERM_C = "maint_test_carrot"
ALL_TERMS = [TERM_A, TERM_B, TERM_C]


def _cleanup():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM
    with SessionLocal() as session:
        for t in ALL_TERMS:
            session.query(PriceHistoryORM).filter_by(query_term=t).delete()
        session.commit()


@pytest.fixture(autouse=True)
def clean(in_memory_mysql_session):
    _cleanup()
    yield
    _cleanup()


# ── 测试 1：price_history 为空时返回 [] ──────────────────────────────────────

def test_get_tracked_terms_empty():
    terms = get_tracked_terms(days=30)
    for t in ALL_TERMS:
        assert t not in terms, f"期望不含 {t}，实际 terms={terms}"
    print("\n[1] PASS — price_history 为空（测试词）时 get_tracked_terms 不含测试词")


# ── 测试 2：插入3条不同 query_term 后返回3个词 ─────────────────────────────────

def test_get_tracked_terms_returns_distinct():
    for term in ALL_TERMS:
        record_price(term, "woolworths", 1.99, "kg")
        record_price(term, "umall",      2.50, "kg")   # 同 term 两条，应去重

    terms = get_tracked_terms(days=30)
    found = [t for t in ALL_TERMS if t in terms]
    print(f"\n[2] 找到 {len(found)} 个测试词: {found}")
    assert len(found) == 3, f"期望3个词，实际 found={found}"
    assert len(set(found)) == 3, "应无重复"
    print("[2] PASS — 3个不同 query_term，去重后正确返回")


# ── 测试 3：days=7 只返回7天内的，30天前的不返回 ───────────────────────────────

def test_get_tracked_terms_day_filter():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM

    # TERM_A: 插2条40天前的记录（满足 min_count=2，但超出7天范围）
    old_time = datetime.now() - timedelta(days=40)
    with SessionLocal() as session:
        for _ in range(2):
            session.add(PriceHistoryORM(
                query_term=TERM_A,
                platform="woolworths",
                price=1.99,
                unit="kg",
                recorded_at=old_time,
            ))
        session.commit()

    # TERM_B: 插2条今天的记录（满足 min_count=2，且在7天范围内）
    record_price(TERM_B, "woolworths", 2.50, "kg")
    record_price(TERM_B, "umall",      2.80, "kg")

    terms_7d  = get_tracked_terms(days=7)
    terms_60d = get_tracked_terms(days=60)

    print(f"\n[3] 7天: {[t for t in ALL_TERMS if t in terms_7d]}")
    print(f"[3] 60天: {[t for t in ALL_TERMS if t in terms_60d]}")

    assert TERM_A not in terms_7d,  "40天前的记录不应在7天范围内"
    assert TERM_B in terms_7d,      "今天的记录应在7天范围内"
    assert TERM_A in terms_60d,     "40天前的记录应在60天范围内"
    print("[3] PASS — 天数过滤正确")


# ── 测试 4：price_history 为空时 daily_price_update 跳过，不报错 ──────────────

@pytest.mark.asyncio
async def test_daily_price_update_skips_when_empty():
    from agents.maintenance_agent import daily_price_update

    # daily_price_update 内部做 `from tools.price_history import get_tracked_terms`
    # patch 模块属性，运行时 import 会拿到 mock 版本
    with patch("tools.price_history.get_tracked_terms", return_value=[]):
        try:
            await daily_price_update()
        except Exception as e:
            pytest.fail(f"daily_price_update 不应抛出，但抛出了: {e}")
    print("\n[4] PASS — price_history 为空时跳过，无异常")


# ── 测试 5：有追踪词时，执行后 price_history 有新记录（mock 网络请求）─────────────

@pytest.mark.asyncio
async def test_daily_price_update_writes_history():
    from agents.maintenance_agent import daily_price_update

    # mock get_ingredient_price：模拟有货结果，side_effect 同时调用真实 record_price
    # 这样既不发网络请求，又能验证 price_history 有新记录
    mock_result = {
        "item": TERM_A,
        "comparable": True,
        "best_price": 2.10,
        "best_platform": "woolworths",
        "unit": "kg",
        "all_results": [
            {"platform": "woolworths", "price": 2.10, "unit": "kg",
             "available": True, "url": None, "product_name": "Mock Apple"},
            {"platform": "umall", "price": None, "unit": "",
             "available": False, "url": None, "product_name": None},
        ],
    }

    def fake_invoke(kwargs):
        # 模拟 get_ingredient_price 内部会调用 record_price 的行为
        record_price(
            query_term=kwargs["ingredient"],
            platform="woolworths",
            price=2.10,
            unit="kg",
            product_name="Mock Apple",
        )
        return mock_result

    mock_tool = MagicMock()
    mock_tool.invoke = fake_invoke

    with patch("tools.price_history.get_tracked_terms", return_value=[TERM_A]):
        with patch("tools.price.get_ingredient_price", mock_tool):
            await daily_price_update()

    # 验证 price_history 有新记录
    history = get_price_history(TERM_A, days=1)
    print(f"\n[5] price_history 记录数: {len(history)}")
    assert len(history) >= 1, "执行后 price_history 应至少有1条记录"
    assert history[0]["platform"] == "woolworths"
    assert history[0]["price"] == pytest.approx(2.10, rel=1e-2)
    print("[5] PASS — daily_price_update 执行后 price_history 有新记录")
