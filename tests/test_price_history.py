"""
tests/test_price_history.py

验证价格历史的写入、查询和边界行为：
  1. record_price 写入后，get_price_history 能读回来
  2. get_price_history 按平台过滤，只返回指定平台记录
  3. get_price_history 按天数过滤，超出范围的记录不返回
  4. record_price 遇到 DB 异常时静默处理，不抛出
  5. get_ingredient_price("tomato") 真实查询后，price_history 表有新记录
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from tools.price_history import record_price, get_price_history, get_tracked_terms

TERM = "test_price_hist_apple"


def _cleanup():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM
    with SessionLocal() as session:
        session.query(PriceHistoryORM).filter_by(query_term=TERM).delete()
        session.commit()


@pytest.fixture(autouse=True)
def clean(in_memory_mysql_session):
    _cleanup()
    yield
    _cleanup()


# ── 测试 1：写入后能读回 ────────────────────────────────────────────────────────

def test_record_and_get():
    record_price(
        query_term=TERM,
        platform="woolworths",
        price=3.99,
        unit="kg",
        product_name="Fresh Apple 1kg",
        currency="AUD",
        url="https://woolworths.com.au/apple",
    )
    rows = get_price_history(TERM)
    print(f"\n[1] 读回记录: {rows}")
    assert len(rows) >= 1
    r = rows[0]
    assert r["query_term"] == TERM
    assert r["platform"] == "woolworths"
    assert r["price"] == pytest.approx(3.99, rel=1e-2)
    assert r["unit"] == "kg"
    assert r["product_name"] == "Fresh Apple 1kg"
    assert r["currency"] == "AUD"
    print("[1] PASS — 写入后 get_price_history 能读回")


# ── 测试 2：按平台过滤 ──────────────────────────────────────────────────────────

def test_filter_by_platform():
    record_price(TERM, "woolworths", 3.99, "kg")
    record_price(TERM, "umall",      4.50, "kg")

    ww_rows = get_price_history(TERM, platform="woolworths")
    um_rows = get_price_history(TERM, platform="umall")
    print(f"\n[2] woolworths={len(ww_rows)} 条, umall={len(um_rows)} 条")

    assert all(r["platform"] == "woolworths" for r in ww_rows)
    assert all(r["platform"] == "umall"      for r in um_rows)
    assert len(ww_rows) >= 1
    assert len(um_rows) >= 1
    print("[2] PASS — 平台过滤正确")


# ── 测试 3：按天数过滤，超出范围不返回 ──────────────────────────────────────────

def test_filter_by_days():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM

    # 直接向 MySQL 插入一条 40 天前的旧记录
    old_time = datetime.now() - timedelta(days=40)
    with SessionLocal() as session:
        session.add(PriceHistoryORM(
            query_term=TERM,
            platform="woolworths",
            price=2.00,
            unit="kg",
            recorded_at=old_time,
        ))
        session.commit()

    # 插一条"今天"的记录
    record_price(TERM, "woolworths", 3.99, "kg")

    rows_30d = get_price_history(TERM, days=30)
    rows_60d = get_price_history(TERM, days=60)

    print(f"\n[3] 30天内: {len(rows_30d)} 条, 60天内: {len(rows_60d)} 条")
    assert len(rows_30d) == 1, f"30天内应只有1条，实际{len(rows_30d)}"
    assert len(rows_60d) == 2, f"60天内应有2条，实际{len(rows_60d)}"
    print("[3] PASS — 天数过滤正确")


# ── 测试 4：DB 异常时静默处理 ───────────────────────────────────────────────────

def test_record_price_silent_on_db_error():
    from core.database import SessionLocal as RealSession

    class BrokenSession:
        def __enter__(self): raise RuntimeError("DB 挂了")
        def __exit__(self, *_): pass

    with patch("tools.price_history.SessionLocal", return_value=BrokenSession()):
        try:
            record_price(TERM, "woolworths", 3.99, "kg")
        except Exception as e:
            pytest.fail(f"record_price 不应抛出异常，但抛出了: {e}")
    print("\n[4] PASS — DB 异常时静默处理")


# ── 测试 5：真实查询 tomato 后，price_history 有新记录 ────────────────────────────

@pytest.mark.network
def test_get_ingredient_price_writes_history():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM

    # 先清掉 tomato 的旧记录，确保计数准确
    with SessionLocal() as session:
        session.query(PriceHistoryORM).filter_by(query_term="tomato").delete()
        session.commit()

    from tools.price import _get_price_multi
    result = _get_price_multi("tomato")
    print(f"\n[5] 查询结果: {result.get('best_price')} {result.get('unit')} @ {result.get('best_platform')}")

    history = get_price_history("tomato", days=1)
    print(f"[5] price_history 记录数: {len(history)}")
    for r in history:
        print(f"    [{r['platform']}] {r['price']} {r['unit']} — {r['product_name']}")

    assert len(history) >= 1, "查询后 price_history 应至少有1条记录"
    assert all(r["query_term"] == "tomato" for r in history)
    print("[5] PASS — 真实查询后 price_history 有记录")


# ── 测试 6：get_tracked_terms min_count 过滤 ──────────────────────────────────

TERM_SINGLE = "test_price_hist_single_record"
TERM_MULTI  = "test_price_hist_multi_record"


def _cleanup_mincount():
    from core.database import SessionLocal
    from models.price_history import PriceHistoryORM
    with SessionLocal() as session:
        session.query(PriceHistoryORM).filter(
            PriceHistoryORM.query_term.in_([TERM_SINGLE, TERM_MULTI])
        ).delete(synchronize_session=False)
        session.commit()


def test_get_tracked_terms_min_count():
    """测试 6：只有1条记录的词不返回，有2条以上的词才返回。"""
    _cleanup_mincount()
    try:
        # TERM_SINGLE：只插1条记录
        record_price(TERM_SINGLE, "woolworths", 1.99, "kg")
        # TERM_MULTI：插2条记录
        record_price(TERM_MULTI, "woolworths", 2.50, "kg")
        record_price(TERM_MULTI, "umall",      2.80, "kg")

        # 默认 min_count=2
        terms = get_tracked_terms(days=1)
        print(f"\n[6] get_tracked_terms(min_count=2) 包含 SINGLE={TERM_SINGLE in terms}, MULTI={TERM_MULTI in terms}")
        assert TERM_SINGLE not in terms, f"1条记录的词不应被返回，实际 terms 含 {TERM_SINGLE!r}"
        assert TERM_MULTI  in terms,     f"2条记录的词应被返回，实际 terms 不含 {TERM_MULTI!r}"

        # min_count=1 时单条记录也返回
        terms1 = get_tracked_terms(days=1, min_count=1)
        print(f"[6] get_tracked_terms(min_count=1) 包含 SINGLE={TERM_SINGLE in terms1}, MULTI={TERM_MULTI in terms1}")
        assert TERM_SINGLE in terms1, f"min_count=1 时1条记录的词也应返回"
        assert TERM_MULTI  in terms1

        print("[6] PASS — min_count 过滤正确")
    finally:
        _cleanup_mincount()
