"""
tests/test_price_chart_api.py

数据层测试（5个）+ API 层测试（4个）：
  1. get_price_chart_single → 无数据时返回 comparable=False, data={}
  2. get_price_chart_single → 单位一致时 comparable=True，data 是列表
  3. get_price_chart_single → 单位不一致时 comparable=False，data 是平台分组字典
  4. get_price_chart_multi  → 无数据时 data={}
  5. get_price_chart_multi  → 有数据时按平台分组返回
  6. GET /api/price-history/tomato?mode=single → 200，含 mode/comparable/data
  7. GET /api/price-history/tomato?mode=multi  → 200，含 mode/data
  8. GET /api/tracked-terms                    → 200，含 terms/count
  9. GET /api/price-history/不存在的商品        → 200，data 为空（不报错）
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tools.price_history import (
    get_price_chart_single,
    get_price_chart_multi,
    record_price,
)

TERM_SAME   = "chart_test_same_unit"    # 单位一致
TERM_DIFF   = "chart_test_diff_unit"    # 单位不一致
TERM_MULTI  = "chart_test_multi"        # 多线测试
TERM_EMPTY  = "chart_test_nonexistent_xyz"
ALL_TERMS   = [TERM_SAME, TERM_DIFF, TERM_MULTI]


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


# ── 数据层测试 ─────────────────────────────────────────────────────────────────

def test_single_no_data():
    result = get_price_chart_single(TERM_EMPTY)
    print(f"\n[1] {result}")
    assert result["query_term"] == TERM_EMPTY
    assert result["mode"] == "single"
    assert result["comparable"] is False
    assert result["data"] == {} or result["data"] == []
    print("[1] PASS — 无数据时 comparable=False, data 为空")


def test_single_same_unit():
    record_price(TERM_SAME, "woolworths", 3.50, "kg")
    record_price(TERM_SAME, "umall",      4.20, "kg")

    result = get_price_chart_single(TERM_SAME)
    print(f"\n[2] {result}")
    assert result["comparable"] is True
    assert result["unit"] == "kg"
    assert isinstance(result["data"], list)
    assert len(result["data"]) >= 1
    entry = result["data"][0]
    assert "date" in entry and "price" in entry
    # 同一天两个平台，取全局最低价 3.50
    assert entry["price"] == pytest.approx(3.50, rel=1e-2)
    print("[2] PASS — 单位一致时 comparable=True，data 是列表，取最低价")


def test_single_diff_unit():
    record_price(TERM_DIFF, "woolworths", 0.50, "pack")
    record_price(TERM_DIFF, "umall",      6.00, "kg")

    result = get_price_chart_single(TERM_DIFF)
    print(f"\n[3] {result}")
    assert result["comparable"] is False
    assert isinstance(result["data"], dict)
    assert "woolworths" in result["data"] or "umall" in result["data"]
    # 每条记录应有 date/price/unit
    for platform, entries in result["data"].items():
        for e in entries:
            assert "date" in e and "price" in e and "unit" in e
    print("[3] PASS — 单位不一致时 comparable=False，data 按平台分组")


def test_multi_no_data():
    result = get_price_chart_multi(TERM_EMPTY)
    print(f"\n[4] {result}")
    assert result["mode"] == "multi"
    assert result["data"] == {}
    print("[4] PASS — 无数据时 multi mode data={}")


def test_multi_by_platform():
    record_price(TERM_MULTI, "woolworths", 3.50, "kg")
    record_price(TERM_MULTI, "umall",      4.20, "kg")

    result = get_price_chart_multi(TERM_MULTI)
    print(f"\n[5] {result}")
    assert result["mode"] == "multi"
    assert isinstance(result["data"], dict)
    assert "woolworths" in result["data"]
    assert "umall" in result["data"]
    for platform, entries in result["data"].items():
        for e in entries:
            assert "date" in e and "price" in e and "unit" in e
    print("[5] PASS — 有数据时 multi 按平台分组返回")


# ── API 层测试 ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def test_api_single(client):
    resp = client.get("/api/price-history/tomato?mode=single&days=30")
    print(f"\n[6] status={resp.status_code}, body={resp.text[:200]}")
    assert resp.status_code == 200
    body = resp.json()
    assert "mode" in body and body["mode"] == "single"
    assert "comparable" in body
    assert "data" in body
    print("[6] PASS — GET /api/price-history/tomato?mode=single 结构正确")


def test_api_multi(client):
    resp = client.get("/api/price-history/tomato?mode=multi&days=30")
    print(f"\n[7] status={resp.status_code}, body={resp.text[:200]}")
    assert resp.status_code == 200
    body = resp.json()
    assert "mode" in body and body["mode"] == "multi"
    assert "data" in body
    print("[7] PASS — GET /api/price-history/tomato?mode=multi 结构正确")


def test_api_tracked_terms(client):
    resp = client.get("/api/tracked-terms?days=30")
    print(f"\n[8] status={resp.status_code}, body={resp.text[:300]}")
    assert resp.status_code == 200
    body = resp.json()
    assert "terms" in body
    assert "count" in body
    assert isinstance(body["terms"], list)
    assert body["count"] == len(body["terms"])
    print(f"[8] PASS — GET /api/tracked-terms 返回 {body['count']} 个词")


def test_api_nonexistent_term(client):
    resp = client.get("/api/price-history/完全不存在的商品xyz123?mode=single")
    print(f"\n[9] status={resp.status_code}, body={resp.text[:200]}")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    # data 为空（列表或字典）
    assert body["data"] == {} or body["data"] == []
    print("[9] PASS — 不存在的商品返回 200 且 data 为空")
