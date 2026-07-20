"""
tests/test_goodguys_source.py

GoodGuysSource 单元测试（10个）+ 网络测试（1个）：

  单元测试（无网络）：
  1.  GoodGuysSource 是 PriceSource 子类
  2.  "iphone case" 被黑名单过滤
  3.  "usb charger" 被黑名单过滤
  4.  "Apple iPhone 17" 通过黑名单
  5.  'Samsung 65" TV' 通过黑名单
  6.  mock 两个产品，取最低价
  7.  unavailable 产品不参与最低价计算
  8.  全 unavailable 返回 available=False
  9.  中文 "iPhone" 正确 passthrough/翻译
  10. 发出的 GraphQL query 含 predictiveSearch 字符串

  网络测试：
  11. 真实请求，price > 0 且 currencyCode == AUD
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.price_source import PriceSource
from tools.sources.goodguys_source import (
    GoodGuysSource,
    _is_blacklisted,
    _build_query,
)
from tools.sources.translate_utils import _translate


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_product(title: str, price: str, available: bool = True) -> dict:
    return {
        "title": title,
        "availableForSale": available,
        "handle": title.lower().replace(" ", "-"),
        "priceRange": {"minVariantPrice": {"amount": price, "currencyCode": "AUD"}},
        "variants": {
            "nodes": [{
                "price": {"amount": price, "currencyCode": "AUD"},
                "availableForSale": available,
            }]
        },
    }


def _mock_response(products: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"predictiveSearch": {"products": products}}
    }
    return mock


# ── 测试 1：继承关系 ──────────────────────────────────────────────────────────

def test_goodguys_is_price_source():
    src = GoodGuysSource()
    print(f"\n[1] isinstance(GoodGuysSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    assert src.get_platform_name() == "goodguys"
    print("[1] PASS")


# ── 测试 2-5：_is_blacklisted ─────────────────────────────────────────────────

def test_title_blacklist_filters_case():
    result = _is_blacklisted("iPhone 17 Case - Clear")
    print(f"\n[2] _is_blacklisted('iPhone 17 Case - Clear') = {result}")
    assert result is True
    print("[2] PASS — 'case' 被过滤")


def test_title_blacklist_filters_charger():
    result = _is_blacklisted("USB Charger 20W")
    print(f"\n[3] _is_blacklisted('USB Charger 20W') = {result}")
    assert result is True
    print("[3] PASS — 'charger' 被过滤")


def test_title_blacklist_allows_iphone():
    result = _is_blacklisted("Apple iPhone 17 256GB Black")
    print(f"\n[4] _is_blacklisted('Apple iPhone 17 256GB Black') = {result}")
    assert result is False
    print("[4] PASS — iPhone 本体通过")


def test_title_blacklist_allows_samsung_tv():
    result = _is_blacklisted('Samsung 65" 4K QLED TV')
    print(f"\n[5] _is_blacklisted('Samsung 65\" 4K QLED TV') = {result}")
    assert result is False
    print('[5] PASS — Samsung TV 通过')


# ── 测试 6：取最低价 ──────────────────────────────────────────────────────────

def test_parse_result_returns_lowest_price():
    products = [
        _make_product("Apple iPhone 17 256GB Black", "1308.0"),
        _make_product("Apple iPhone 17 128GB Black", "1099.0"),  # 更低
    ]
    with patch("tools.sources.goodguys_source.httpx.post", return_value=_mock_response(products)):
        src = GoodGuysSource()
        result = src.get_price("iphone")

    print(f"\n[6] price={result.price}, product={result.product_name!r}")
    assert result.available is True
    assert result.price == 1099.0, f"期望 1099.0，实际 {result.price}"
    assert "128GB" in result.product_name
    print("[6] PASS — 返回最低价 1099.0")


# ── 测试 7：unavailable 产品被跳过 ────────────────────────────────────────────

def test_parse_result_skips_unavailable():
    products = [
        _make_product("Apple iPhone 17 256GB Black", "1308.0", available=True),
        _make_product("Apple iPhone 17 Pro 256GB", "899.0", available=False),  # 更低但下架
    ]
    with patch("tools.sources.goodguys_source.httpx.post", return_value=_mock_response(products)):
        src = GoodGuysSource()
        result = src.get_price("iphone")

    print(f"\n[7] price={result.price}, product={result.product_name!r}")
    assert result.available is True
    assert result.price == 1308.0, f"下架产品不应参与比价，期望 1308.0，实际 {result.price}"
    assert "256GB Black" in result.product_name
    print("[7] PASS — 下架产品被跳过，返回在售的 1308.0")


# ── 测试 8：全部 unavailable → available=False ────────────────────────────────

def test_parse_result_all_unavailable():
    products = [
        _make_product("Apple iPhone 17 256GB", "1308.0", available=False),
        _make_product("Apple iPhone 17 512GB", "1797.0", available=False),
    ]
    with patch("tools.sources.goodguys_source.httpx.post", return_value=_mock_response(products)):
        src = GoodGuysSource()
        result = src.get_price("iphone")

    print(f"\n[8] available={result.available}, price={result.price}")
    assert result.available is False
    assert result.price is None
    print("[8] PASS — 全部下架，返回 available=False")


# ── 测试 9：中文翻译 ──────────────────────────────────────────────────────────

def test_translate_chinese():
    result = _translate("iPhone")
    print(f"\n[9] _translate('iPhone') = {result!r}")
    assert result == "iphone"
    print("[9] PASS — 'iPhone' → 'iphone'")


# ── 测试 10：GraphQL query 结构 ───────────────────────────────────────────────

def test_graphql_query_structure():
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"data": {"predictiveSearch": {"products": []}}}
        return mock

    with patch("tools.sources.goodguys_source.httpx.post", side_effect=fake_post):
        src = GoodGuysSource()
        src.get_price("iphone")

    query_str = captured.get("body", {}).get("query", "")
    print(f"\n[10] query 前80字符: {query_str[:80]!r}")
    assert "predictiveSearch" in query_str, f"query 里没有 predictiveSearch: {query_str[:200]}"
    assert "iphone" in query_str.lower()
    print("[10] PASS — query 包含 predictiveSearch 和搜索词")


# ── 网络测试 ──────────────────────────────────────────────────────────────────

@pytest.mark.network
def test_network_iphone_price():
    """测试 11：真实请求 iphone，返回 price > 0 且 currency=AUD。"""
    src = GoodGuysSource()
    result = src.get_price("iphone")

    print(f"\n[11] get_price('iphone') →")
    print(f"     available    = {result.available}")
    print(f"     price        = {result.price}")
    print(f"     currency     = {result.currency}")
    print(f"     product_name = {result.product_name!r}")
    print(f"     url          = {result.url!r}")

    assert result.available is True, f"期望 available=True，实际 {result.available}"
    assert result.price is not None and result.price > 0, f"期望 price>0，实际 {result.price}"
    assert result.currency == "AUD"
    assert result.unit == "each"
    assert result.platform == "goodguys"
    print(f"[11] PASS — {result.price} AUD each — {result.product_name}")
