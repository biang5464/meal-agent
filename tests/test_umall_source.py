"""
tests/test_umall_source.py

验证 UmallSource、_extract_unit_price、_translate、_is_fresh：

  1-5b. _extract_unit_price 单元测试（无网络）
  6.    UmallSource 是 PriceSource 子类
  7.    get_price("tomato") 真实网络请求 → available=True, price>0, unit="kg"
  8.    get_price("xyzabc不存在的商品") → available=False
  9.    get_price('番茄') 观察打印（无硬断言）
  11-13. _translate 映射测试
  14-15. 品类白名单测试
  16.   get_price("番茄") 中文查询 → available=True（验证映射生效）

测试 7、8、9、16 标记 @pytest.mark.network，CI 无网络时可跳过：
  pytest tests/test_umall_source.py -m "not network"

运行全部：.venv\\Scripts\\python.exe -m pytest tests/test_umall_source.py -v -s
"""

import pytest

from tools.price_source import PriceSource
from tools.sources.umall_source import UmallSource
from tools.sources.translate_utils import ZH_TO_EN, _translate
from tools.sources.unit_utils import _extract_unit_price, _is_fresh

# ── 单元测试 1-5b：_extract_unit_price ───────────────────────────────────────

def test_extract_500g():
    price, unit = _extract_unit_price("Cherry Tomato 500g", 3.00)
    print(f"\n[1] 500g $3.00 → ${price}/{unit}")
    assert unit == "kg"
    assert abs(price - 6.00) < 0.01, f"期望 6.00，实际 {price}"
    print("[1] PASS")


def test_extract_1kg():
    price, unit = _extract_unit_price("Pork Belly 1kg", 12.00)
    print(f"\n[2] 1kg $12.00 → ${price}/{unit}")
    assert unit == "kg"
    assert abs(price - 12.00) < 0.01, f"期望 12.00，实际 {price}"
    print("[2] PASS")


def test_extract_1_5kg():
    price, unit = _extract_unit_price("Beef Mince 1.5kg", 15.00)
    print(f"\n[3] 1.5kg $15.00 → ${price}/{unit}")
    assert unit == "kg"
    assert abs(price - 10.00) < 0.01, f"期望 10.00，实际 {price}"
    print("[3] PASS")


def test_extract_tuna_not_in_whitelist():
    # "tuna" 不在 FRESH_KEYWORDS → 白名单过滤，退回原价 pack
    price, unit = _extract_unit_price("Ayam Tuna 160g", 4.09)
    print(f"\n[4] Ayam Tuna 160g $4.09 → ${price}/{unit}  (tuna 不在白名单)")
    assert unit == "pack", f"期望 pack，实际 {unit}"
    assert abs(price - 4.09) < 0.01, f"期望 4.09，实际 {price}"
    print("[4] PASS — tuna 不在白名单，正确退回原价")


def test_extract_no_weight():
    price, unit = _extract_unit_price("Soy Sauce", 3.50)
    print(f"\n[5] 无克重 → ${price}/{unit}")
    assert unit == "pack"
    assert abs(price - 3.50) < 0.01, f"期望 3.50，实际 {price}"
    print("[5] PASS")


def test_extract_triggers_over_200():
    # "spice" 不在白名单，直接退回（顺带验证白名单过滤先于阈值）
    price, unit = _extract_unit_price("Spice Sample 1g", 0.30)
    print(f"\n[5b] Spice 1g $0.30 → ${price}/{unit}  (不在白名单，应退回)")
    assert unit == "pack"
    assert abs(price - 0.30) < 0.01
    print("[5b] PASS")


# ── 测试 6：isinstance 继承关系 ───────────────────────────────────────────────

def test_umall_is_price_source():
    src = UmallSource()
    print(f"\n[6] isinstance(UmallSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    assert src.get_platform_name() == "umall"
    print("[6] PASS")


# ── 测试 7-8：真实网络请求 ───────────────────────────────────────────────────

@pytest.mark.network
def test_get_price_tomato_real():
    src = UmallSource()
    result = src.get_price("tomato")
    d = result.to_dict()

    print(f"\n[7] get_price('tomato') →")
    for k, v in d.items():
        print(f"    {k}: {v!r}")

    assert result.available is True, f"期望 available=True，实际 {result.available}"
    assert result.price is not None and result.price > 0, f"price 应 > 0，实际 {result.price}"
    assert result.unit in ("kg", "pack"), f"unit 应为 kg 或 pack，实际 {result.unit!r}"
    assert result.currency == "AUD"
    assert result.platform == "umall"
    assert result.url and result.url.startswith("https://www.umall.com.au")
    print(f"[7] PASS — {result.price} {result.unit} / AUD")


@pytest.mark.network
def test_get_price_nonexistent():
    src = UmallSource()
    result = src.get_price("xyzabc不存在的商品")
    print(f"\n[8] get_price('xyzabc不存在的商品') → available={result.available}")
    assert result.available is False
    assert result.price is None
    print("[8] PASS")


@pytest.mark.network
def test_get_price_fanqie_print():
    """打印 get_price('番茄') 的真实返回（历史对比用，非硬断言）。"""
    src = UmallSource()
    result = src.get_price("番茄")
    print(f"\n[9/观察] get_price('番茄') →")
    for k, v in result.to_dict().items():
        print(f"    {k}: {v!r}")


# ── 测试 11-13：_translate 中英映射 ──────────────────────────────────────────

def test_translate_chinese():
    result = _translate("番茄")
    print(f"\n[11] _translate('番茄') → {result!r}")
    assert result == "tomato", f"期望 'tomato'，实际 {result!r}"
    print("[11] PASS")


def test_translate_english_passthrough():
    result = _translate("tomato")
    print(f"\n[12] _translate('tomato') → {result!r}")
    assert result == "tomato", f"英文应原样返回，实际 {result!r}"
    print("[12] PASS")


def test_translate_unknown_passthrough():
    result = _translate("未知食材xyz")
    print(f"\n[13] _translate('未知食材xyz') → {result!r}")
    assert result == "未知食材xyz", f"未知词应原样返回，实际 {result!r}"
    print("[13] PASS")


# ── 测试 14-15：品类白名单 ─────────────────────────────────────────────────────

def test_whitelist_tuna_pack():
    # tuna 不在 FRESH_KEYWORDS → pack
    price, unit = _extract_unit_price("Ayam Tuna 160g", 4.09)
    print(f"\n[14] Ayam Tuna 160g $4.09 → ${price}/{unit}")
    assert unit == "pack"
    assert abs(price - 4.09) < 0.01
    print("[14] PASS — tuna 不在白名单")


def test_whitelist_tomato_kg():
    # tomato 在 FRESH_KEYWORDS → 换算 /kg
    price, unit = _extract_unit_price("Fresh Vine Tomatoes 1kg", 6.29)
    print(f"\n[15] Fresh Vine Tomatoes 1kg $6.29 → ${price}/{unit}")
    assert unit == "kg"
    assert abs(price - 6.29) < 0.01, f"期望 6.29，实际 {price}"
    print("[15] PASS — tomato 在白名单，1kg 换算正确")


# ── 测试 16：中文查询真实网络（验证映射生效）─────────────────────────────────

@pytest.mark.network
def test_get_price_chinese_query():
    src = UmallSource()
    result = src.get_price("番茄")
    print(f"\n[16] get_price('番茄') →")
    for k, v in result.to_dict().items():
        print(f"    {k}: {v!r}")

    assert result.available is True, (
        f"期望 available=True（中英映射应把'番茄'转为'tomato'），"
        f"实际 {result.available}"
    )
    assert result.price is not None and result.price > 0
    print(f"[16] PASS — 中文查询映射生效，{result.price} {result.unit} / AUD")
