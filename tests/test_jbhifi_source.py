"""
tests/test_jbhifi_source.py

JBHiFiSource 单元测试（6个）+ 网络测试（4个）：

  单元测试（无网络）：
  1. JBHiFiSource 是 PriceSource 子类
  2. get_platform_name() == "jbhifi"
  3. _translate("手机") == "phone"
  4. _translate("iPhone") == "iphone"
  5. _translate("电脑") == "laptop"
  6. _translate("未知词xyz") == "未知词xyz"

  网络测试（真实请求）：
  7. get_price("iphone")   → available=True, price>0, unit="each"
  8. get_price("laptop")   → available=True, price>0
  9. get_price("手机")     → available=True（验证中文翻译生效）
  10. get_price("xyzabc不存在") → available=False

运行：.venv\\Scripts\\python.exe -m pytest tests/test_jbhifi_source.py -v -s
网络测试：.venv\\Scripts\\python.exe -m pytest tests/test_jbhifi_source.py -v -s -m network
"""

import pytest

from tools.price_source import PriceSource
from tools.sources.jbhifi_source import JBHiFiSource
from tools.sources.translate_utils import _translate


# ── 单元测试 ──────────────────────────────────────────────────────────────────

def test_jbhifi_is_price_source():
    """测试 1：JBHiFiSource 是 PriceSource 的子类。"""
    src = JBHiFiSource()
    print(f"\n[1] isinstance(JBHiFiSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    print("[1] PASS")


def test_jbhifi_platform_name():
    """测试 2：get_platform_name() 返回 'jbhifi'。"""
    src = JBHiFiSource()
    name = src.get_platform_name()
    print(f"\n[2] get_platform_name() = {name!r}")
    assert name == "jbhifi"
    print("[2] PASS")


def test_translate_phone():
    """测试 3：_translate('手机') → 'phone'。"""
    result = _translate("手机")
    print(f"\n[3] _translate('手机') = {result!r}")
    assert result == "phone"
    print("[3] PASS")


def test_translate_iphone():
    """测试 4：_translate('iPhone') → 'iphone'。"""
    result = _translate("iPhone")
    print(f"\n[4] _translate('iPhone') = {result!r}")
    assert result == "iphone"
    print("[4] PASS")


def test_translate_laptop():
    """测试 5：_translate('电脑') → 'laptop'。"""
    result = _translate("电脑")
    print(f"\n[5] _translate('电脑') = {result!r}")
    assert result == "laptop"
    print("[5] PASS")


def test_translate_unknown():
    """测试 6：_translate('未知词xyz') 原样返回。"""
    result = _translate("未知词xyz")
    print(f"\n[6] _translate('未知词xyz') = {result!r}")
    assert result == "未知词xyz"
    print("[6] PASS")


# ── 网络测试 ──────────────────────────────────────────────────────────────────

@pytest.mark.network
def test_get_price_iphone():
    """测试 7：get_price('iphone') → available=True, price>0, unit='each'。"""
    src = JBHiFiSource()
    result = src.get_price("iphone")

    print(f"\n[7] get_price('iphone') →")
    print(f"    available    = {result.available}")
    print(f"    price        = {result.price}")
    print(f"    unit         = {result.unit!r}")
    print(f"    product_name = {result.product_name!r}")
    print(f"    url          = {result.url!r}")

    assert result.available is True, f"期望 available=True，实际 {result.available}"
    assert result.price is not None and result.price > 0, f"期望 price>0，实际 {result.price}"
    assert result.unit == "each", f"期望 unit='each'，实际 {result.unit!r}"
    assert result.platform == "jbhifi"
    assert result.currency == "AUD"
    print(f"[7] PASS — {result.price} AUD each — {result.product_name}")


@pytest.mark.network
def test_get_price_laptop():
    """测试 8：get_price('laptop') → available=True, price>0。"""
    src = JBHiFiSource()
    result = src.get_price("laptop")

    print(f"\n[8] get_price('laptop') →")
    print(f"    available    = {result.available}")
    print(f"    price        = {result.price}")
    print(f"    product_name = {result.product_name!r}")

    assert result.available is True, f"期望 available=True，实际 {result.available}"
    assert result.price is not None and result.price > 0
    assert result.unit == "each"
    print(f"[8] PASS — {result.price} AUD each — {result.product_name}")


@pytest.mark.network
def test_get_price_chinese_phone():
    """测试 9：get_price('手机') → available=True（验证中文→phone 翻译生效）。"""
    src = JBHiFiSource()
    result = src.get_price("手机")

    print(f"\n[9] get_price('手机') →")
    print(f"    available    = {result.available}")
    print(f"    price        = {result.price}")
    print(f"    product_name = {result.product_name!r}")

    assert result.available is True, (
        f"期望 available=True（手机→phone 应有结果），实际 {result.available}"
    )
    assert result.price is not None and result.price > 0
    print(f"[9] PASS — 手机→phone 翻译生效，{result.price} AUD — {result.product_name}")


@pytest.mark.network
def test_get_price_nonexistent():
    """测试 10：get_price('xyzabc不存在') → available=False。"""
    src = JBHiFiSource()
    result = src.get_price("xyzabc不存在")

    print(f"\n[10] get_price('xyzabc不存在') →")
    print(f"    available = {result.available}")
    print(f"    price     = {result.price}")

    assert result.available is False
    assert result.price is None
    print("[10] PASS — 不存在的商品返回 available=False")


# ── 品类过滤测试 ──────────────────────────────────────────────────────────────

@pytest.mark.network
def test_phone_no_lanyard():
    """测试 11：get_price('手机') → product_name 不包含 'Lanyard'（配件被过滤）。"""
    src = JBHiFiSource()
    result = src.get_price("手机")

    print(f"\n[11] get_price('手机') → product_name={result.product_name!r}")

    assert result.available is True
    assert result.product_name is not None
    assert "Lanyard" not in result.product_name, (
        f"配件未被过滤，返回了 {result.product_name!r}"
    )
    print(f"[11] PASS — 配件已被过滤，返回 {result.product_name!r}")


@pytest.mark.network
def test_phone_price_above_50():
    """测试 12：get_price('手机') → price > 50（phone 本体，不是 $15 配件）。"""
    src = JBHiFiSource()
    result = src.get_price("手机")

    print(f"\n[12] get_price('手机') → price={result.price}, product_name={result.product_name!r}")

    assert result.available is True
    assert result.price is not None and result.price > 50, (
        f"期望 price>50（phone 本体），实际 price={result.price}（可能仍是配件）"
    )
    print(f"[12] PASS — price={result.price} > 50，是 phone 本体而非配件")


@pytest.mark.network
def test_laptop_price_above_200():
    """测试 13：get_price('laptop') → price > 200（笔记本本体，不是充电器）。"""
    src = JBHiFiSource()
    result = src.get_price("laptop")

    print(f"\n[13] get_price('laptop') → price={result.price}, product_name={result.product_name!r}")

    assert result.available is True
    assert result.price is not None and result.price > 200, (
        f"期望 price>200（笔记本本体），实际 price={result.price}（可能仍是配件）"
    )
    print(f"[13] PASS — price={result.price} > 200，是笔记本本体")


@pytest.mark.network
def test_charger_not_filtered():
    """测试 14：get_price('charger') → available=True（充电器查询不过滤，正常返回）。"""
    src = JBHiFiSource()
    result = src.get_price("charger")

    print(f"\n[14] get_price('charger') → available={result.available}, price={result.price}, product_name={result.product_name!r}")

    assert result.available is True, f"期望 available=True，实际 {result.available}"
    assert result.price is not None and result.price > 0
    print(f"[14] PASS — 充电器查询正常返回 {result.price} AUD — {result.product_name}")
