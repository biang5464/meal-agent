"""
tests/test_price_source.py

验证 PriceSource 抽象层和 MockSource / WoolworthsSource 实现：

  1. MockSource 查已知商品 → available=True，price 不为 None，to_dict() 包含所有字段
  2. MockSource 查未知商品 → available=False，price 为 None
  3. get_prices 批量查询 → 返回列表长度等于输入长度
  4. WoolworthsSource 和 MockSource 都是 PriceSource 的子类（isinstance 断言）

  17. get_ingredient_price 返回包含 best_price/best_platform/unit/all_results（mock 网络）
  18. all_results 长度 == 3
  19. all_results 里 coles 的 available == False
  20. best_platform 是 all_results 里 price 最低的那个平台
  21. get_ingredient_price("tomato") 真实网络 → best_price 不为 None，平台在白名单内

运行：.venv\\Scripts\\python.exe -m pytest tests/test_price_source.py -v -s
"""

import pytest
from unittest.mock import patch

from tools.price_source import PriceResult, PriceSource
from tools.sources.jbhifi_source import JBHiFiSource
from tools.sources.mock_source import MOCK_DATA, MockSource
from tools.sources.woolworths_source import WoolworthsSource


# ---------- 辅助：构造 PriceResult ----------

def _make_pr(platform: str, price, unit: str, available: bool) -> PriceResult:
    return PriceResult(
        platform=platform,
        item_name="test_item",
        price=price,
        unit=unit,
        currency="AUD",
        available=available,
        timestamp="2026-01-01T00:00:00",
        url=f"https://{platform}.example.com" if available else None,
    )

# ── 测试1：MockSource 查已知商品 ─────────────────────────────────────────────

def test_mock_known_item_available():
    src = MockSource()
    known = next(iter(MOCK_DATA))          # 取 MOCK_DATA 第一个 key
    result = src.get_price(known)

    print(f"\n[1] get_price({known!r}) → {result.to_dict()}")

    assert isinstance(result, PriceResult)
    assert result.available is True
    assert result.price is not None
    assert result.price > 0
    assert result.platform == "mock"
    assert result.item_name == known
    print("[1] PASS")


def test_mock_known_item_to_dict_fields():
    src = MockSource()
    known = next(iter(MOCK_DATA))
    d = src.get_price(known).to_dict()

    required_keys = {"platform", "item_name", "price", "unit", "currency",
                     "available", "timestamp", "url"}
    missing = required_keys - d.keys()

    print(f"\n[2] to_dict() keys: {set(d.keys())}")
    assert not missing, f"to_dict() 缺少字段: {missing}"
    print("[2] PASS — to_dict() 包含全部字段")


# ── 测试2：MockSource 查未知商品 ─────────────────────────────────────────────

def test_mock_unknown_item_unavailable():
    src = MockSource()
    result = src.get_price("根本不存在的商品XYZ")

    print(f"\n[3] get_price('XYZ') → available={result.available}, price={result.price}")

    assert result.available is False
    assert result.price is None
    assert result.item_name == "根本不存在的商品XYZ"
    print("[3] PASS")


# ── 测试3：get_prices 批量查询长度一致 ──────────────────────────────────────

def test_mock_get_prices_length():
    src = MockSource()
    items = ["番茄", "鸡蛋", "不存在商品A", "不存在商品B"]
    results = src.get_prices(items)

    print(f"\n[4] get_prices({items}) → {len(results)} 条结果")
    for r in results:
        print(f"    {r.item_name}: available={r.available}, price={r.price}")

    assert len(results) == len(items), (
        f"期望 {len(items)} 条，实际 {len(results)} 条"
    )
    # 前两个是已知商品，应 available
    assert results[0].available is True
    assert results[1].available is True
    # 后两个未知，应 unavailable
    assert results[2].available is False
    assert results[3].available is False
    print("[4] PASS")


# ── 测试4：isinstance 继承关系断言 ───────────────────────────────────────────

def test_mock_source_is_price_source():
    src = MockSource()
    print(f"\n[5] isinstance(MockSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    print("[5] PASS")


def test_woolworths_source_is_price_source():
    src = WoolworthsSource()
    print(f"\n[6] isinstance(WoolworthsSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    print("[6] PASS")


def test_platform_names():
    assert MockSource().get_platform_name() == "mock"
    assert WoolworthsSource().get_platform_name() == "woolworths"
    print("\n[7] platform names 正确 PASS")


# ── 测试 17-20：get_ingredient_price 多平台结构验证（mock 网络）────────────────

@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_get_ingredient_price_fields(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 17：返回 dict 包含 best_price/best_platform/unit/all_results。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    print(f"\n[17] get_ingredient_price 结构: {list(result.keys())}")
    for field in ("best_price", "best_platform", "unit", "all_results"):
        assert field in result, f"缺少字段 {field!r}"
    print("[17] PASS — 四个必要字段均存在")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_all_results_length(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 18：all_results 长度 == 5（ww + umall + jbhifi + goodguys + coles 占位）。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    n = len(result["all_results"])
    print(f"\n[18] all_results 长度: {n}")
    assert n == 5, f"期望 5，实际 {n}"
    platforms = [r["platform"] for r in result["all_results"]]
    assert "woolworths" in platforms
    assert "umall" in platforms
    assert "jbhifi" in platforms
    assert "goodguys" in platforms
    assert "coles" in platforms
    print("[18] PASS")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_coles_always_unavailable(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 19：all_results 里 coles 的 available 始终为 False。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    coles = next(r for r in result["all_results"] if r["platform"] == "coles")
    print(f"\n[19] coles entry: {coles}")
    assert coles["available"] is False
    assert coles["price"] is None
    print("[19] PASS — coles 占位 available=False")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_best_platform_is_cheapest(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 20：best_platform 是 all_results 里价格最低的那个。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    available = [r for r in result["all_results"] if r["available"]]
    cheapest = min(available, key=lambda r: r["price"])

    print(f"\n[20] best_platform={result['best_platform']!r}, cheapest={cheapest['platform']!r}")
    assert result["best_platform"] == cheapest["platform"]
    assert result["best_price"] == cheapest["price"]
    print("[20] PASS — best_platform 确实是价格最低平台")


# ── 测试 21：真实网络请求 ─────────────────────────────────────────────────────

@pytest.mark.network
def test_get_ingredient_price_real_network():
    """测试 21：真实请求 tomato → all_results 有数据，comparable 字段存在且结构正确。"""
    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    print(f"\n[21] get_ingredient_price('tomato') →")
    print(f"    comparable:    {result['comparable']}")
    print(f"    best_price:    {result['best_price']}")
    print(f"    best_platform: {result['best_platform']}")
    print(f"    note:          {result.get('note')}")
    for r in result["all_results"]:
        print(f"    [{r['platform']}] price={r['price']} unit={r['unit']} available={r['available']}")

    # 结构断言
    assert "comparable" in result
    assert isinstance(result["all_results"], list) and len(result["all_results"]) == 5

    # 至少一个平台有结果
    available = [r for r in result["all_results"] if r["available"]]
    assert len(available) >= 1, "至少一个平台应有数据"

    if result["comparable"]:
        assert result["best_price"] is not None and result["best_price"] > 0
        assert result["best_platform"] in ("woolworths", "umall", "jbhifi", "goodguys")
        assert result.get("note") is None
        print(f"[21] PASS — comparable=True, best={result['best_price']} {result['unit']} @ {result['best_platform']}")
    else:
        assert result["best_price"] is None
        assert result["best_platform"] is None
        assert "note" in result
        print(f"[21] PASS — comparable=False（单位不同），各平台原始价格见 all_results")


# ── 测试 22-24：unit_utils 共用层与跨平台单位对齐 ─────────────────────────────

def test_unit_utils_importable():
    """测试 22：unit_utils 公开符号可正常导入。"""
    from tools.sources.unit_utils import FRESH_KEYWORDS, _extract_unit_price, _is_fresh

    assert isinstance(FRESH_KEYWORDS, list) and len(FRESH_KEYWORDS) > 0
    price, unit = _extract_unit_price("Tomato 500g", 2.50)
    assert unit == "kg"
    assert abs(price - 5.00) < 0.01
    price2, unit2 = _extract_unit_price("Coca Cola 1.25L", 3.50)
    assert unit2 == "pack"
    print("\n[22] PASS — unit_utils 导入正常，换算逻辑正确")


@pytest.mark.network
def test_woolworths_unit_is_kg_or_pack():
    """测试 23：WoolworthsSource.get_price('tomato') 返回 unit 不为空字符串。"""
    src = WoolworthsSource()
    result = src.get_price("tomato")

    print(f"\n[23] WoolworthsSource.get_price('tomato') → price={result.price} unit={result.unit!r} available={result.available}")

    if result.available:
        assert result.unit in ("kg", "pack"), (
            f"unit 应为 'kg' 或 'pack'，实际 {result.unit!r}"
        )
        assert result.price is not None and result.price > 0
    else:
        print("[23] SKIP — Woolworths 无结果（网络或封锁），跳过断言")

    print("[23] PASS")


@pytest.mark.network
def test_cross_platform_unit_alignment():  # test 24
    """测试 24：WW 和 Umall 同时有结果时，unit 字段均为 'kg' 或 'pack'（不为空）。"""
    from tools.sources.umall_source import UmallSource
    from tools.sources.woolworths_source import WoolworthsSource

    ww = WoolworthsSource().get_price("tomato")
    um = UmallSource().get_price("番茄")

    print(f"\n[24] WW   → price={ww.price} unit={ww.unit!r} available={ww.available}")
    print(f"[24] Umall → price={um.price} unit={um.unit!r} available={um.available}")

    for result, name in [(ww, "woolworths"), (um, "umall")]:
        if result.available:
            assert result.unit in ("kg", "pack"), (
                f"{name} unit 应为 'kg' 或 'pack'，实际 {result.unit!r}"
            )

    if ww.available and um.available:
        print(f"[24] 两平台均有结果，单位均已规范化为 kg/pack")

    print("[24] PASS")


# ── 测试 25-27：comparable 字段逻辑 ───────────────────────────────────────────

@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_comparable_false_when_units_differ(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 25：WW=pack、Umall=kg → comparable=False，best 为 None，note 存在。"""
    ww_pr = _make_pr("woolworths", 0.54, "pack", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "番茄"})

    print(f"\n[25] comparable={result['comparable']!r}  best_price={result['best_price']!r}")
    print(f"     note={result.get('note')!r}")

    assert result["comparable"] is False
    assert result["best_price"] is None
    assert result["best_platform"] is None
    assert "note" in result and result["note"] is not None
    print("[25] PASS — comparable=False 时 best=None，note 字段存在")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_comparable_true_when_units_match(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 26：WW=kg、Umall=kg → comparable=True，best_price 是较低那个，note 为 None。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "tomato"})

    print(f"\n[26] comparable={result['comparable']!r}  best_price={result['best_price']!r}  best_platform={result['best_platform']!r}")

    assert result["comparable"] is True
    assert result["best_price"] == 6.29
    assert result["best_platform"] == "umall"
    assert result.get("note") is None
    print("[26] PASS — comparable=True，best 是单价更低的 umall")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_comparable_true_single_platform(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 27：只有一个平台 available → comparable=True，返回该平台价格。"""
    ww_pr = _make_pr("woolworths", 5.00, "kg", True)
    umall_pr = _make_pr("umall", None, "", False)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", None, "", False)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "chicken"})

    print(f"\n[27] comparable={result['comparable']!r}  best_price={result['best_price']!r}  best_platform={result['best_platform']!r}")

    assert result["comparable"] is True
    assert result["best_price"] == 5.00
    assert result["best_platform"] == "woolworths"
    assert result.get("note") is None
    print("[27] PASS — 单平台 available 时 comparable=True")


# ── 测试 28-30：translate_utils 公共化 + WW 中文查询 ─────────────────────────

def test_translate_utils_importable():
    """测试 28：_translate 从 translate_utils 可正常导入，映射行为正确。"""
    from tools.sources.translate_utils import _translate

    assert _translate("番茄") == "tomato"
    assert _translate("鸡腿") == "chicken leg"
    assert _translate("tomato") == "tomato"       # 英文直接返回
    assert _translate("未知食材xyz") == "未知食材xyz"  # 未知词直接返回
    print("\n[28] PASS — translate_utils 导入正常，映射行为符合预期")


@pytest.mark.network
def test_woolworths_chinese_query():
    """测试 29：WoolworthsSource.get_price('番茄') → available=True（中文查询生效）。"""
    src = WoolworthsSource()
    result = src.get_price("番茄")

    print(f"\n[29] WoolworthsSource.get_price('番茄') → price={result.price} unit={result.unit!r} available={result.available}")

    assert result.available is True, (
        f"期望 available=True，实际 available={result.available}（中文翻译后仍无结果，请检查网络或 WW 响应）"
    )
    assert result.price is not None and result.price > 0
    assert result.unit in ("kg", "pack")
    print(f"[29] PASS — 番茄→tomato 翻译生效，WW 返回 {result.price} {result.unit}")


@pytest.mark.network
def test_both_platforms_available_fanqie():
    """测试 30：get_ingredient_price('番茄') → WW 和 Umall 都 available=True。"""
    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "番茄"})

    print(f"\n[30] get_ingredient_price('番茄') →")
    print(f"    comparable:    {result['comparable']}")
    print(f"    best_price:    {result['best_price']}")
    print(f"    best_platform: {result['best_platform']}")
    print(f"    note:          {result.get('note')}")
    for r in result["all_results"]:
        print(f"    [{r['platform']}] price={r['price']} unit={r['unit']!r} available={r['available']}")

    ww = next(r for r in result["all_results"] if r["platform"] == "woolworths")
    um = next(r for r in result["all_results"] if r["platform"] == "umall")

    assert ww["available"] is True, "woolworths 应 available=True"
    assert um["available"] is True, "umall 应 available=True"
    print(f"[30] PASS — 两平台均 available，comparable={result['comparable']}")


# ── 新增：JBHiFiSource 集成测试 ───────────────────────────────────────────────

def test_jbhifi_source_is_price_source():
    """测试 31：JBHiFiSource 是 PriceSource 的子类。"""
    src = JBHiFiSource()
    print(f"\n[31] isinstance(JBHiFiSource(), PriceSource) = {isinstance(src, PriceSource)}")
    assert isinstance(src, PriceSource)
    assert src.get_platform_name() == "jbhifi"
    print("[31] PASS")


@patch("tools.price._goodguys_source")
@patch("tools.price._jbhifi_source")
@patch("tools.price._umall_source")
@patch("tools.price._ww_source")
def test_all_results_contains_jbhifi(mock_ww, mock_umall, mock_jbhifi, mock_goodguys):
    """测试 32：all_results 里包含 jbhifi 平台条目。"""
    ww_pr = _make_pr("woolworths", 7.50, "kg", True)
    umall_pr = _make_pr("umall", 6.29, "kg", True)
    mock_ww.get_price.return_value = ww_pr
    mock_ww.get_price_strict.return_value = ww_pr
    mock_umall.get_price.return_value = umall_pr
    mock_umall.get_price_strict.return_value = umall_pr
    mock_jbhifi.get_price.return_value = _make_pr("jbhifi", 899.00, "each", True)
    mock_goodguys.get_price.return_value = _make_pr("goodguys", None, "", False)

    from tools.price import get_ingredient_price
    result = get_ingredient_price.invoke({"ingredient": "iphone"})

    platforms = [r["platform"] for r in result["all_results"]]
    print(f"\n[32] all_results 平台列表: {platforms}")
    assert "jbhifi" in platforms, "all_results 应包含 jbhifi 平台"

    jb = next(r for r in result["all_results"] if r["platform"] == "jbhifi")
    print(f"[32] jbhifi entry: {jb}")
    assert jb["available"] is True
    assert jb["price"] == 899.00
    print("[32] PASS — all_results 包含 jbhifi 平台条目，数据正确")
