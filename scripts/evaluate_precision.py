import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from tools.price import get_electronics_prices

# 评测集：query + 期望关键词 + 禁止关键词
TEST_CASES = [
    {
        "query": "MacBook Air M3",
        "expected": ["macbook"],
        "forbidden": ["case", "protector", "sleeve", "bag", "cover", "skin",
                      "hub", "dock", "stand", "cable", "adapter", "keyboard cover",
                      "内胆", "保护", "贴膜", "支架", "扩展"],
    },
    {
        "query": "iPhone 16",
        "expected": ["iphone", "16"],
        "forbidden": ["case", "protector", "screen", "cover", "cable",
                      "charger", "adapter", "lens", "film", "保护", "贴膜"],
    },
    {
        "query": "iPhone 16 Pro",
        "expected": ["iphone", "16 pro"],
        "forbidden": ["case", "protector", "screen", "cover", "cable",
                      "charger", "lens", "film", "保护", "贴膜"],
    },
    {
        "query": "Samsung S25",
        "expected": ["samsung", "s25"],
        "forbidden": ["case", "protector", "cover", "cable", "charger",
                      "保护", "贴膜"],
    },
    {
        "query": "Sony WH-1000XM5",
        "expected": ["wh-1000xm5"],
        "forbidden": ["case", "cable", "adapter", "pouch", "保护"],
    },
    {
        "query": "iPad Air",
        "expected": ["ipad"],
        "forbidden": ["case", "protector", "cover", "keyboard", "pencil",
                      "sleeve", "保护", "贴膜", "键盘"],
    },
    {
        "query": "AirPods Pro",
        "expected": ["airpods", "pro"],
        "forbidden": ["case", "cover", "cable", "tip", "保护"],
    },
    {
        "query": "Dell XPS 15",
        "expected": ["dell", "xps"],
        "forbidden": ["case", "sleeve", "adapter", "cable", "保护"],
    },
]

def is_relevant(name: str, expected: list[str], forbidden: list[str]) -> bool:
    name_lower = name.lower()
    has_expected = all(kw in name_lower for kw in expected)
    has_forbidden = any(kw in name_lower for kw in forbidden)
    return has_expected and not has_forbidden

def is_forbidden(name: str, forbidden: list[str]) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in forbidden)

async def evaluate():
    print("=" * 60)
    print("电子产品查价 Precision 评测报告")
    print("=" * 60)

    total_results = 0
    total_relevant = 0
    case_precisions = []

    for case in TEST_CASES:
        query = case["query"]
        results = await get_electronics_prices(query)

        if not results:
            print(f"\n【{query}】→ 无结果")
            case_precisions.append(None)
            continue

        relevant = [r for r in results if is_relevant(r["name"], case["expected"], case["forbidden"])]
        noise = [r for r in results if is_forbidden(r["name"], case["forbidden"])]
        precision = len(relevant) / len(results)
        case_precisions.append(precision)

        total_results += len(results)
        total_relevant += len(relevant)

        print(f"\n【{query}】")
        print(f"  返回 {len(results)} 条，相关 {len(relevant)} 条，噪音 {len(noise)} 条")
        print(f"  Precision: {precision:.0%}")

        if noise:
            print(f"  ⚠ 噪音结果：")
            for r in noise:
                print(f"    - [{r['platform']}] {r['name']} ${r['price']}")

        if relevant:
            print(f"  ✓ 相关结果：")
            for r in relevant:
                print(f"    - [{r['platform']}] {r['name']} ${r['price']}")

    # 汇总
    valid_cases = [p for p in case_precisions if p is not None]
    avg_precision = sum(valid_cases) / len(valid_cases) if valid_cases else 0
    overall_precision = total_relevant / total_results if total_results else 0

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"测试用例数：{len(TEST_CASES)}")
    print(f"有结果用例：{len(valid_cases)}")
    print(f"平均 Precision（各用例平均）：{avg_precision:.0%}")
    print(f"整体 Precision（所有结果合并）：{overall_precision:.0%}")

    # 评级
    if avg_precision >= 0.8:
        grade = "✅ 优秀（≥80%）"
    elif avg_precision >= 0.6:
        grade = "⚠ 一般（60-80%），建议扩充黑名单"
    else:
        grade = "❌ 较差（<60%），需要重点优化过滤逻辑"
    print(f"评级：{grade}")
    print("=" * 60)

asyncio.run(evaluate())
