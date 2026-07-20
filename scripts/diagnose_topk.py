import sys
sys.path.insert(0, '.')
from tools.nutrition import search_nutrition, search_food_safety

def diagnose(label, search_fn, queries, n_results=5):
    print(f"\n{'='*60}")
    print(f"【{label}】top_k 诊断报告")
    print(f"{'='*60}")
    for query in queries:
        print(f"\n问题：{query}")
        print("-" * 40)
        results = search_fn(query, n_results=n_results)
        for i, r in enumerate(results):
            marker = "✓" if r['distance'] < 0.35 else ("△" if r['distance'] < 0.55 else "✗")
            print(f"  [{i+1}] {marker} distance={r['distance']:.3f} | {r['source']} | {r['content'][:60]}...")

        distances = [r['distance'] for r in results]
        useful = sum(1 for d in distances if d < 0.35)
        print(f"\n  → 有效chunk数（distance<0.35）: {useful}/{n_results}")
        if len(distances) >= 2:
            jump = distances[-1] - distances[0]
            print(f"  → distance跨度: {distances[0]:.3f} ~ {distances[-1]:.3f}（跨度{jump:.3f}）")
        if useful <= 1:
            print("  ⚠ 建议：相关内容太少，检查文档是否正确入库")
        elif useful >= 4:
            print("  ⚠ 建议：相关chunk较多，top_k可调大到4-5")
        else:
            print("  ✓ 建议：top_k=3 基本合理")

# nutrition 诊断
diagnose(
    label="nutrition知识库",
    search_fn=search_nutrition,
    queries=[
        "糖尿病能吃白米饭吗",
        "高血压患者能吃什么",
        "痛风发作期间饮食注意事项",
    ]
)

# food_safety 诊断
diagnose(
    label="food_safety知识库",
    search_fn=search_food_safety,
    queries=[
        "我在吃头孢能喝酒吗",
        "菠菜和豆腐能一起吃吗",
        "隔夜菜能不能吃",
        "四季豆没煮熟会怎样",
    ]
)

print(f"\n{'='*60}")
print("诊断完成")
print(f"{'='*60}\n")
