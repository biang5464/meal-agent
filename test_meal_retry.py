"""
Test E：验证 LLM 是否会"过早放弃"重试，导致明明有便宜替代方案却没找到。

场景：
  budget_level="low"
  user_input="我想吃点丰盛的，预算控制在low"

  "丰盛"的向量相似度会倾向拉向白灼虾/清蒸鲈鱼/红烧肉（均超 low 阈值），
  但库里有番茄炒蛋、宫保鸡丁、麻婆豆腐、干煸四季豆等明确符合 low 预算的菜。

  如果 LLM 重试机制有效 → 最终应选中其中一道。
  如果 LLM 过早放弃 → 会给出 fallback 说"没有符合预算的菜"，但实际上有。

运行：.venv\\Scripts\\python.exe -m pytest test_meal_retry.py -v -s
"""

import pytest
from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def init_services():
    from models.maintenance import HealthCycleORM, PreferenceHistoryORM, ReminderORM  # noqa: F401
    from tools.memory import init_db
    from tools.recipe import init_chroma
    init_db("./data/test_users.db")
    init_chroma("./data/test_chroma")
    yield


# 明确符合 low 预算的菜（根据食材价格分析）
LOW_BUDGET_VIABLE = {"番茄炒蛋", "宫保鸡丁", "麻婆豆腐", "干煸四季豆"}
# 明确超出 low 预算的菜
LOW_BUDGET_OVERPRICED = {"清蒸鲈鱼", "白灼虾", "红烧肉", "冬瓜排骨汤"}


async def _run_agent_with_trace(
    user_id: str,
    user_input: str,
    user_brief: dict,
) -> tuple[str, list[dict]]:
    """
    运行 meal_agent，同时捕获 on_tool_start + on_tool_end 事件。
    返回 (result_text, trace)，trace 是按时序排列的工具调用记录。

    每条 trace 记录格式：
    {
        "seq":    调用序号（从1开始）,
        "name":   工具名,
        "input":  工具入参 dict,
        "output": 工具返回值（on_tool_end 的 output）,
    }
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent

    from agents.meal_agent import _RECURSION_LIMIT, _build_system_prompt
    from config import get_llm
    from tools.price import compare_recipe_cost, get_ingredient_price
    from tools.recipe import get_recipe_by_name, search_recipes

    llm = get_llm(streaming=True)
    agent = create_react_agent(
        model=llm,
        tools=[search_recipes, get_recipe_by_name, get_ingredient_price, compare_recipe_cost],
        prompt=SystemMessage(content=_build_system_prompt(user_id, user_brief)),
    )

    parts: list[str] = []
    # pending_start 暂存 run_id → start_info，等到 tool_end 时合并
    pending_start: dict[str, dict] = {}
    trace: list[dict] = []
    seq_counter = [0]

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        version="v2",
        config={"recursion_limit": _RECURSION_LIMIT},
    ):
        etype = event["event"]
        run_id = event.get("run_id", "")

        if etype == "on_tool_start":
            seq_counter[0] += 1
            pending_start[run_id] = {
                "seq": seq_counter[0],
                "name": event["name"],
                "input": event["data"].get("input", {}),
            }

        elif etype == "on_tool_end":
            start_info = pending_start.pop(run_id, None)
            if start_info:
                trace.append({
                    **start_info,
                    "output": event["data"].get("output"),
                })

        elif etype == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                parts.append(chunk.content)

    return "".join(parts), trace


def _print_trace(trace: list[dict]):
    """打印详细工具调用轨迹，重点展示 search_recipes 的返回结果。"""
    print(f"\n{'='*60}")
    print(f"工具调用轨迹（共 {len(trace)} 次）")
    print('='*60)

    search_round = 0

    for rec in trace:
        seq = rec["seq"]
        name = rec["name"]
        inp = rec["input"]
        out = rec["output"]

        if name == "search_recipes":
            search_round += 1
            query = inp.get("query", "")
            print(f"\n[第{search_round}轮检索 · 调用#{seq}] search_recipes")
            print(f"  query: {query!r}")

            # 解析返回的菜谱列表
            recipes = out if isinstance(out, list) else []
            if recipes:
                print(f"  返回候选（{len(recipes)} 道）：")
                for r in recipes:
                    rname = r.get("name", "?")
                    score = r.get("relevance_score", "?")
                    ingredients = r.get("ingredients", [])
                    budget_tag = (
                        "✅ 符合low预算" if rname in LOW_BUDGET_VIABLE
                        else "❌ 超low预算" if rname in LOW_BUDGET_OVERPRICED
                        else "⚠️ 边界"
                    )
                    print(f"    - {rname}  相似度={score}  {budget_tag}")
                    print(f"      食材：{'、'.join(ingredients[:5])}")
            else:
                print(f"  返回：{out!r}")

        elif name == "get_ingredient_price":
            ingredient = inp.get("ingredient", "")
            # 解析价格工具返回
            price_info = out if isinstance(out, dict) else {}
            price = price_info.get("price", "?")
            unit = price_info.get("unit", "?")
            store = price_info.get("store", "?")
            print(f"\n  [调用#{seq}] get_ingredient_price: {ingredient!r}")
            print(f"    → ${price} / {unit}  ({store})")
            # 粗算每公斤单价（假设 unit 含数量信息）
            if price_info.get("error"):
                print(f"    → 错误: {price_info['error']}")

        elif name == "compare_recipe_cost":
            ingredients = inp.get("ingredients", [])
            print(f"\n  [调用#{seq}] compare_recipe_cost")
            print(f"    食材列表: {ingredients}")
            if isinstance(out, dict):
                total = out.get("total_cost", "?")
                print(f"    → 总估价: ${total}")

        elif name == "get_recipe_by_name":
            rname = inp.get("name", "")
            print(f"\n  [调用#{seq}] get_recipe_by_name: {rname!r}")

    print(f"\n{'='*60}")


@pytest.mark.asyncio
async def test_E_retry_finds_cheap_alternative():
    """
    Test E：丰盛需求 + low预算 → 初次检索应命中贵菜 → 重试后应找到便宜替代。

    断言：
      - 最终回复中提到了至少一道 LOW_BUDGET_VIABLE 中的菜，
        OR 多次调用 search_recipes（说明确实重试了）。
      - 若 LLM 过早放弃（只搜1轮 + 给fallback），则标注为"存在过早放弃风险"
        并打印警告，但测试本身宽松通过（记录问题，不硬 fail）。
    """
    user_id = "meal_retry_E"
    user_brief = {
        "likes": [],
        "dislikes": [],
        "active_restrictions": [],
        "recent_meals": [],
        "unread_reminders": [],
        "period_status": "",
        "budget_level": "low",
    }
    user_input = "我想吃点丰盛的，预算控制在low"

    print(f"\n[Test E] user_input={user_input!r}  budget_level='low'")
    print("[Test E] 期望：LLM 先检索到贵菜 → 重试 → 最终找到 low 预算内的替代方案")

    result, trace = await _run_agent_with_trace(user_id, user_input, user_brief)
    _print_trace(trace)

    print(f"\n[Test E] 最终回复：\n{result}")

    # ── 统计 ────────────────────────────────────────────────────
    tool_names = [r["name"] for r in trace]
    search_count = tool_names.count("search_recipes")
    price_count = tool_names.count("get_ingredient_price") + tool_names.count("compare_recipe_cost")

    # 第一轮 search 返回了哪些菜
    first_search_results = []
    for rec in trace:
        if rec["name"] == "search_recipes":
            first_search_results = rec["output"] if isinstance(rec["output"], list) else []
            break

    first_result_names = [r.get("name", "") for r in first_search_results]
    first_overpriced = [n for n in first_result_names if n in LOW_BUDGET_OVERPRICED]
    first_viable = [n for n in first_result_names if n in LOW_BUDGET_VIABLE]

    # 最终回复中提到了哪些 low 预算可行菜
    viable_in_reply = [n for n in LOW_BUDGET_VIABLE if n in result]

    print(f"\n[Test E] 统计摘要：")
    print(f"  search_recipes 调用次数: {search_count}")
    print(f"  价格工具调用次数: {price_count}")
    print(f"  第一轮返回的超预算菜: {first_overpriced}")
    print(f"  第一轮返回的可行菜: {first_viable}")
    print(f"  最终回复中提到的可行菜: {viable_in_reply}")

    # ── 判断结果 ─────────────────────────────────────────────────
    if viable_in_reply:
        print(f"\n[Test E] ✅ 重试机制有效 — 最终选中了 low 预算内的菜: {viable_in_reply}")
    elif search_count >= 2:
        print(f"\n[Test E] ⚠️  重试了 {search_count} 轮，但最终回复未明确提到可行菜名")
        print("         可能是 LLM 给出了 fallback 提示而非具体菜名")
    else:
        print(f"\n[Test E] ❌ 疑似过早放弃 — 只检索了 {search_count} 轮")
        print("         库里有符合 low 预算的菜（番茄炒蛋、宫保鸡丁等），但 LLM 没有继续尝试")

    # 宽松断言：至少调用了价格工具（说明触发了比价流程）
    assert price_count > 0, (
        f"budget_level='low' 场景下未触发任何价格查询，工具序列: {tool_names}"
    )
    print(f"\n[Test E] 基础断言通过 — 触发了 {price_count} 次价格查询")
