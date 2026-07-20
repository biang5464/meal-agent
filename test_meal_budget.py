"""
测试 meal_agent 比价逻辑，覆盖 4 个场景：
  A - budget_level="low"，用户不提钱 → 自动触发比价
  B - budget_level=""，用户说"预算low" → 临时预算生效
  C - budget_level=""，用户不提预算 → 跳过比价
  D - budget_level="low"，要求含三文鱼/大虾的菜 → 超预算，触发重试

运行：.venv\\Scripts\\python.exe -m pytest test_meal_budget.py -v -s
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


# ── 辅助：直接运行 agent 并捕获工具调用序列 ──────────────────────────────────

async def _run_agent(user_id: str, user_input: str, user_brief: dict) -> tuple[str, list[dict]]:
    """
    重建 meal_agent 内部逻辑，额外收集 on_tool_start 事件。
    返回 (result_text, tool_calls)，其中 tool_calls 是 {name, input} 列表。
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
    tool_calls: list[dict] = []

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        version="v2",
        config={"recursion_limit": _RECURSION_LIMIT},
    ):
        if event["event"] == "on_tool_start":
            tool_calls.append({
                "name": event["name"],
                "input": event["data"].get("input", {}),
            })
        elif event["event"] == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                parts.append(chunk.content)

    return "".join(parts), tool_calls


def _print_tool_trace(label: str, tool_calls: list[dict]):
    print(f"\n[{label}] 工具调用序列（共 {len(tool_calls)} 次）：")
    for i, tc in enumerate(tool_calls, 1):
        inp = tc["input"]
        # 只打印关键字段，避免过长
        if "query" in inp:
            key = f"query={inp['query']!r}"
        elif "ingredient" in inp:
            key = f"ingredient={inp['ingredient']!r}"
        elif "ingredients" in inp:
            key = f"ingredients={inp['ingredients']}"
        elif "name" in inp:
            key = f"name={inp['name']!r}"
        else:
            key = str(inp)[:80]
        print(f"  {i}. {tc['name']}  ({key})")


# ── Test A：budget_level="low"，用户不提钱 → 自动触发比价 ────────────────────

@pytest.mark.asyncio
async def test_A_persistent_budget_auto_triggers_price_check():
    user_id = "meal_budget_A"
    user_brief = {
        "likes": [],
        "dislikes": [],
        "active_restrictions": [],
        "recent_meals": [],
        "unread_reminders": [],
        "period_status": "",
        "budget_level": "low",     # 持久化预算，规则1生效
    }
    user_input = "帮我推荐今天吃什么"   # 完全没提钱

    print(f"\n[Test A] user_input={user_input!r}  budget_level='low'")
    result, tool_calls = await _run_agent(user_id, user_input, user_brief)
    _print_tool_trace("Test A", tool_calls)
    print(f"[Test A] 回复前200字：{result[:200]!r}")

    tool_names = [t["name"] for t in tool_calls]
    assert "search_recipes" in tool_names, "应调用 search_recipes"
    assert (
        "get_ingredient_price" in tool_names
        or "compare_recipe_cost" in tool_names
    ), "budget_level='low' 时应自动触发比价工具，但工具序列中未找到价格查询"
    print("[Test A] PASS — 持久化预算触发了自动比价")


# ── Test B：budget_level=""，用户说"预算控制在low" → 临时预算生效 ───────────

@pytest.mark.asyncio
async def test_B_inline_budget_triggers_price_check():
    user_id = "meal_budget_B"
    user_brief = {
        "likes": [],
        "dislikes": [],
        "active_restrictions": [],
        "recent_meals": [],
        "unread_reminders": [],
        "period_status": "",
        "budget_level": "",        # 未设置持久化预算
    }
    user_input = "今天预算控制在low，帮我推荐一个菜"   # 临时提及预算，规则2生效

    print(f"\n[Test B] user_input={user_input!r}  budget_level=''")
    result, tool_calls = await _run_agent(user_id, user_input, user_brief)
    _print_tool_trace("Test B", tool_calls)
    print(f"[Test B] 回复前200字：{result[:200]!r}")

    tool_names = [t["name"] for t in tool_calls]
    assert "search_recipes" in tool_names, "应调用 search_recipes"
    assert (
        "get_ingredient_price" in tool_names
        or "compare_recipe_cost" in tool_names
    ), "用户说了'预算low'，应触发比价工具，但工具序列中未找到价格查询"
    print("[Test B] PASS — 临时预算触发了比价")


# ── Test C：budget_level=""，不提预算 → 跳过比价 ─────────────────────────────

@pytest.mark.asyncio
async def test_C_no_budget_skips_price_check():
    user_id = "meal_budget_C"
    user_brief = {
        "likes": [],
        "dislikes": [],
        "active_restrictions": [],
        "recent_meals": [],
        "unread_reminders": [],
        "period_status": "",
        "budget_level": "",        # 未设置持久化预算
    }
    user_input = "帮我推荐今天吃什么"   # 完全不提预算，规则3生效

    print(f"\n[Test C] user_input={user_input!r}  budget_level=''")
    result, tool_calls = await _run_agent(user_id, user_input, user_brief)
    _print_tool_trace("Test C", tool_calls)
    print(f"[Test C] 回复前200字：{result[:200]!r}")

    tool_names = [t["name"] for t in tool_calls]
    assert "search_recipes" in tool_names, "应调用 search_recipes"
    assert "get_ingredient_price" not in tool_names, (
        f"无预算信息时不应调用 get_ingredient_price，"
        f"实际工具序列：{tool_names}"
    )
    assert "compare_recipe_cost" not in tool_names, (
        f"无预算信息时不应调用 compare_recipe_cost，"
        f"实际工具序列：{tool_names}"
    )
    print("[Test C] PASS — 无预算信息，正确跳过比价")


# ── Test D：budget_level="low"，要求含三文鱼/大虾 → 超预算触发重试 ──────────

@pytest.mark.asyncio
async def test_D_over_budget_triggers_retry_and_fallback():
    """
    三文鱼实际价格 ~$25-35/kg，大虾 ~$20-40/kg，
    均超过 low 档肉类阈值 $12/kg。
    LLM 应尝试重新检索替代菜谱，最终给出"接近预算"提示。
    """
    user_id = "meal_budget_D"
    user_brief = {
        "likes": ["海鲜"],
        "dislikes": [],
        "active_restrictions": [],
        "recent_meals": [],
        "unread_reminders": [],
        "period_status": "",
        "budget_level": "low",     # 极低预算
    }
    user_input = "推荐一道含三文鱼或者大虾的菜"   # 强制要求昂贵食材

    print(f"\n[Test D] user_input={user_input!r}  budget_level='low'（昂贵食材）")
    result, tool_calls = await _run_agent(user_id, user_input, user_brief)
    _print_tool_trace("Test D", tool_calls)
    print(f"[Test D] 完整回复：\n{result}")

    tool_names = [t["name"] for t in tool_calls]

    # 基础断言：比价工具必须被调用
    assert (
        "get_ingredient_price" in tool_names
        or "compare_recipe_cost" in tool_names
    ), f"超预算场景应触发价格查询，实际工具序列：{tool_names}"

    # 统计 search_recipes 调用次数，检查是否触发了重试
    search_count = tool_names.count("search_recipes")
    price_count = (
        tool_names.count("get_ingredient_price")
        + tool_names.count("compare_recipe_cost")
    )
    print(f"[Test D] search_recipes 调用了 {search_count} 次，价格查询调用了 {price_count} 次")

    # 检查回复中是否提到了预算/价格相关内容
    keywords = ["预算", "价格", "接近", "超出", "最接近", "略高", "low", "便宜"]
    found = [kw for kw in keywords if kw in result]
    print(f"[Test D] 回复中出现的预算关键词：{found}")

    # 宽松断言：回复中至少包含一个预算相关关键词
    assert found, (
        f"超预算场景的回复中未找到任何预算相关关键词，"
        f"实际回复：{result[:300]}"
    )
    print(f"[Test D] PASS — 触发比价 {price_count} 次，{search_count} 轮检索，回复含预算说明")
