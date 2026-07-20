"""Meal Agent：基于用户画像做菜谱推荐，支持自动比价和预算过滤。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.context_consumer import MEAL_CONSTRAINTS, build_human_message, get_agent_context
from agents.react_deadline import run_react_with_deadline
from agents.state import AgentState
from config import get_llm
from tools.price import compare_recipe_cost, get_ingredient_price
from tools.recipe import get_recipe_by_name, search_recipes
from tools.timeout_config import TimeoutConfig

# recursion_limit 推导：
#   每次工具调用 = 2 supersteps（LLM 决策节点 + tools 节点）
#   最差路径：search(2) + 2×price_check(4) + 4次重试×(search+2×check)(24) + final(1) = 31
#   取 30：覆盖实际路径，走满 4 次完整重试时 GraphRecursionError 是有意义的上限信号
_RECURSION_LIMIT = 30

_PRICE_THRESHOLDS = (
    "蔬菜类（AUD/kg）：\n"
    "  low  <= $6    （土豆、洋葱、胡萝卜、卷心菜、白菜）\n"
    "  mid  <= $10   （西红柿、西兰花、青椒、茄子、南瓜、豆角）\n"
    "  high <= $20   （芦笋、菠菜、小香菇、荷兰豆、小番茄）\n"
    "\n"
    "肉类（AUD/kg）：\n"
    "  low  <= $12   （鸡腿、鸡翅、猪肉末、普通鸡胸）\n"
    "  mid  <= $22   （猪排、散养鸡胸、牛肉末、羊肉卷）\n"
    "  high <= $40   （牛排、羊腿、三文鱼、大虾）"
)


def _build_system_prompt(user_id: str, user_brief: dict) -> str:
    likes = "、".join(user_brief.get("likes", [])) or "无"
    dislikes = "、".join(user_brief.get("dislikes", [])) or "无"
    restrictions = "、".join(user_brief.get("active_restrictions", [])) or "无"
    recent = "、".join(user_brief.get("recent_meals", [])) or "暂无"
    period = user_brief.get("period_status", "") or ""
    budget_level = user_brief.get("budget_level", "") or ""

    period_line = f"\n健康状态：{period}" if period else ""
    budget_line = f"\n预算档位：{budget_level}" if budget_level else "\n预算档位：未设置"

    return f"""\
你是专业个人饮食推荐助手。当前用户 ID：{user_id}

【用户画像】
喜欢：{likes}
忌口：{dislikes}
当前饮食限制：{restrictions}
近期吃过：{recent}{period_line}{budget_line}

【判断是否需要比价的规则】
1. 如果"预算档位"字段有值（low/mid/high 之一），说明用户有持久化预算偏好，
   本次推荐必须自动进行比价，无需用户在这句话里重新提及预算。
2. 如果"预算档位"为"未设置"，检查用户这句话里有没有提到预算或金额
   （如"预算低一点"、"便宜的"、"$15以内"、"控制在low"），
   如果提到，按此次提及的预算等级或金额进行比价。
3. 如果以上两种情况都没有预算信息，跳过比价步骤，
   直接按口味偏好推荐，绝对不要主动调用 get_ingredient_price 或 compare_recipe_cost。

【比价流程（仅在触发比价时执行）】
第一步：调用 search_recipes 检索候选菜谱，过滤忌口食材。
第二步：识别候选菜谱里的主要原材料（蔬菜类或肉类）。
第三步：调用 get_ingredient_price 查询每种主要食材的价格。
        注意：返回的 price 是商品包装价格，需结合 unit 字段换算为每公斤单价，
        然后对照以下阈值表判断是否超出对应类别和档位的阈值：

{_PRICE_THRESHOLDS}

第四步：如果有原材料超出对应阈值，重新调用 search_recipes 检索替代菜谱，
        最多重试 4 次。
第五步：4 次重试后仍超预算，选择最接近预算的方案，
        推荐时明确告知用户【这是预算内最接近的选择，价格略高于预期】。

【无需比价时的任务】
调用 search_recipes 检索匹配菜谱，严格过滤忌口和饮食限制，
推荐 2-3 道菜并分别说明推荐理由，语气亲切自然。\
"""


async def meal_agent(state: AgentState, queue: asyncio.Queue = None) -> AgentState:
    user_id = state["user_id"]
    user_input = state["user_input"]
    user_brief = state.get("user_brief", {})

    context = get_agent_context(state, "food", MEAL_CONSTRAINTS)

    llm = get_llm(streaming=True)
    agent = create_react_agent(
        model=llm,
        tools=[search_recipes, get_recipe_by_name, get_ingredient_price, compare_recipe_cost],
        prompt=SystemMessage(content=_build_system_prompt(user_id, user_brief)),
    )

    input_messages = list(state.get("messages", [])) + [
        HumanMessage(content=build_human_message(user_input, context))
    ]

    def _on_token(token: str) -> None:
        if queue is not None:
            queue.put_nowait(token)

    result = await run_react_with_deadline(
        agent,
        {"messages": input_messages},
        deadline=TimeoutConfig.MEAL_AGENT_DEADLINE,
        config={"recursion_limit": _RECURSION_LIMIT},
        on_token=_on_token,
    )
    return {**state, "result": result}
