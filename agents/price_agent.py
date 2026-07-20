"""Price Agent：调用 Coles / Woolworths 比价工具，结果写入 state.result。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.context_consumer import PRICE_CONSTRAINTS, build_human_message, get_agent_context
from agents.react_deadline import run_react_with_deadline
from agents.state import AgentState
from config import get_llm
from tools.price import compare_recipe_cost, get_ingredient_price
from tools.timeout_config import TimeoutConfig

_SYSTEM_PROMPT = (
    "你是价格查询助手。从用户消息中提取食材或商品关键词（英文搜索效果最佳），"
    "调用 get_ingredient_price 对比 Coles 和 Woolworths 的单品价格，"
    "或调用 compare_recipe_cost 估算整道菜的食材总成本。\n"
    "用清晰的列表或表格呈现结果，标注推荐购买渠道和可节省金额。"
)


async def price_agent(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    context = get_agent_context(state, "food", PRICE_CONSTRAINTS)

    llm = get_llm(streaming=True)
    agent = create_react_agent(
        model=llm,
        tools=[get_ingredient_price, compare_recipe_cost],
        prompt=SystemMessage(content=_SYSTEM_PROMPT),
    )

    input_messages = list(state.get("messages", [])) + [
        HumanMessage(content=build_human_message(user_input, context))
    ]

    result = await run_react_with_deadline(
        agent,
        {"messages": input_messages},
        deadline=TimeoutConfig.PRICE_AGENT_DEADLINE,
    )
    return {**state, "result": result}
