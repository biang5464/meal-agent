from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.context_consumer import (
    ELECTRONICS_CONSTRAINTS,
    build_electronics_input,
    get_agent_context,
)
from agents.state import AgentState
from tools.price import get_electronics_prices_result

PLATFORM_DISPLAY_NAMES = {
    "jbhifi": "JB Hi-Fi",
    "goodguys": "The Good Guys",
    "woolworths": "Woolworths",
    "coles": "Coles",
    "umall": "Umall",
}

_KEYWORD_SYSTEM_PROMPT = (
    "从用户输入里提取电子产品的搜索关键词。"
    "只输出关键词本身，不要输出任何其他内容。"
    "例如：'apple价格多少' → 'apple'，"
    "'帮我查iPhone 16手机' → 'iPhone 16'，"
    "'索尼耳机推荐' → '索尼耳机'"
)


async def _keyword_impl(messages: list):
    """Async LLM call; ToolExecutor wraps this with asyncio.wait_for for timeout."""
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_retries=0,
        request_timeout=65,
    )
    return await llm.ainvoke(messages)


async def _extract_query_keyword(user_input: str, last_turn: dict | None = None) -> str:
    """从用户输入里提取电子产品搜索关键词。"""
    from tools.tool_executor import tool_executor
    messages = [
        SystemMessage(content=_KEYWORD_SYSTEM_PROMPT),
        HumanMessage(content=user_input[:200]),
    ]
    result = await tool_executor.execute(
        _keyword_impl,
        messages,
        policy_name="llm_classification",
        tool_name="extract_keyword",
        fallback_data=None,
    )
    if result.ok and result.data is not None:
        keyword = result.data.content.strip()
        return keyword if keyword else user_input
    return user_input


async def run_electronics_price_agent(state: AgentState) -> dict:
    """
    电子产品查价 agent。
    只查询 JB Hi-Fi 和 The Good Guys，跳过 Woolworths/Umall。
    结果存入 state["result"]，由 run_with_queue 统一写入 queue。
    """
    user_input = state.get("user_input", "")
    context = get_agent_context(state, "electronics", ELECTRONICS_CONSTRAINTS)
    electronics_input = build_electronics_input(user_input, context)
    query = await _extract_query_keyword(electronics_input, last_turn=state.get("last_turn", {}))

    try:
        tool_result = await get_electronics_prices_result(query)
    except Exception:
        return {**state, "result": "查询电子产品价格时出现错误，请稍后再试。"}

    if not tool_result.ok:
        return {**state, "result": "查询电子产品价格时出现错误，请稍后再试。"}

    data = tool_result.data or {}
    items = data.get("items", [])
    sources = data.get("sources", [])

    if not items:
        result_text = f'抱歉，暂时没有找到"{query}"的相关电子产品价格信息。'
    else:
        lines = [f"**{query}** 电子产品比价结果：\n"]
        for r in items:
            platform_name = PLATFORM_DISPLAY_NAMES.get(r["platform"], r["platform"])
            lines.append(
                f"- **{platform_name}**：{r['name']} "
                f"**${r['price']} AUD**"
            )
        # 有平台不可用时：从 ToolResult.data.sources 取明确状态，不再自行推断
        unavailable_names = [
            PLATFORM_DISPLAY_NAMES.get(s["source"], s["source"])
            for s in sources
            if not s["ok"]
        ]
        if unavailable_names:
            lines.append(f"\n> ⚠️ {', '.join(unavailable_names)} 暂时无法获取，仅显示可用平台数据。")
        lines.append("\n> 价格来源：JB Hi-Fi、The Good Guys，实时查询仅供参考。")
        result_text = "\n".join(lines)

    return {**state, "result": result_text}
