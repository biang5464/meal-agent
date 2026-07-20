"""Update Agent：识别偏好变更并写入数据库，结果写入 state.result。

如果 state.needs_meal == True，图会在本节点之后继续路由到 meal_agent。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.react_deadline import run_react_with_deadline
from agents.state import AgentState
from config import get_llm
from core.update_service import invalidate_cache_result
from tools.memory import update_user_profile
from tools.timeout_config import TimeoutConfig


def _build_system_prompt(user_id: str) -> str:
    return (
        f"你是用户偏好管理助手。当前用户 ID：{user_id}\n\n"
        "从用户消息中识别饮食偏好变更，调用 update_user_profile 保存到数据库。\n\n"
        "update_user_profile 支持的字段：\n"
        "  likes          - 喜欢的口味/食材/菜系，传 JSON 数组字符串，如 '[\"辣\",\"川菜\"]'\n"
        "  dislikes       - 忌口列表，传 JSON 数组字符串，如 '[\"香菜\",\"榴莲\"]'\n"
        "  budget         - 预算等级，只能是 low / mid / high\n"
        "  diet_restriction - 饮食限制，如 \"素食\"；无限制传空字符串\n\n"
        "识别并保存完成后，向用户简洁确认已记录的内容。"
    )


async def update_agent(state: AgentState) -> AgentState:
    user_id = state["user_id"]

    llm = get_llm(streaming=True)
    agent = create_react_agent(
        model=llm,
        tools=[update_user_profile],
        prompt=SystemMessage(content=_build_system_prompt(user_id)),
    )

    input_messages = list(state.get("messages", [])) + [
        HumanMessage(content=state["user_input"])
    ]

    result = await run_react_with_deadline(
        agent,
        {"messages": input_messages},
        deadline=TimeoutConfig.UPDATE_AGENT_DEADLINE,
    )

    # 画像已更新，删除 Redis 缓存；失败时记录日志，不影响主流程
    inv_result = await invalidate_cache_result(user_id)
    if not inv_result.ok:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[update_agent] Redis 缓存失效失败（部分成功）: %s", inv_result.error
        )

    # 写入确认消息；若 needs_meal=True，graph 会继续路由到 meal_agent
    return {**state, "result": result}
