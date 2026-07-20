"""动态路由器：意图分类 → 工具选择 → LangGraph ReAct Agent 流式推理。"""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from tools import ALL_TOOLS
from tools.memory import MEMORY_TOOLS, get_user_profile, update_user_profile
from tools.price import PRICE_TOOLS
from tools.recipe import RECIPE_TOOLS

# ---------- 意图 → 工具子集 ----------

_INTENT_TOOLS = {
    # 比价：价格工具 + 读画像（获取预算偏好）
    "compare_price": PRICE_TOOLS + [get_user_profile],
    # 推荐：菜谱检索 + 全部画像工具（读/写历史）
    "recommend_meal": RECIPE_TOOLS + MEMORY_TOOLS,
    # 偏好更新：只需画像读写
    "update_preference": MEMORY_TOOLS,
}

# ---------- 意图分类 ----------

_CLASSIFY_PROMPT = """\
你是意图分类器。根据用户消息，从以下三种意图中选一个，只输出意图名称，不要解释：

- compare_price    ：用户询问食材或商品的价格、哪里更便宜
- recommend_meal   ：用户想知道吃什么、要饮食推荐、问今日菜单
- update_preference：用户在表达口味偏好、忌口、饮食限制

用户消息：{message}

意图："""


async def classify_intent(message: str) -> str:
    """返回三种意图字符串之一；无法识别时默认 recommend_meal。"""
    llm = get_llm(streaming=False)
    resp = await llm.ainvoke(
        [HumanMessage(content=_CLASSIFY_PROMPT.format(message=message))]
    )
    raw = resp.content.strip().lower()
    for intent in ("compare_price", "recommend_meal", "update_preference"):
        if intent in raw:
            return intent
    return "recommend_meal"


# ---------- 系统提示构建 ----------

def _build_system_prompt(intent: str, user_id: str, profile: dict) -> str:
    likes = "、".join(profile.get("likes", [])) or "无"
    dislikes = "、".join(profile.get("dislikes", [])) or "无"
    budget = {"low": "低预算", "mid": "中等预算", "high": "高预算"}.get(
        profile.get("budget", "mid"), "中等预算"
    )
    restriction = profile.get("diet_restriction") or "无"
    recent = profile.get("meal_history", [])[-5:]
    recent_str = "、".join(recent) if recent else "暂无"

    profile_block = (
        f"【用户画像】\n"
        f"用户 ID：{user_id}\n"
        f"喜欢：{likes}\n"
        f"忌口：{dislikes}\n"
        f"预算：{budget}\n"
        f"饮食限制：{restriction}\n"
        f"近期吃过：{recent_str}\n"
    )

    if intent == "compare_price":
        task = (
            "当前任务：查询食材价格并对比。调用价格工具获取 Coles 和 Woolworths 报价，"
            "结合用户预算偏好给出购买建议，标注哪家更划算。"
        )
    elif intent == "recommend_meal":
        task = (
            "当前任务：为用户推荐今日餐食。先调用 search_recipes 检索菜谱，"
            "结合用户喜好过滤后推荐 2-3 道菜并说明理由，推荐完成后调用 append_meal_history 记录。"
        )
    elif intent == "update_preference":
        task = (
            "当前任务：更新用户口味偏好。从对话中识别用户的喜好或忌口，"
            "调用 update_user_profile 保存，并向用户确认已记录。"
        )
    else:
        task = "请根据用户需求调用合适的工具给出回答。"

    return f"你是专业的个人饮食推荐助手。\n\n{profile_block}\n{task}"


# ---------- Agent 构建 ----------

def build_agent(user_id: str, intent: str, profile: dict):
    """
    根据意图和用户画像构建 LangGraph ReAct Agent。

    Args:
        user_id: 用户唯一标识（注入系统提示供 LLM 调用工具时使用）
        intent:  已分类的意图字符串
        profile: 已加载的用户画像字典

    Returns:
        LangGraph CompiledGraph，支持 astream_events。
    """
    llm = get_llm(streaming=True)
    tools = _INTENT_TOOLS.get(intent, ALL_TOOLS)
    system_content = _build_system_prompt(intent, user_id, profile)

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_content),
    )


# ---------- 偏好后处理 ----------

_EXTRACT_PROMPT = """\
从下面这句话中提取用户的饮食偏好更新。
返回 JSON 数组，每项格式：{{"field": "字段名", "value": "值"}}

可用字段及格式：
- likes           : JSON 数组字符串，如 ["辣", "川菜"]
- dislikes        : JSON 数组字符串，如 ["香菜", "榴莲"]
- budget          : 字符串，只能是 low / mid / high
- diet_restriction: 字符串，如 "素食"，无限制则 ""

如果消息里没有明确的偏好信息，返回 []。

用户消息：{message}

JSON："""


async def _extract_and_update_preferences(user_id: str, message: str) -> None:
    """用 LLM 从消息中提取偏好字段并持久化，失败时静默忽略。"""
    llm = get_llm(streaming=False)
    resp = await llm.ainvoke(
        [HumanMessage(content=_EXTRACT_PROMPT.format(message=message))]
    )
    raw = resp.content.strip()

    # 兼容 LLM 在 JSON 外包了 markdown 代码块的情况
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return

    try:
        updates: list[dict] = json.loads(match.group())
    except json.JSONDecodeError:
        return

    for item in updates:
        field = item.get("field", "")
        value = item.get("value")
        if field and value is not None:
            update_user_profile.invoke(
                {"user_id": user_id, "field": field, "value": str(value)}
            )


# ---------- 流式推理入口 ----------

async def stream_recommendation(
    user_id: str,
    message: str,
    chat_history: list | None = None,
) -> AsyncIterator[str]:
    """
    完整推理流程：意图分类 → 加载画像 → 构建 Agent → 流式 yield token。

    Args:
        user_id:      当前用户 ID
        message:      用户本轮输入
        chat_history: 历史消息列表（LangChain Message 格式，可选）

    Yields:
        Agent 生成的文本片段（不含工具调用中间过程）
    """
    # 1. 意图分类
    intent = await classify_intent(message)

    # 2. 加载用户画像（同步 SQLite，速度快）
    profile: dict = get_user_profile.invoke({"user_id": user_id})

    # 3. 构建 Agent
    agent = build_agent(user_id, intent, profile)

    # 4. 组装输入：历史消息 + 当前轮
    input_messages = list(chat_history or []) + [HumanMessage(content=message)]

    # 5. 流式推理
    #    on_chat_model_stream 事件在 LLM 输出最终回答时触发；
    #    工具调用阶段 chunk.content 为空，自然过滤掉。
    async for event in agent.astream_events(
        {"messages": input_messages},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                yield chunk.content

    # 6. 偏好意图：流式完成后再做后处理提取
    if intent == "update_preference":
        await _extract_and_update_preferences(user_id, message)
