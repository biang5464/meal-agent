"""Supervisor Agent：意图分类，决定后续走哪条子 Agent 分支。"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from config import get_llm

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "MEAL", "PRICE", "MEMORY_UPDATE", "UPDATE_AND_MEAL",
    "HEALTH_CYCLE", "CHAT", "NUTRITION", "FOOD_SAFETY",
    "ELECTRONICS_PRICE",
}

# 模块级 LLM，初始为 None，支持测试时 patch
llm = None

# 超时/全部重试失败时的安全降级状态（路由层可安全处理 CHAT+LOW）
_SUPERVISOR_FALLBACK = {
    "intent": "CHAT",
    "confidence": "LOW",
    "missing_slots": [],
    "top2": None,
}

_CLASSIFY_PROMPT = """\
你是一个意图识别助手，分析用户输入并输出结构化判断。

可选意图：
MEAL - 用户请求餐食推荐或菜谱建议
PRICE - 仅用于食材、生鲜、日用品、超市商品的价格查询
ELECTRONICS_PRICE - 仅用于电子产品、数码产品的价格查询（手机/电脑/耳机/电视/相机等）
NUTRITION - 用户询问营养成分或疾病饮食建议
FOOD_SAFETY - 用户询问食材安全、储存或药食禁忌
MEMORY_UPDATE - 用户要求记录或更新个人偏好
UPDATE_AND_MEAL - 用户同时更新偏好并要求推荐
HEALTH_CYCLE - 用户记录健康状态或查询健康历史
CHAT - 其他闲聊或无关内容

严格按以下格式输出，不要输出任何其他内容：
INTENT: <意图>
CONFIDENCE: <HIGH|MEDIUM|LOW>
MISSING_SLOTS: <缺失的关键信息，多个用逗号分隔，无则填NONE>
TOP2: <第二可能的意图，无则填NONE>

置信度判断标准：
HIGH: 意图明确且关键信息完整
MEDIUM: 意图明确但信息不足，或有轻微歧义
LOW: 意图模糊，无法确定

缺失槽位判断规则：
PRICE 意图：
  - 完全没有提到任何商品 → MISSING_SLOTS: product
  - 提到了商品（如"苹果手机"、"猪肉"、"牛奶"）→ MISSING_SLOTS: NONE（食材/日用品不需要型号）

ELECTRONICS_PRICE 意图：
  - 完全没有提到任何商品 → MISSING_SLOTS: product
  - 提到了品牌但没有具体型号（如"苹果手机"、"索尼耳机"、"戴尔电脑"）→ MISSING_SLOTS: model
  - 提到了具体型号（如"iPhone 16"、"MacBook Air M3"、"Samsung S25"）→ MISSING_SLOTS: NONE

NUTRITION 意图：缺少具体疾病或食物名 → MISSING_SLOTS: topic
FOOD_SAFETY 意图：缺少具体食材或药物名 → MISSING_SLOTS: ingredient
其他意图：通常填 NONE

示例：
"苹果手机多少钱" → INTENT: ELECTRONICS_PRICE, CONFIDENCE: MEDIUM, MISSING_SLOTS: model
"iPhone 16价格"  → INTENT: ELECTRONICS_PRICE, CONFIDENCE: HIGH, MISSING_SLOTS: NONE
"价格多少"       → INTENT: ELECTRONICS_PRICE, CONFIDENCE: LOW, MISSING_SLOTS: product
"猪肉多少钱"     → INTENT: PRICE, CONFIDENCE: HIGH, MISSING_SLOTS: NONE
"""


def _parse_supervisor_output(raw: str) -> dict:
    lines = raw.strip().split('\n')
    result = {
        "intent": "CHAT",
        "confidence": "HIGH",
        "missing_slots": [],
        "top2": None,
    }
    for line in lines:
        if line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip().upper()
            result["intent"] = intent if intent in VALID_INTENTS else "CHAT"
        elif line.startswith("CONFIDENCE:"):
            conf = line.replace("CONFIDENCE:", "").strip().upper()
            result["confidence"] = conf if conf in {"HIGH", "MEDIUM", "LOW"} else "HIGH"
        elif line.startswith("MISSING_SLOTS:"):
            slots_raw = line.replace("MISSING_SLOTS:", "").strip()
            result["missing_slots"] = (
                [] if slots_raw.upper() == "NONE"
                else [s.strip() for s in slots_raw.split(",")]
            )
        elif line.startswith("TOP2:"):
            top2 = line.replace("TOP2:", "").strip().upper()
            result["top2"] = top2 if top2 != "NONE" and top2 in VALID_INTENTS else None
    return result


def _build_session_hint(session_ctx: dict) -> str:
    if not isinstance(session_ctx, dict) or not session_ctx.get("entity"):
        return ""
    entity = _sanitize_str(session_ctx.get("entity", ""))
    if not entity:
        return ""
    domain = _sanitize_str(session_ctx.get("domain", ""), max_len=30)
    excluded_str = ', '.join(_sanitize_list(session_ctx.get('excluded', []))) or '无'
    constraint_parts: list[str] = []
    raw = session_ctx.get("constraints", {})
    if isinstance(raw, dict):
        for k, v in list(raw.items())[:_HINT_MAX_LIST_ITEMS]:
            if isinstance(k, str) and isinstance(v, (str, int, float, bool)):
                ks = _sanitize_str(k, 20)
                vs = _sanitize_str(str(v), 20)
                if ks and vs:
                    constraint_parts.append(f"{ks}={vs}")
    constraints_str = ', '.join(constraint_parts) or '无'
    topic_turns = session_ctx.get("topic_turns", 1)
    if not isinstance(topic_turns, int):
        try:
            topic_turns = int(topic_turns)
        except (ValueError, TypeError):
            topic_turns = 1
    return (
        f'\n\n【当前会话上下文】\n'
        f'话题域：{domain}\n'
        f'当前实体：{entity}\n'
        f'已排除：{excluded_str}\n'
        f'约束条件：{constraints_str}\n'
        f'连续轮次：{topic_turns}\n'
        f'\n请结合以上上下文理解用户当前输入。\n'
    )


# Candidate Hint 安全边界常量
_HINT_MAX_STR_LEN = 50    # 单个字符串字段最大字符数
_HINT_MAX_LIST_ITEMS = 5  # excluded/constraints 最多展示条数
_HINT_ITEM_MAX_LEN = 30   # 每个列表项最大字符数
_CTRL_TRANS = str.maketrans("", "", "\x00\r\n\t\x0b\x0c")  # 去除控制字符


def _sanitize_str(value, max_len: int = _HINT_MAX_STR_LEN) -> str:
    """去除控制字符并截断到 max_len。非 str 类型先转 str。"""
    if not isinstance(value, str):
        value = str(value)
    return value.translate(_CTRL_TRANS).strip()[:max_len]


def _sanitize_list(items, max_items: int = _HINT_MAX_LIST_ITEMS, item_max_len: int = _HINT_ITEM_MAX_LEN) -> list[str]:
    """过滤非字符串项、去控制字符、截断，最多取 max_items 条。None 或不可迭代返回空列表。"""
    if not items:
        return []
    result = []
    try:
        for item in items:
            if not isinstance(item, str):
                continue
            s = _sanitize_str(item, item_max_len)
            if s:
                result.append(s)
            if len(result) >= max_items:
                break
    except TypeError:
        return []
    return result


def _build_candidate_hint(candidate: dict) -> str:
    """从 context_candidate 构建 Supervisor 分类参考提示。

    标注为"候选"，不冒充已提交的 session_context。
    不暴露内部技术字段（match_type 等）给 LLM。
    """
    if not isinstance(candidate, dict) or not candidate:
        return ""
    entity = _sanitize_str(candidate.get("entity", ""))
    turn_excluded = _sanitize_list(candidate.get("turn_excluded", []))
    existing_excluded = _sanitize_list(candidate.get("existing_excluded", []))

    turn_constraint_parts: list[str] = []
    turn_remove_parts: list[str] = []
    raw_turn_constraints = candidate.get("turn_constraints", {})
    if isinstance(raw_turn_constraints, dict):
        for k, v in list(raw_turn_constraints.items())[:_HINT_MAX_LIST_ITEMS]:
            if not isinstance(k, str):
                continue
            ks = _sanitize_str(k, 20)
            if not ks:
                continue
            if v is None:
                turn_remove_parts.append(ks)
            elif isinstance(v, (str, int, float, bool)):
                vs = _sanitize_str(str(v), 20)
                if vs:
                    turn_constraint_parts.append(f"{ks}={vs}")

    historical_constraint_parts: list[str] = []
    raw_constraints = candidate.get("existing_constraints", {})
    if isinstance(raw_constraints, dict):
        for k, v in list(raw_constraints.items())[:_HINT_MAX_LIST_ITEMS]:
            if isinstance(k, str) and isinstance(v, (str, int, float, bool)):
                ks = _sanitize_str(k, 20)
                vs = _sanitize_str(str(v), 20)
                if ks and vs:
                    historical_constraint_parts.append(f"{ks}={vs}")

    if not entity and not turn_excluded and not existing_excluded and not turn_constraint_parts and not turn_remove_parts and not historical_constraint_parts:
        return ""
    parts = ["\n\n【候选上下文（仅供分类参考）】"]
    if entity:
        parts.append(f"候选实体：{entity}")
    if turn_excluded:
        parts.append(f"本轮排除：{', '.join(turn_excluded)}")
    if existing_excluded:
        parts.append(f"历史排除：{', '.join(existing_excluded)}")
    if turn_constraint_parts:
        parts.append(f"本轮约束：{', '.join(turn_constraint_parts)}")
    if turn_remove_parts:
        parts.append(f"本轮移除：{', '.join(turn_remove_parts)}")
    if historical_constraint_parts:
        parts.append(f"历史约束：{', '.join(historical_constraint_parts)}")
    return "\n".join(parts)


def _build_last_turn_hint(last_turn: dict) -> str:
    if not last_turn or not last_turn.get("user_input"):
        return ""
    intent_line = f"识别意图：{last_turn['intent']}\n" if last_turn.get("intent") else ""
    preview_line = f"回复摘要：{last_turn['result_preview']}\n" if last_turn.get("result_preview") else ""
    return (
        f'\n\n【上一轮对话参考】\n'
        f'用户说：{last_turn["user_input"]}\n'
        f'{intent_line}'
        f'{preview_line}'
        f'\n请结合上一轮对话理解当前用户输入，特别注意：\n'
        f'- 如果用户说"不是XX"、"不对"、"重新查"、"换一个"等纠正性表达，\n'
        f'  应保持上一轮意图，并理解为对结果的修正\n'
        f'- 如果用户说"那XX呢"、"还有"等追问，应保持上一轮意图继续查询\n'
    )


async def _classify_impl(messages: list):
    """同步/异步皆可：异步调用 LLM，由 ToolExecutor 套 asyncio.wait_for 控超时。"""
    _llm = get_llm(streaming=False)
    return await _llm.ainvoke(messages)


async def supervisor_agent(state: AgentState) -> AgentState:
    memory_context = state.get("conversation_memory", "")
    memory_hint = f"\n\n【用户历史记忆】\n{memory_context}" if memory_context else ""
    last_turn_hint = _build_last_turn_hint(state.get("last_turn", {}))
    session_hint = _build_session_hint(state.get("session_context", {}))
    candidate_hint = _build_candidate_hint(state.get("context_candidate", {}))

    messages = [
        HumanMessage(content=_CLASSIFY_PROMPT + memory_hint + last_turn_hint + session_hint + candidate_hint),
        HumanMessage(content=f"用户输入：{state['user_input'][:1000]}"),
    ]

    from tools.tool_executor import tool_executor
    result = await tool_executor.execute(
        _classify_impl,
        messages,
        policy_name="llm_classification",
        tool_name="supervisor_classify",
        fallback_data=None,
    )

    if result.ok and result.data is not None:
        parsed = _parse_supervisor_output(result.data.content)
    else:
        logger.warning("[supervisor] LLM 分类失败，降级为 CHAT: %s", result.error)
        parsed = dict(_SUPERVISOR_FALLBACK)

    needs_meal = parsed["intent"] == "UPDATE_AND_MEAL"
    return {
        **state,
        "intent": parsed["intent"],
        "confidence": parsed.get("confidence", "LOW"),
        "missing_slots": parsed.get("missing_slots", []),
        "top2": parsed.get("top2"),
        "needs_meal": needs_meal,
    }


def run_supervisor(state: AgentState) -> AgentState:
    """同步版意图分类，供单元测试使用。"""
    global llm
    if llm is None:
        llm = get_llm(streaming=False)

    last_turn_hint = _build_last_turn_hint(state.get("last_turn", {}))
    session_hint = _build_session_hint(state.get("session_context", {}))
    candidate_hint = _build_candidate_hint(state.get("context_candidate", {}))

    messages = [
        HumanMessage(content=_CLASSIFY_PROMPT + last_turn_hint + session_hint + candidate_hint),
        HumanMessage(content=f"用户输入：{state['user_input'][:1000]}"),
    ]

    resp = llm.invoke(messages)
    parsed = _parse_supervisor_output(resp.content)

    needs_meal = parsed["intent"] == "UPDATE_AND_MEAL"
    return {
        **state,
        "intent": parsed["intent"],
        "confidence": parsed["confidence"],
        "missing_slots": parsed["missing_slots"],
        "top2": parsed["top2"],
        "needs_meal": needs_meal,
    }
