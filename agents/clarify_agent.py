from agents.slots import CLARIFY_TEMPLATES, INTENT_DISPLAY, REQUIRED_SLOTS
from agents.state import AgentState


def run_clarify_agent(state: AgentState) -> dict:
    """
    处理需要澄清的情况，生成追问内容返回给用户。
    不调用任何 LLM，纯规则生成。
    """
    confidence = state.get("confidence", "HIGH")
    missing_slots = state.get("missing_slots", [])
    intent = state.get("intent", "CHAT")
    top2 = state.get("top2")

    # LOW 置信度：展示 Top2 让用户选择
    if confidence == "LOW" and top2:
        intent_a = INTENT_DISPLAY.get(intent, intent)
        intent_b = INTENT_DISPLAY.get(top2, top2)
        result = CLARIFY_TEMPLATES["low_confidence"].format(
            intent_a=intent_a,
            intent_b=intent_b,
        )
        return {**state, "result": result}

    # MEDIUM + 缺失强制槽位：追问最关键的1个
    if missing_slots:
        slot = missing_slots[0]
        template_key = f"missing_{slot}"
        if template_key in CLARIFY_TEMPLATES:
            result = CLARIFY_TEMPLATES[template_key]
        else:
            result = "请问您能提供更多信息吗？"
        return {**state, "result": result}

    # 兜底
    return {**state, "result": "请问您能描述得更具体一些吗？"}
