from agents.slots import REQUIRED_SLOTS, PROFILE_FILLABLE_SLOTS
from agents.state import AgentState


def should_clarify(state: AgentState) -> str:
    """
    判断是否需要追问，返回路由目标节点名称。

    返回值：
    - "clarify"：需要追问
    - 对应 intent 小写：直接路由到 agent
    """
    confidence = state.get("confidence", "HIGH")
    missing_slots = state.get("missing_slots", [])
    intent = state.get("intent", "CHAT")
    user_brief = state.get("user_brief", {})

    if not isinstance(user_brief, dict):
        user_brief = {}

    # LOW 置信度直接追问
    if confidence == "LOW":
        return "clarify"

    # HIGH 直接路由
    if confidence == "HIGH":
        return intent.lower()

    # MEDIUM：检查必填槽位能否从画像补全
    required = REQUIRED_SLOTS.get(intent, [])
    truly_missing = []
    for slot in missing_slots:
        if slot in required:
            profile_key = PROFILE_FILLABLE_SLOTS.get(slot)
            if profile_key and user_brief.get(profile_key):
                continue  # 画像能补全，跳过
            truly_missing.append(slot)

    if truly_missing:
        return "clarify"

    return intent.lower()
