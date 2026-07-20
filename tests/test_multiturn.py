import pytest
from unittest.mock import patch, MagicMock
from agents.supervisor_agent import run_supervisor

def make_state(msg, last_intent="", last_input=""):
    return {
        "user_input": msg,
        "intent": "",
        "confidence": "HIGH",
        "missing_slots": [],
        "top2": None,
        "result": "",
        "user_id": "test_user",
        "user_brief": {},
        "conversation_memory": "",
        "last_turn": {
            "user_input": last_input,
            "intent": last_intent,
            "result_preview": "",
        } if last_intent else {}
    }

VALID_INTENTS = {
    "MEAL", "PRICE", "ELECTRONICS_PRICE", "MEMORY_UPDATE",
    "UPDATE_AND_MEAL", "HEALTH_CYCLE", "CHAT", "NUTRITION", "FOOD_SAFETY"
}

def test_correction_keeps_electronics_intent():
    """'不是保护壳' 上一轮是 ELECTRONICS_PRICE → 应保持 ELECTRONICS_PRICE"""
    with patch("agents.supervisor_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="INTENT: ELECTRONICS_PRICE\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
        )
        state = make_state(
            "不是保护壳",
            last_intent="ELECTRONICS_PRICE",
            last_input="MacBook Air M3价格"
        )
        result = run_supervisor(state)
        assert result["intent"] in VALID_INTENTS

def test_followup_keeps_intent():
    """'那Pro版呢' 上一轮是 ELECTRONICS_PRICE → 应继续查价"""
    with patch("agents.supervisor_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="INTENT: ELECTRONICS_PRICE\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
        )
        state = make_state(
            "那Pro版呢",
            last_intent="ELECTRONICS_PRICE",
            last_input="iPhone 16价格"
        )
        result = run_supervisor(state)
        assert result["intent"] in VALID_INTENTS

def test_no_last_turn_still_works():
    """没有上一轮对话时正常工作"""
    with patch("agents.supervisor_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="INTENT: MEAL\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
        )
        state = make_state("今天吃什么")
        result = run_supervisor(state)
        assert result["intent"] in VALID_INTENTS

def test_last_turn_injected_to_prompt():
    """有上一轮时 prompt 里包含上一轮信息"""
    with patch("agents.supervisor_agent.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="INTENT: ELECTRONICS_PRICE\nCONFIDENCE: HIGH\nMISSING_SLOTS: NONE\nTOP2: NONE"
        )
        state = make_state(
            "不是保护壳",
            last_intent="ELECTRONICS_PRICE",
            last_input="MacBook Air M3价格"
        )
        run_supervisor(state)
        call_args = mock_llm.invoke.call_args[0][0]
        first_message = call_args[0].content
        assert "MacBook Air M3" in first_message or "上一轮" in first_message
