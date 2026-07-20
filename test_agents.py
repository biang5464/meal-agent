"""
独立测试每个 Agent 节点。

运行方式:
    .venv\\Scripts\\python.exe -m pytest test_agents.py -v -s
"""

import pytest

# 确保 .env 加载（init_db / init_chroma 在 fixture 里调用）
from dotenv import load_dotenv
load_dotenv()

from tools.memory import init_db
from tools.recipe import init_chroma

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def init_services():
    """初始化 SQLite 和 ChromaDB（测试用临时路径）。"""
    # 必须先 import 维护表 ORM，create_all 才能建出所有表（与 main.py 一致）
    from models.maintenance import HealthCycleORM, PreferenceHistoryORM, ReminderORM  # noqa: F401
    init_db("./data/test_users.db")
    init_chroma("./data/test_chroma")
    yield


# 每个 test 用同一个 user_id 方便追踪状态
TEST_USER = "test_user_001"

# 最小合法的初始 state（memory_agent 会覆盖 user_brief）
def _base_state(user_input: str = "") -> dict:
    return {
        "user_id": TEST_USER,
        "user_input": user_input,
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
    }


# ── 1. memory_agent ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_agent():
    from agents.memory_agent import memory_agent

    state = _base_state("随便问一下")
    print(f"\n[memory_agent] 入参 user_id={state['user_id']}")

    out = await memory_agent(state)

    print(f"[memory_agent] 返回 user_brief 字段：")
    for k, v in out["user_brief"].items():
        print(f"  {k}: {v!r}")

    brief = out["user_brief"]
    assert "likes" in brief, "user_brief 缺少 likes"
    assert "dislikes" in brief, "user_brief 缺少 dislikes"
    assert "active_restrictions" in brief, "user_brief 缺少 active_restrictions"
    assert "recent_meals" in brief, "user_brief 缺少 recent_meals"
    assert "unread_reminders" in brief, "user_brief 缺少 unread_reminders"
    assert "period_status" in brief, "user_brief 缺少 period_status"
    print("[memory_agent] [OK] user_brief 字段完整")


# ── 2. supervisor_agent 意图分类（5条） ───────────────────────────────────────

_INTENT_CASES = [
    ("帮我推荐今天吃什么",           "MEAL"),
    ("西红柿多少钱",                  "PRICE"),
    ("我不喜欢香菜",                  "MEMORY_UPDATE"),
    ("今天不想吃香菜，推荐一个菜",    "UPDATE_AND_MEAL"),
    ("你好",                          "CHAT"),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("message,expected", _INTENT_CASES)
async def test_supervisor_agent(message, expected):
    from agents.memory_agent import memory_agent
    from agents.supervisor_agent import supervisor_agent

    state = _base_state(message)
    state = await memory_agent(state)   # supervisor 依赖 user_brief

    print(f"\n[supervisor] 入参: {message!r}")
    out = await supervisor_agent(state)
    intent = out["intent"]
    needs_meal = out["needs_meal"]
    print(f"[supervisor] 返回 intent={intent!r}  needs_meal={needs_meal}")

    assert intent == expected, (
        f"意图错误：期望 {expected!r}，实际 {intent!r}（消息：{message!r}）"
    )
    if expected == "UPDATE_AND_MEAL":
        assert needs_meal is True, "UPDATE_AND_MEAL 时 needs_meal 应为 True"
    print(f"[supervisor] [OK] {message!r} → {intent}")


# ── 3. meal_agent ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meal_agent():
    from agents.meal_agent import meal_agent
    from agents.memory_agent import memory_agent

    state = _base_state("帮我推荐今天吃什么")
    state = await memory_agent(state)
    state["intent"] = "MEAL"

    print(f"\n[meal_agent] 入参 user_input={state['user_input']!r}")
    print(f"[meal_agent] user_brief={state['user_brief']}")

    out = await meal_agent(state)
    result = out["result"]
    print(f"[meal_agent] 返回 result（前300字）：\n{result[:300]}")

    assert isinstance(result, str) and len(result) > 10, "result 为空或过短"
    print("[meal_agent] [OK] 返回非空推荐内容")


# ── 4. price_agent ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_price_agent():
    from agents.memory_agent import memory_agent
    from agents.price_agent import price_agent

    state = _base_state("chicken breast 多少钱")
    state = await memory_agent(state)
    state["intent"] = "PRICE"

    print(f"\n[price_agent] 入参 user_input={state['user_input']!r}")

    out = await price_agent(state)
    result = out["result"]
    print(f"[price_agent] 返回 result（前300字）：\n{result[:300]}")

    assert isinstance(result, str) and len(result) > 5, "result 为空"
    print("[price_agent] [OK] 返回非空价格信息")


# ── 5. update_agent ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_agent():
    from agents.memory_agent import memory_agent
    from agents.update_agent import update_agent

    state = _base_state("我不喜欢香菜")
    state = await memory_agent(state)
    state["intent"] = "MEMORY_UPDATE"

    print(f"\n[update_agent] 入参 user_input={state['user_input']!r}")

    out = await update_agent(state)
    result = out["result"]
    print(f"[update_agent] 返回 result：{result!r}")

    assert isinstance(result, str) and len(result) > 5, "result 为空"
    print("[update_agent] [OK] 返回确认消息")


# ── 6. chat_agent ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_agent():
    from agents.chat_agent import chat_agent
    from agents.memory_agent import memory_agent

    state = _base_state("你好")
    state = await memory_agent(state)
    state["intent"] = "CHAT"

    print(f"\n[chat_agent] 入参 user_input={state['user_input']!r}")
    print(f"[chat_agent] user_brief={state['user_brief']}")

    out = await chat_agent(state)
    result = out["result"]
    print(f"[chat_agent] 返回 result：{result!r}")

    assert isinstance(result, str) and len(result) > 5, "result 为空"
    print("[chat_agent] [OK] 返回非空回复")
