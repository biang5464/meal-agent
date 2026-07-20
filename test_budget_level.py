"""
最小测试：确认 memory_agent 正确把 user_brief.budget_level 读出来。

覆盖三种情况：
  Case 1 - 用户显式设置了 budget = "low"
  Case 2 - 用户显式设置了 budget = "high"
  Case 3 - 新用户（从未写过 budget），ORM 默认值 "mid" 也能读出来，不报错

运行方式：
    .venv\\Scripts\\python.exe -m pytest test_budget_level.py -v -s
"""

import pytest
from dotenv import load_dotenv
load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def init_services():
    from models.maintenance import HealthCycleORM, PreferenceHistoryORM, ReminderORM  # noqa: F401
    from tools.memory import init_db
    from tools.recipe import init_chroma
    init_db("./data/test_users.db")
    init_chroma("./data/test_chroma")
    yield


def _base_state(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "user_input": "test",
        "user_brief": {},
        "intent": "",
        "result": "",
        "messages": [],
        "needs_meal": False,
    }


# ── 辅助：直接写 budget 到 DB，绕过 LangChain tool 层 ──────────────────────

def _set_budget(user_id: str, budget: str | None):
    """直接操作 ORM 设置 budget，方便构造测试数据。"""
    from tools.memory import _session, _get_or_create
    with _session() as session:
        row = _get_or_create(session, user_id)
        row.budget = budget
        session.commit()


# ── Case 1: budget = "low" ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_budget_level_low():
    from agents.memory_agent import memory_agent

    user_id = "budget_test_low"
    _set_budget(user_id, "low")

    state = _base_state(user_id)
    out = await memory_agent(state)

    budget_level = out["user_brief"].get("budget_level")
    print(f"\n[Case 1] budget='low'  → user_brief.budget_level={budget_level!r}")

    assert budget_level == "low", f"期望 'low'，实际 {budget_level!r}"
    print("[Case 1] PASS")


# ── Case 2: budget = "high" ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_budget_level_high():
    from agents.memory_agent import memory_agent

    user_id = "budget_test_high"
    _set_budget(user_id, "high")

    state = _base_state(user_id)
    out = await memory_agent(state)

    budget_level = out["user_brief"].get("budget_level")
    print(f"\n[Case 2] budget='high' → user_brief.budget_level={budget_level!r}")

    assert budget_level == "high", f"期望 'high'，实际 {budget_level!r}"
    print("[Case 2] PASS")


# ── Case 3: 新用户，budget 从未设置（ORM 默认值 "mid"） ─────────────────────

@pytest.mark.asyncio
async def test_budget_level_new_user():
    """
    新用户第一次写入时 ORM default="mid" 会生效。
    要确认的是：不会抛异常，budget_level 是字符串（"mid" 或 ""），不是 None。
    """
    from agents.memory_agent import memory_agent

    # 使用一个全新 user_id，让 _get_or_create 走新建分支
    user_id = "budget_test_brand_new_user_xyz"

    state = _base_state(user_id)
    out = await memory_agent(state)

    budget_level = out["user_brief"].get("budget_level")
    print(f"\n[Case 3] 新用户  → user_brief.budget_level={budget_level!r}")

    # ORM default="mid"，所以新用户拿到 "mid" 是合理的；
    # 但无论是 "mid" 还是 ""，都不能是 None 或抛异常
    assert isinstance(budget_level, str), (
        f"budget_level 应为字符串，实际类型 {type(budget_level)}"
    )
    assert budget_level != "None", "budget_level 不应是字符串 'None'"
    print(f"[Case 3] PASS  (新用户 ORM 默认值 → {budget_level!r}，符合预期)")
