"""
验证 budget default=None 后的完整数据流：
  1. 新建用户 → DB 里 budget 是 NULL
  2. get_user_profile 读取 → profile["budget"] 是 None
  3. memory_agent 组装 → user_brief["budget_level"] 是 ""
  4. 显式 update_user_profile budget="mid" → 再读，得到 "mid"，与"未设置"区分

运行：.venv\\Scripts\\python.exe -m pytest test_budget_null.py -v -s
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


# ── 验证 1：新建用户 DB 里 budget 是 NULL ─────────────────────────────────────

def test_new_user_budget_is_null_in_db():
    import sqlite3
    from tools.memory import _session, _get_or_create

    user_id = "null_budget_db_check"

    # 通过 ORM 创建用户（走 _get_or_create 分支）
    with _session() as session:
        _get_or_create(session, user_id)
        session.commit()

    # 直接查 SQLite，确认 budget 是 NULL
    import os
    db_path = "./data/test_users.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT budget FROM user_profiles WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()

    raw_value = row[0] if row else "NOT FOUND"
    print(f"\n[验证1] DB 原始值: {raw_value!r}")

    assert raw_value is None, f"期望 NULL，实际 {raw_value!r}"
    print("[验证1] PASS — 新建用户 budget 在 DB 里是 NULL")


# ── 验证 2：get_user_profile 返回 budget=None ────────────────────────────────

def test_get_profile_returns_none_budget():
    from tools.memory import get_user_profile

    user_id = "null_budget_profile_check"
    profile = get_user_profile.invoke({"user_id": user_id})
    budget_val = profile.get("budget")

    print(f"\n[验证2] get_user_profile 返回 budget: {budget_val!r}")

    assert budget_val is None, f"期望 None，实际 {budget_val!r}"
    print("[验证2] PASS — profile['budget'] 是 None")


# ── 验证 3：memory_agent 组装 budget_level = "" ──────────────────────────────

@pytest.mark.asyncio
async def test_memory_agent_budget_level_empty():
    from agents.memory_agent import memory_agent

    user_id = "null_budget_brief_check"
    state = _base_state(user_id)
    out = await memory_agent(state)

    budget_level = out["user_brief"].get("budget_level")
    print(f"\n[验证3] user_brief['budget_level']: {budget_level!r}")

    assert budget_level == "", f"期望 ''，实际 {budget_level!r}"
    print("[验证3] PASS — budget_level 是空字符串，与'已设置'用户区分开")


# ── 验证 4：显式设置 budget='mid'，能与"未设置"区分 ─────────────────────────

@pytest.mark.asyncio
async def test_explicit_mid_differs_from_unset():
    from agents.memory_agent import memory_agent
    from tools.memory import _session, update_user_profile
    from models.user import UserProfileORM

    user_id = "null_budget_explicit_mid"
    state = _base_state(user_id)

    # 先重置 budget 为 NULL，确保跨运行可重复
    with _session() as session:
        row = session.get(UserProfileORM, user_id)
        if row is not None:
            row.budget = None
            session.commit()

    # 先确认初始状态是空
    out_before = await memory_agent(state)
    level_before = out_before["user_brief"].get("budget_level")
    print(f"\n[验证4] 设置前 budget_level: {level_before!r}")
    assert level_before == "", f"设置前期望 ''，实际 {level_before!r}"

    # 显式设置 budget='mid'
    result = update_user_profile.invoke({
        "user_id": user_id,
        "field": "budget",
        "value": "mid",
    })
    print(f"[验证4] update_user_profile 返回: {result!r}")

    # 再读，应该拿到 'mid'
    out_after = await memory_agent(state)
    level_after = out_after["user_brief"].get("budget_level")
    print(f"[验证4] 设置后 budget_level: {level_after!r}")

    assert level_after == "mid", f"设置后期望 'mid'，实际 {level_after!r}"
    print("[验证4] PASS — '' 和 'mid' 已能区分")
