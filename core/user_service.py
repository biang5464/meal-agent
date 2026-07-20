"""
core/user_service.py
Phase 6A：SQLite 用户业务数据只读服务层，遵循 ToolResult 协议。

所有 impl 函数：同步，内部创建并关闭 Session，由 ToolExecutor 经 asyncio.to_thread 包装。
所有 result 函数：异步，调用 tool_executor.execute() 返回 ToolResult。

约束：
  - impl 函数不接受外部 Session 参数，自行创建关闭
  - result 函数不自己 to_thread，由 ToolExecutor 负责线程包装
  - SQLite 关键读取失败必须返回 ok=False，不得伪装成"空用户"
  - 成功无数据（ok=True, data=None）与依赖故障（ok=False）必须区分
"""
from __future__ import annotations

from datetime import date, datetime, timedelta


# ── 内部工具（从 maintenance_agent._cycle_phase 内联，避免跨层导入）────────────

def _cycle_phase(cycle, today: date) -> dict:
    if cycle.last_start is None:
        return {
            "phase": "unknown",
            "cycle_day": None,
            "days_until_next": None,
            "next_start": None,
        }
    days_since = (today - cycle.last_start).days
    cycle_day = days_since % cycle.cycle_days + 1
    days_until_next = cycle.cycle_days - days_since % cycle.cycle_days
    next_start = cycle.last_start + timedelta(
        days=((days_since // cycle.cycle_days) + 1) * cycle.cycle_days
    )
    if cycle_day <= cycle.period_days:
        phase = "in_period"
    elif cycle_day <= cycle.period_days + 7:
        phase = "post_period"
    elif days_until_next <= 3:
        phase = "pre_period"
    else:
        phase = "normal"
    return {
        "phase": phase,
        "cycle_day": cycle_day,
        "days_until_next": days_until_next,
        "next_start": next_start.isoformat(),
    }


# ── 用户画像（SQLite）─────────────────────────────────────────────────────────

def _get_user_profile_impl(user_id: str) -> dict | None:
    from core.database import SessionLocal
    from models.user import UserProfileORM, orm_to_dict
    with SessionLocal() as session:
        row = session.get(UserProfileORM, user_id)
    if row is None:
        return None
    return orm_to_dict(row)


async def get_user_profile_result(user_id: str):
    """SQLite 用户画像读取。

    - ok=True, data=dict   → 用户存在
    - ok=True, data=None   → 新用户，无记录（正常情况，非错误）
    - ok=False             → SQLite 故障（不伪装成空用户）
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _get_user_profile_impl,
        user_id,
        policy_name="database_read",
        tool_name="get_user_profile",
        source="sqlite",
    )


# ── 饮食偏好历史（SQLite）────────────────────────────────────────────────────

def _get_preference_history_impl(user_id: str) -> list[dict]:
    from core.database import SessionLocal
    from models.maintenance import PreferenceHistoryORM
    from sqlalchemy import or_
    now = datetime.utcnow()
    with SessionLocal() as session:
        rows = (
            session.query(PreferenceHistoryORM)
            .filter(
                PreferenceHistoryORM.user_id == user_id,
                PreferenceHistoryORM.preference_type == "restriction",
                or_(
                    PreferenceHistoryORM.expires_at.is_(None),
                    PreferenceHistoryORM.expires_at > now,
                ),
            )
            .all()
        )
        return [
            {
                "value": r.value,
                "reason": r.reason or "",
                "weight": r.weight,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            }
            for r in rows
        ]


async def get_preference_history_result(user_id: str):
    """活跃饮食限制读取（preference_type='restriction'，未过期）。

    - ok=True, data=[...]  → 读取成功（可能为空列表）
    - ok=False             → SQLite 故障
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _get_preference_history_impl,
        user_id,
        policy_name="database_read",
        tool_name="get_preference_history",
        source="sqlite",
    )


# ── 未读提醒（SQLite，只读，不标已读）────────────────────────────────────────

def _get_unread_reminders_readonly_impl(user_id: str) -> list[dict]:
    """只查询未读提醒，不修改 is_read 状态，不执行写入操作。"""
    from core.database import SessionLocal
    from models.maintenance import ReminderORM
    with SessionLocal() as session:
        rows = (
            session.query(ReminderORM)
            .filter(
                ReminderORM.user_id == user_id,
                ReminderORM.is_read == False,  # noqa: E712
            )
            .order_by(ReminderORM.created_at.desc())
            .limit(5)
            .all()
        )
        return [
            {
                "id": r.id,
                "message": r.message,
                "reminder_type": r.reminder_type,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


async def get_unread_reminders_readonly_result(user_id: str):
    """只读查询未读提醒，不标记已读。供 memory_agent 组装 user_brief 使用。

    - ok=True, data=[...]  → 读取成功（可能为空列表）
    - ok=False             → SQLite 故障
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _get_unread_reminders_readonly_impl,
        user_id,
        policy_name="database_read",
        tool_name="get_unread_reminders_readonly",
        source="sqlite",
    )


# ── 健康周期预测（SQLite，只读）──────────────────────────────────────────────

def _predict_next_period_impl(user_id: str) -> dict | None:
    from core.database import SessionLocal
    from models.maintenance import HealthCycleORM
    with SessionLocal() as session:
        row = session.get(HealthCycleORM, user_id)
    if row is None or row.last_start is None:
        return None  # 未设置周期，正常情况（非错误）
    info = _cycle_phase(row, date.today())
    return {
        "user_id": user_id,
        "next_start": info["next_start"],
        "days_until_next": info["days_until_next"],
        "in_period": info["phase"] == "in_period",
        "cycle_day": info["cycle_day"],
        "phase": info["phase"],
    }


async def predict_next_period_result(user_id: str):
    """健康周期预测，不依赖 maintenance_agent @tool。

    - ok=True, data=dict   → 有周期记录，返回预测
    - ok=True, data=None   → 未设置健康周期（正常情况）
    - ok=False             → SQLite 故障
    """
    from tools.tool_executor import tool_executor
    return await tool_executor.execute(
        _predict_next_period_impl,
        user_id,
        policy_name="database_read",
        tool_name="predict_next_period",
        source="sqlite",
    )
