"""定时维护 Agent：健康周期管理、偏好权重衰减、经期提醒。"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from models.maintenance import (
    HealthCycleORM,
    PreferenceHistoryORM,
    ReminderORM,
    decode_symptoms,
)
from models.user import UserProfileORM
from tools.memory import _session  # 共享同一个 SQLAlchemy engine

logger = logging.getLogger(__name__)

# ── APScheduler ──────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

# ── DB 内部工具 ───────────────────────────────────────────────────────────────

def _get_or_create_cycle(session: Session, user_id: str) -> HealthCycleORM:
    row = session.get(HealthCycleORM, user_id)
    if row is None:
        row = HealthCycleORM(user_id=user_id)
        session.add(row)
        session.flush()
    return row


def _set_temporary_restriction(
    session: Session,
    user_id: str,
    value: str,
    reason: str,
    duration_days: int,
) -> None:
    entry = PreferenceHistoryORM(
        user_id=user_id,
        preference_type="restriction",
        value=value,
        reason=reason,
        expires_at=datetime.utcnow() + timedelta(days=duration_days),
        weight=1.0,
    )
    session.add(entry)


def _add_reminder(
    session: Session, user_id: str, message: str, reminder_type: str
) -> None:
    session.add(
        ReminderORM(user_id=user_id, message=message, reminder_type=reminder_type)
    )


def _cycle_phase(cycle: HealthCycleORM, today: date) -> dict:
    """
    计算当前处于经期周期的哪个阶段。

    Returns dict with keys:
        phase: "in_period" | "pre_period" | "post_period" | "normal"
        cycle_day: 当前周期第几天（1-indexed）
        days_until_next: 距下次经期天数（负数表示已在经期中）
        next_start: 预测下次经期日期
    """
    if cycle.last_start is None:
        return {
            "phase": "unknown",
            "cycle_day": None,
            "days_until_next": None,
            "next_start": None,
        }

    days_since = (today - cycle.last_start).days
    cycle_day = days_since % cycle.cycle_days + 1  # 1-indexed
    days_until_next = cycle.cycle_days - days_since % cycle.cycle_days

    # 预测下次经期起始日
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


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def set_health_cycle(
    user_id: str,
    cycle_type: str = "menstrual",
    cycle_days: int = 28,
    period_days: int = 5,
    last_start: str = "",
) -> str:
    """
    设置用户的健康周期信息。

    Args:
        user_id:     用户唯一标识
        cycle_type:  周期类型，menstrual 或 custom
        cycle_days:  完整周期天数，默认 28
        period_days: 经期持续天数，默认 5
        last_start:  上次经期开始日期，格式 YYYY-MM-DD；留空则不更新

    Returns:
        操作结果描述。
    """
    with _session() as session:
        row = _get_or_create_cycle(session, user_id)
        row.cycle_type = cycle_type
        row.cycle_days = cycle_days
        row.period_days = period_days
        if last_start:
            try:
                row.last_start = date.fromisoformat(last_start)
            except ValueError:
                return f"错误：last_start 格式不正确，应为 YYYY-MM-DD，收到 '{last_start}'"
        session.commit()

    return (
        f"已设置 {user_id} 的健康周期：{cycle_type}，"
        f"周期 {cycle_days} 天，经期 {period_days} 天。"
    )


@tool
def predict_next_period(user_id: str) -> dict:
    """
    根据上次经期日期预测下次经期，并判断当前是否在经期中。

    Args:
        user_id: 用户唯一标识

    Returns:
        包含 next_start（下次经期日期）、days_until_next（距今天数）、
        in_period（是否在经期中）、cycle_day（当前周期第几天）的字典。
    """
    with _session() as session:
        row = session.get(HealthCycleORM, user_id)

    if row is None or row.last_start is None:
        return {"error": "尚未设置健康周期，请先调用 set_health_cycle。"}

    info = _cycle_phase(row, date.today())
    return {
        "user_id": user_id,
        "next_start": info["next_start"],
        "days_until_next": info["days_until_next"],
        "in_period": info["phase"] == "in_period",
        "cycle_day": info["cycle_day"],
        "phase": info["phase"],
    }


@tool
def get_period_diet_advice(user_id: str) -> dict:
    """
    根据当前经期阶段返回个性化饮食建议。

    阶段划分：
      - 经前3天：补铁补血，推荐红枣、菠菜、瘦肉
      - 经期中：温热食物，避免辛辣生冷
      - 经期后7天：滋补恢复，推荐鸡蛋、牛奶、鱼
      - 其他：常规均衡饮食

    Args:
        user_id: 用户唯一标识

    Returns:
        包含 phase（阶段）、advice（建议文本）、recommended_foods（推荐食材）、
        avoid_foods（避免食材）的字典。
    """
    with _session() as session:
        row = session.get(HealthCycleORM, user_id)

    if row is None or row.last_start is None:
        return {
            "phase": "unknown",
            "advice": "尚未设置健康周期，无法给出阶段性建议。",
            "recommended_foods": [],
            "avoid_foods": [],
        }

    info = _cycle_phase(row, date.today())
    phase = info["phase"]

    advice_map = {
        "pre_period": {
            "label": "经前期（前3天）",
            "advice": "经期即将来临，建议补铁补血，保持情绪稳定，避免过度劳累。",
            "recommended_foods": ["红枣", "菠菜", "瘦肉", "黑豆", "桂圆"],
            "avoid_foods": ["辛辣食物", "酒精", "咖啡因", "生冷食物"],
        },
        "in_period": {
            "label": "经期中",
            "advice": "经期中应以温热食物为主，促进血液循环，减轻不适。",
            "recommended_foods": ["红糖姜茶", "鸡汤", "山药", "红豆", "温热粥品"],
            "avoid_foods": ["冷饮", "生冷水果", "辛辣食物", "冰激凌", "螃蟹"],
        },
        "post_period": {
            "label": "经期后（恢复期）",
            "advice": "经期结束后注重滋补恢复，补充流失的铁质和蛋白质。",
            "recommended_foods": ["鸡蛋", "牛奶", "鱼", "猪肝", "黑芝麻", "枸杞"],
            "avoid_foods": [],
        },
        "normal": {
            "label": "卵泡期/黄体期",
            "advice": "当前处于周期平稳阶段，保持均衡饮食即可。",
            "recommended_foods": ["新鲜蔬菜", "优质蛋白", "全谷物", "坚果"],
            "avoid_foods": [],
        },
    }

    entry = advice_map.get(phase, advice_map["normal"])
    return {
        "phase": entry["label"],
        "cycle_day": info["cycle_day"],
        "days_until_next": info["days_until_next"],
        "advice": entry["advice"],
        "recommended_foods": entry["recommended_foods"],
        "avoid_foods": entry["avoid_foods"],
    }


@tool
def get_unread_reminders(user_id: str) -> List[dict]:
    """
    获取用户所有未读提醒，并将其标记为已读。

    Args:
        user_id: 用户唯一标识

    Returns:
        未读提醒列表，每项包含 id、message、reminder_type、created_at。
    """
    with _session() as session:
        rows = (
            session.query(ReminderORM)
            .filter(
                ReminderORM.user_id == user_id,
                ReminderORM.is_read == False,  # noqa: E712
            )
            .order_by(ReminderORM.created_at.desc())
            .all()
        )

        result = [
            {
                "id": r.id,
                "message": r.message,
                "reminder_type": r.reminder_type,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]

        for r in rows:
            r.is_read = True
        session.commit()

    return result


# ── 维护任务（内部函数，由 Scheduler 调用）────────────────────────────────────

def clean_expired_restrictions(user_id: str) -> int:
    """删除已过期的临时限制记录，返回删除数量。"""
    with _session() as session:
        now = datetime.utcnow()
        rows = (
            session.query(PreferenceHistoryORM)
            .filter(
                PreferenceHistoryORM.user_id == user_id,
                PreferenceHistoryORM.expires_at.isnot(None),
                PreferenceHistoryORM.expires_at < now,
            )
            .all()
        )
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()

    if count:
        _add_reminder_direct(
            user_id=user_id,
            message=f"已自动清除 {count} 条过期饮食限制。",
            reminder_type="restriction_expired",
        )
    return count


def update_preference_weights(user_id: str) -> None:
    """对每条偏好历史重新计算时间衰减权重。超过90天未强化降至 0.1。"""
    with _session() as session:
        rows = (
            session.query(PreferenceHistoryORM)
            .filter(
                PreferenceHistoryORM.user_id == user_id,
                PreferenceHistoryORM.expires_at.is_(None),  # 仅处理永久记录
            )
            .all()
        )
        for row in rows:
            days_since = (datetime.utcnow() - row.last_reinforced).days
            # 线性衰减：0天→1.0，90天→0.1
            row.weight = round(max(0.1, 1.0 - days_since / 90 * 0.9), 4)
        session.commit()


def check_period_reminder(user_id: str) -> None:
    """预测经期，提前3天自动设置临时饮食限制并写入提醒。"""
    result = predict_next_period.invoke({"user_id": user_id})
    if result.get("error"):
        return

    phase = result.get("phase", "normal")
    days_until = result.get("days_until_next", 999)
    in_period = result.get("in_period", False)

    if not (in_period or 0 < days_until <= 3):
        return

    with _session() as session:
        # 防重：今天是否已写过经期提醒
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        already = (
            session.query(ReminderORM)
            .filter(
                ReminderORM.user_id == user_id,
                ReminderORM.reminder_type == "period",
                ReminderORM.created_at >= today_start,
            )
            .first()
        )
        if already:
            return

        _set_temporary_restriction(
            session,
            user_id=user_id,
            value="不能吃辛辣生冷",
            reason="经期即将开始",
            duration_days=8,
        )

        if in_period:
            msg = "当前处于经期，请保持饮食温热清淡，避免辛辣生冷（已自动设置限制）。"
        else:
            msg = (
                f"经期提醒：预计 {days_until} 天后来月经，"
                "已自动设置饮食限制（避免辛辣生冷，持续8天）。"
            )
        _add_reminder(session, user_id=user_id, message=msg, reminder_type="period")
        session.commit()


def _add_reminder_direct(user_id: str, message: str, reminder_type: str) -> None:
    """直接写提醒（无需外层 session）。"""
    with _session() as session:
        _add_reminder(session, user_id=user_id, message=message, reminder_type=reminder_type)
        session.commit()


# ── Scheduler Jobs ────────────────────────────────────────────────────────────

async def daily_price_update() -> None:
    """
    每日凌晨批量查价：
    1. 从 price_history 提取最近30天内查过的不重复商品
    2. 对每个商品重新查所有平台价格（record_price 在内部自动写入）
    3. price_history 为空时直接跳过（冷启动场景）
    """
    from tools.price_history import get_tracked_terms
    from tools.price import get_ingredient_price

    terms = get_tracked_terms(days=30)
    if not terms:
        print("[maintenance] price_history 为空，跳过批量查价")
        return

    print(f"[maintenance] 开始批量查价，共 {len(terms)} 个商品")
    for term in terms:
        try:
            result = get_ingredient_price.invoke({"ingredient": term})
            available_count = len([r for r in result["all_results"] if r["available"]])
            print(f"[maintenance] {term} → {available_count} 个平台有价格")
        except Exception as e:
            print(f"[maintenance] {term} 查价失败: {e}")

    print("[maintenance] 批量查价完成")


async def daily_budget_tier_update() -> None:
    """
    每日凌晨4点重新计算所有菜谱的 budget_tier 并更新 ChromaDB。
    在 daily_price_update（3点）完成后运行，确保使用最新价格。
    """
    import json as _json
    from tools.budget_tier import calc_recipe_budget_tier
    from tools.recipe import get_all_recipes, update_recipe_budget_tier

    print("[maintenance] 开始更新菜谱 budget_tier...")
    recipes = get_all_recipes()
    updated = 0

    for recipe in recipes:
        name = recipe["name"]
        ingredients = recipe["ingredients"]  # 已由 _meta_to_dict 反序列化为 list

        try:
            tier = calc_recipe_budget_tier(ingredients)
            success = update_recipe_budget_tier(name, tier)
            if success:
                updated += 1
                print(f"[maintenance] {name} → {tier}")
        except Exception as e:
            print(f"[maintenance] {name} 计算失败: {e}")

    print(f"[maintenance] budget_tier 更新完成，共 {updated}/{len(recipes)} 条")


async def _daily_maintenance() -> None:
    """凌晨2点：遍历所有用户，执行清理/权重更新/经期检查。"""
    try:
        with _session() as session:
            user_ids = {
                row[0] for row in session.query(UserProfileORM.user_id).all()
            } | {
                row[0] for row in session.query(HealthCycleORM.user_id).all()
            }

        for uid in user_ids:
            try:
                deleted = clean_expired_restrictions(uid)
                update_preference_weights(uid)
                check_period_reminder(uid)
                logger.debug(
                    "[maintenance] %s: expired=%d cleaned", uid, deleted
                )
            except Exception:
                logger.exception("[maintenance] task failed for user %s", uid)

    except Exception:
        logger.exception("[maintenance] daily job failed to start")


def start_scheduler() -> None:
    """在 FastAPI lifespan 启动时调用。重复调用安全（已运行则直接返回）。"""
    if scheduler.running:
        logger.info("Scheduler already running, skipping duplicate start.")
        return

    from apscheduler.triggers.cron import CronTrigger
    from tools.memory import cleanup_expired_conversation_memory

    scheduler.add_job(
        _daily_maintenance,
        trigger="cron",
        hour=2,
        minute=0,
        id="daily_maintenance",
        replace_existing=True,
    )
    scheduler.add_job(
        daily_price_update,
        trigger="cron",
        hour=3,
        minute=0,
        id="daily_price_update",
        replace_existing=True,
    )
    scheduler.add_job(
        daily_budget_tier_update,
        trigger="cron",
        hour=4,
        minute=0,
        id="daily_budget_tier_update",
        replace_existing=True,
    )
    scheduler.add_job(
        cleanup_expired_conversation_memory,
        trigger=CronTrigger(hour=4, minute=10),
        id="cleanup_conversation_memory",
        name="清理过期对话记忆",
        replace_existing=True,
    )

    from agents.daily_recommendation_agent import (
        generate_daily_recommendations,
        retry_daily_recommendations,
    )
    scheduler.add_job(
        generate_daily_recommendations,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_recommendation_generate",
        name="每日推荐生成",
        replace_existing=True,
    )
    scheduler.add_job(
        retry_daily_recommendations,
        trigger=CronTrigger(hour=8, minute=5),
        id="daily_recommendation_retry_1",
        name="每日推荐重试1",
        replace_existing=True,
    )
    scheduler.add_job(
        retry_daily_recommendations,
        trigger=CronTrigger(hour=8, minute=15),
        id="daily_recommendation_retry_2",
        name="每日推荐重试2",
        replace_existing=True,
    )
    scheduler.add_job(
        retry_daily_recommendations,
        trigger=CronTrigger(hour=8, minute=30),
        id="daily_recommendation_retry_3",
        name="每日推荐重试3",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("MaintenanceAgent scheduler started (daily at 02:00 & 03:00 & 08:00 CST).")


def stop_scheduler() -> None:
    """在 FastAPI lifespan 关闭时调用。"""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("MaintenanceAgent scheduler stopped.")


# ── Tools 导出 ────────────────────────────────────────────────────────────────

MAINTENANCE_TOOLS = [
    set_health_cycle,
    predict_next_period,
    get_period_diet_advice,
    get_unread_reminders,
]
