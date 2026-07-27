"""每日餐饮推荐 Agent：过滤、选菜、LLM 推理、DB 写入、重试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import secrets
import time
from datetime import date, datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models.daily_recommendation import DailyRecommendationORM
from tools.dead_letter import write_dead_letter_result
from tools.memory import _session
from tools.recipe import get_all_recipes

logger = logging.getLogger(__name__)

MAIN_DISH_CATEGORIES = {"荤菜", "肉类", "海鲜", "禽类"}
VEGGIE_CATEGORIES    = {"蔬菜", "半荤素", "素菜", "豆腐"}
SOUP_CATEGORIES      = {"汤", "副菜", "轻食", "粥"}

_RETRY_QUEUE_KEY = "daily_rec_retry_queue"

_LOCK_TTL = 180          # lock TTL in seconds; covers LLM timeout (60s) + full generation margin
_LOCK_WAIT_TIMEOUT = 12  # max seconds to poll for lock acquisition
_LOCK_POLL_INTERVAL = 3  # seconds between poll attempts

_LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _normalize_recipe_steps(value) -> list[str]:
    """Normalize recipe steps to list[str] for stable JSON storage.

    - str  → single-element list (stripped); empty string → []
    - list → filter non-str / blank items, strip each
    - None / other → []
    Does not modify the original object.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    result.append(stripped)
        return result
    return []


class LockBusy(Exception):
    """Lock held by another request; wait timeout elapsed."""


class LockDegraded(Exception):
    """Redis unavailable when distributed lock is required for generation."""


async def _try_acquire_lock(lock_key: str, token: str):
    """Single SET NX attempt. Returns ToolResult: data=True if acquired, data=None if busy."""
    from core.cache import _client as _redis_client
    from tools.tool_executor import tool_executor

    def _set_nx(k, t):
        return _redis_client.set(k, t, nx=True, ex=_LOCK_TTL)

    return await tool_executor.execute(
        _set_nx, lock_key, token,
        policy_name="redis_write",
        tool_name="acquire_daily_rec_lock",
    )


async def _release_daily_rec_lock(lock_key: str, token: str) -> None:
    """CAS-delete the lock only if our token still matches (prevents releasing another request's lock)."""
    from core.cache import _client as _redis_client
    from tools.tool_executor import tool_executor

    def _cas_del(k, t):
        return _redis_client.eval(_LUA_RELEASE, 1, k, t)

    try:
        await tool_executor.execute(
            _cas_del, lock_key, token,
            policy_name="redis_write",
            tool_name="release_daily_rec_lock",
        )
    except Exception as e:
        logger.warning("[daily_rec] lock release failed key=%s: %s", lock_key, e)


# ── 核心过滤 ───────────────────────────────────────────────────────────────────

def _filter_recipes(
    recipes: list[dict],
    dislikes: list[str] | None = None,
    restrictions: list[str] | None = None,
    budget: str = "",
    recent_dishes: list[str] | None = None,
    same_day_dishes: list[str] | None = None,
) -> list[dict]:
    """
    过滤菜谱：
    - 硬过滤：忌口食材（dislikes）、饮食限制关键词（restrictions）、同天另一餐（same_day_dishes，永不放宽）
    - 软过滤：预算档位（budget）、近期去重（recent_dishes）
    """
    dislikes     = [d.strip() for d in (dislikes or [])     if d.strip()]
    restrictions = [r.strip() for r in (restrictions or []) if r.strip()]
    recent       = set(recent_dishes or [])
    same_day     = set(same_day_dishes or [])

    result = []
    for r in recipes:
        ingredients_str = " ".join(r.get("ingredients", []))
        flavor = r.get("flavor", "")
        steps  = r.get("steps", "")

        # 硬过滤：忌口食材
        if any(d in ingredients_str for d in dislikes):
            continue

        # 硬过滤：饮食限制关键词（命中 ingredient/flavor/steps 任一则排除）
        if any(
            rk in ingredients_str or rk in flavor or rk in steps
            for rk in restrictions
        ):
            continue

        # 硬过滤：同天另一餐菜（永不放宽，防止午晚餐重复）
        if r.get("name") in same_day:
            continue

        # 软过滤：预算档位（unknown 视为可接受任何预算）
        tier = r.get("budget_tier", "unknown")
        if budget in ("low", "mid", "high") and tier not in ("unknown", budget):
            continue

        # 软过滤：近期去重
        if r.get("name") in recent:
            continue

        result.append(r)

    return result


# ── 结构化选菜 ─────────────────────────────────────────────────────────────────

def _select_by_structure(recipes: list[dict]) -> list[dict]:
    """
    从候选菜谱按结构选 1主菜 + 1素菜/副菜 + 1汤。
    某类不足时从剩余菜中随机补充，保证不重复。
    """
    mains   = [r for r in recipes if r.get("category") in MAIN_DISH_CATEGORIES]
    veggies = [r for r in recipes if r.get("category") in VEGGIE_CATEGORIES]
    soups   = [r for r in recipes if r.get("category") in SOUP_CATEGORIES]

    selected: list[dict] = []
    used: set[str] = set()

    def _pick(pool: list[dict]) -> dict | None:
        available = [r for r in pool if r.get("name") not in used]
        if not available:
            return None
        choice = random.choice(available)
        used.add(choice["name"])
        return choice

    for pool in (mains, veggies, soups):
        dish = _pick(pool)
        if dish:
            selected.append(dish)

    # 不足3道时从剩余任意菜补充
    if len(selected) < 3:
        remaining = [r for r in recipes if r.get("name") not in used]
        random.shuffle(remaining)
        for r in remaining:
            if len(selected) >= 3:
                break
            selected.append(r)
            used.add(r["name"])

    return selected[:3]


# ── 评分兜底 ───────────────────────────────────────────────────────────────────

def _rule_fallback(recipes: list[dict], n: int = 3) -> list[dict]:
    """
    评分兜底选菜：low 预算加分，加随机扰动，取 top-n。
    """
    tier_score = {"low": 3, "mid": 2, "high": 1, "unknown": 2}
    scored = [
        (r, tier_score.get(r.get("budget_tier", "unknown"), 2) + random.random())
        for r in recipes
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:n]]


# ── LLM 推理 ───────────────────────────────────────────────────────────────────

_REASONING_SYSTEM = (
    "你是专业营养师助手。根据用户画像和推荐菜品，生成简短的 JSON 推荐理由。\n"
    "输出格式（严格 JSON，不要包含其他内容）：\n"
    '{"dishes": [{"dish": "菜名", "reason": "推荐原因"}], '
    '"summary": "整体营养评价（1-2句）"}\n'
    "注意：dish 字段必须与输入菜名完全一致。"
)


async def _generate_reasoning(
    dishes: list[dict],
    user_profile: dict,
    meal_type: str,
) -> str:
    """调用 LLM 生成 JSON 推荐理由；失败时返回简单兜底 JSON。"""
    dish_names = [d.get("name", "") for d in dishes]
    prompt = (
        f"餐类：{meal_type}（{'午餐' if meal_type == 'lunch' else '晚餐'}）\n"
        f"推荐菜品：{', '.join(dish_names)}\n"
        f"用户偏好：喜欢 {user_profile.get('likes', [])}，"
        f"忌口 {user_profile.get('dislikes', [])}，"
        f"预算 {user_profile.get('budget', 'mid')}\n"
        "请生成推荐理由 JSON。"
    )
    async def _reasoning_impl(messages: list):
        llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.3,
            max_retries=0,
            request_timeout=65,
        )
        return await llm.ainvoke(messages)

    from tools.tool_executor import tool_executor
    messages = [SystemMessage(content=_REASONING_SYSTEM), HumanMessage(content=prompt)]
    tool_result = await tool_executor.execute(
        _reasoning_impl,
        messages,
        policy_name="llm_generation",
        tool_name="generate_reasoning",
        fallback_data=None,
    )
    fallback = {
        "dishes": [{"dish": n, "reason": "营养均衡，适合今日"} for n in dish_names],
        "summary": "今日推荐兼顾荤素搭配，营养均衡。",
    }
    if not tool_result.ok or tool_result.data is None:
        logger.warning("[daily_rec] 生成推理失败: %s", tool_result.error)
        return json.dumps(fallback, ensure_ascii=False)
    try:
        raw = tool_result.data.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.warning("[daily_rec] 生成推理 JSON 解析失败: %s", e)
        return json.dumps(fallback, ensure_ascii=False)


# ── 数据库辅助 ─────────────────────────────────────────────────────────────────

def _get_user_profile(user_id: str) -> dict:
    """直接查 SQLite 获取用户画像，避免 LangChain 工具调用开销。"""
    from models.user import UserProfileORM, orm_to_dict
    try:
        with _session() as session:
            row = session.get(UserProfileORM, user_id)
            return orm_to_dict(row) if row else {}
    except Exception:
        return {}


def _get_recent_dishes(user_id: str, days: int = 7) -> tuple[list[str], str]:
    """获取用户最近 N 天已推荐的菜名。返回 (names, status)。"""
    since = date.today() - timedelta(days=days)
    try:
        with _session() as session:
            rows = (
                session.query(DailyRecommendationORM)
                .filter(
                    DailyRecommendationORM.user_id == user_id,
                    DailyRecommendationORM.date >= since,
                )
                .all()
            )
        names: list[str] = []
        for row in rows:
            if row.dishes:
                names.extend(d.get("name", "") for d in row.dishes)
        return names, ("success" if names else "empty")
    except Exception:
        return [], "read_degraded"


def _get_same_day_dishes(user_id: str, today: date, meal_type: str) -> tuple[list[str], str]:
    """查询同天另一餐的已推荐菜名，用于跨餐硬约束去重。返回 (names, status)。"""
    other_meal = "dinner" if meal_type == "lunch" else "lunch"
    try:
        with _session() as session:
            rows = (
                session.query(DailyRecommendationORM)
                .filter(
                    DailyRecommendationORM.user_id == user_id,
                    DailyRecommendationORM.date == today,
                    DailyRecommendationORM.meal_type == other_meal,
                )
                .all()
            )
        names: list[str] = []
        for row in rows:
            if row.dishes:
                names.extend(d.get("name", "") for d in row.dishes)
        return names, ("success" if names else "empty")
    except Exception:
        return [], "read_degraded"


def _orm_to_dict(rec: DailyRecommendationORM) -> dict:
    return {
        "id":           rec.id,
        "user_id":      rec.user_id,
        "date":         rec.date.isoformat(),
        "meal_type":    rec.meal_type,
        "dishes":       rec.dishes,
        "reasoning":    rec.reasoning,
        "generated_by": rec.generated_by,
        "source_status": rec.source_status,
        "is_read":      rec.is_read,
        "created_at":   rec.created_at.isoformat() if rec.created_at else None,
    }


# ── 主生成函数 ─────────────────────────────────────────────────────────────────

async def generate_for_user(
    user_id: str,
    meal_type: str = "lunch",
    date_str: str | None = None,
    generated_by: str = "scheduled",
) -> dict | None:
    """
    为单个用户生成今日推荐（幂等：已存在则直接返回已有记录）。
    """
    from sqlalchemy.exc import IntegrityError

    today = date.fromisoformat(date_str) if date_str else date.today()
    today_str = today.isoformat()

    # 幂等检查（无锁快速路径）
    with _session() as session:
        existing = (
            session.query(DailyRecommendationORM)
            .filter(
                DailyRecommendationORM.user_id == user_id,
                DailyRecommendationORM.date == today,
                DailyRecommendationORM.meal_type == meal_type,
            )
            .first()
        )
        if existing:
            return _orm_to_dict(existing)

    # 分布式锁（按 user_id+date 粒度，覆盖午晚两餐，防止并发生成导致跨餐重复）
    lock_key = f"daily_rec_lock:{user_id}:{today_str}"
    token = secrets.token_urlsafe(16)
    deadline = time.monotonic() + _LOCK_WAIT_TIMEOUT
    lock_acquired = False

    try:
        while True:
            acquire_result = await _try_acquire_lock(lock_key, token)
            if not acquire_result.ok:
                raise LockDegraded(
                    f"Redis unavailable when acquiring lock for {user_id}/{today_str}: {acquire_result.error}"
                )
            if acquire_result.data is True:
                lock_acquired = True
                break
            # 锁被占用：检查当前 meal_type 是否已生成，已生成则提前返回
            with _session() as session:
                existing = (
                    session.query(DailyRecommendationORM)
                    .filter(
                        DailyRecommendationORM.user_id == user_id,
                        DailyRecommendationORM.date == today,
                        DailyRecommendationORM.meal_type == meal_type,
                    )
                    .first()
                )
                if existing:
                    return _orm_to_dict(existing)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockBusy(
                    f"Lock busy for {user_id}/{today_str}/{meal_type} after {_LOCK_WAIT_TIMEOUT}s"
                )
            await asyncio.sleep(min(_LOCK_POLL_INTERVAL, remaining))

        # 获得锁后再次确认幂等（等待期间另一个请求可能已写入）
        with _session() as session:
            existing = (
                session.query(DailyRecommendationORM)
                .filter(
                    DailyRecommendationORM.user_id == user_id,
                    DailyRecommendationORM.date == today,
                    DailyRecommendationORM.meal_type == meal_type,
                )
                .first()
            )
            if existing:
                return _orm_to_dict(existing)

        # 用户画像
        profile = _get_user_profile(user_id)
        dislikes = profile.get("dislikes", [])
        budget   = profile.get("budget", "")
        diet_restriction = profile.get("diet_restriction", "") or ""
        restrictions = [kw for kw in diet_restriction.replace("，", ",").split(",") if kw.strip()]

        # 全量菜谱
        all_recipes = get_all_recipes()

        # 同天另一餐的菜（硬约束，永不放宽）- 在锁内读取，确保看到最新写入
        same_day_dishes, same_day_status = _get_same_day_dishes(user_id, today, meal_type)

        # 最近7天已推荐菜（软约束，可放宽）
        recent_dishes, history_read_status = _get_recent_dishes(user_id, days=7)

        # 硬约束基础集：dislikes + restrictions + same_day（三者均不可放宽）
        after_safety = _filter_recipes(
            all_recipes, dislikes=dislikes, restrictions=restrictions,
        )
        hard_base = _filter_recipes(after_safety, same_day_dishes=same_day_dishes)

        # 软约束层叠，逐级放宽
        lv0 = _filter_recipes(hard_base, budget=budget, recent_dishes=recent_dishes)
        lv1 = _filter_recipes(hard_base, budget=budget)   # 放宽 7-day 历史
        lv2 = hard_base                                    # 放宽预算 + 历史

        # 诊断用：仅去近期（不含预算）
        after_recent = _filter_recipes(hard_base, recent_dishes=recent_dishes)

        # 选择最优 level（same_day 约束在所有 level 中均已应用）
        if len(lv0) >= 3:
            filtered, fallback_level, source_status = lv0, 0, "success"
        elif len(lv1) >= 3:
            filtered, fallback_level, source_status = lv1, 1, "partial"
        elif len(lv2) >= 3:
            filtered, fallback_level, source_status = lv2, 2, "fallback"
        else:
            filtered, fallback_level, source_status = lv2, 3, "fallback"

        logger.info(
            "[daily_rec] user=%s meal=%s date=%s "
            "total_candidates=%d after_safety=%d after_same_day=%d after_recent=%d "
            "selected_count=%d fallback_level=%d source_status=%s "
            "history_status=%s same_day_status=%s",
            user_id, meal_type, today.isoformat(),
            len(all_recipes), len(after_safety), len(hard_base), len(after_recent),
            len(filtered), fallback_level, source_status,
            history_read_status, same_day_status,
        )

        # 选菜
        if len(filtered) >= 3:
            selected = _select_by_structure(filtered)
        else:
            selected = _rule_fallback(filtered) if filtered else []

        if not selected:
            if hard_base:
                selected = _rule_fallback(hard_base, n=min(3, len(hard_base)))
            else:
                logger.warning("[daily_rec] 硬约束后无候选 user=%s meal=%s", user_id, meal_type)
                return None

        # LLM 推理
        reasoning = await _generate_reasoning(selected, profile, meal_type)

        # 构建 dishes 数据（快照：不修改原始 recipe dict）
        dishes_data = [
            {
                "name":        r.get("name"),
                "category":    r.get("category"),
                "ingredients": list(r.get("ingredients", [])),
                "cuisine":     r.get("cuisine"),
                "flavor":      r.get("flavor"),
                "budget_tier": r.get("budget_tier"),
                "steps":       _normalize_recipe_steps(r.get("steps")),
            }
            for r in selected
        ]

        rec = DailyRecommendationORM(
            user_id=user_id,
            date=today,
            meal_type=meal_type,
            dishes=dishes_data,
            reasoning=reasoning,
            generated_by=generated_by,
            source_status=source_status,
            is_read=0,
        )

        try:
            with _session() as session:
                session.add(rec)
                session.commit()
                session.refresh(rec)
                return _orm_to_dict(rec)
        except IntegrityError:
            logger.info("[daily_rec] 并发冲突，user=%s meal=%s 已存在，返回已有记录", user_id, meal_type)
            with _session() as session:
                existing = (
                    session.query(DailyRecommendationORM)
                    .filter(
                        DailyRecommendationORM.user_id == user_id,
                        DailyRecommendationORM.date == today,
                        DailyRecommendationORM.meal_type == meal_type,
                    )
                    .first()
                )
                return _orm_to_dict(existing) if existing else None
        except Exception as e:
            logger.error("[daily_rec] 写入失败 user=%s: %s", user_id, e)
            raise  # 向上传播，使 generate_daily_recommendations 可触发 _enqueue_retry_result

    finally:
        if lock_acquired:
            await _release_daily_rec_lock(lock_key, token)


# ── 定时任务 ───────────────────────────────────────────────────────────────────

def _error_code_str(error) -> str:
    """从 ToolError 中提取可读错误码字符串（用于 Dead-letter error_code 字段）。"""
    if error is None:
        return "INTERNAL"
    code = getattr(error, "code", None)
    if code is None:
        return "INTERNAL"
    return getattr(code, "value", str(code))


async def generate_daily_recommendations() -> None:
    """8:00 定时任务：为所有用户生成今日午餐+晚餐推荐。"""
    from models.user import UserProfileORM

    try:
        with _session() as session:
            user_ids = [row[0] for row in session.query(UserProfileORM.user_id).all()]
    except Exception:
        logger.exception("[daily_rec] 获取用户列表失败")
        return

    # 业务日期：与 generate_for_user 内部 date.today() 一致，计算一次供所有子任务共用
    today_str = date.today().isoformat()
    logger.info("[daily_rec] 开始生成今日推荐 date=%s 共 %d 个用户", today_str, len(user_ids))

    for uid in user_ids:
        for meal_type in ("lunch", "dinner"):
            try:
                await generate_for_user(uid, meal_type=meal_type, date_str=today_str)
            except Exception:
                logger.exception("[daily_rec] 生成失败 user=%s meal=%s", uid, meal_type)
                retry_r = await _enqueue_retry_result(uid, meal_type)
                if not retry_r.ok:
                    uid_hash = hashlib.sha256(uid.encode()).hexdigest()[:8]
                    logger.warning(
                        "[daily_rec] 重试入队失败 uid_hash=%s meal=%s: %s",
                        uid_hash, meal_type, retry_r.error,
                    )
                    # Redis 入队失败 → 持久化到 Dead-letter SQLite 兜底
                    dl_r = await write_dead_letter_result(
                        uid, meal_type, today_str,
                        error_code=_error_code_str(retry_r.error),
                    )
                    if not dl_r.ok:
                        logger.error(
                            "[daily_rec] Dead-letter 写入失败 uid_hash=%s meal=%s: %s",
                            uid_hash, meal_type, dl_r.error,
                        )

    logger.info("[daily_rec] 今日推荐生成完成")


async def _enqueue_retry_result(user_id: str, meal_type: str):
    """将失败的推荐任务入队重试，返回 ToolResult。redis_write 策略，不重试。
    入队失败只记录日志，不覆盖原始生成失败原因。
    """
    from core.cache import _client as _redis_client
    from tools.tool_executor import tool_executor

    def _impl(uid: str, mt: str) -> None:
        _redis_client.rpush(_RETRY_QUEUE_KEY, f"{uid}:{mt}")

    return await tool_executor.execute(
        _impl,
        user_id,
        meal_type,
        policy_name="redis_write",
        tool_name="enqueue_retry",
        source="retry_queue",
    )


async def retry_daily_recommendations() -> None:
    """8:05/8:15/8:30 定时任务：处理 Redis 重试队列中失败的用户。"""
    try:
        from core.cache import _client as redis_client
        items: list[str] = []
        while True:
            item = redis_client.lpop(_RETRY_QUEUE_KEY)
            if item is None:
                break
            items.append(item.decode() if isinstance(item, bytes) else item)
    except Exception:
        logger.exception("[daily_rec] 读取重试队列失败")
        return

    if not items:
        return

    logger.info("[daily_rec] 重试队列有 %d 个任务", len(items))
    for item in items:
        parts = item.split(":", 1)
        uid       = parts[0]
        meal_type = parts[1] if len(parts) > 1 else "lunch"
        try:
            await generate_for_user(uid, meal_type=meal_type, generated_by="retry")
        except Exception:
            logger.exception("[daily_rec] 重试失败 user=%s meal=%s", uid, meal_type)
