"""FastAPI 入口：挂载路由、初始化依赖、提供 SSE 流式推荐接口。"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated

import core.runtime_paths as _rp

from core.web_config import get_allowed_origins
from core.api_security import get_api_security_config, AuthMiddleware
from core.rate_limit import get_rate_limit_config, RateLimitMiddleware
from core.user_identity import require_current_user_id
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.maintenance_agent import start_scheduler, stop_scheduler
from models.maintenance import HealthCycleORM, PreferenceHistoryORM, ReminderORM  # noqa: F401 — 确保建表
from models.daily_recommendation import DailyRecommendationORM  # noqa: F401 — 确保建表
from agents.graph import run as run_graph, run_with_queue
from tools.dead_letter import init_dead_letter_db
from tools.memory import init_db
from tools.timeout_config import TimeoutConfig
from tools.memory import init_conversation_memory_chroma
from tools.nutrition import (
    init_food_safety_chroma,
    init_nutrition_chroma,
    load_food_safety_documents,
    load_nutrition_documents,
)
from tools.recipe import init_chroma
from tools.price_history import (
    get_tracked_terms,
    get_price_chart_single,
    get_price_chart_multi,
)

load_dotenv()

# Initialize security configs at startup (raises ValueError if production misconfigured)
_AUTH_CONFIG = get_api_security_config()
_RL_CONFIG = get_rate_limit_config()


def _log_rss(phase: str) -> None:
    """Log process RSS after each startup phase. Linux only, no psutil needed."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    import logging
                    logging.getLogger(__name__).info(
                        "startup %s rss_mb=%.0f", phase, int(line.split()[1]) / 1024
                    )
                    break
    except Exception:
        pass


# ---------- 生命周期 ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _rp.ensure_runtime_dirs()
    persist_str = str(_rp.chroma_dir())

    init_db(str(_rp.sqlite_path()))
    init_dead_letter_db(str(_rp.dead_letter_path()))
    init_chroma(persist_str)
    _log_rss("recipe_chroma")
    init_conversation_memory_chroma()
    _log_rss("conv_memory")
    init_nutrition_chroma(persist_str)
    _log_rss("nutrition_chroma")
    load_nutrition_documents(str(_rp.nutrition_dir()), persist_str, replace=False)
    _log_rss("nutrition_docs")
    init_food_safety_chroma(persist_str)
    _log_rss("food_safety_chroma")
    load_food_safety_documents(str(_rp.food_safety_dir()), persist_str, replace=False)
    _log_rss("food_safety_docs")

    scheduler_started = False
    if _rp.env_flag("RUN_SCHEDULER", default=True):
        start_scheduler()
        scheduler_started = True

    try:
        yield
    finally:
        if scheduler_started:
            stop_scheduler()


# ---------- App ----------

app = FastAPI(
    title="Meal Agent",
    description="个人饮食推荐 Agent，基于用户画像提供流式菜谱推荐。",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware order: last added = outermost (processes request first).
# Request flow: AuthMiddleware → RateLimitMiddleware → CORSMiddleware → Router
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.add_middleware(RateLimitMiddleware, config=_RL_CONFIG)
app.add_middleware(AuthMiddleware, config=_AUTH_CONFIG)


# ---------- 请求/响应模型 ----------

class RecommendRequest(BaseModel):
    message: str = Field(..., max_length=1000, min_length=1)
    chat_history: list = []


class GenerateRecommendationRequest(BaseModel):
    meal_type: str = "lunch"


# ---------- 路由 ----------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "meal-agent"}


@app.post("/recommend")
async def recommend(
    req: RecommendRequest,
    current_user_id: Annotated[str, Depends(require_current_user_id)],
):
    """
    SSE 流式饮食推荐接口。

    客户端使用 EventSource 或 fetch+ReadableStream 消费，
    每个 data: 事件携带一个 token 片段，最后以 [DONE] 结束。
    用户身份由代理注入的 X-Meal-Agent-User-ID header 提供，
    不信任请求体中的任何 user_id 字段。
    """
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        message = req.message.strip()
        task = asyncio.create_task(run_with_queue(current_user_id, message, queue))

        # Graph watchdog：整体执行超过 GRAPH 秒时，向 queue 注入降级消息并取消 task
        async def _watchdog():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=TimeoutConfig.GRAPH)
            except asyncio.TimeoutError:
                await queue.put("抱歉，处理时间较长，请稍后重试。")
                await queue.put("[DONE]")
                if not task.done():
                    task.cancel()

        watchdog = asyncio.create_task(_watchdog())

        try:
            while True:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=TimeoutConfig.SSE_IDLE)
                    if token == "[DONE]":
                        yield {"data": "[DONE]"}
                        break
                    yield {"data": token}
                except asyncio.TimeoutError:
                    yield {"data": "[DONE]"}
                    break
        except Exception as e:
            yield {"data": f"[ERROR] {str(e)}"}
        finally:
            if not watchdog.done():
                watchdog.cancel()
                try:
                    await watchdog
                except (asyncio.CancelledError, Exception):
                    pass
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return EventSourceResponse(event_generator())


@app.get("/api/price-history/{query_term}")
async def price_history_chart(
    query_term: str,
    mode: str = "single",
    days: int = 30,
):
    """
    获取商品价格历史曲线数据。
    - mode=single：单线模式（单位一致取最低价，不一致按平台展开）
    - mode=multi：多线模式（每个平台各一条线）
    """
    if mode == "multi":
        return get_price_chart_multi(query_term, days=days)
    return get_price_chart_single(query_term, days=days)


@app.get("/api/tracked-terms")
async def tracked_terms(days: int = 30):
    """获取可追踪的商品关键词列表（给前端做下拉选择用）。"""
    terms = get_tracked_terms(days=days)
    return {"terms": terms, "count": len(terms)}


@app.get("/api/daily-recommendation")
async def get_daily_recommendation(
    current_user_id: Annotated[str, Depends(require_current_user_id)],
    date: str | None = None,
    meal_type: str = "lunch",
):
    """
    获取用户今日（或指定日期）推荐。
    若当日推荐尚未生成，自动触发生成后返回。
    用户身份由代理注入的 X-Meal-Agent-User-ID header 提供。
    查询参数中的 user_id 不受信任且被忽略。
    """
    from datetime import date as _date
    from agents.daily_recommendation_agent import generate_for_user
    from tools.memory import _session

    target_date = date or _date.today().isoformat()

    with _session() as session:
        rec = (
            session.query(DailyRecommendationORM)
            .filter(
                DailyRecommendationORM.user_id == current_user_id,
                DailyRecommendationORM.date == target_date,
                DailyRecommendationORM.meal_type == meal_type,
            )
            .first()
        )
        if rec:
            rec.is_read = 1
            session.commit()
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

    from agents.daily_recommendation_agent import LockBusy, LockDegraded
    try:
        result = await generate_for_user(current_user_id, meal_type=meal_type, date_str=target_date, generated_by="lazy")
    except LockBusy:
        raise HTTPException(status_code=503, detail="推荐生成繁忙，请稍后重试")
    except LockDegraded:
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试")
    if result is None:
        raise HTTPException(status_code=500, detail="推荐生成失败，请稍后重试")
    return result


@app.post("/api/daily-recommendation/generate")
async def trigger_daily_recommendation(
    req: GenerateRecommendationRequest,
    current_user_id: Annotated[str, Depends(require_current_user_id)],
):
    """
    手动触发为当前用户生成今日推荐。
    用户身份由代理注入的 X-Meal-Agent-User-ID header 提供。
    请求体中的 user_id 不受信任且被忽略。
    """
    from agents.daily_recommendation_agent import generate_for_user

    from agents.daily_recommendation_agent import LockBusy, LockDegraded
    try:
        result = await generate_for_user(current_user_id, meal_type=req.meal_type, generated_by="manual")
    except LockBusy:
        raise HTTPException(status_code=503, detail="推荐生成繁忙，请稍后重试")
    except LockDegraded:
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试")
    if result is None:
        raise HTTPException(status_code=500, detail="推荐生成失败，请稍后重试")
    return result


@app.get("/users/me/profile")
async def get_profile(
    current_user_id: Annotated[str, Depends(require_current_user_id)],
):
    """获取当前用户画像（调试用）。"""
    # TODO: 调用 get_user_profile tool 或直接查库
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.put("/users/me/profile")
async def update_profile(
    current_user_id: Annotated[str, Depends(require_current_user_id)],
    payload: dict,
):
    """更新当前用户画像字段。"""
    # TODO: 调用 update_user_profile tool
    raise HTTPException(status_code=501, detail="Not implemented yet")
