"""FastAPI 入口：挂载路由、初始化依赖、提供 SSE 流式推荐接口。"""

import asyncio
import os
from contextlib import asynccontextmanager

import core.runtime_paths as _rp

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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


# ---------- 生命周期 ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _rp.ensure_runtime_dirs()
    persist_str = str(_rp.chroma_dir())

    init_db(str(_rp.sqlite_path()))
    init_dead_letter_db(str(_rp.dead_letter_path()))
    init_chroma(persist_str)
    init_conversation_memory_chroma()
    init_nutrition_chroma(persist_str)
    load_nutrition_documents(str(_rp.nutrition_dir()), persist_str, replace=False)
    init_food_safety_chroma(persist_str)
    load_food_safety_documents(str(_rp.food_safety_dir()), persist_str, replace=False)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求/响应模型 ----------

class RecommendRequest(BaseModel):
    user_id: str
    message: str = Field(..., max_length=1000, min_length=1)
    chat_history: list = []


# ---------- 路由 ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """
    SSE 流式饮食推荐接口。

    客户端使用 EventSource 或 fetch+ReadableStream 消费，
    每个 data: 事件携带一个 token 片段，最后以 [DONE] 结束。
    """
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        message = req.message.strip()
        task = asyncio.create_task(run_with_queue(req.user_id, message, queue))

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
    user_id: str,
    date: str | None = None,
    meal_type: str = "lunch",
):
    """
    获取用户今日（或指定日期）推荐。
    若当日推荐尚未生成，自动触发生成后返回。
    """
    from datetime import date as _date
    from agents.daily_recommendation_agent import generate_for_user
    from tools.memory import _session

    target_date = date or _date.today().isoformat()

    with _session() as session:
        rec = (
            session.query(DailyRecommendationORM)
            .filter(
                DailyRecommendationORM.user_id == user_id,
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

    result = await generate_for_user(user_id, meal_type=meal_type, date_str=target_date, generated_by="lazy")
    if result is None:
        raise HTTPException(status_code=500, detail="推荐生成失败，请稍后重试")
    return result


@app.post("/api/daily-recommendation/generate")
async def trigger_daily_recommendation(payload: dict):
    """
    手动触发为指定用户生成今日推荐。
    payload: {"user_id": "...", "meal_type": "lunch"|"dinner"}
    """
    from agents.daily_recommendation_agent import generate_for_user

    user_id   = payload.get("user_id", "")
    meal_type = payload.get("meal_type", "lunch")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    result = await generate_for_user(user_id, meal_type=meal_type, generated_by="manual")
    if result is None:
        raise HTTPException(status_code=500, detail="推荐生成失败，请稍后重试")
    return result


@app.get("/users/{user_id}/profile")
async def get_profile(user_id: str):
    """获取用户画像（调试用）。"""
    # TODO: 调用 get_user_profile tool 或直接查库
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.put("/users/{user_id}/profile")
async def update_profile(user_id: str, payload: dict):
    """更新用户画像字段。"""
    # TODO: 调用 update_user_profile tool
    raise HTTPException(status_code=501, detail="Not implemented yet")
