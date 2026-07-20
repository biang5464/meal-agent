"""价格历史工具：写入和查询 price_history 表。"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import desc, func

from models.price_history import PriceHistoryORM
from core.database import SessionLocal


def record_price(
    query_term: str,
    platform: str,
    price: float,
    unit: str,
    product_name: str = None,
    currency: str = "AUD",
    url: str = None,
) -> None:
    """写入一条价格历史记录，失败时静默处理不影响主流程。"""
    try:
        with SessionLocal() as session:
            session.add(PriceHistoryORM(
                query_term=query_term,
                platform=platform,
                product_name=product_name,
                price=price,
                unit=unit,
                currency=currency,
                url=url,
            ))
            session.commit()
    except Exception:
        pass


def get_tracked_terms(days: int = 30, min_count: int = 2) -> list[str]:
    """
    从 price_history 提取最近 N 天内查过的不重复商品关键词。
    min_count: 至少有这么多条记录才算有效追踪词（过滤偶发噪音）。
    """
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as session:
            rows = (
                session.query(PriceHistoryORM.query_term)
                .filter(PriceHistoryORM.recorded_at >= since)
                .group_by(PriceHistoryORM.query_term)
                .having(func.count(PriceHistoryORM.id) >= min_count)
                .all()
            )
            return [r.query_term for r in rows]
    except Exception:
        return []


def get_price_chart_single(query_term: str, days: int = 30) -> dict:
    """
    单线模式：按日期聚合，判断单位是否一致。
    - 单位一致 → 每天取最低价，返回单条数据列表
    - 单位不一致 → 每个平台各自返回每天最低价，data 为平台分组字典
    """
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as session:
            rows = (
                session.query(
                    PriceHistoryORM.platform,
                    PriceHistoryORM.unit,
                    func.date(PriceHistoryORM.recorded_at).label("date"),
                    func.min(PriceHistoryORM.price).label("min_price"),
                )
                .filter(
                    PriceHistoryORM.query_term == query_term,
                    PriceHistoryORM.recorded_at >= since,
                    PriceHistoryORM.price.isnot(None),
                )
                .group_by(
                    PriceHistoryORM.platform,
                    PriceHistoryORM.unit,
                    func.date(PriceHistoryORM.recorded_at),
                )
                .order_by(func.date(PriceHistoryORM.recorded_at))
                .all()
            )
    except Exception:
        return {"query_term": query_term, "mode": "single", "comparable": False, "data": {}}

    if not rows:
        return {"query_term": query_term, "mode": "single", "comparable": False, "data": {}}

    units = set(r.unit for r in rows if r.unit)
    comparable = len(units) == 1

    if comparable:
        daily: dict[str, float] = {}
        for r in rows:
            date_str = str(r.date)
            price = float(r.min_price)
            if date_str not in daily or price < daily[date_str]:
                daily[date_str] = price
        return {
            "query_term": query_term,
            "mode": "single",
            "comparable": True,
            "unit": units.pop(),
            "data": [{"date": d, "price": p} for d, p in sorted(daily.items())],
        }
    else:
        by_platform: dict[str, list] = {}
        for r in rows:
            platform = r.platform
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append({
                "date": str(r.date),
                "price": float(r.min_price),
                "unit": r.unit,
            })
        return {
            "query_term": query_term,
            "mode": "single",
            "comparable": False,
            "data": by_platform,
        }


def get_price_chart_multi(query_term: str, days: int = 30) -> dict:
    """多线模式：每个平台各自返回每天最低价，不做单位判断。"""
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as session:
            rows = (
                session.query(
                    PriceHistoryORM.platform,
                    PriceHistoryORM.unit,
                    func.date(PriceHistoryORM.recorded_at).label("date"),
                    func.min(PriceHistoryORM.price).label("min_price"),
                )
                .filter(
                    PriceHistoryORM.query_term == query_term,
                    PriceHistoryORM.recorded_at >= since,
                    PriceHistoryORM.price.isnot(None),
                )
                .group_by(
                    PriceHistoryORM.platform,
                    PriceHistoryORM.unit,
                    func.date(PriceHistoryORM.recorded_at),
                )
                .order_by(func.date(PriceHistoryORM.recorded_at))
                .all()
            )
    except Exception:
        return {"query_term": query_term, "mode": "multi", "data": {}}

    by_platform: dict[str, list] = {}
    for r in rows:
        platform = r.platform
        if platform not in by_platform:
            by_platform[platform] = []
        by_platform[platform].append({
            "date": str(r.date),
            "price": float(r.min_price),
            "unit": r.unit,
        })

    by_platform = {k: v if isinstance(v, list) else [] for k, v in by_platform.items()}
    return {
        "query_term": query_term,
        "mode": "multi",
        "data": by_platform,
    }


def get_price_history(
    query_term: str,
    platform: str = None,
    days: int = 30,
) -> list[dict]:
    """读取某商品的价格历史，支持按平台过滤，默认最近30天。"""
    since = datetime.now() - timedelta(days=days)
    try:
        with SessionLocal() as session:
            q = session.query(PriceHistoryORM).filter(
                PriceHistoryORM.query_term == query_term,
                PriceHistoryORM.recorded_at >= since,
            )
            if platform:
                q = q.filter(PriceHistoryORM.platform == platform)
            rows = q.order_by(desc(PriceHistoryORM.recorded_at)).all()
            return [
                {
                    "query_term": r.query_term,
                    "platform": r.platform,
                    "product_name": r.product_name,
                    "price": float(r.price) if r.price else None,
                    "unit": r.unit,
                    "currency": r.currency,
                    "url": r.url,
                    "recorded_at": r.recorded_at.isoformat(),
                }
                for r in rows
            ]
    except Exception:
        return []
