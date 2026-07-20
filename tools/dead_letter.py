"""Dead-letter 持久化：Redis 重试入队失败时的本地 SQLite 兜底。

独立 SQLite（data/dead_letter.db），与主业务 DB 完全隔离。
延迟初始化：生产时在 lifespan 中调用 init_dead_letter_db()，测试时传入 tmp_path。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

if TYPE_CHECKING:
    from core.tool_protocol import ToolResult

logger = logging.getLogger(__name__)

_engine = None
_Session = None


# ── ORM ────────────────────────────────────────────────────────────────────────

class _DLBase(DeclarativeBase):
    pass


class DeadLetterORM(_DLBase):
    __tablename__ = "dead_letter_tasks"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    idempotency_key = Column(String(256), unique=True, nullable=False, index=True)
    task_type       = Column(String(64),  nullable=False, default="daily_recommendation")
    user_id         = Column(String(256), nullable=False)
    meal_type       = Column(String(32),  nullable=False)
    target_date     = Column(String(32),  nullable=False)
    error_code      = Column(String(64),  nullable=False, default="INTERNAL")
    status          = Column(String(32),  nullable=False, default="pending")
    attempts        = Column(Integer,     nullable=False, default=1)
    created_at      = Column(DateTime,    nullable=False)
    updated_at      = Column(DateTime,    nullable=False)


# ── 初始化 ──────────────────────────────────────────────────────────────────────

def init_dead_letter_db(path: str = "./data/dead_letter.db") -> None:
    """创建 engine 并建表（幂等）。测试时传入 tmp_path 下的路径。"""
    global _engine, _Session
    _engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )
    _DLBase.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)


# ── 核心写入 ────────────────────────────────────────────────────────────────────

def _write_dead_letter_sync(
    user_id: str,
    meal_type: str,
    target_date: str,
    error_code: str,
) -> None:
    """同步写入，由 ToolExecutor 在 asyncio.to_thread 中执行。
    幂等：idempotency_key 重复时递增 attempts，不抛错。
    """
    if _Session is None:
        raise RuntimeError("Dead-letter DB 未初始化（请先调用 init_dead_letter_db()）")

    idempotency_key = f"daily_recommendation:{user_id}:{target_date}:{meal_type}"
    now = datetime.utcnow()

    with _Session() as session:
        try:
            session.add(
                DeadLetterORM(
                    idempotency_key=idempotency_key,
                    task_type="daily_recommendation",
                    user_id=user_id,
                    meal_type=meal_type,
                    target_date=target_date,
                    error_code=error_code,
                    status="pending",
                    attempts=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = (
                session.query(DeadLetterORM)
                .filter_by(idempotency_key=idempotency_key)
                .first()
            )
            if existing is not None:
                existing.attempts += 1
                existing.updated_at = now
                existing.error_code = error_code
                session.commit()


async def write_dead_letter_result(
    user_id: str,
    meal_type: str,
    target_date: str,
    error_code: str = "INTERNAL",
) -> ToolResult:
    """ToolExecutor 包装的 Dead-letter 写入，返回 ToolResult。
    写入失败只记录日志，不向外抛出。
    """
    from tools.tool_executor import tool_executor

    return await tool_executor.execute(
        _write_dead_letter_sync,
        user_id,
        meal_type,
        target_date,
        error_code,
        policy_name="database_write",
        tool_name="write_dead_letter",
        fallback_data=None,
    )
