# 在所有 tests/ 测试开始前，初始化表结构（只执行一次）
import socket
import os

import pytest

import models.daily_recommendation  # noqa: F401 — 注册 DailyRecommendationORM
import models.maintenance           # noqa: F401 — 注册 ORM
import models.chat_history          # noqa: F401 — 注册 ChatHistoryORM
import models.price_history         # noqa: F401 — 注册 PriceHistoryORM
from tools.memory import init_db


def _mysql_reachable(host: str = "localhost", port: int = 3306, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_mysql_host = os.getenv("MYSQL_HOST", "localhost")
_mysql_port = int(os.getenv("MYSQL_PORT", "3306"))

if _mysql_reachable(_mysql_host, _mysql_port):
    init_db()  # MySQL 可用：使用 MySQL（db_url=None 走 MySQL 路径）
else:
    # MySQL 不可用：降级到 SQLite，令 MySQL 无关的测试仍可运行
    init_db("./data/test_fallback.db")


# ── 共享 Fake 基础设施 Fixture ─────────────────────────────────────────────────

@pytest.fixture
def fake_redis_client():
    """Returns a FakeRedis client (decode_responses=True) for test isolation."""
    import fakeredis
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def in_memory_mysql_session(monkeypatch):
    """Patches core.database.SessionLocal and tools.price_history.SessionLocal
    with an in-memory SQLite session factory. Each test gets a fresh, empty DB.

    Uses StaticPool so all sessions share the same single in-memory connection —
    without it, each Session() call gets a NEW connection (fresh empty database)
    and the tables created by create_all() become invisible to subsequent sessions.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from models.user import Base
    import models.maintenance    # noqa: F401 — ensure HealthCycleORM etc. registered
    import models.chat_history   # noqa: F401 — ensure ChatHistoryORM registered
    import models.price_history  # noqa: F401 — ensure PriceHistoryORM registered

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr("core.database.SessionLocal", Session)
    monkeypatch.setattr("tools.price_history.SessionLocal", Session)

    yield Session

    engine.dispose()
