import os
from collections.abc import Mapping

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker

load_dotenv()


def build_database_url(env: Mapping[str, str] | None = None) -> URL:
    """Build a mysql+pymysql SQLAlchemy URL from environment variables.

    Priority:
    1. DATABASE_URL  — full connection string
    2. MYSQL_URL     — Railway MySQL plugin URL
    3. Component variables:
       Local names (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE)
       checked first; Railway names (MYSQLHOST, MYSQLPORT, MYSQLUSER, MYSQLPASSWORD,
       MYSQLDATABASE) used as fallback.
    """
    _env: Mapping[str, str] = env if env is not None else os.environ

    for key in ("DATABASE_URL", "MYSQL_URL"):
        raw = _env.get(key)
        if raw:
            parsed = make_url(raw)
            if parsed.drivername in ("mysql", "mysql+mysqldb"):
                parsed = parsed.set(drivername="mysql+pymysql")
            return parsed

    host = _env.get("MYSQL_HOST") or _env.get("MYSQLHOST") or "localhost"
    port = int(_env.get("MYSQL_PORT") or _env.get("MYSQLPORT") or "3306")
    user = _env.get("MYSQL_USER") or _env.get("MYSQLUSER") or "root"
    password = (
        _env["MYSQL_PASSWORD"]
        if "MYSQL_PASSWORD" in _env
        else _env.get("MYSQLPASSWORD")
    )
    database = _env.get("MYSQL_DATABASE") or _env.get("MYSQLDATABASE") or "meal_agent"

    return URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password if password else None,
        host=host,
        port=port,
        database=database,
        query={"charset": "utf8mb4"},
    )


engine = create_engine(
    build_database_url(),
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(bind=engine)
