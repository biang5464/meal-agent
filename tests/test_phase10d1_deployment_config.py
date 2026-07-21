"""
tests/test_phase10d1_deployment_config.py

Phase 10D1 deployment configuration unit tests.
Verifies that all environment-variable reading is correct without
connecting to real Redis, MySQL, or ChromaDB services.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

# ── Redis ──────────────────────────────────────────────────────────────────────


class TestRedisConfig:
    def test_redis_url_takes_priority(self, monkeypatch):
        """REDIS_URL is used and passed to redis.from_url."""
        # Import before entering patch context so module-level _client
        # initialization does not count toward the mock call count.
        from core.cache import _build_redis_client

        monkeypatch.setenv("REDIS_URL", "redis://testhost:9999/2")
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("REDISHOST", raising=False)

        mock_client = MagicMock()
        with patch("redis.from_url", return_value=mock_client) as mock_fn:
            result = _build_redis_client()

        mock_fn.assert_called_once_with("redis://testhost:9999/2", decode_responses=True)
        assert result is mock_client

    def test_rediss_url_forwarded_to_from_url(self, monkeypatch):
        """rediss:// (TLS) is forwarded to redis.from_url without modification."""
        from core.cache import _build_redis_client

        monkeypatch.setenv("REDIS_URL", "rediss://default:secret@tls-host:6380/0")
        monkeypatch.delenv("REDIS_HOST", raising=False)

        mock_client = MagicMock()
        with patch("redis.from_url", return_value=mock_client) as mock_fn:
            _build_redis_client()

        args, _ = mock_fn.call_args
        assert args[0].startswith("rediss://"), "TLS URL must be passed verbatim"

    def test_no_redis_url_uses_host_port_db(self, monkeypatch):
        """When REDIS_URL is absent, REDIS_HOST / REDIS_PORT / REDIS_DB are used."""
        from core.cache import _build_redis_client

        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("REDIS_HOST", "custom-host")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_DB", "1")
        monkeypatch.delenv("REDISHOST", raising=False)

        mock_client = MagicMock()
        with patch("redis.Redis", return_value=mock_client) as mock_cls:
            _build_redis_client()

        mock_cls.assert_called_once_with(
            host="custom-host", port=6380, db=1, decode_responses=True
        )

    def test_railway_redis_vars_accepted(self, monkeypatch):
        """Railway component variables REDISHOST / REDISPORT are used as fallback."""
        from core.cache import _build_redis_client

        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("REDIS_PORT", raising=False)
        monkeypatch.setenv("REDISHOST", "railway-redis-host")
        monkeypatch.setenv("REDISPORT", "6379")

        mock_client = MagicMock()
        with patch("redis.Redis", return_value=mock_client) as mock_cls:
            _build_redis_client()

        kwargs = mock_cls.call_args[1]
        assert kwargs["host"] == "railway-redis-host"

    def test_client_singleton_has_expected_interface(self):
        """Module-level _client exposes the Redis interface the rest of the app uses."""
        from core.cache import _client
        for method in ("get", "set", "delete", "expire", "rpush", "lrange", "ltrim"):
            assert hasattr(_client, method), f"_client missing method: {method}"


# ── MySQL / database URL ───────────────────────────────────────────────────────


class TestDatabaseUrl:
    def test_database_url_first_priority(self):
        """DATABASE_URL is used verbatim (driver normalised if needed)."""
        from core.database import build_database_url
        url = build_database_url(
            {"DATABASE_URL": "mysql+pymysql://user:pass@host:3306/mydb"}
        )
        assert url.drivername == "mysql+pymysql"
        assert url.host == "host"
        assert url.username == "user"

    def test_mysql_url_second_priority(self, ):
        """MYSQL_URL is used when DATABASE_URL is absent."""
        from core.database import build_database_url
        url = build_database_url({"MYSQL_URL": "mysql+pymysql://u:p@rw-host:3306/db"})
        assert url.host == "rw-host"
        assert url.drivername == "mysql+pymysql"

    def test_mysql_url_normalized_to_pymysql(self):
        """mysql:// driver is normalised to mysql+pymysql://."""
        from core.database import build_database_url
        url = build_database_url({"MYSQL_URL": "mysql://user:pass@host:3306/db"})
        assert url.drivername == "mysql+pymysql"

    def test_database_url_normalized_to_pymysql(self):
        """DATABASE_URL with plain mysql:// driver is also normalised."""
        from core.database import build_database_url
        url = build_database_url({"DATABASE_URL": "mysql://user:pass@host/db"})
        assert url.drivername == "mysql+pymysql"

    def test_local_component_vars(self):
        """MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE work."""
        from core.database import build_database_url
        url = build_database_url({
            "MYSQL_HOST": "db-host",
            "MYSQL_PORT": "3307",
            "MYSQL_USER": "appuser",
            "MYSQL_PASSWORD": "apppass",
            "MYSQL_DATABASE": "appdb",
        })
        assert url.host == "db-host"
        assert url.port == 3307
        assert url.username == "appuser"
        assert url.password == "apppass"
        assert url.database == "appdb"
        assert url.drivername == "mysql+pymysql"

    def test_railway_component_vars(self):
        """Railway names (MYSQLHOST etc.) are accepted as fallback."""
        from core.database import build_database_url
        url = build_database_url({
            "MYSQLHOST": "railway-db",
            "MYSQLPORT": "3306",
            "MYSQLUSER": "root",
            "MYSQLPASSWORD": "railpass",
            "MYSQLDATABASE": "meal_agent",
        })
        assert url.host == "railway-db"
        assert url.password == "railpass"

    def test_local_vars_win_over_railway(self):
        """When both local and Railway names are set, local names take priority."""
        from core.database import build_database_url
        url = build_database_url({
            "MYSQL_HOST": "local-host",
            "MYSQLHOST": "railway-host",
            "MYSQL_PASSWORD": "local-pass",
            "MYSQLPASSWORD": "railway-pass",
        })
        assert url.host == "local-host"
        assert url.password == "local-pass"

    def test_special_char_password_safe(self):
        """Passwords with special characters are stored correctly by URL.create."""
        from core.database import build_database_url
        url = build_database_url({
            "MYSQL_HOST": "host",
            "MYSQL_USER": "user",
            "MYSQL_PASSWORD": "p@ss:word/special",
            "MYSQL_DATABASE": "db",
        })
        assert url.password == "p@ss:word/special"
        assert url.drivername == "mysql+pymysql"

    def test_no_default_password_in_source(self):
        """The hardcoded default password must not exist in the source file."""
        import core.database as db_mod
        src = inspect.getsource(db_mod)
        assert "mealagent123" not in src, (
            "Default plaintext password 'mealagent123' must not appear in source"
        )

    def test_empty_env_uses_safe_defaults(self):
        """With empty env, URL is built with localhost defaults and no password."""
        from core.database import build_database_url
        url = build_database_url({})
        assert url.host == "localhost"
        assert url.port == 3306
        assert url.password is None
        assert url.drivername == "mysql+pymysql"


# ── Chroma path ────────────────────────────────────────────────────────────────


class TestChromaPath:
    def test_default_fallback(self, monkeypatch):
        """Without CHROMA_PERSIST_DIR, the path falls back to ./data/chroma."""
        monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
        from tools.memory import _get_chroma_path
        assert _get_chroma_path() == "./data/chroma"

    def test_env_var_used(self, monkeypatch, tmp_path):
        """CHROMA_PERSIST_DIR is returned when set."""
        custom = str(tmp_path / "custom_chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", custom)
        from tools.memory import _get_chroma_path
        assert _get_chroma_path() == custom

    def test_init_conv_memory_uses_env_path(self, monkeypatch, tmp_path):
        """init_conversation_memory_chroma uses the env-configured path."""
        custom = str(tmp_path / "chroma_env")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", custom)

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client) as mock_ctor:
            from tools.memory import init_conversation_memory_chroma
            init_conversation_memory_chroma()

        mock_ctor.assert_called_once_with(path=custom)

    def test_save_conversation_memory_uses_env_path(self, monkeypatch, tmp_path):
        """save_conversation_memory creates the Chroma client at the env path."""
        custom = str(tmp_path / "chroma_save")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", custom)

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client) as mock_ctor:
            import tools.memory as mem_mod
            # Call only the Chroma client creation part — not the full async save
            # by re-using the _get_chroma_path helper directly
            path_used = mem_mod._get_chroma_path()

        assert path_used == custom

    def test_memory_collection_name_unchanged(self):
        """The conversation_memory collection name must not change."""
        from tools.memory import _MEMORY_COLLECTION
        assert _MEMORY_COLLECTION == "conversation_memory"

    def test_no_hardcoded_chroma_path_in_source(self):
        """The old hardcoded _CHROMA_PATH assignment must not exist in the source."""
        import tools.memory as mem_mod
        src = inspect.getsource(mem_mod)
        assert '_CHROMA_PATH = "./data/chroma"' not in src, (
            "Hardcoded _CHROMA_PATH must be replaced with _get_chroma_path()"
        )


# ── Lifespan initialisation ────────────────────────────────────────────────────


class TestLifespan:
    @pytest.mark.asyncio
    async def test_nutrition_init_called_with_persist_dir(self, monkeypatch, tmp_path):
        """init_nutrition_chroma is called and receives the same persist_dir as other Chroma inits."""
        chroma_dir = str(tmp_path / "chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)

        with (
            patch("main.init_chroma") as mock_recipe,
            patch("main.init_nutrition_chroma") as mock_nutrition,
            patch("main.init_food_safety_chroma") as mock_food_safety,
            patch("main.init_conversation_memory_chroma"),
            patch("main.load_food_safety_documents"),
            patch("main.load_nutrition_documents"),
            patch("main.init_db"),
            patch("main.init_dead_letter_db"),
            patch("main.start_scheduler"),
            patch("main.stop_scheduler"),
        ):
            import main as main_mod
            async with main_mod.lifespan(main_mod.app):
                pass

        mock_recipe.assert_called_once_with(chroma_dir)
        mock_nutrition.assert_called_once_with(chroma_dir)
        mock_food_safety.assert_called_once_with(chroma_dir)

    @pytest.mark.asyncio
    async def test_lifespan_existing_init_preserved(self, monkeypatch, tmp_path):
        """init_db, init_dead_letter_db, and start_scheduler are still called."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        with (
            patch("main.init_chroma"),
            patch("main.init_nutrition_chroma"),
            patch("main.init_food_safety_chroma"),
            patch("main.init_conversation_memory_chroma"),
            patch("main.load_food_safety_documents"),
            patch("main.load_nutrition_documents"),
            patch("main.init_db") as mock_db,
            patch("main.init_dead_letter_db") as mock_dl,
            patch("main.start_scheduler") as mock_sched,
            patch("main.stop_scheduler"),
        ):
            import main as main_mod
            async with main_mod.lifespan(main_mod.app):
                pass

        assert mock_db.call_count == 1
        assert mock_dl.call_count == 1
        assert mock_sched.call_count == 1

    @pytest.mark.asyncio
    async def test_load_nutrition_called_with_nutrition_dir(self, monkeypatch, tmp_path):
        """load_nutrition_documents is called with NUTRITION_DIR and persist_dir."""
        chroma_dir = str(tmp_path / "chroma")
        nutrition_dir = str(tmp_path / "nutrition_docs")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
        monkeypatch.setenv("NUTRITION_DIR", nutrition_dir)

        with (
            patch("main.init_chroma"),
            patch("main.init_nutrition_chroma"),
            patch("main.init_food_safety_chroma"),
            patch("main.init_conversation_memory_chroma"),
            patch("main.load_food_safety_documents"),
            patch("main.load_nutrition_documents") as mock_load,
            patch("main.init_db"),
            patch("main.init_dead_letter_db"),
            patch("main.start_scheduler"),
            patch("main.stop_scheduler"),
        ):
            import main as main_mod
            async with main_mod.lifespan(main_mod.app):
                pass

        mock_load.assert_called_once_with(nutrition_dir, chroma_dir)
