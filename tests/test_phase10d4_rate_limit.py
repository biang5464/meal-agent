"""
tests/test_phase10d4_rate_limit.py

Phase 10D4 rate limiting tests.
All Redis interactions are mocked — no real network connections.
Tests cover: config, hashing, middleware behavior, Redis failure modes.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helper: build isolated test app with RateLimitMiddleware ──────────────────

def _make_rl_app(enabled: bool, recommend_per_min: int = 10, read_per_min: int = 60,
                  app_env: str = "development") -> FastAPI:
    from core.rate_limit import RateLimitMiddleware, RateLimitConfig
    config = RateLimitConfig(
        enabled=enabled,
        recommend_per_minute=recommend_per_min,
        read_per_minute=read_per_min,
        app_env=app_env,
    )
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, config=config)

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    @test_app.post("/recommend")
    def recommend():
        return {"ok": True}

    @test_app.get("/api/tracked-terms")
    def tracked_terms():
        return {"terms": []}

    @test_app.get("/api/daily-recommendation")
    def daily_rec():
        return {"id": 1}

    return test_app


def _make_mock_redis(count: int = 1, ttl: int = 55):
    """Return a mock Redis client whose eval returns [count, ttl]."""
    mock = MagicMock()
    mock.eval.return_value = [count, ttl]
    return mock


# ── get_rate_limit_config ─────────────────────────────────────────────────────

class TestGetRateLimitConfig:
    def test_dev_defaults_disabled(self, monkeypatch):
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
        from core.rate_limit import get_rate_limit_config
        cfg = get_rate_limit_config()
        assert cfg.enabled is False

    def test_production_default_enabled(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
        from core.rate_limit import get_rate_limit_config
        cfg = get_rate_limit_config()
        assert cfg.enabled is True

    def test_custom_limits(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RECOMMEND_PER_MINUTE", "5")
        monkeypatch.setenv("RATE_LIMIT_READ_PER_MINUTE", "30")
        from core.rate_limit import get_rate_limit_config
        cfg = get_rate_limit_config()
        assert cfg.recommend_per_minute == 5
        assert cfg.read_per_minute == 30

    def test_defaults_are_sensible(self, monkeypatch):
        monkeypatch.delenv("RATE_LIMIT_RECOMMEND_PER_MINUTE", raising=False)
        monkeypatch.delenv("RATE_LIMIT_READ_PER_MINUTE", raising=False)
        from core.rate_limit import get_rate_limit_config
        cfg = get_rate_limit_config()
        assert cfg.recommend_per_minute == 10
        assert cfg.read_per_minute == 60


# ── hash_client_id ────────────────────────────────────────────────────────────

class TestHashClientId:
    def test_hash_is_deterministic(self):
        from core.rate_limit import hash_client_id
        assert hash_client_id("1.2.3.4") == hash_client_id("1.2.3.4")

    def test_different_ips_different_hashes(self):
        from core.rate_limit import hash_client_id
        assert hash_client_id("1.2.3.4") != hash_client_id("5.6.7.8")

    def test_hash_does_not_contain_raw_ip(self):
        from core.rate_limit import hash_client_id
        raw = "192.168.1.100"
        result = hash_client_id(raw)
        assert raw not in result

    def test_hash_is_hex_string(self):
        from core.rate_limit import hash_client_id
        result = hash_client_id("10.0.0.1")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_ip_produces_hash(self):
        from core.rate_limit import hash_client_id
        result = hash_client_id("")
        assert isinstance(result, str)
        assert len(result) > 0


# ── Middleware: disabled ───────────────────────────────────────────────────────

class TestRateLimitMiddlewareDisabled:
    def setup_method(self):
        self.client = TestClient(_make_rl_app(enabled=False))

    def test_recommend_passes_through(self):
        res = self.client.post("/recommend")
        assert res.status_code == 200

    def test_health_passes_through(self):
        res = self.client.get("/health")
        assert res.status_code == 200

    def test_tracked_terms_passes_through(self):
        res = self.client.get("/api/tracked-terms")
        assert res.status_code == 200


# ── Middleware: exempt paths ───────────────────────────────────────────────────

class TestRateLimitExemptPaths:
    def setup_method(self):
        mock_redis = _make_mock_redis(count=999, ttl=55)
        self._patch = patch("tools.tool_executor.asyncio.to_thread", side_effect=Exception("should not call"))
        self._patch.start()
        # Exempt paths never reach Redis
        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        self.client = TestClient(app, raise_server_exceptions=False)

    def teardown_method(self):
        self._patch.stop()

    def test_health_never_rate_limited(self):
        """/health must be accessible even if Redis is completely broken."""
        res = self.client.get("/health")
        assert res.status_code == 200


# ── Middleware: within limit ───────────────────────────────────────────────────

class TestRateLimitWithinLimit:
    def _make_client(self, count: int, limit: int):
        mock_redis = _make_mock_redis(count=count, ttl=55)

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        app = _make_rl_app(enabled=True, recommend_per_min=limit, read_per_min=60)
        client = TestClient(app)

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            return client, mock_redis

    def test_request_within_limit_passes(self):
        mock_redis = _make_mock_redis(count=5, ttl=55)

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            client = TestClient(app)
            res = client.post("/recommend")
        assert res.status_code == 200


# ── Middleware: over limit ────────────────────────────────────────────────────

def _make_over_limit_app(limit: int) -> FastAPI:
    """Build a FastAPI app with rate limiting enabled (limit applies to all routes)."""
    from core.rate_limit import RateLimitMiddleware, RateLimitConfig
    config = RateLimitConfig(enabled=True, recommend_per_minute=limit,
                              read_per_minute=limit, app_env="development")
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, config=config)

    @app.post("/recommend")
    def recommend():
        return {"ok": True}

    @app.get("/api/tracked-terms")
    def terms():
        return {"terms": []}

    return app


class TestRateLimitOverLimit:
    """Patches must stay active for the duration of the request (context-managed)."""

    def _run_over_limit(self, path: str, method: str = "POST",
                         count: int = 11, limit: int = 10):
        """Run one request with the mock active during the call, return response."""
        mock_redis = _make_mock_redis(count=count, ttl=30)

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        app = _make_over_limit_app(limit=limit)
        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            client = TestClient(app)
            if method == "POST":
                res = client.post(path)
            else:
                res = client.get(path)
        return res

    def test_over_limit_returns_429(self):
        res = self._run_over_limit("/recommend", count=11, limit=10)
        assert res.status_code == 429

    def test_429_has_retry_after_header(self):
        res = self._run_over_limit("/recommend", count=11, limit=10)
        assert res.status_code == 429
        assert "retry-after" in {k.lower() for k in res.headers}
        retry_after = int(res.headers.get("retry-after", "0"))
        assert retry_after >= 1

    def test_429_has_ratelimit_headers(self):
        res = self._run_over_limit("/recommend", count=11, limit=10)
        assert res.status_code == 429
        assert "x-ratelimit-limit" in {k.lower() for k in res.headers}
        assert "x-ratelimit-remaining" in {k.lower() for k in res.headers}
        assert res.headers.get("x-ratelimit-remaining") == "0"

    def test_429_response_is_json(self):
        res = self._run_over_limit("/recommend", count=11, limit=10)
        assert res.status_code == 429
        data = res.json()
        assert "detail" in data

    def test_recommend_uses_strict_limit(self):
        """count=5 over a limit of 3 → 429."""
        res = self._run_over_limit("/recommend", count=5, limit=3)
        assert res.status_code == 429

    def test_read_api_uses_read_limit(self):
        """count=11 is within read limit (60) → 200."""
        mock_redis = _make_mock_redis(count=11, ttl=30)

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.get("/api/tracked-terms")
        def terms():
            return {"terms": []}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            client = TestClient(app)
            res = client.get("/api/tracked-terms")
        assert res.status_code == 200


# ── Redis failure handling ────────────────────────────────────────────────────

class TestRateLimitRedisFailure:
    def _run_with_error(self, path: str, method: str, app_env: str):
        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env=app_env)
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        if method == "POST":
            @app.post(path)
            def handler():
                return {"ok": True}
        else:
            @app.get(path)
            def handler():
                return {"ok": True}

        async def raise_error(*args, **kwargs):
            raise ConnectionError("Redis down")

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=raise_error):
            client = TestClient(app, raise_server_exceptions=False)
            if method == "POST":
                return client.post(path)
            return client.get(path)

    def test_dev_redis_failure_fails_open(self):
        """In development, Redis failure → fail-open (request passes through)."""
        res = self._run_with_error("/recommend", "POST", "development")
        assert res.status_code == 200

    def test_prod_redis_failure_cost_bearing_returns_503(self):
        """In production, Redis failure on cost-bearing endpoint → 503."""
        res = self._run_with_error("/recommend", "POST", "production")
        assert res.status_code == 503

    def test_prod_redis_failure_cost_bearing_response_is_json(self):
        res = self._run_with_error("/recommend", "POST", "production")
        assert res.status_code == 503
        data = res.json()
        assert "detail" in data

    def test_prod_redis_failure_read_api_fails_open(self):
        """Non-cost-bearing endpoints fail-open even in production."""
        res = self._run_with_error("/api/tracked-terms", "GET", "production")
        assert res.status_code == 200


# ── Non-idempotent counter ────────────────────────────────────────────────────

class TestNonIdempotentCounter:
    def test_redis_eval_called_exactly_once_per_request(self):
        """Rate limit Lua script must not be retried on failure (non-idempotent)."""
        call_count = 0
        original_to_thread = None

        async def counting_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("first call fails")
            return [1, 55]  # would succeed on retry

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=counting_to_thread):
            client = TestClient(app, raise_server_exceptions=False)
            client.post("/recommend")

        # Must be called exactly once — no retry on failure
        assert call_count == 1


# ── Redis key privacy ─────────────────────────────────────────────────────────

class TestRedisKeyPrivacy:
    def test_redis_key_does_not_contain_raw_ip(self):
        """The Redis key must not contain a raw IP address."""
        raw_ip = "203.0.113.42"
        captured_key = []

        mock_redis = MagicMock()

        def capture_eval(script, numkeys, *args):
            captured_key.append(args[0])  # first KEYS arg
            return [1, 55]

        mock_redis.eval.side_effect = capture_eval

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            client = TestClient(app, headers={"X-Client-IP": raw_ip})
            client.post("/recommend")

        assert len(captured_key) == 1
        key = captured_key[0]
        assert raw_ip not in key, f"Raw IP found in Redis key: {key!r}"

    def test_anonymous_fallback_when_no_ip(self):
        """When no client IP is available, an anonymous bucket is used."""
        captured_key = []

        mock_redis = MagicMock()

        def capture_eval(script, numkeys, *args):
            captured_key.append(args[0])
            return [1, 55]

        mock_redis.eval.side_effect = capture_eval

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis), \
             patch("core.rate_limit._get_client_id", return_value="anon"):
            client = TestClient(app)
            client.post("/recommend")

        assert len(captured_key) == 1
        assert "anon" in captured_key[0]


# ── ToolExecutor protocol verification ───────────────────────────────────────

class TestRateLimitToolExecutor:
    """Verify that rate limiting uses ToolExecutor, not bare asyncio.wait_for."""

    def test_rate_limit_write_policy_exists(self):
        from core.tool_policy import TOOL_POLICIES
        assert "rate_limit_write" in TOOL_POLICIES

    def test_rate_limit_write_policy_timeout(self):
        from core.tool_policy import get_tool_policy
        p = get_tool_policy("rate_limit_write")
        assert p.timeout == 0.5

    def test_rate_limit_write_policy_no_retries(self):
        from core.tool_policy import get_tool_policy
        p = get_tool_policy("rate_limit_write")
        assert p.retries == 0

    def test_rate_limit_write_policy_not_idempotent(self):
        from core.tool_policy import get_tool_policy
        p = get_tool_policy("rate_limit_write")
        assert p.idempotent is False

    def test_no_direct_asyncio_wait_for_in_call(self):
        """RateLimitMiddleware.__call__ must not call asyncio.wait_for directly."""
        import inspect
        from core.rate_limit import RateLimitMiddleware
        source = inspect.getsource(RateLimitMiddleware.__call__)
        assert "asyncio.wait_for" not in source

    def test_tool_executor_execute_called_on_request(self):
        """tool_executor.execute must be invoked for rate-limited requests."""
        mock_redis = _make_mock_redis(count=1, ttl=55)

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="development")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        from tools.tool_executor import tool_executor
        from unittest.mock import patch as _patch
        with patch("tools.tool_executor.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("core.cache._client", mock_redis):
            with _patch.object(tool_executor, "execute", wraps=tool_executor.execute) as spy:
                client = TestClient(app)
                res = client.post("/recommend")
                assert res.status_code == 200
                assert spy.called, "tool_executor.execute was not called"
                call_kwargs = spy.call_args
                assert call_kwargs.kwargs.get("policy_name") == "rate_limit_write"

    def test_tool_result_failure_prod_cost_bearing_returns_503(self):
        """ToolResult(ok=False) on cost-bearing endpoint in production → 503."""
        async def always_fail(*args, **kwargs):
            raise ConnectionError("Redis down")

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="production")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=always_fail):
            client = TestClient(app, raise_server_exceptions=False)
            res = client.post("/recommend")
        assert res.status_code == 503

    def test_tool_result_failure_does_not_execute_downstream(self):
        """503 response must not reach route handler."""
        handler_called = []

        async def always_fail(*args, **kwargs):
            raise ConnectionError("Redis down")

        from core.rate_limit import RateLimitMiddleware, RateLimitConfig
        config = RateLimitConfig(enabled=True, recommend_per_minute=10,
                                  read_per_minute=60, app_env="production")
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, config=config)

        @app.post("/recommend")
        def recommend():
            handler_called.append(True)
            return {"ok": True}

        with patch("tools.tool_executor.asyncio.to_thread", side_effect=always_fail):
            client = TestClient(app, raise_server_exceptions=False)
            client.post("/recommend")
        assert not handler_called, "Route handler was executed despite 503 fail-close"
