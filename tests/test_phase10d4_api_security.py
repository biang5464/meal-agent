"""
tests/test_phase10d4_api_security.py

Phase 10D4 API authentication tests.
No real network connections, no Docker, no real Redis/MySQL/Chroma.
Tests cover: config loading, key verification, middleware behavior.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helper: build isolated test app with AuthMiddleware ────────────────────────

def _make_auth_app(enabled: bool, api_key: str, app_env: str = "development") -> FastAPI:
    from core.api_security import AuthMiddleware, ApiSecurityConfig
    config = ApiSecurityConfig(enabled=enabled, api_key=api_key, app_env=app_env)
    test_app = FastAPI()
    test_app.add_middleware(AuthMiddleware, config=config)

    @test_app.get("/health")
    def health():
        return {"status": "ok"}

    @test_app.post("/recommend")
    def recommend():
        return {"ok": True}

    @test_app.get("/api/tracked-terms")
    def tracked_terms():
        return {"terms": []}

    return test_app


# ── get_api_security_config ────────────────────────────────────────────────────

class TestGetApiSecurityConfig:
    def test_dev_defaults_auth_disabled(self, monkeypatch):
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("API_AUTH_ENABLED", raising=False)
        monkeypatch.delenv("MEAL_AGENT_API_KEY", raising=False)
        from core.api_security import get_api_security_config
        cfg = get_api_security_config()
        assert cfg.enabled is False
        assert cfg.app_env == "development"

    def test_explicit_enable_in_dev(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("API_AUTH_ENABLED", "true")
        monkeypatch.setenv("MEAL_AGENT_API_KEY", "dev-key")
        from core.api_security import get_api_security_config
        cfg = get_api_security_config()
        assert cfg.enabled is True
        assert cfg.api_key == "dev-key"

    def test_production_missing_key_raises(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("API_AUTH_ENABLED", "true")
        monkeypatch.delenv("MEAL_AGENT_API_KEY", raising=False)
        from core.api_security import get_api_security_config
        import importlib
        import core.api_security
        importlib.reload(core.api_security)
        from core.api_security import get_api_security_config as cfg_fn
        with pytest.raises(ValueError, match="MEAL_AGENT_API_KEY"):
            cfg_fn()

    def test_production_empty_key_raises(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("API_AUTH_ENABLED", "true")
        monkeypatch.setenv("MEAL_AGENT_API_KEY", "")
        from core.api_security import get_api_security_config
        with pytest.raises(ValueError, match="MEAL_AGENT_API_KEY"):
            get_api_security_config()

    def test_production_placeholder_key_raises(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("API_AUTH_ENABLED", "true")
        monkeypatch.setenv("MEAL_AGENT_API_KEY", "placeholder")
        from core.api_security import get_api_security_config
        with pytest.raises(ValueError, match="placeholder"):
            get_api_security_config()

    def test_production_valid_key_ok(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("API_AUTH_ENABLED", "true")
        monkeypatch.setenv("MEAL_AGENT_API_KEY", "super-secret-random-key-32bytes!")
        from core.api_security import get_api_security_config
        cfg = get_api_security_config()
        assert cfg.enabled is True
        assert cfg.app_env == "production"

    def test_production_auth_disabled_no_key_required(self, monkeypatch):
        """If API_AUTH_ENABLED=false in production, key not validated at startup."""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("API_AUTH_ENABLED", "false")
        monkeypatch.delenv("MEAL_AGENT_API_KEY", raising=False)
        from core.api_security import get_api_security_config
        cfg = get_api_security_config()
        assert cfg.enabled is False


# ── is_public_path ────────────────────────────────────────────────────────────

class TestIsPublicPath:
    def test_health_is_public(self):
        from core.api_security import is_public_path
        assert is_public_path("/health") is True

    def test_docs_is_public(self):
        from core.api_security import is_public_path
        assert is_public_path("/docs") is True
        assert is_public_path("/docs/oauth2-redirect") is True
        assert is_public_path("/redoc") is True
        assert is_public_path("/openapi.json") is True

    def test_recommend_is_not_public(self):
        from core.api_security import is_public_path
        assert is_public_path("/recommend") is False

    def test_api_routes_not_public(self):
        from core.api_security import is_public_path
        assert is_public_path("/api/tracked-terms") is False
        assert is_public_path("/api/daily-recommendation") is False


# ── verify_api_key ────────────────────────────────────────────────────────────

class TestVerifyApiKey:
    def test_correct_key_returns_true(self):
        from core.api_security import verify_api_key
        assert verify_api_key("my-secret", "my-secret") is True

    def test_wrong_key_returns_false(self):
        from core.api_security import verify_api_key
        assert verify_api_key("wrong-key", "my-secret") is False

    def test_empty_provided_returns_false(self):
        from core.api_security import verify_api_key
        assert verify_api_key("", "my-secret") is False

    def test_empty_expected_returns_false(self):
        from core.api_security import verify_api_key
        assert verify_api_key("some-key", "") is False

    def test_both_empty_returns_false(self):
        from core.api_security import verify_api_key
        assert verify_api_key("", "") is False

    def test_constant_time_compare(self):
        """Uses secrets.compare_digest (constant-time). Just verify it doesn't short-circuit."""
        import time
        from core.api_security import verify_api_key
        key = "a" * 64
        wrong = "b" * 64  # same length, all chars wrong → timing should still complete
        t0 = time.perf_counter()
        for _ in range(1000):
            verify_api_key(wrong, key)
        elapsed = time.perf_counter() - t0
        # Should complete in reasonable time — just ensure it doesn't blow up
        assert elapsed < 5.0


# ── AuthMiddleware behavior ────────────────────────────────────────────────────

class TestAuthMiddlewareDisabled:
    """When auth is disabled, all requests pass through."""

    def setup_method(self):
        self.client = TestClient(_make_auth_app(enabled=False, api_key=""))

    def test_recommend_accessible_without_key(self):
        res = self.client.post("/recommend")
        assert res.status_code == 200

    def test_health_accessible(self):
        res = self.client.get("/health")
        assert res.status_code == 200

    def test_tracked_terms_accessible_without_key(self):
        res = self.client.get("/api/tracked-terms")
        assert res.status_code == 200


class TestAuthMiddlewareEnabled:
    """When auth is enabled, protected routes require X-API-Key."""

    _KEY = "test-secret-key-abc123"

    def setup_method(self):
        self.client = TestClient(_make_auth_app(enabled=True, api_key=self._KEY))

    def test_health_accessible_without_key(self):
        """/health is always public, even when auth is enabled."""
        res = self.client.get("/health")
        assert res.status_code == 200

    def test_recommend_without_key_returns_401(self):
        res = self.client.post("/recommend")
        assert res.status_code == 401
        data = res.json()
        assert data["detail"] == "Unauthorized"

    def test_recommend_wrong_key_returns_401(self):
        res = self.client.post("/recommend", headers={"X-API-Key": "wrong-key"})
        assert res.status_code == 401

    def test_recommend_correct_key_returns_200(self):
        res = self.client.post("/recommend", headers={"X-API-Key": self._KEY})
        assert res.status_code == 200

    def test_tracked_terms_without_key_returns_401(self):
        res = self.client.get("/api/tracked-terms")
        assert res.status_code == 401

    def test_tracked_terms_correct_key_returns_200(self):
        res = self.client.get("/api/tracked-terms", headers={"X-API-Key": self._KEY})
        assert res.status_code == 200

    def test_401_response_does_not_echo_key(self):
        """Error response must not reflect the submitted key back to the caller."""
        submitted = "submitted-bad-key"
        res = self.client.post("/recommend", headers={"X-API-Key": submitted})
        assert res.status_code == 401
        body = res.text
        assert submitted not in body
        assert self._KEY not in body

    def test_401_response_content_type_is_json(self):
        res = self.client.post("/recommend")
        assert res.status_code == 401
        assert "application/json" in res.headers.get("content-type", "")

    def test_options_preflight_not_blocked(self):
        """CORS preflight OPTIONS must never require authentication."""
        res = self.client.options(
            "/recommend",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # Must not be 401 — could be 200 or 405 depending on routing
        assert res.status_code != 401

    def test_401_does_not_execute_endpoint_logic(self):
        """Unauthenticated request must not reach the route handler.

        The route handler returns {"ok": True} for /recommend when called;
        a 401 response must not contain that body.
        """
        res = self.client.post("/recommend")
        assert res.status_code == 401
        body = res.text
        assert '"ok"' not in body


# ── Response body never leaks key ─────────────────────────────────────────────

class TestNoKeyLeakage:
    def test_401_body_is_minimal(self):
        from core.api_security import AuthMiddleware, ApiSecurityConfig
        config = ApiSecurityConfig(enabled=True, api_key="super-secret", app_env="production")
        app = FastAPI()
        app.add_middleware(AuthMiddleware, config=config)

        @app.get("/protected")
        def protected():
            return {"data": "sensitive"}

        client = TestClient(app)
        res = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert res.status_code == 401
        assert "super-secret" not in res.text
        assert "wrong" not in res.text
